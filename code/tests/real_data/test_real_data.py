"""Checks on the downloaded raw data (marker real_data; run with `pytest -m real_data`).

Results of these runs are recorded in the Implementation Plan's Verification log.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import pandas as pd
import pytest

from fqsel.data.dedup import item_text, shingles
from fqsel.data.financebench import load_financebench
from fqsel.data.finqa import load_finqa
from fqsel.data.tatqa import load_tatqa
from fqsel.schemas import ITEM_COLUMNS, TABLES
from fqsel.verify import verify_item
from tests.conftest import REAL_RAW, requires_real_data

pytestmark = [pytest.mark.real_data, requires_real_data()]


@pytest.mark.spec("S1-01")
def test_manifest_hashes_match():
    lines = [ln for ln in (REAL_RAW / "MANIFEST").read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 8
    for line in lines:
        digest, rel = line.split(maxsplit=1)
        assert hashlib.sha256((REAL_RAW / rel).read_bytes()).hexdigest() == digest, rel


@pytest.mark.spec("S1-04")
def test_finqa_executor_coverage_on_train():
    items = load_finqa(REAL_RAW, {"train": "finqa/train.json"})
    failures = Counter()
    for it in items:
        v = verify_item(it)
        if not v.verified:
            failures[v.failure_reason] += 1
    rate = 1 - sum(failures.values()) / len(items)
    print(f"FinQA train executor coverage {rate:.4f} over {len(items)} items; failures {dict(failures)}")
    assert rate >= 0.99


@pytest.mark.spec("S1-05")
def test_tatqa_evaluator_coverage_on_train():
    items = load_tatqa(REAL_RAW, {"train": "tatqa/tatqa_dataset_train.json"})
    ok, n = Counter(), Counter()
    for it in items:
        group = "arith" if it["source_answer_type"] == "arithmetic" else (
            "span" if it["source_answer_type"] in ("span", "multi-span") else "count")
        n[group] += 1
        ok[group] += bool(verify_item(it).verified)
    rates = {g: ok[g] / n[g] for g in n}
    print(f"TAT-QA train verification rates {rates} over {dict(n)}")
    assert rates["arith"] >= 0.97 and rates["span"] >= 0.99


def test_all_real_items_fit_the_pool_schema():
    items = load_finqa(REAL_RAW, {s: f"finqa/{s}.json" for s in ("train", "dev", "test")})
    items += load_tatqa(REAL_RAW, {"train": "tatqa/tatqa_dataset_train.json", "dev": "tatqa/tatqa_dataset_dev.json",
                                   "test": "tatqa/tatqa_dataset_test_gold.json"})
    fb, dropped = load_financebench(REAL_RAW, {"test": "financebench/financebench_open_source.jsonl"})
    df = pd.DataFrame(items + fb, columns=list(ITEM_COLUMNS))
    df["split"] = "pool"
    df["qa_tokens"] = 1
    TABLES["pool"].pandera().validate(df, lazy=True)
    print(f"{len(df)} items; FinanceBench kept {len(fb)}, filtered {dropped}")
    assert len(fb) >= 1


@pytest.mark.spec("S1-03")
def test_leakage_exact_check_on_real_run():
    """After a real-data pool build (profile run into data/), confirm with exact Jaccard over an inverted
    shingle index, independent of MinHash, that no pool item reaches 0.9 with a probe or test item."""
    import os

    from fqsel.config import load_config
    from fqsel.io import read_table, table_path

    root = os.environ.get("FQSEL_REAL_ROOT")  # a real-data build elsewhere, e.g. a scratch run
    cfg = load_config([f"paths.root={root}"] if root else [])
    if not table_path(cfg, "pool").exists():
        pytest.skip("no real-data pool build found (set FQSEL_REAL_ROOT or build into data/)")
    pool = read_table("pool", cfg)
    n, thr = cfg.data.dedup.shingle_words, cfg.data.dedup.jaccard_threshold
    held = pool[pool["split"].isin(["probe", "test"])]
    index: dict[str, list[int]] = defaultdict(list)
    held_sizes = []
    for k, (_, row) in enumerate(held.iterrows()):
        sh = shingles(item_text(row), n)
        held_sizes.append(len(sh))
        for s in sh:
            index[s].append(k)
    for _, row in pool[(pool["split"] == "pool") & ~pool["is_derived"]].iterrows():
        sh = shingles(item_text(row), n)
        overlap = Counter(k for s in sh for k in index.get(s, ()))
        for k, inter in overlap.items():
            assert inter / (len(sh) + held_sizes[k] - inter) < thr, (row["item_id"], held.iloc[k]["item_id"])


@pytest.fixture(scope="module")
def base_tokenizer():
    """The model.base tokenizer at its pinned revision (downloads tokenizer files only, no weights)."""
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from fqsel.config import CONF_DIR

    base = OmegaConf.load(CONF_DIR / "model" / "base.yaml")
    return AutoTokenizer.from_pretrained(base.name, revision=base.revision)


@pytest.mark.spec("S2-09")
def test_base_chat_template_has_no_reasoning_block(base_tokenizer):
    """Spec 2: the chat template renders no thinking/reasoning block, in either prompt format."""
    from fqsel.prompts import build_messages
    from tests.fixtures.golden_items import golden_items

    item = golden_items()["tatqa:q-arith"]
    for fmt, kw in (("direct", {"with_answer": True}), ("reason", {})):
        text = base_tokenizer.apply_chat_template(build_messages(item, fmt, **kw), tokenize=False,
                                                  add_generation_prompt=not kw)
        lowered = text.lower()
        assert "<think" not in lowered and "reasoning" not in lowered.split("<|im_start|>user")[0]
        assert "<|im_start|>user\n" in text
    direct = base_tokenizer.apply_chat_template(build_messages(item, "direct", with_answer=True), tokenize=False)
    assert direct.endswith("<|im_start|>assistant\nAnswer: 12.76 percent" + base_tokenizer.eos_token)


def test_token_budget_with_base_tokenizer(base_tokenizer):
    """Real exclusion rates under the Spec 1 budget; recorded in the Verification log."""
    from fqsel.prompts import render_item

    items = load_finqa(REAL_RAW, {s: f"finqa/{s}.json" for s in ("train", "dev", "test")})
    items += load_tatqa(REAL_RAW, {"train": "tatqa/tatqa_dataset_train.json", "dev": "tatqa/tatqa_dataset_dev.json",
                                   "test": "tatqa/tatqa_dataset_test_gold.json"})
    fb, _ = load_financebench(REAL_RAW, {"test": "financebench/financebench_open_source.jsonl"})
    total, over = Counter(), Counter()
    for it in items + fb:
        n = len(base_tokenizer(render_item(it, "qa"), add_special_tokens=False)["input_ids"])
        key = (it["dataset"], it["source_split"])
        total[key] += 1
        over[key] += n > (4096 if it["source_split"] == "test" else 2048)
    rates = {f"{d}/{s}": over[(d, s)] / total[(d, s)] for d, s in sorted(total)}
    print(f"over-budget rates with model.base tokenizer: {rates}")
    assert all(r < 0.05 for r in rates.values())
    fb_kept = total[("financebench", "test")] - over[("financebench", "test")]
    assert fb_kept >= 50, "Spec 4 needs >= 50 FinanceBench items to hypothesis-test out-of-distribution results"



@pytest.mark.spec("S2-02")
def test_answer_span_with_the_base_tokenizer(base_tokenizer):
    """The 20 hand-picked answers, through OLMo 3's own chat template and tokenizer."""
    from fqsel.nll import answer_span
    from tests.unit.test_nll import ANSWERS, KnownLogprobs, _messages

    backend = KnownLogprobs(base_tokenizer)
    adjusted = 0
    for answer in ANSWERS:
        span = answer_span(backend, _messages(answer))
        adjusted += span.boundary_adjusted
        if not span.boundary_adjusted:
            assert base_tokenizer.decode(span.ids[span.start: span.end]) == " " + answer
    assert adjusted == 0, f"{adjusted} answers needed boundary adjustment"
