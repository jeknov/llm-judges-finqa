"""Stages pool, verify and q3 on synthetic raw data (Spec 1 acceptance; Implementation Plan M2 tests)."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from fqsel.data.dedup import item_text, jaccard, shingles
from fqsel.data.splits import allocate
from fqsel.io import read_table, table_path
from fqsel.match import answers_match
from fqsel.perturb import gold_for_match
from fqsel.stages import run_stage


@pytest.fixture(scope="module")
def built(tmp_path_factory, synth_raw, model_cache):
    from fqsel.config import load_config

    root = tmp_path_factory.mktemp("pool_run")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={model_cache}"])
    for stage in ("pool", "verify", "q3"):
        run_stage(stage, cfg)
    return cfg, read_table("pool", cfg)


@pytest.mark.spec("S1-02")
def test_ids_unique_and_one_split_each(built):
    _, pool = built
    assert pool["item_id"].is_unique
    originals = pool[~pool["is_derived"]]
    assert originals.groupby("item_id")["split"].nunique().max() == 1
    assert set(pool.loc[pool["is_derived"], "split"]) == {"pool"}


@pytest.mark.spec("S1-12")
def test_split_sizes_reach_targets(built):
    cfg, pool = built
    sizes = pool[~pool["is_derived"]].groupby(["dataset", "split"]).size().to_dict()
    for ds in ("finqa", "tatqa"):
        assert sizes[(ds, "probe")] == cfg.data.sizes.probe_per_dataset
        assert sizes[(ds, "fixed_demo")] == cfg.data.sizes.fixed_demo_per_dataset
        assert sizes[(ds, "test")] == cfg.data.sizes.test_per_dataset
        assert sizes[(ds, "pool")] == cfg.data.sizes.pool_per_dataset
    assert sizes[("financebench", "test")] == cfg.data.sizes.financebench_max


def test_probe_stratified_like_pool_candidates(built):
    cfg, pool = built
    manifest = json.loads((table_path(cfg, "pool").parent / "split_manifest.json").read_text())
    for ds in ("finqa", "tatqa"):
        comp = manifest["datasets"][ds]["pool_composition"]
        expected = allocate(comp, cfg.data.sizes.probe_per_dataset)
        got = pool[(pool["dataset"] == ds) & (pool["split"] == "probe")]["answer_type"].value_counts().to_dict()
        assert got == {k: v for k, v in expected.items() if v}


@pytest.mark.spec("S1-11")
def test_token_budget(built):
    cfg, pool = built
    test = pool[pool["split"] == "test"]
    other = pool[pool["split"] != "test"]
    assert (test["qa_tokens"] <= cfg.data.token_budget.test).all()
    assert (other["qa_tokens"] <= cfg.data.token_budget.pool).all()
    assert (pool["qa_tokens"] > 0).all()


@pytest.mark.spec("S1-03")
def test_no_pool_item_leaks_into_probe_or_test(built):
    cfg, pool = built
    n = cfg.data.dedup.shingle_words
    held = [shingles(item_text(r), n) for _, r in pool[pool["split"].isin(["probe", "test"])].iterrows()]
    for _, r in pool[(pool["split"] == "pool") & ~pool["is_derived"]].iterrows():
        sh = shingles(item_text(r), n)
        assert all(jaccard(sh, h) < cfg.data.dedup.jaccard_threshold for h in held)


@pytest.mark.spec("S1-15")
def test_negatives(built):
    cfg, pool = built
    neg = pool[pool["is_derived"]]
    assert len(neg) == cfg.data.negatives.n
    assert neg["item_id"].str.endswith("#neg").all()
    by_id = pool.set_index("item_id")
    for _, n in neg.iterrows():
        base = by_id.loc[n["base_item_id"]]
        assert base["split"] == "pool" and not base["is_derived"]
        ref = gold_for_match(base)
        cand = json.loads(n["gold_value"]) if isinstance(ref, list) else n["gold_answer"]
        assert not answers_match(cand, ref, base["answer_type"])
        assert n["context_text"] == base["context_text"] and n["question"] == base["question"]


def test_verified_table(built):
    cfg, pool = built
    ver = read_table("verified", cfg).set_index("item_id")
    assert set(ver.index) == set(pool["item_id"])
    originals = pool[~pool["is_derived"] & (pool["dataset"] != "financebench")]["item_id"]
    assert ver.loc[originals, "verified"].map(lambda v: v is True).all()
    fb = pool[pool["dataset"] == "financebench"]["item_id"]
    assert ver.loc[fb, "verified"].isna().all()
    finqa_neg = pool[pool["is_derived"] & (pool["dataset"] == "finqa")]["item_id"]
    assert ver.loc[finqa_neg, "failure_reason"].eq("mismatch").all()


def test_q3_one_check_per_pool_item_and_negative(built):
    cfg, pool = built
    q3 = read_table("q3_checks", cfg)
    targets = set(pool.loc[pool["split"] == "pool", "item_id"])
    assert set(q3["item_id"]) == targets
    assert q3["validated"].isna().all()
    assert {"template", "llm"} <= set(q3["source"])


@pytest.mark.spec("S1-10")
def test_rebuild_from_raw_is_byte_identical(built, make_cfg, tmp_path):
    cfg, _ = built
    other = make_cfg(root=tmp_path / "again")
    run_stage("pool", other)
    assert table_path(other, "pool").read_bytes() == table_path(cfg, "pool").read_bytes()


def test_exclusions_conserve_counts(make_cfg, tmp_path):
    cfg = make_cfg("data.sizes.pool_per_dataset=null", root=tmp_path / "all")
    run_stage("pool", cfg)
    pool = read_table("pool", cfg)
    base = table_path(cfg, "pool").parent
    manifest = json.loads((base / "split_manifest.json").read_text())
    excl = pd.read_csv(base / "exclusions.csv")
    for ds in ("finqa", "tatqa"):
        loaded = manifest["loaded"][f"{ds}/train"]
        excluded = int(excl[(excl["dataset"] == ds) & (excl["source_split"] == "train")]["n"].sum())
        kept = int(((pool["dataset"] == ds) & (pool["split"] == "pool") & ~pool["is_derived"]).sum())
        assert loaded == kept + excluded


def test_planted_duplicate_and_leak_are_removed(make_cfg, synth_raw, tmp_path, built):
    cfg0, pool0 = built
    raw = tmp_path / "raw"
    shutil.copytree(synth_raw, raw)
    test_records = json.loads((raw / "finqa" / "test.json").read_text())
    sampled = set(pool0.loc[(pool0["dataset"] == "finqa") & (pool0["split"] == "test"), "item_id"])
    target = next(r for r in test_records if f"finqa:{r['id']}" in sampled)
    train = json.loads((raw / "finqa" / "train.json").read_text())
    # A long context of distinct sentences, so that one changed question word keeps Jaccard above 0.9.
    long_context = [f"segment {k} reported figure {k * 7} for quarter {k % 4} and region {k * 3} ." for k in range(25)]
    target_long = dict(target, pre_text=long_context)
    exact = json.loads(json.dumps(target_long)) | {"id": "ZZZ/dup/page_1.pdf-1"}
    near = json.loads(json.dumps(target_long)) | {"id": "ZZZ/near/page_1.pdf-1"}
    near["qa"] = dict(near["qa"], question=near["qa"]["question"] + " please")
    test_records = [target_long if r is target else r for r in test_records]
    (raw / "finqa" / "test.json").write_text(json.dumps(test_records))
    (raw / "finqa" / "train.json").write_text(json.dumps(train + [exact, near]))
    cfg = make_cfg(root=tmp_path / "dups", raw=raw)
    run_stage("pool", cfg)
    log = pd.read_csv(table_path(cfg, "pool").parent / "dedup_log.csv")
    assert f"finqa:{target['id']}" in set(read_table("pool", cfg).query("split == 'test'")["item_id"])
    # Exact copy: dropped by exact dedup, the test item (first by item_id) is kept.
    exact_rows = log[log["kind"] == "exact"]
    assert ("finqa:ZZZ/dup/page_1.pdf-1", f"finqa:{target['id']}") in set(
        zip(exact_rows["dropped_id"], exact_rows["matched_id"], strict=True))
    # Near copy: dropped by leakage removal against the sampled test item.
    leak = log[log["kind"] == "leakage"].set_index("dropped_id")
    assert leak.loc["finqa:ZZZ/near/page_1.pdf-1", "matched_id"] == f"finqa:{target['id']}"
    assert leak.loc["finqa:ZZZ/near/page_1.pdf-1", "jaccard"] >= 0.9
    pool = read_table("pool", cfg)
    assert not set(pool["item_id"]) & {"finqa:ZZZ/near/page_1.pdf-1", "finqa:ZZZ/dup/page_1.pdf-1"}
