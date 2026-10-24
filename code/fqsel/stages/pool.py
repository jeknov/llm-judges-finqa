"""Stage `pool` (Spec 1, Pool construction): load, exclude, sample splits, remove leakage, build negatives.

Order: (1) load FinQA, TAT-QA and FinanceBench; (2) exclusions — empty gold, failed verification, token
budget, exact duplicates — applied before any sampling; (3) per dataset, sample probe, then fixed_demo
(disjoint from probe) from dev, and test from test, stratified by answer_type in proportion to the pool
candidates; (4) drop pool candidates leaking into a sampled probe or test item; (5) the pool is every
surviving train item (or a stratified sample under profile=tiny); (6) build the validation negatives.

Outputs: data/pool.parquet, data/dedup_log.csv, data/exclusions.csv, data/split_manifest.json.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from fqsel.backends import load_tokenizer
from fqsel.config import raw_dir, root_dir
from fqsel.data.dedup import exact_duplicates, leakage
from fqsel.data.financebench import load_financebench
from fqsel.data.finqa import load_finqa
from fqsel.data.splits import SamplingError, allocate, composition, sample_stratified
from fqsel.data.tatqa import load_tatqa
from fqsel.ids import derived_id, rng
from fqsel.io import write_table
from fqsel.perturb import make_wrong_answer
from fqsel.prompts import render_item
from fqsel.schemas import ITEM_COLUMNS
from fqsel.verify import verify_item


def load_all(cfg: DictConfig) -> tuple[list[dict], int]:
    raw = raw_dir(cfg)
    files = cfg.data.raw_files
    items = load_finqa(raw, dict(files.finqa)) + load_tatqa(raw, dict(files.tatqa))
    fb, fb_dropped = load_financebench(raw, dict(files.financebench))
    return items + fb, fb_dropped


def apply_exclusions(items: list[dict], cfg: DictConfig) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (survivors, exclusion log rows, dedup log rows). Sets qa_tokens on every item."""
    tokenizer = load_tokenizer(cfg)
    budget = cfg.data.token_budget
    survivors, excluded = [], []
    for it in items:
        it["qa_tokens"] = len(tokenizer(render_item(it, "qa"), add_special_tokens=False)["input_ids"])
        limit = budget.test if it["source_split"] == "test" else budget.pool
        reason = None
        if not it["gold_answer"].strip():
            reason = "empty_gold"
        elif verify_item(it).verified is False:
            reason = "unverified"
        elif it["qa_tokens"] > limit:
            reason = "token_budget"
        if reason:
            excluded.append({"item_id": it["item_id"], "dataset": it["dataset"],
                             "source_split": it["source_split"], "reason": reason})
        else:
            survivors.append(it)
    dups = exact_duplicates(survivors)
    dropped = {d for d, _ in dups}
    by_id = {it["item_id"]: it for it in survivors}
    for d, _ in dups:
        excluded.append({"item_id": d, "dataset": by_id[d]["dataset"],
                         "source_split": by_id[d]["source_split"], "reason": "exact_duplicate"})
    dedup_log = [{"dropped_id": d, "matched_id": k, "kind": "exact", "jaccard": 1.0} for d, k in dups]
    return [it for it in survivors if it["item_id"] not in dropped], excluded, dedup_log


def build_negatives(pool: list[dict], cfg: DictConfig) -> list[dict]:
    ncfg = cfg.data.negatives
    strata: dict[str, list[dict]] = {}
    for it in pool:
        strata.setdefault(f"{it['dataset']}|{it['answer_type']}", []).append(it)
    alloc = allocate({k: len(v) for k, v in strata.items()}, int(ncfg.n))
    negatives = []
    for stratum in sorted(alloc):
        members = sorted(strata[stratum], key=lambda it: it["item_id"])
        order = rng(cfg.global_seed, "negatives", stratum).permutation(len(members))
        made = 0
        for i in order:
            if made == alloc[stratum]:
                break
            original = members[i]
            wrong = make_wrong_answer(original, ncfg.tag, list(ncfg.factors), int(ncfg.max_redraws))
            if wrong is None:
                continue  # Spec 1: drop the item and draw a replacement
            neg = dict(original)
            neg.update(
                item_id=derived_id(original["item_id"], "neg"),
                gold_answer=wrong["gold_answer"],
                gold_value=json.dumps(wrong["gold_value"], sort_keys=True),
                is_derived=True,
                base_item_id=original["item_id"],
                perturbation=wrong["perturbation"],
            )
            negatives.append(neg)
            made += 1
        if made < alloc[stratum]:
            raise SamplingError(f"negatives: stratum {stratum} produced {made} of {alloc[stratum]}")
    return negatives


def run(cfg: DictConfig) -> None:
    items, fb_dropped = load_all(cfg)
    survivors, excluded, dedup_log = apply_exclusions(items, cfg)
    sizes = cfg.data.sizes
    gs = cfg.global_seed
    chosen: dict[str, str] = {}
    pool_candidates: dict[str, list[dict]] = {}
    manifest: dict = {"tatqa_test_source": cfg.data.raw_files.tatqa.test,
                      "financebench_filtered_non_numeric_or_no_evidence": fb_dropped,
                      "datasets": {}}

    for ds in cfg.data.datasets:
        mine = [it for it in survivors if it["dataset"] == ds]
        train = [it for it in mine if it["source_split"] == "train"]
        dev = [it for it in mine if it["source_split"] == "dev"]
        test = [it for it in mine if it["source_split"] == "test"]
        comp = composition(train, "answer_type")
        probe = sample_stratified(dev, comp, sizes.probe_per_dataset, "answer_type", [gs, "probe", ds])
        probe_ids = set(probe)
        rest = [it for it in dev if it["item_id"] not in probe_ids]
        fixed = sample_stratified(rest, comp, sizes.fixed_demo_per_dataset, "answer_type", [gs, "fixed_demo", ds])
        test_ids = sample_stratified(test, comp, sizes.test_per_dataset, "answer_type", [gs, "test", ds])
        for split, ids in (("probe", probe), ("fixed_demo", fixed), ("test", test_ids)):
            chosen.update({i: split for i in ids})
        pool_candidates[ds] = train
        manifest["datasets"][ds] = {"pool_composition": comp}

    fb = sorted((it for it in survivors if it["dataset"] == "financebench"), key=lambda it: it["item_id"])
    if sizes.financebench_max is not None and len(fb) > sizes.financebench_max:
        order = rng(gs, "test", "financebench").permutation(len(fb))
        fb = sorted((fb[i] for i in order[: sizes.financebench_max]), key=lambda it: it["item_id"])
    chosen.update({it["item_id"]: "test" for it in fb})

    by_id = {it["item_id"]: it for it in survivors}
    held_out = [by_id[i] for i in chosen]
    all_train = [it for ds in pool_candidates for it in pool_candidates[ds]]
    dd = cfg.data.dedup
    leaks = leakage(all_train, held_out, dd.jaccard_threshold, dd.num_perm, dd.shingle_words)
    leaked = {p for p, _, _ in leaks}
    for p, o, j in leaks:
        excluded.append({"item_id": p, "dataset": by_id[p]["dataset"], "source_split": "train", "reason": "leakage"})
        dedup_log.append({"dropped_id": p, "matched_id": o, "kind": "leakage", "jaccard": j})

    for ds, train in pool_candidates.items():
        train = [it for it in train if it["item_id"] not in leaked]
        if sizes.pool_per_dataset is not None and len(train) > sizes.pool_per_dataset:
            ids = sample_stratified(train, composition(train, "answer_type"), sizes.pool_per_dataset,
                                    "answer_type", [gs, "pool", ds])
        else:
            ids = [it["item_id"] for it in train]
        chosen.update({i: "pool" for i in ids})

    selected = []
    for item_id, split in chosen.items():
        it = dict(by_id[item_id])
        it["split"] = split
        selected.append(it)
    pool_items = [it for it in selected if it["split"] == "pool"]
    negatives = build_negatives(pool_items, cfg)
    rows = sorted(selected + negatives, key=lambda it: it["item_id"])
    df = pd.DataFrame(rows, columns=list(ITEM_COLUMNS))
    write_table("pool", df, cfg)

    root = root_dir(cfg)
    pd.DataFrame(dedup_log, columns=["dropped_id", "matched_id", "kind", "jaccard"]).sort_values(
        ["kind", "dropped_id"]).to_csv(root / "data" / "dedup_log.csv", index=False)
    counts = Counter((e["dataset"], e["source_split"], e["reason"]) for e in excluded)
    pd.DataFrame([{"dataset": d, "source_split": s, "reason": r, "n": n} for (d, s, r), n in sorted(counts.items())],
                 columns=["dataset", "source_split", "reason", "n"]).to_csv(root / "data" / "exclusions.csv", index=False)
    loaded = Counter((it["dataset"], it["source_split"]) for it in items)
    manifest["loaded"] = {f"{d}/{s}": n for (d, s), n in sorted(loaded.items())}
    manifest["split_sizes"] = {f"{d}/{s}": n for (d, s), n in sorted(
        Counter((it["dataset"], it["split"]) for it in selected).items())}
    manifest["negatives"] = len(negatives)
    manifest["sizes_config"] = OmegaConf.to_container(sizes)
    (root / "data" / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"pool: wrote {len(rows)} rows ({len(negatives)} negatives); {len(excluded)} exclusions")
