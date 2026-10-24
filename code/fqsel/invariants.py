"""`python -m fqsel check`: re-validate every existing output and the cross-table invariants
(Implementation Plan, Testing policy rule 7). Returns a list of problems; empty means all good."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from fqsel.gates import gates_path, read_gates
from fqsel.ids import base_id
from fqsel.io import read_table, table_path
from fqsel.schemas import TABLES


def _request_problems(long: pd.DataFrame, reqs: pd.DataFrame) -> list[str]:
    """Spec 1: each scored item has one batched six-question request per repeat (or six single-question
    requests per repeat), and every request ID in the scores appears in jev_requests."""
    problems = []
    used = {r for ids in long["request_id"] for r in ids.split(";")}
    if used != set(reqs["request_id"]):
        problems.append("jev_requests: request IDs differ from those referenced by jev_scores_long")
    n_calls = long.groupby("item_id")["n_calls"].first()
    batched = bool(long["batched"].all())
    if long["batched"].nunique() != 1:
        problems.append("jev_scores_long: mixed batched and single-question rows")
    expected_q = 6 if batched else 1
    if (reqs["n_questions"] != expected_q).any():
        problems.append(f"jev_requests: requests without exactly {expected_q} question(s)")
    per_item = reqs.groupby("item_id").size()
    expected = n_calls * (1 if batched else 6)
    if not per_item.reindex(expected.index).fillna(0).astype(int).equals(expected.astype(int)):
        problems.append("jev_requests: an item does not have the expected number of requests")
    return problems


def run_checks(cfg: DictConfig) -> list[str]:
    problems: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, tdef in TABLES.items():
        if "{" in tdef.path or not table_path(cfg, name).exists():
            continue
        try:
            tables[name] = read_table(name, cfg)
        except Exception as e:  # noqa: BLE001 - report every failure, do not stop at the first
            problems.append(f"{name}: schema validation failed: {e}")

    pool = tables.get("pool")
    if pool is not None:
        ids = set(pool["item_id"])
        derived = pool[pool["is_derived"]]
        for _, row in derived.iterrows():
            if row["base_item_id"] != base_id(row["item_id"]) or row["base_item_id"] not in ids:
                problems.append(f"pool: derived item {row['item_id']} has no valid base item")
        originals = pool[~pool["is_derived"]]
        if (originals["base_item_id"] != originals["item_id"]).any():
            problems.append("pool: an original item has base_item_id != item_id")
        over = originals[
            ((originals["split"] == "test") & (originals["qa_tokens"] > cfg.data.token_budget.test))
            | ((originals["split"] != "test") & (originals["qa_tokens"] > cfg.data.token_budget.pool))
        ]
        if len(over):
            problems.append(f"pool: {len(over)} items exceed the token budget")
        targets = set(pool.loc[pool["split"] == "pool", "item_id"])
        q3 = tables.get("q3_checks")
        if q3 is not None and not set(q3["item_id"]) <= targets:
            problems.append("q3_checks: rows for items that are not pool items or negatives")
        long = tables.get("jev_scores_long")
        if long is not None:
            counts = long.groupby("item_id").size()
            if (counts != 6).any():
                problems.append(f"jev_scores_long: {int((counts != 6).sum())} items without exactly 6 rows")
            if ((long["p_yes"] + long["p_no"] - 1).abs() > 1e-6).any():
                problems.append("jev_scores_long: p_yes + p_no differs from 1")
            if q3 is not None and set(long["item_id"]) != set(q3["item_id"]):
                problems.append("jev_scores_long: scored items differ from items with a Q3 check")
            reqs = tables.get("jev_requests")
            if reqs is not None:
                problems += _request_problems(long, reqs)
        verified = tables.get("verified")
        if verified is not None and set(verified["item_id"]) != ids:
            problems.append("verified: items differ from pool.parquet")

    if gates_path(cfg).exists():
        try:
            read_gates(cfg)
        except Exception as e:  # noqa: BLE001
            problems.append(f"gates.json: {e}")
    return problems
