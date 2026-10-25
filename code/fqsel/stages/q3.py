"""Stage `q3` (Spec 1, Q3 check generation): one check per pool item and validation negative.

Items whose LLM fallback fails twice are dropped and logged to data/q3_dropped.csv; they have no Q3 row,
and the jev stage scores only items with a check. `validated` stays null until the human validation
labels are imported (python -m fqsel annotate import q3_validation).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from omegaconf import DictConfig

from fqsel.checks import generate_check
from fqsel.clients.strong_llm import make_strong
from fqsel.config import root_dir
from fqsel.io import read_table, write_table

COLUMNS = ["item_id", "check_text", "check_type", "source", "template_op", "generator_model", "validated"]


def run(cfg: DictConfig) -> None:
    pool = read_table("pool", cfg)
    targets = pool[pool["split"] == "pool"]
    llm = make_strong(cfg)
    rows, dropped = [], []
    items = [item for _, item in targets.iterrows()]
    # each check depends only on its own item and seed, so parallel requests give the same table
    with ThreadPoolExecutor(max_workers=int(cfg.model.strong.max_workers)) as pool:
        results = list(pool.map(lambda item: generate_check(item, cfg, llm), items))
    for item, check in zip(items, results, strict=True):
        if check is None:
            dropped.append({"item_id": item["item_id"], "reason": "llm_fallback_failed_twice"})
            continue
        rows.append({"item_id": item["item_id"], **check, "validated": None})
    write_table("q3_checks", pd.DataFrame(rows, columns=COLUMNS).sort_values("item_id"), cfg)
    pd.DataFrame(dropped, columns=["item_id", "reason"]).to_csv(root_dir(cfg) / "data" / "q3_dropped.csv", index=False)
    n_tpl = sum(r["source"] == "template" for r in rows)
    print(f"q3: {len(rows)} checks ({n_tpl} template, {len(rows) - n_tpl} llm); {len(dropped)} dropped")
