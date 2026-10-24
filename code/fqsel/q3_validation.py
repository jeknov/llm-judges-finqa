"""Human validation of Q3 checks (Spec 1, Validation), through the CSV sheets in fqsel.annotate.

- q3_validation: validation_n LLM-generated checks, stratified by dataset; the generator is accepted if at
  least accept_rate of them are labeled "yes" by both annotators. Labels set q3_checks.validated.
- q3_templates: template_check_n template checks per template operation, for the hand-check that each
  template must pass.
"""

from __future__ import annotations

import json

import pandas as pd
from omegaconf import DictConfig

from fqsel.annotate import export_sheets, import_sheets
from fqsel.config import root_dir
from fqsel.data.splits import allocate
from fqsel.ids import rng
from fqsel.io import read_table, write_table
from fqsel.prompts import render_item


def _sheet_items(checks: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = checks.merge(pool, on="item_id")
    merged["item_text"] = [render_item(row, "judge") for _, row in merged.iterrows()]
    return merged[["item_id", "item_text", "check_text"]]


def _sample(ids: list[str], k: int, *seed_parts) -> list[str]:
    ids = sorted(ids)
    return sorted(ids[i] for i in rng(*seed_parts).permutation(len(ids))[:k])


def export_task(task: str, cfg: DictConfig) -> list:
    checks, pool = read_table("q3_checks", cfg), read_table("pool", cfg)
    with_ds = checks.merge(pool[["item_id", "dataset"]], on="item_id")
    chosen: list[str] = []
    if task == "q3_validation":
        llm = with_ds[with_ds["source"] == "llm"]
        counts = llm.groupby("dataset").size().to_dict()
        total = min(int(cfg.data.q3.validation_n), len(llm))
        for ds, k in allocate(counts, total).items():
            chosen += _sample(list(llm.loc[llm["dataset"] == ds, "item_id"]), k, cfg.global_seed, task, ds)
    elif task == "q3_templates":
        tpl = with_ds[with_ds["source"] == "template"]
        for op, grp in tpl.groupby("template_op"):
            chosen += _sample(list(grp["item_id"]), int(cfg.data.q3.template_check_n), cfg.global_seed, task, op)
    else:
        raise ValueError(f"unknown Q3 annotation task {task!r}")
    return export_sheets(task, _sheet_items(checks[checks["item_id"].isin(chosen)], pool), cfg)


def import_task(task: str, cfg: DictConfig) -> dict:
    labels = import_sheets(task, cfg)
    wide = labels.pivot_table(index="item_id", columns="annotator", values="label", aggfunc="first")
    both_yes = (wide == "yes").all(axis=1)
    checks = read_table("q3_checks", cfg)
    summary: dict = {"task": task, "n_items": int(len(wide)), "share_both_yes": float(both_yes.mean())}
    if task == "q3_validation":
        summary["accept_rate"] = float(cfg.data.q3.accept_rate)
        summary["accepted"] = summary["share_both_yes"] >= summary["accept_rate"]
        checks["validated"] = [
            bool(both_yes[i]) if i in both_yes.index else None for i in checks["item_id"]
        ]
        write_table("q3_checks", checks, cfg)
    else:
        ops = checks.set_index("item_id")["template_op"]
        summary["share_both_yes_by_template"] = {
            op: float(both_yes[[i for i in both_yes.index if ops[i] == op]].mean())
            for op in sorted({ops[i] for i in both_yes.index})
        }
    out = root_dir(cfg) / "results" / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
