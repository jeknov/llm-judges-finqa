"""CBI hand audit (workshop protocol, Experiment 8; owner decision 2026-10-26), through the CSV sheets in
fqsel.annotate.

Population: the strong model's confident errors (CBI) as notebook 03 defines them, rebuilt from its frozen
snapshots. An item is CBI when its 5 self-consistency answers are unanimous (1 - H/log n over the normalised
answers >= consistency_threshold; unparsed answers never agree) and fewer than 3 of 5 match the gold under
exact match plus numeric equivalence. Only items Jev also judged with Qwen's modal answer are included.

Strata, from Jev's verdicts (P(yes | Q1) >= supported_threshold): A, Jev rejects the gold and supports
Qwen's answer; B, every other CBI item, sampled in proportion to its three verdict cells. n_per_stratum
items per stratum; double_per_stratum of them go on both sheets (agreement), and the rest are split evenly
between the two annotators. The sheets show the item, the gold answer and derivation, and Qwen's answer
and reasoning; never the stratum, Jev's scores or the item ID. The design is written to
annotation/cbi_audit/_design.csv, private like the key file.

Labels (one per row): model_error, scoring_artefact, ambiguous_question, questionable_gold.

Adjudication (owner decision 2026-10-31): the double-annotated items on which the two annotators disagree go
on one sheet for a third annotator, annotation/cbi_audit/adjudication/annotator_3.csv, with the same row IDs,
item columns and row hashes as the original sheets and both earlier labels and notes shown next to them.
import_adjudication merges everything into one final label per item: the single annotator's label, the
agreed label, or the third annotator's label where the two disagreed.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import cohen_kappa_score

from fqsel.annotate import (
    AnnotationError,
    _read_csv,
    _task,
    export_sheets,
    import_sheets,
    row_hash,
    task_dir,
)
from fqsel.config import root_dir
from fqsel.data.splits import allocate
from fqsel.ids import rng
from fqsel.prompts import render_item

TASK = "cbi_audit"
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
CELLS = ("jev_sides_with_model", "both_supported", "gold_only", "neither")

INSTRUCTIONS = """# CBI hand audit: instructions for annotators

Each row is a financial question that a strong language model answered **the same way in five out of five
attempts**, with an answer that does **not** match the dataset's gold answer. Your task is to decide **why**
the model's answer counts as wrong. You see the context (text and table), the question, the gold answer
with the calculation that produced it, and the model's answer with its reasoning.

Put exactly one of these labels in the `category` column:

| Label | Use it when |
| --- | --- |
| `model_error` | The question is clear, the gold answer is right, and the model's answer is wrong. |
| `scoring_artefact` | The model's answer is actually correct, just written differently (format, units, rounding, wording). |
| `ambiguous_question` | The question can reasonably be read in more than one way, and the model's answer is right under a defensible reading. |
| `questionable_gold` | The gold answer or its calculation looks wrong given the context and the question. |

If two labels seem to apply, choose the one that best explains the disagreement, and explain in `note`.
Use `note` for anything else that matters (for example, which reading of the question each answer uses).

Do not edit, add, delete or reorder rows, and do not change any column other than `category` and `note`.
Return the file under the same name.
"""


def _numbers(s: str | None) -> list[float]:
    out = []
    for m in re.finditer(r"-?\d[\d,]*\.?\d*", s or ""):
        try:
            v = float(m.group().replace(",", "").rstrip("."))
        except ValueError:
            continue
        tail = (s[m.end():m.end() + 12] or "").strip().lower()
        mult = next((k for k in SCALE if tail.startswith(k)), None)
        out.append(v * (SCALE[mult] if mult else 1))
    return out


def numeric_equivalent(pred: str | None, gold: str, scale: str) -> bool:
    """Notebook 03's numeric equivalence: every number in the answer equals a gold number (scale words applied)."""
    gn, pn = _numbers(gold), _numbers(pred)
    if not gn or not pn or len(pn) < len(gn):
        return False
    targets = set(gn) | ({v * SCALE[scale] for v in gn} if scale in SCALE else set())
    return all(any(abs(v - w) <= 1e-6 * max(1.0, abs(w)) for w in targets) for v in pn)


def _norm(a: str) -> str:
    return re.sub(r"\s+", " ", a.strip().lower())


def entropy_consistency(answers: list[str | None]) -> float:
    labels = [_norm(a) if a is not None else f"__unparsed_{i}" for i, a in enumerate(answers)]
    p = np.array(list(Counter(labels).values()), float) / len(labels)
    return float(1 - (-(p * np.log(p)).sum()) / np.log(len(labels)))


def _snap(cfg: DictConfig, key: str):
    return root_dir(cfg) / "runs" / cfg.annotation.tasks[TASK].snapshots[key]


def unanimous_items(cfg: DictConfig) -> pd.DataFrame:
    """Every item the strong model answers unanimously (notebook 03's 1,730), with `cbi` marking the
    confident errors, Jev's verdict cell, the gold and model answers, and the model's first reasoning
    sample. The judge CBI replication (fqsel.comparators.judges_cbi) scores exactly these items."""
    spec = cfg.annotation.tasks[TASK]
    pool = pd.read_parquet(_snap(cfg, "pool_and_jev") / "data" / "pool.parquet").set_index("item_id")
    r = pd.read_parquet(_snap(cfg, "pool_and_jev") / "scores" / "jev_scores.parquet").set_index("item_id")["R"]
    jq = pd.read_parquet(_snap(cfg, "jev_on_qwen") / "jev_on_qwen.parquet").set_index("item_id")
    smp = pd.read_parquet(_snap(cfg, "self_consistency") / "scores" / "sc_samples.parquet")
    smp = smp.join(pool[["gold_answer", "scale"]], on="item_id").sort_values(["item_id", "k"])
    smp["c_num"] = [bool(c) or (a is not None and numeric_equivalent(a, g, s or ""))
                    for a, g, s, c in zip(smp["answer"], smp["gold_answer"], smp["scale"], smp["correct"], strict=True)]
    g = smp.groupby("item_id")
    t = pd.DataFrame({"consistency": g["answer"].apply(lambda a: entropy_consistency(list(a))),
                      "acc_num": g["c_num"].mean(), "model_reasoning": g["text"].first()})
    t = t.join(jq[["p_q1", "stated_answer"]], how="inner").join(r)
    t = t[t["p_q1"].notna()]
    una = t[t["consistency"] >= float(spec.consistency_threshold)].copy()
    una["cbi"] = una["acc_num"] < float(spec.wrong_below)
    sup = float(spec.supported_threshold)
    gold_ok, model_ok = una["R"] >= sup, una["p_q1"] >= sup
    una["cell"] = np.select([~gold_ok & model_ok, gold_ok & model_ok, gold_ok & ~model_ok], list(CELLS[:3]), CELLS[3])
    una["stratum"] = np.where(una["cell"] == "jev_sides_with_model", "A", "B")
    return una.join(pool, rsuffix="_pool").rename(columns={"stated_answer": "model_answer"}).sort_index()


def cbi_items(cfg: DictConfig) -> pd.DataFrame:
    """The confident errors among the unanimous items, as sampled by the audit."""
    una = unanimous_items(cfg)
    return una[una["cbi"]]


def design(cbi: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """The sampled items, their stratum and cell, and which sheets they go on."""
    spec = cfg.annotation.tasks[TASK]
    n, n_double = int(spec.n_per_stratum), int(spec.double_per_stratum)
    a1, a2 = list(cfg.annotation.annotators)
    rows = []
    for stratum, grp in cbi.groupby("stratum", sort=True):
        if stratum == "A":
            quota = {"jev_sides_with_model": n}
        else:
            quota = allocate(grp["cell"].value_counts().to_dict(), n)
        chosen = []
        for cell, k in sorted(quota.items()):
            ids = sorted(grp.index[grp["cell"] == cell])
            if k > len(ids):
                raise ValueError(f"cell {cell} has {len(ids)} items, {k} requested")
            chosen += [ids[i] for i in rng(cfg.global_seed, TASK, stratum, cell).permutation(len(ids))[:k]]
        chosen = sorted(chosen)
        order = [chosen[i] for i in rng(cfg.global_seed, TASK, "assign", stratum).permutation(len(chosen))]
        double, single = order[:n_double], order[n_double:]
        half = len(single) // 2
        for i in double:
            rows.append({"item_id": i, "stratum": stratum, "cell": grp.loc[i, "cell"], "double": True, "annotators": f"{a1};{a2}"})
        for j, i in enumerate(single):
            who = a1 if j < half else a2
            rows.append({"item_id": i, "stratum": stratum, "cell": grp.loc[i, "cell"], "double": False, "annotators": who})
    return pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)


def _sheet_items(cbi: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    rows = []
    for i in ids:
        it = cbi.loc[i]
        deriv = it["program"] if isinstance(it["program"], str) else (it["derivation"] if isinstance(it["derivation"], str) else "")
        rows.append({"item_id": i, "item_text": render_item(it, "qa"), "gold_answer": it["gold_answer"],
                     "gold_derivation": deriv or "(none: the gold answer is a span read from the context)",
                     "model_answer": it["model_answer"], "model_reasoning": it["model_reasoning"]})
    return pd.DataFrame(rows)


def export_task(cfg: DictConfig) -> list:
    cbi = cbi_items(cfg)
    plan = design(cbi, cfg)
    out = task_dir(cfg, TASK)
    out.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out / "_design.csv", index=False)
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    assignment = {a: [r.item_id for r in plan.itertuples() if a in r.annotators.split(";")]
                  for a in cfg.annotation.annotators}
    paths = export_sheets(TASK, _sheet_items(cbi, list(plan["item_id"])), cfg, assignment=assignment)
    return [*paths, out / "_design.csv", out / "INSTRUCTIONS.md"]


ADJUDICATOR = "annotator_3"
ADJUDICATION_INSTRUCTIONS = """# CBI hand audit: adjudication instructions for the third annotator

The two annotators gave different labels to the rows on this sheet. Each row shows the item exactly as they
saw it, followed by both of their labels and notes (`annotator_1_label`, `annotator_1_note`,
`annotator_2_label`, `annotator_2_note`). Decide which label is right, or choose another one, and put it in
the `category` column.

The labels and their meaning are the same as in the original instructions:

| Label | Use it when |
| --- | --- |
| `model_error` | The question is clear, the gold answer is right, and the model's answer is wrong. |
| `scoring_artefact` | The model's answer is actually correct, just written differently (format, units, rounding, wording). |
| `ambiguous_question` | The question can reasonably be read in more than one way, and the model's answer is right under a defensible reading. |
| `questionable_gold` | The gold answer or its calculation looks wrong given the context and the question. |

Use `note` to say briefly why. Do not edit, add, delete or reorder rows, and do not change any column other
than `category` and `note`. Return the file under the same name.
"""


def disagreements(cfg: DictConfig) -> pd.DataFrame:
    """Double-annotated items whose two labels differ, from the imported labels (run import_task first)."""
    labels = pd.read_parquet(task_dir(cfg, TASK) / "labels.parquet")
    a1, a2 = cfg.annotation.annotators
    wide = labels.pivot_table(index="item_id", columns="annotator", values=["label", "note"],
                              aggfunc="first", dropna=False)
    both = labels.groupby("item_id")["annotator"].nunique() == 2
    out = pd.DataFrame({f"{a}_{f}": wide[(f, a)] for a in (a1, a2) for f in ("label", "note")}).loc[both[both].index]
    return out[out[f"{a1}_label"] != out[f"{a2}_label"]].fillna("").sort_index()


def export_adjudication(cfg: DictConfig) -> list[Path]:
    item_cols, label_cols, _ = _task(cfg, TASK)
    base = task_dir(cfg, TASK)
    out_dir = base / "adjudication"
    path = out_dir / f"{ADJUDICATOR}.csv"
    if path.exists():                                   # never overwrite a third annotator's labels
        existing = _read_csv(path)
        if any(existing[c].str.strip().ne("").any() for c in [*label_cols, "note"] if c in existing):
            raise AnnotationError(f"{path} already has entries; move it away before re-exporting")
    dis = disagreements(cfg)
    key = _read_csv(base / "_key.csv").set_index("item_id")["row_id"]
    sent = pd.concat([_read_csv(base / "returned" / f"{a}.csv") for a in cfg.annotation.annotators])
    sent = sent.drop_duplicates("row_id").set_index("row_id")
    rows = []
    for item_id, d in dis.iterrows():
        rid = key[item_id]
        rows.append({"row_id": rid, **{c: sent.loc[rid, c] for c in item_cols}, "row_hash": sent.loc[rid, "row_hash"],
                     **d.to_dict(), **{c: "" for c in label_cols}, "note": ""})
    order = rng(TASK, ADJUDICATOR).permutation(len(rows))
    header = ["row_id", *item_cols, "row_hash", *dis.columns, *label_cols, "note"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rows[i] for i in order], columns=header).to_csv(path, index=False, quoting=csv.QUOTE_ALL,
                                                                  encoding="utf-8")
    (out_dir / "INSTRUCTIONS.md").write_text(ADJUDICATION_INSTRUCTIONS)
    return [path, out_dir / "INSTRUCTIONS.md"]


def import_adjudication(cfg: DictConfig) -> dict:
    """Reads the returned adjudication sheet (adjudication/returned/annotator_3.csv), checks it against the
    sheet that was sent, and writes final per-item labels and their summary by stratum."""
    from statsmodels.stats.proportion import proportion_confint

    item_cols, label_cols, labels = _task(cfg, TASK)
    base = task_dir(cfg, TASK)
    sent = _read_csv(base / "adjudication" / f"{ADJUDICATOR}.csv")
    path = base / "adjudication" / "returned" / f"{ADJUDICATOR}.csv"
    if not path.exists():
        raise AnnotationError(f"{TASK}: no returned adjudication file at {path}")
    back = _read_csv(path)
    shown = [c for c in back.columns if c.endswith(("_label", "_note")) and c not in label_cols]
    if list(back.columns) != list(sent.columns) or list(back["row_id"]) != list(sent["row_id"]):
        raise AnnotationError(f"{path}: columns or rows changed")
    for i, row in back.iterrows():
        if row_hash([row[c] for c in item_cols]) != row["row_hash"] or row["row_hash"] != sent.loc[i, "row_hash"] \
                or any(row[c] != sent.loc[i, c] for c in shown):
            raise AnnotationError(f"{path}: row {row['row_id']} was edited outside category and note")
        if row["category"].strip().lower() not in labels:
            raise AnnotationError(f"{path}: row {row['row_id']} has invalid label {row['category']!r}")
    key = _read_csv(base / "_key.csv").set_index("row_id")["item_id"]
    third = {key[r]: c.strip().lower() for r, c in zip(back["row_id"], back["category"], strict=True)}
    dis = disagreements(cfg)
    if set(third) != set(dis.index):
        raise AnnotationError(f"{path}: rows do not match the current disagreements")
    first = pd.read_parquet(base / "labels.parquet").groupby("item_id")["label"].agg(list)
    plan = pd.read_csv(base / "_design.csv").set_index("item_id")
    final = pd.DataFrame({"label": [third.get(i, ls[0]) for i, ls in first.items()],
                          "source": ["adjudicated" if i in third else ("agreed" if len(ls) == 2 else "single")
                                     for i, ls in first.items()]}, index=first.index).join(plan[["stratum", "cell"]])
    final.index.name = "item_id"
    final.reset_index().to_parquet(base / "final_labels.parquet", index=False)
    item_problem = ("ambiguous_question", "questionable_gold")
    summary = {"task": TASK, "n_items": int(len(final)), "sources": final["source"].value_counts().sort_index().to_dict(),
               "by_stratum": {}}
    for stratum, g in final.groupby("stratum"):
        n = len(g)
        shares = {}
        for name, k in [*((lab, int((g["label"] == lab).sum())) for lab in labels),
                        ("item_problem", int(g["label"].isin(item_problem).sum()))]:
            lo, hi = proportion_confint(k, n, method="wilson")
            shares[name] = {"n": k, "share": k / n, "wilson_95": [float(lo), float(hi)]}
        summary["by_stratum"][stratum] = {"n_items": n, "labels": shares}
    out = root_dir(cfg) / "results" / f"{TASK}_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def import_task(cfg: DictConfig) -> dict:
    labels = import_sheets(TASK, cfg)
    plan = pd.read_csv(task_dir(cfg, TASK) / "_design.csv").set_index("item_id")
    wide = labels.pivot_table(index="item_id", columns="annotator", values="label", aggfunc="first")
    both = wide.dropna()
    summary = {"task": TASK, "n_items": int(len(wide)), "n_double": int(len(both)),
               "agreement": float((both.iloc[:, 0] == both.iloc[:, 1]).mean()) if len(both) else None,
               "cohen_kappa": float(cohen_kappa_score(both.iloc[:, 0], both.iloc[:, 1])) if len(both) > 1 else None,
               "labels_by_stratum": labels.join(plan["stratum"], on="item_id")
                                          .groupby(["stratum", "label"]).size().unstack(fill_value=0).to_dict("index")}
    out = root_dir(cfg) / "results" / f"{TASK}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
