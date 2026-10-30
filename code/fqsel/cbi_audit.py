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
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import cohen_kappa_score

from fqsel.annotate import export_sheets, import_sheets, task_dir
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
