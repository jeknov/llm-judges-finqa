"""CBI replication with LLM judges (workshop protocol, Experiment 7; owner decision 2026-10-29, minimal scope).

Notebook 03 found that Jev, judging the gold answer, separates the strong model's confident errors (CBI)
from its reliable unanimous answers (AUROC 0.75), but judging the model's own answer it barely does (0.62).
This comparator asks other judges the same two things, on the same items, in one run on one endpoint:

- items: notebook 03's unanimous items (fqsel.cbi_audit.unanimous_items: 1,730, of which 281 CBI);
- gold arm: Q1 and Q3 on the `judge` state with the gold answer, and Q2 in its v2 wording on the `qa`
  state (no stated answer), verbatim from the frozen battery v2;
- model arm: Q1 on the `judge` state with the strong model's modal answer as the stated answer.

Q3 is asked of the gold arm only: its check is generated from the gold answer. Each request wraps the state
and question in prompts/judge_yesno.txt; P(yes) comes from the first answer token's log-probabilities
(fqsel.clients.judge_llm), with the same client, cache and settings as the judge comparison. Determinism:
judges.cbi.determinism_n_items items, drawn with rng(global_seed, "judges_cbi_determinism") and so the same
for every judge, have both arms re-sent once; the mean |dP| is reported and changes nothing.

Outputs per judge: scores/judges_cbi/<judge>_long.parquet, scores/judges_cbi/<judge>.parquet and
results/judges_cbi/<judge>_manifest.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from fqsel.cbi_audit import unanimous_items
from fqsel.clients.judge_llm import JudgeClient, make_judge
from fqsel.comparators.judges import _send, judge_prompt, load_judge_prompt
from fqsel.config import root_dir
from fqsel.ids import rng
from fqsel.io import read_table, write_table
from fqsel.prompts import render_item
from fqsel.stages.jev import load_battery

BATTERY = "v2"
# (arm, question, state): the requests asked of every item
QUESTIONS = (("gold", "q1", "judge"), ("gold", "q2", "qa"), ("gold", "q3", "judge"), ("model", "q1", "judge"))


def population(cfg: DictConfig) -> pd.DataFrame:
    """The unanimous items with their Q3 check (from the snapshot the battery was scored on)."""
    una = unanimous_items(cfg)
    snap = root_dir(cfg) / "runs" / cfg.annotation.tasks.cbi_audit.snapshots.pool_and_jev
    checks = pd.read_parquet(snap / "data" / "q3_checks.parquet").set_index("item_id")["check_text"]
    missing = una.index.difference(checks.index)
    if len(missing):
        raise ValueError(f"{len(missing)} unanimous items have no Q3 check, e.g. {list(missing[:3])}")
    return una.assign(check_text=checks.reindex(una.index))


def build_requests(items: pd.DataFrame, template: str) -> list[dict]:
    battery = load_battery(BATTERY)["orig"]
    out = []
    for item_id, it in items.iterrows():
        item = it.to_dict()
        states = {"gold": {"judge": render_item(item, "judge"), "qa": render_item(item, "qa")},
                  "model": {"judge": render_item(item | {"gold_answer": it["model_answer"]}, "judge")}}
        for arm, q, state in QUESTIONS:
            text = battery[q].format(check_text=it["check_text"]) if q == "q3" else battery[q]
            out.append({"item_id": item_id, "arm": arm, "question": q, "state": state,
                        "stated_answer": it["gold_answer"] if arm == "gold" else it["model_answer"],
                        "prompt": judge_prompt(template, states[arm][state], text)})
    return out


def determinism_check(client: JudgeClient, cfg: DictConfig, requests: list[dict], answers: dict) -> dict:
    ids = sorted({r["item_id"] for r in requests})
    n = min(int(cfg.judges.cbi.determinism_n_items), len(ids))
    sample = {ids[i] for i in rng(cfg.global_seed, "judges_cbi_determinism").permutation(len(ids))[:n]}
    prompts = [r["prompt"] for r in requests if r["item_id"] in sample]
    again = _send(client, prompts, 1, int(cfg.judges.max_workers))
    diffs = {arm: [abs(answers[r["prompt"]].p_yes - again[r["prompt"]].p_yes) for r in requests
                   if r["item_id"] in sample and r["arm"] == arm
                   and answers[r["prompt"]].p_yes is not None and again[r["prompt"]].p_yes is not None]
             for arm in ("gold", "model")}
    all_d = diffs["gold"] + diffs["model"]
    return {"n_items": n, "n_requests": len(again), "n_compared": len(all_d),
            "mean_abs_diff": float(np.mean(all_d)) if all_d else None,
            "max_abs_diff": float(np.max(all_d)) if all_d else None,
            "mean_abs_diff_by_arm": {a: (float(np.mean(d)) if d else None) for a, d in diffs.items()}}


def long_table(judge: str, requests: list[dict], answers: dict) -> pd.DataFrame:
    rows = []
    for r in requests:
        a = answers[r["prompt"]]
        rows.append({"item_id": r["item_id"], "judge": judge, "arm": r["arm"], "question": r["question"],
                     "state": r["state"], "stated_answer": r["stated_answer"], "p_yes": a.p_yes, "mass": a.mass,
                     "top_token": a.top_token, "provider": a.provider, "served_model": a.served_model,
                     "request_id": a.request_id, "input_tokens": a.input_tokens, "cost": a.cost,
                     "timestamp": a.timestamp})
    return pd.DataFrame(rows).sort_values(["item_id", "arm", "question"]).reset_index(drop=True)


def wide_table(judge: str, long: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    p = {(r.item_id, r.arm, r.question): r.p_yes for r in long.itertuples()}
    mass = long.groupby("item_id")["mass"].min().to_dict()

    def mean(a, b):
        return None if a is None or b is None or pd.isna(a) or pd.isna(b) else (a + b) / 2

    rows = []
    for item_id, it in items.iterrows():
        q2, q3 = p[(item_id, "gold", "q2")], p[(item_id, "gold", "q3")]
        rows.append({"item_id": item_id, "judge": judge, "cbi": bool(it["cbi"]), "jev_cell": it["cell"],
                     "R_gold": p[(item_id, "gold", "q1")], "Q2_v2": q2, "Q3": q3, "I_v2": mean(q2, q3),
                     "R_model": p[(item_id, "model", "q1")], "min_mass": float(mass[item_id])})
    return pd.DataFrame(rows)


def run_judge(cfg: DictConfig, judge: str, client: JudgeClient | None = None) -> dict:
    client = client or make_judge(cfg, judge)
    items = population(cfg)
    requests = build_requests(items, load_judge_prompt(cfg))
    answers = _send(client, [r["prompt"] for r in requests], 0, int(cfg.judges.max_workers))
    determinism = determinism_check(client, cfg, requests, answers)

    long = long_table(judge, requests, answers)
    providers = sorted(set(long["provider"]))
    if len(providers) != 1:
        raise RuntimeError(f"{judge}: answers came from several providers: {providers}")
    write_table("judge_cbi_long", long, cfg, path=f"scores/judges_cbi/{judge}_long.parquet")
    write_table("judge_cbi", wide_table(judge, long, items), cfg, path=f"scores/judges_cbi/{judge}.parquet")

    m = cfg.judges.models[judge]
    sent = long.drop_duplicates("request_id")
    manifest = {
        "judge": judge, "model_version": client.model_version, "weights": m.weights, "family": m.family,
        "provider": providers[0], "served_models": sorted(set(long["served_model"])), "battery": BATTERY,
        "n_items": len(items), "n_cbi": int(items["cbi"].sum()), "n_rows": len(long),
        "n_distinct_requests": len(sent), "input_tokens": int(sent["input_tokens"].sum()),
        "cost_usd": float(sent["cost"].sum()) if sent["cost"].notna().any() else None,
        "share_p_yes_null": float(long["p_yes"].isna().mean()),
        "share_first_token_not_yes_no": float((~long["top_token"].str.strip().str.lower().isin(["yes", "no"])).mean()),
        "mass_quantiles": {str(q): float(long["mass"].quantile(q)) for q in (0.01, 0.05, 0.5)},
        "determinism": determinism, "requests_sent_this_run": client.requests_sent,
    }
    out = root_dir(cfg) / "results" / "judges_cbi" / f"{judge}_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"judges-cbi: {judge} scored {len(items)} items ({manifest['n_cbi']} CBI) x {len(QUESTIONS)} questions "
          f"({len(sent)} distinct requests, {manifest['input_tokens']} input tokens, cost {manifest['cost_usd']}); "
          f"P(yes) missing {manifest['share_p_yes_null']:.1%}; determinism mean |dP| {determinism['mean_abs_diff']}")
    return manifest


def smoke(cfg: DictConfig, judge: str, n_items: int, client: JudgeClient | None = None) -> pd.DataFrame:
    """Score n_items unanimous items, half of them CBI, and return the answers; writes no table."""
    client = client or make_judge(cfg, judge)
    items = population(cfg)
    k = max(1, n_items // 2)
    picked = pd.concat([items[items["cbi"]].head(k), items[~items["cbi"]].head(n_items - k)])
    requests = build_requests(picked, load_judge_prompt(cfg))
    answers = _send(client, [r["prompt"] for r in requests], 0, int(cfg.judges.max_workers))
    long = long_table(judge, requests, answers).join(picked["cbi"], on="item_id")
    return long[["item_id", "cbi", "arm", "question", "stated_answer", "p_yes", "mass", "top_token", "provider",
                 "served_model", "input_tokens", "cost"]]


def run(cfg: DictConfig, judges: list[str] | None = None) -> list[dict]:
    return [run_judge(cfg, j) for j in (judges or list(cfg.judges.cbi.judges))]


def read_scores(cfg: DictConfig, judge: str) -> pd.DataFrame:
    return read_table("judge_cbi", cfg, path=f"scores/judges_cbi/{judge}.parquet")
