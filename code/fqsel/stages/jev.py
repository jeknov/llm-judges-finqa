"""Stage `jev` (Spec 1, Jev battery scoring): six yes/no questions per pool item and negative.

Each prompt is the `judge` rendering of the item followed by the question. Before the full run, a
determinism check scores n_items items twice; if the mean absolute difference in P(yes) exceeds the
tolerance, every item is scored `repeats` times and the mean is stored.

R_b = P(yes | Q1_b); I_b = mean(P(yes | Q2_b), P(yes | Q3_b)); V_b = R_b x I_b;
indecision_b = max over the three questions of the binary entropy of P(yes), in bits.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig

from fqsel.clients.jev import JevClient, make_jev
from fqsel.config import PROMPTS_DIR, PlaceholderError, prompt_status, root_dir
from fqsel.ids import rng
from fqsel.io import read_table, write_table
from fqsel.prompts import render_item
from fqsel.provenance import now_iso

BATTERIES = ("orig", "para")
QUESTIONS = ("q1", "q2", "q3")


def load_battery() -> dict:
    if prompt_status("jev_battery.yaml") != "FINAL":
        raise PlaceholderError("prompts/jev_battery.yaml must be FINAL")
    return yaml.safe_load((PROMPTS_DIR / "jev_battery.yaml").read_text())


def build_prompt(item, battery: dict, b: str, q: str, check_text: str) -> str:
    question = battery[b][q]
    if q == "q3":
        question = question.format(check_text=check_text)
    return render_item(item, "judge") + "\n\n" + question


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _score(client: JevClient, prompts: dict, options: list[str], repeats: list[int], workers: int) -> dict:
    keys = [(k, r) for k in sorted(prompts) for r in repeats]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda kr: client.decide(prompts[kr[0]], options, repeat=kr[1]), keys))
    return dict(zip(keys, results, strict=True))


def determinism_check(client, prompts, item_ids, cfg) -> dict:
    dcfg = cfg.jev.determinism
    n = min(int(dcfg.n_items), len(item_ids))
    sample = sorted(item_ids[i] for i in rng(cfg.global_seed, "jev_determinism").permutation(len(item_ids))[:n])
    chosen = set(sample)
    subset = {k: v for k, v in prompts.items() if k[0] in chosen}
    opt = list(cfg.jev.options)
    res = _score(client, subset, opt, [0, 1], cfg.jev.max_workers)
    diffs = [abs(res[(k, 0)][opt[0]] - res[(k, 1)][opt[0]]) for k in sorted(subset)]
    mad = float(np.mean(diffs)) if diffs else 0.0
    return {"n_items": n, "mean_abs_diff": mad, "tolerance": float(dcfg.tolerance),
            "triplicate": mad > float(dcfg.tolerance), "item_ids": sample}


def aggregate(long: pd.DataFrame, negative_ids: set[str]) -> pd.DataFrame:
    rows = []
    for item_id, grp in long.groupby("item_id", sort=True):
        p = {(r.battery, r.question): r.p_yes for r in grp.itertuples()}
        e = {(r.battery, r.question): r.entropy_bits for r in grp.itertuples()}
        row = {"item_id": item_id, "is_negative": item_id in negative_ids}
        for b, suffix in (("orig", ""), ("para", "_para")):
            r_ = p[(b, "q1")]
            i_ = (p[(b, "q2")] + p[(b, "q3")]) / 2
            row[f"R{suffix}"] = r_
            row[f"I{suffix}"] = i_
            row[f"V{suffix}"] = r_ * i_
            row[f"indecision{suffix}"] = max(e[(b, q)] for q in QUESTIONS)
        rows.append(row)
    cols = ["item_id", "is_negative", "R", "I", "V", "indecision", "R_para", "I_para", "V_para", "indecision_para"]
    return pd.DataFrame(rows, columns=cols)


def run(cfg: DictConfig, client: JevClient | None = None) -> None:
    client = client or make_jev(cfg)
    battery = load_battery()
    pool = read_table("pool", cfg)
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"].to_dict()
    targets = pool[(pool["split"] == "pool") & pool["item_id"].isin(checks)]
    options = list(cfg.jev.options)
    prompts = {}
    for _, item in targets.iterrows():
        for b in BATTERIES:
            for q in QUESTIONS:
                prompts[(item["item_id"], b, q)] = build_prompt(item, battery, b, q, checks[item["item_id"]])

    item_ids = sorted(targets["item_id"])
    det = determinism_check(client, prompts, item_ids, cfg)
    repeats = list(range(int(cfg.jev.determinism.repeats))) if det["triplicate"] else [0]
    results = _score(client, prompts, options, repeats, cfg.jev.max_workers)

    stamp = now_iso()
    long_rows = []
    for key in sorted(prompts):
        item_id, b, q = key
        p_yes = float(np.mean([results[(key, r)][options[0]] for r in repeats]))
        long_rows.append({
            "item_id": item_id, "battery": b, "question": q, "p_yes": p_yes, "p_no": 1.0 - p_yes,
            "entropy_bits": binary_entropy(p_yes), "n_calls": len(repeats),
            "jev_model_version": client.model_version, "timestamp": stamp,
        })
    long = pd.DataFrame(long_rows)
    write_table("jev_scores_long", long, cfg, extra_meta={"determinism": {k: v for k, v in det.items() if k != "item_ids"}})
    negatives = set(targets.loc[targets["is_derived"], "item_id"])
    write_table("jev_scores", aggregate(long, negatives), cfg)
    out = root_dir(cfg) / "results" / "jev_determinism.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**det, "jev_model_version": client.model_version}, indent=2, sort_keys=True) + "\n")
    skipped = int((pool["split"] == "pool").sum()) - len(targets)
    print(f"jev: scored {len(targets)} items x 6 questions (repeats={len(repeats)}); {skipped} without a Q3 check")
