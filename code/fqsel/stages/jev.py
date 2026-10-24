"""Stage `jev` (Spec 1, Jev battery scoring): six noul questions per pool item and negative.

The state is the `judge` rendering of the item, identical for all six questions; each item is scored in
one batched request carrying q1, q2, q3, q1p, q2p, q3p. P(no) is stored as 1 - P(yes).

Before the full run, on n_items items: (1) determinism — score them twice in batched requests; if the
mean absolute difference in P(yes) exceeds the tolerance, every item is scored `repeats` times and the
mean stored; (2) batching — score each question in its own request and confirm that batched and
single-question probabilities differ by no more than the repeat difference plus `batching_margin`;
otherwise every item is scored one question per request. Both results and the selected mode are written
to results/jev_manifest.json.

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

from fqsel.clients.jev import JevClient, JevResponse, make_jev
from fqsel.config import PROMPTS_DIR, PlaceholderError, prompt_status, root_dir
from fqsel.ids import rng
from fqsel.io import read_table, write_table
from fqsel.prompts import render_item

QUESTIONS = ("q1", "q2", "q3")
# Spec 1 question IDs: the original battery and its paraphrase ("p").
QIDS = {("orig", q): q for q in QUESTIONS} | {("para", q): f"{q}p" for q in QUESTIONS}
QID_TO_KEY = {v: k for k, v in QIDS.items()}


def load_battery() -> dict:
    if prompt_status("jev_battery.yaml") != "FINAL":
        raise PlaceholderError("prompts/jev_battery.yaml must be FINAL")
    return yaml.safe_load((PROMPTS_DIR / "jev_battery.yaml").read_text())


def build_questions(battery: dict, check_text: str) -> dict[str, str]:
    """The six noul questions, keyed by question ID, with {check_text} filled in."""
    return {qid: battery[b][q].format(check_text=check_text) if q == "q3" else battery[b][q]
            for (b, q), qid in QIDS.items()}


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _requests(states: dict, questions: dict, item_ids, batched: bool) -> list[tuple[str, dict[str, str]]]:
    """(item_id, question subset) per request: one per item when batched, else one per question."""
    if batched:
        return [(i, questions[i]) for i in item_ids]
    return [(i, {qid: text}) for i in item_ids for qid, text in questions[i].items()]


def _send(client: JevClient, states, plan, repeats, workers) -> list[tuple[str, int, JevResponse]]:
    """Requests in parallel within a repeat round; rounds run one after another, so identical
    requests are never in flight together and repeat r is always the r-th time a request is sent."""
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in repeats:
            resps = list(pool.map(lambda j, r=r: client.ask(states[j[0]], j[1], repeat=r), plan))
            out += [(item_id, r, resp) for (item_id, _), resp in zip(plan, resps, strict=True)]
    return out


def _probs_by_repeat(results) -> dict[tuple[str, str, int], float]:
    out = {}
    for item_id, r, resp in results:
        for qid, p in resp.probs.items():
            out[(item_id, qid, r)] = p
    return out


def preflight_checks(client, states, questions, item_ids, cfg) -> dict:
    """Determinism (batched x2) and batching (batched vs single-question) checks on a sample."""
    dcfg = cfg.jev.determinism
    n = min(int(dcfg.n_items), len(item_ids))
    sample = sorted(item_ids[i] for i in rng(cfg.global_seed, "jev_determinism").permutation(len(item_ids))[:n])
    workers = cfg.jev.max_workers
    batched = _send(client, states, _requests(states, questions, sample, True), [0, 1], workers)
    single = _send(client, states, _requests(states, questions, sample, False), [0], workers)
    pb, ps = _probs_by_repeat(batched), _probs_by_repeat(single)
    keys = [(i, qid) for i in sample for qid in questions[i]]
    repeat_diff = float(np.mean([abs(pb[(i, q, 0)] - pb[(i, q, 1)]) for i, q in keys])) if keys else 0.0
    batch_diff = float(np.mean([abs(pb[(i, q, 0)] - ps[(i, q, 0)]) for i, q in keys])) if keys else 0.0
    allowed = repeat_diff + float(cfg.jev.batching_margin)
    return {
        "n_items": n,
        "item_ids": sample,
        "determinism": {"mean_abs_diff": repeat_diff, "tolerance": float(dcfg.tolerance),
                        "triplicate": repeat_diff > float(dcfg.tolerance)},
        "batching": {"mean_abs_diff_batched_vs_single": batch_diff, "allowed": allowed,
                     "mode": "batched" if batch_diff <= allowed else "single_question"},
        "check_requests": len({resp.request_id for _, _, resp in batched + single}),
        "check_input_tokens": int(sum(resp.input_tokens for _, _, resp in batched + single)),
    }


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
    states = {row["item_id"]: render_item(row, "judge") for _, row in targets.iterrows()}
    questions = {i: build_questions(battery, checks[i]) for i in states}
    item_ids = sorted(states)

    pre = preflight_checks(client, states, questions, item_ids, cfg)
    repeats = list(range(int(cfg.jev.determinism.repeats))) if pre["determinism"]["triplicate"] else [0]
    batched = pre["batching"]["mode"] == "batched"
    results = _send(client, states, _requests(states, questions, item_ids, batched), repeats, cfg.jev.max_workers)

    versions = {resp.model_version for _, _, resp in results}
    if len(versions) != 1:
        raise RuntimeError(f"Jev reported several model versions in one run: {sorted(versions)}")
    version = versions.pop()

    probs: dict[tuple[str, str], list[float]] = {}
    rids: dict[tuple[str, str], list[str]] = {}
    stamps: dict[tuple[str, str], str] = {}
    requests = {}
    for item_id, _, resp in results:
        for qid, p in resp.probs.items():
            probs.setdefault((item_id, qid), []).append(p)
            rids.setdefault((item_id, qid), []).append(resp.request_id)
            stamps[(item_id, qid)] = max(stamps.get((item_id, qid), ""), resp.timestamp)
        requests[resp.request_id] = {"request_id": resp.request_id, "item_id": item_id,
                                     "n_questions": len(resp.probs), "input_tokens": int(resp.input_tokens),
                                     "jev_model_version": resp.model_version, "timestamp": resp.timestamp}

    long_rows = []
    for item_id, qid in sorted(probs):
        battery_name, q = QID_TO_KEY[qid]
        p_yes = float(np.mean(probs[(item_id, qid)]))
        long_rows.append({
            "item_id": item_id, "battery": battery_name, "question": q, "p_yes": p_yes, "p_no": 1.0 - p_yes,
            "entropy_bits": binary_entropy(p_yes), "n_calls": len(probs[(item_id, qid)]), "batched": batched,
            "jev_model_version": version, "request_id": ";".join(sorted(rids[(item_id, qid)])),
            "timestamp": stamps[(item_id, qid)],
        })
    long = pd.DataFrame(long_rows)
    summary = {k: v for k, v in pre.items() if k != "item_ids"}
    write_table("jev_scores_long", long, cfg, extra_meta={"preflight": summary})
    req_df = pd.DataFrame(sorted(requests.values(), key=lambda r: r["request_id"]))
    write_table("jev_requests", req_df, cfg)
    negatives = set(targets.loc[targets["is_derived"], "item_id"])
    write_table("jev_scores", aggregate(long, negatives), cfg)

    manifest = {**pre, "jev_model": client.model, "jev_model_version": version,
                "full_run": {"mode": pre["batching"]["mode"], "repeats": len(repeats), "requests": len(req_df),
                             "input_tokens": int(req_df["input_tokens"].sum())}}
    out = root_dir(cfg) / "results" / "jev_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    skipped = int((pool["split"] == "pool").sum()) - len(targets)
    print(f"jev: scored {len(targets)} items x 6 questions ({pre['batching']['mode']}, repeats={len(repeats)}, "
          f"{len(req_df)} requests, {manifest['full_run']['input_tokens']} input tokens); "
          f"{skipped} without a Q3 check")
