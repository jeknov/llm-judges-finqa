"""Exploratory (pre-specified in the plan, 2026-10-25 19:48 UTC): Jev judges the strong model's own answers.

For every self-consistency subsample item with a modal answer, Jev answers Q1 and Q1' (support; frozen
wording, identical in battery v0/v1/v2) on the `judge` rendering of the item with Qwen's modal answer as
the stated answer. One batched request per item, Spec 1's client, cache and pinned model, single scoring.
Writes code/runs/jev_on_qwen_2026-10-25/ (with SHA256SUMS). Not part of the pre-registered pipeline.

Usage (from code/): python scripts/jev_on_qwen_answers.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from fqsel.clients.jev import make_jev
from fqsel.config import CODE_DIR, load_config
from fqsel.prompts import render_item
from fqsel.stages import get_stage
from fqsel.stages.jev import build_questions, load_battery

SC_DIR = CODE_DIR / "runs" / "self_consistency_2026-10-25"
POOL = CODE_DIR / "runs" / "full_v2_2026-10-25" / "data" / "pool.parquet"
OUT = CODE_DIR / "runs" / "jev_on_qwen_2026-10-25"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build the requests and print counts; no API calls")
    args = ap.parse_args()
    # Spec 1 scope, plus the exploratory judge-comparison settings another workstream added, which this run never reads
    cfg = load_config(["profile=full"], ignore=(*get_stage("jev").ignores, "judges", "judge_yesno.txt"))
    battery = load_battery(cfg.jev.battery_version)
    sc = pd.read_parquet(SC_DIR / "scores" / "self_consistency.parquet")
    pool = pd.read_parquet(POOL).set_index("item_id")
    sc = sc[sc["modal_answer"].notna()].sort_values("item_id")
    jobs = []
    for _, row in sc.iterrows():
        item = pool.loc[row["item_id"]].to_dict() | {"item_id": row["item_id"], "gold_answer": row["modal_answer"]}
        qs = {k: v for k, v in build_questions(battery, "").items() if k in ("q1", "q1p")}
        jobs.append((row["item_id"], row["modal_answer"], render_item(item, "judge"), qs))
    print(f"items with a modal answer: {len(jobs)} of {len(pd.read_parquet(SC_DIR / 'scores' / 'self_consistency.parquet'))}; "
          f"questions {sorted(jobs[0][3])}; model {cfg.jev.model}; battery file {cfg.jev.battery_version} (Q1/Q1' identical in all versions)")
    if args.dry_run:
        print("example state (end):", jobs[0][2][-200:].replace("\n", " | "))
        return
    client = make_jev(cfg)

    def one(job):
        item_id, answer, state, qs = job
        r = client.ask(state, qs)
        return {"item_id": item_id, "stated_answer": answer, "p_q1": r.probs["q1"], "p_q1p": r.probs["q1p"],
                "request_id": r.request_id, "jev_model_version": r.model_version, "input_tokens": int(r.input_tokens),
                "timestamp": r.timestamp}

    with ThreadPoolExecutor(max_workers=int(cfg.jev.max_workers)) as ex:
        rows = list(ex.map(one, jobs))
    out = pd.DataFrame(rows)
    versions = set(out["jev_model_version"])
    if versions != {cfg.jev.model}:
        raise RuntimeError(f"unexpected Jev versions {versions}")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "jev_on_qwen.parquet", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "pre_specified": "specs/IMPLEMENTATION_PLAN.md, 'Exploratory: Jev judging the strong model's own answers' (2026-10-25 19:48 UTC)",
        "jev_model": cfg.jev.model, "questions": {k: battery["orig" if k == "q1" else "para"]["q1"] for k in ("q1", "q1p")},
        "items": len(out), "requests_sent": client.requests_sent, "input_tokens": int(out["input_tokens"].sum()),
        "source": str(SC_DIR.relative_to(CODE_DIR))}, indent=2) + "\n")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.name}" for p in sorted(OUT.iterdir()) if p.name != "SHA256SUMS"]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"scored {len(out)} items; requests sent {client.requests_sent}; input tokens {out['input_tokens'].sum():,}")


if __name__ == "__main__":
    main()
