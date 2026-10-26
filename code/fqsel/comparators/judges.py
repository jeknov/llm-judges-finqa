"""Judge comparison (owner decision 2026-10-25): the Jev battery asked of LLM judges of two sizes per family.

Items: the constructed negatives that have a Q3 check, and their originals (Spec 3's A1 pairs). Questions,
one per request, verbatim from the frozen batteries: Q1 and Q3, whose wording every version shares, on the
`judge` state; Q2 in its v0 and v1 wordings on the `judge` state and in its v2 wording on the `qa` state,
without the stated answer. With judges.include_paraphrase, the same again for Q1', Q2' and Q3'. An original
and its negative share the v2 request, which is sent once.

Each request wraps the state and question in prompts/judge_yesno.txt. P(yes) comes from the first answer
token's log-probabilities (fqsel.clients.judge_llm). Determinism: n_items items, drawn with
rng(global_seed, "judge_determinism") and so the same for every judge, are re-sent once; the mean |dP| is
reported in the manifest, and scoring is unchanged whatever its value.

Outputs per judge: scores/judges/<judge>_long.parquet, scores/judges/<judge>.parquet (R = Q1,
I_<v> = mean(Q2_<v>, Q3)) and results/judges/<judge>_manifest.json.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from fqsel.clients.judge_llm import JudgeClient, JudgeResponse, make_judge
from fqsel.config import PROMPTS_DIR, prompt_status, root_dir
from fqsel.ids import rng
from fqsel.io import read_table, write_table
from fqsel.prompts import load_prompt, render_item
from fqsel.stages.jev import load_battery

Q2_VERSIONS = ("v0", "v1", "v2")
BATTERIES = {"orig": "", "para": "p"}
_FIELD = re.compile(r"\{(state|question)\}")


def question_set(include_paraphrase: bool) -> list[dict]:
    """The questions asked of every item: battery, question, variant, state kind and wording."""
    files = {v: load_battery(v) for v in Q2_VERSIONS}
    out = []
    for battery in ("orig", "para") if include_paraphrase else ("orig",):
        for q in ("q1", "q3"):
            texts = {files[v][battery][q] for v in Q2_VERSIONS}
            if len(texts) != 1:
                raise ValueError(f"{battery} {q} differs between battery versions; the comparison assumes it is shared")
            out.append({"battery": battery, "question": q, "variant": "all", "state": "judge", "text": texts.pop()})
        for v in Q2_VERSIONS:
            state = "qa" if "q2" in files[v].get("answer_free", []) else "judge"
            out.append({"battery": battery, "question": "q2", "variant": v, "state": state, "text": files[v][battery]["q2"]})
    return out


def judge_prompt(template: str, state: str, question: str) -> str:
    """Fill {state} and {question} in one pass, so braces inside the item text are left alone."""
    return _FIELD.sub(lambda m: state if m.group(1) == "state" else question, template)


def load_judge_prompt(cfg: DictConfig, allow_draft: bool = False) -> str:
    """The FINAL wrapper prompt; a smoke test may use a DRAFT, whose responses are cached under its own text."""
    name = cfg.judges.prompt
    if allow_draft and prompt_status(name) == "DRAFT":
        lines = (PROMPTS_DIR / name).read_text().splitlines()
        while lines and lines[0].startswith("# "):
            lines.pop(0)
        return "\n".join(lines).strip() + "\n"
    return load_prompt(name, cfg)


def select_targets(cfg: DictConfig) -> pd.DataFrame:
    """Negatives with a Q3 check and their originals, when the original has one too."""
    pool = read_table("pool", cfg)
    checks = set(read_table("q3_checks", cfg)["item_id"])
    pool = pool[(pool["split"] == "pool") & pool["item_id"].isin(checks)]
    negatives = pool[pool["is_derived"]]
    originals = pool[~pool["is_derived"] & pool["item_id"].isin(negatives["base_item_id"])]
    negatives = negatives[negatives["base_item_id"].isin(originals["item_id"])]
    return pd.concat([originals, negatives]).sort_values("item_id").reset_index(drop=True)


def build_requests(cfg: DictConfig, targets: pd.DataFrame, template: str) -> list[dict]:
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"].to_dict()
    questions = question_set(bool(cfg.judges.include_paraphrase))
    out = []
    for _, item in targets.iterrows():
        states = {kind: render_item(item, kind) for kind in ("judge", "qa")}
        for q in questions:
            text = q["text"].format(check_text=checks[item["item_id"]]) if q["question"] == "q3" else q["text"]
            out.append({"item_id": item["item_id"], **{k: q[k] for k in ("battery", "question", "variant", "state")},
                        "prompt": judge_prompt(template, states[q["state"]], text)})
    return out


def _send(client: JudgeClient, prompts: list[str], repeat: int, workers: int) -> dict[str, JudgeResponse]:
    """Each distinct prompt once (an original and its negative share the answer-free one)."""
    unique = sorted(set(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(unique, pool.map(lambda p: client.ask(p, repeat=repeat), unique), strict=True))


def determinism_check(client: JudgeClient, cfg: DictConfig, requests: list[dict], answers: dict) -> dict:
    ids = sorted({r["item_id"] for r in requests})
    n = min(int(cfg.judges.determinism.n_items), len(ids))
    sample = {ids[i] for i in rng(cfg.global_seed, "judge_determinism").permutation(len(ids))[:n]}
    prompts = [r["prompt"] for r in requests if r["item_id"] in sample]
    again = _send(client, prompts, 1, int(cfg.judges.max_workers))
    diffs = [abs(answers[p].p_yes - again[p].p_yes) for p in set(prompts)
             if answers[p].p_yes is not None and again[p].p_yes is not None]
    return {"n_items": n, "n_requests": len(again), "n_compared": len(diffs),
            "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
            "max_abs_diff": float(np.max(diffs)) if diffs else None}


def long_table(judge: str, requests: list[dict], answers: dict[str, JudgeResponse]) -> pd.DataFrame:
    rows = []
    for r in requests:
        a = answers[r["prompt"]]
        rows.append({"item_id": r["item_id"], "judge": judge, "battery": r["battery"], "question": r["question"],
                     "variant": r["variant"], "state": r["state"], "p_yes": a.p_yes, "mass": a.mass,
                     "top_token": a.top_token, "provider": a.provider, "served_model": a.served_model,
                     "request_id": a.request_id, "input_tokens": a.input_tokens, "cost": a.cost,
                     "timestamp": a.timestamp})
    return pd.DataFrame(rows).sort_values(["item_id", "battery", "question", "variant"]).reset_index(drop=True)


def wide_table(judge: str, long: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    orig = long[long["battery"] == "orig"]
    p = {(r.item_id, r.question, r.variant): r.p_yes for r in orig.itertuples()}
    mass = orig.groupby("item_id")["mass"].min().to_dict()

    def mean(a, b):
        return None if a is None or b is None or pd.isna(a) or pd.isna(b) else (a + b) / 2

    rows = []
    for _, item in targets.iterrows():
        i = item["item_id"]
        row = {"item_id": i, "judge": judge, "is_negative": bool(item["is_derived"]),
               "base_item_id": item["base_item_id"], "R": p[(i, "q1", "all")]}
        row |= {f"Q2_{v}": p[(i, "q2", v)] for v in Q2_VERSIONS}
        row["Q3"] = p[(i, "q3", "all")]
        row |= {f"I_{v}": mean(row[f"Q2_{v}"], row["Q3"]) for v in Q2_VERSIONS}
        row["min_mass"] = float(mass[i])
        rows.append(row)
    return pd.DataFrame(rows)


def run_judge(cfg: DictConfig, judge: str, client: JudgeClient | None = None) -> dict:
    client = client or make_judge(cfg, judge)
    targets = select_targets(cfg)
    requests = build_requests(cfg, targets, load_judge_prompt(cfg))
    answers = _send(client, [r["prompt"] for r in requests], 0, int(cfg.judges.max_workers))
    determinism = determinism_check(client, cfg, requests, answers)

    long = long_table(judge, requests, answers)
    providers = sorted(set(long["provider"]))
    if len(providers) != 1:
        raise RuntimeError(f"{judge}: answers came from several providers: {providers}")
    write_table("judge_scores_long", long, cfg, path=f"scores/judges/{judge}_long.parquet")
    write_table("judge_scores", wide_table(judge, long, targets), cfg, path=f"scores/judges/{judge}.parquet")

    m = cfg.judges.models[judge]
    sent = long.drop_duplicates("request_id")
    manifest = {
        "judge": judge, "model_version": client.model_version, "weights": m.weights, "family": m.family,
        "provider": providers[0], "served_models": sorted(set(long["served_model"])),
        "prompt": cfg.judges.prompt, "prompt_status": prompt_status(cfg.judges.prompt),
        "include_paraphrase": bool(cfg.judges.include_paraphrase),
        "n_items": len(targets), "n_negatives": int(targets["is_derived"].sum()),
        "n_rows": len(long), "n_distinct_requests": len(sent),
        "input_tokens": int(sent["input_tokens"].sum()),
        "cost_usd": float(sent["cost"].sum()) if sent["cost"].notna().any() else None,
        "share_p_yes_null": float(long["p_yes"].isna().mean()),
        "mass_quantiles": {str(q): float(long["mass"].quantile(q)) for q in (0.01, 0.05, 0.5)},
        "determinism": determinism,
        "requests_sent_this_run": client.requests_sent,
    }
    out = root_dir(cfg) / "results" / "judges" / f"{judge}_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"judges: {judge} scored {len(targets)} items x {len(long) // len(targets)} questions "
          f"({len(sent)} distinct requests, {manifest['input_tokens']} input tokens, cost {manifest['cost_usd']}); "
          f"P(yes) missing {manifest['share_p_yes_null']:.1%}; determinism mean |dP| {determinism['mean_abs_diff']}")
    return manifest


def smoke(cfg: DictConfig, judge: str, n_pairs: int, client: JudgeClient | None = None) -> pd.DataFrame:
    """Score the first n_pairs originals and their negatives (DRAFT prompt allowed) and return the answers;
    writes no table. Responses are cached, keyed by the prompt text."""
    client = client or make_judge(cfg, judge)
    targets = select_targets(cfg)
    originals = targets.loc[~targets["is_derived"], "item_id"].head(n_pairs)
    picked = targets[targets["item_id"].isin(originals) | targets["base_item_id"].isin(originals)]
    requests = build_requests(cfg, picked, load_judge_prompt(cfg, allow_draft=True))
    answers = _send(client, [r["prompt"] for r in requests], 0, int(cfg.judges.max_workers))
    long = long_table(judge, requests, answers)
    return long[["item_id", "question", "variant", "state", "p_yes", "mass", "top_token", "provider",
                 "served_model", "input_tokens", "cost"]]


def run(cfg: DictConfig, judges: list[str] | None = None) -> list[dict]:
    return [run_judge(cfg, j) for j in (judges or list(cfg.judges.models))]
