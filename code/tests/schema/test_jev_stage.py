"""Stage jev (Spec 1, Jev battery scoring; Implementation Plan M3 tests)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.jev import JevClient, MockJev
from fqsel.io import read_metadata, read_table
from fqsel.stages import jev as jev_stage
from fqsel.stages import run_stage
from fqsel.stages.jev import aggregate, binary_entropy, build_prompt, load_battery


class RecordingJev(MockJev):
    model_version = "recording-1"

    def __init__(self, noise: float = 0.0):
        self.prompts: list[str] = []
        self.noise = noise
        self.n = 0

    def decide(self, prompt, options):
        self.prompts.append(prompt)
        probs, info = super().decide(prompt, options)
        if self.noise:
            self.n += 1
            p = min(0.99, max(0.01, probs[options[0]] + (self.noise if self.n % 2 else -self.noise)))
            probs = {options[0]: p, options[1]: 1 - p}
        return probs, info


@pytest.fixture(scope="module")
def scored(tmp_path_factory, synth_raw, model_cache):
    from fqsel.config import load_config

    root = tmp_path_factory.mktemp("jev_run")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={root / 'cache'}"])
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    backend = RecordingJev()
    client = JevClient(backend, JsonCache(root / "jevcache"))
    jev_stage.run(cfg, client=client)
    return cfg, client, backend


@pytest.mark.spec("S1-08")
def test_six_rows_per_item_summing_to_one(scored):
    cfg, _, _ = scored
    long = read_table("jev_scores_long", cfg)
    pool = read_table("pool", cfg)
    targets = set(pool.loc[pool["split"] == "pool", "item_id"])
    assert set(long["item_id"]) == targets           # every pool item and every negative
    assert (long.groupby("item_id").size() == 6).all()
    assert ((long["p_yes"] + long["p_no"] - 1).abs() <= 1e-6).all()
    assert set(zip(long["battery"], long["question"], strict=True)) == {
        (b, q) for b in ("orig", "para") for q in ("q1", "q2", "q3")}


@pytest.mark.spec("S1-09")
def test_scores_in_unit_interval_no_nulls(scored):
    cfg, _, _ = scored
    wide = read_table("jev_scores", cfg)
    cols = ["R", "I", "V", "indecision", "R_para", "I_para", "V_para", "indecision_para"]
    assert wide[cols].notna().all().all()
    assert ((wide[cols] >= 0) & (wide[cols] <= 1)).all().all()
    pool = read_table("pool", cfg)
    negs = set(pool.loc[pool["is_derived"], "item_id"])
    assert set(wide.loc[wide["is_negative"], "item_id"]) == negs


def test_aggregation_against_hand_computed_values():
    rows = []
    p = {("orig", "q1"): 0.9, ("orig", "q2"): 0.6, ("orig", "q3"): 0.8,
         ("para", "q1"): 0.5, ("para", "q2"): 0.2, ("para", "q3"): 0.4}
    for (b, q), v in p.items():
        rows.append({"item_id": "finqa:a", "battery": b, "question": q, "p_yes": v,
                     "entropy_bits": binary_entropy(v)})
    out = aggregate(pd.DataFrame(rows), set()).iloc[0]
    assert out["R"] == 0.9 and out["I"] == pytest.approx(0.7) and out["V"] == pytest.approx(0.63)
    assert out["indecision"] == pytest.approx(binary_entropy(0.6))   # 0.6 is the least decided
    assert out["R_para"] == 0.5 and out["I_para"] == pytest.approx(0.3)
    assert out["indecision_para"] == pytest.approx(1.0)              # P(yes) = 0.5 is 1 bit
    assert out["is_negative"] is False or out["is_negative"] == False  # noqa: E712


def test_binary_entropy():
    assert binary_entropy(0.5) == 1.0 and binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0
    assert binary_entropy(0.11) == pytest.approx(-(0.11 * math.log2(0.11) + 0.89 * math.log2(0.89)))


def test_prompts_are_judge_rendering_plus_question(scored):
    cfg, _, backend = scored
    pool = read_table("pool", cfg).set_index("item_id")
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"]
    neg_id = next(i for i in pool.index if i.endswith("#neg"))
    neg = pool.loc[neg_id]
    battery = load_battery()
    q1 = build_prompt(neg, battery, "orig", "q1", checks[neg_id])
    q3 = build_prompt(neg, battery, "para", "q3", checks[neg_id])
    assert q1 in backend.prompts and q3 in backend.prompts
    assert f"Stated answer: {neg['gold_answer']}" in q1
    base_gold = pool.loc[neg["base_item_id"], "gold_answer"]
    assert f"Stated answer: {base_gold}\n" not in q1
    assert q3.endswith(f"Statement: {checks[neg_id]}")


def test_warm_cache_sends_no_requests(scored):
    cfg, client, backend = scored
    before = len(backend.prompts)
    jev_stage.run(cfg, client=client)
    assert len(backend.prompts) == before


@pytest.mark.spec("S1-10")
def test_rescoring_reproduces_scores(scored, tmp_path):
    cfg, _, _ = scored
    first = read_table("jev_scores", cfg)
    fresh = JevClient(RecordingJev(), JsonCache(tmp_path / "fresh"))
    jev_stage.run(cfg, client=fresh)
    again = read_table("jev_scores", cfg)
    num = first.select_dtypes("number").columns
    assert np.abs(first[num].to_numpy() - again[num].to_numpy()).max() <= cfg.jev.determinism.tolerance


def test_determinism_check_triggers_triplicate(scored, tmp_path):
    cfg, _, _ = scored
    noisy = JevClient(RecordingJev(noise=0.05), JsonCache(tmp_path / "noisy"))
    jev_stage.run(cfg, client=noisy)
    long = read_table("jev_scores_long", cfg)
    assert (long["n_calls"] == cfg.jev.determinism.repeats).all()
    det = json.loads(read_metadata("jev_scores_long", cfg)["fqsel.extra"])["determinism"]
    assert det["triplicate"] is True and det["mean_abs_diff"] > det["tolerance"]
    quiet = JevClient(RecordingJev(noise=0.004), JsonCache(tmp_path / "quiet"))
    jev_stage.run(cfg, client=quiet)
    assert (read_table("jev_scores_long", cfg)["n_calls"] == 1).all()
