"""Stage jev (Spec 1, Jev battery scoring; Implementation Plan M3 tests)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.jev import JevClient, JevResponse, MockJev
from fqsel.config import root_dir
from fqsel.invariants import run_checks
from fqsel.io import read_table
from fqsel.prompts import render_item
from fqsel.stages import jev as jev_stage
from fqsel.stages import run_stage
from fqsel.stages.jev import QIDS, aggregate, binary_entropy, build_questions, load_battery


class NoisyJev(MockJev):
    """Shifts a question's answer up the 1st, 3rd, ... time it is asked and down the 2nd, 4th, ..., so
    repeats disagree, while batched and single-question answers stay aligned (noise is per question)."""

    def __init__(self, noise: float):
        super().__init__(model="noisy-1")
        self.noise, self.seen = noise, {}

    def ask(self, state, questions):
        r = super().ask(state, questions)
        probs = {}
        with self._lock:
            for q, p in r.probs.items():
                k = self.seen.get((state, questions[q]), 0)
                self.seen[(state, questions[q])] = k + 1
                probs[q] = min(0.99, max(0.01, p + (self.noise if k % 2 == 0 else -self.noise)))
        return JevResponse(probs, r.request_id, r.model_version, r.input_tokens, r.timestamp)


class BatchSensitiveJev(MockJev):
    """Answers differently when several questions share a request: batching would change the scores."""

    def ask(self, state, questions):
        r = super().ask(state, questions)
        if len(questions) == 1:
            return r
        probs = {q: min(0.99, p + 0.2) for q, p in r.probs.items()}
        return JevResponse(probs, r.request_id, r.model_version, r.input_tokens, r.timestamp)


def _manifest(cfg) -> dict:
    return json.loads((root_dir(cfg) / "results" / "jev_manifest.json").read_text())


@pytest.fixture(scope="module")
def scored(tmp_path_factory, synth_raw):
    from fqsel.config import load_config

    root = tmp_path_factory.mktemp("jev_run")
    # battery v0: one state, one six-question request per item (v0 and v1 share this path; v2 is tested below)
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={root / 'cache'}",
                       "jev.battery_version=v0"])
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    backend = MockJev()
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
    assert set(zip(long["battery"], long["question"], strict=True)) == set(QIDS)


@pytest.mark.spec("S1-09")
def test_scores_in_unit_interval_no_nulls(scored):
    cfg, _, _ = scored
    wide = read_table("jev_scores", cfg)
    cols = ["R", "I", "V", "indecision", "R_para", "I_para", "V_para", "indecision_para"]
    assert wide[cols].notna().all().all()
    assert ((wide[cols] >= 0) & (wide[cols] <= 1)).all().all()
    pool = read_table("pool", cfg)
    assert set(wide.loc[wide["is_negative"], "item_id"]) == set(pool.loc[pool["is_derived"], "item_id"])


@pytest.mark.spec("S1-17")
def test_one_batched_six_question_request_per_item(scored):
    cfg, _, _ = scored
    reqs = read_table("jev_requests", cfg)
    long = read_table("jev_scores_long", cfg)
    assert (reqs.groupby("item_id").size() == 1).all()
    assert (reqs["n_questions"] == 6).all()
    assert set(reqs["item_id"]) == set(long["item_id"])
    assert long["batched"].all() and (long["n_calls"] == 1).all()
    assert (long.groupby("item_id")["request_id"].nunique() == 1).all()
    assert (reqs["input_tokens"] > 0).all() and set(reqs["jev_model_version"]) == {"mock-jev-1"}
    assert run_checks(cfg) == []


@pytest.mark.spec("S1-17")
def test_answers_map_to_the_right_question_ids(scored):
    """Each stored P(yes) is the answer to that question's own wording against the item's state."""
    cfg, _, backend = scored
    pool = read_table("pool", cfg).set_index("item_id")
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"]
    battery = load_battery(cfg.jev.battery_version)
    long = read_table("jev_scores_long", cfg)
    for _, row in long.iterrows():
        state = render_item(pool.loc[row["item_id"]].to_dict(), "judge")
        text = build_questions(battery, checks[row["item_id"]])[QIDS[(row["battery"], row["question"])]]
        assert row["p_yes"] == pytest.approx(backend.answer(state, text))


def test_state_is_judge_rendering_and_q3_carries_the_check(scored):
    cfg, _, _ = scored
    pool = read_table("pool", cfg).set_index("item_id")
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"]
    neg_id = next(i for i in pool.index if i.endswith("#neg"))
    state = render_item(pool.loc[neg_id].to_dict(), "judge")
    assert state.endswith(f"Stated answer: {pool.loc[neg_id, 'gold_answer']}")
    qs = build_questions(load_battery(cfg.jev.battery_version), checks[neg_id])
    assert set(qs) == {"q1", "q2", "q3", "q1p", "q2p", "q3p"}
    assert qs["q3"].endswith(f"Check: {checks[neg_id]}") and qs["q3p"].endswith(f"Statement: {checks[neg_id]}")


@pytest.mark.spec("S1-18")
def test_batching_check_recorded_and_batched_mode_used(scored):
    cfg, _, _ = scored
    m = _manifest(cfg)
    assert m["batching"]["mode"] == "batched" and m["full_run"]["mode"] == "batched"
    assert m["batching"]["mean_abs_diff_batched_vs_single"] <= m["batching"]["allowed"]
    assert m["n_items"] == min(cfg.jev.determinism.n_items, len(read_table("jev_scores", cfg)))
    assert m["check_requests"] > 0 and m["check_input_tokens"] > 0
    assert m["full_run"]["input_tokens"] == int(read_table("jev_requests", cfg)["input_tokens"].sum())


@pytest.mark.spec("S1-18")
def test_batching_check_switches_to_single_question_requests(scored, tmp_path):
    cfg, _, _ = scored
    jev_stage.run(cfg, client=JevClient(BatchSensitiveJev(), JsonCache(tmp_path / "bs")))
    m = _manifest(cfg)
    assert m["batching"]["mode"] == "single_question" and m["full_run"]["mode"] == "single_question"
    reqs = read_table("jev_requests", cfg)
    long = read_table("jev_scores_long", cfg)
    assert (reqs["n_questions"] == 1).all() and (reqs.groupby("item_id").size() == 6).all()
    assert not long["batched"].any()
    assert run_checks(cfg) == []


def test_batched_requests_cost_less_than_single_question_requests(scored, tmp_path):
    """The point of batching: the state is billed once per item instead of six times."""
    cfg, _, _ = scored
    jev_stage.run(cfg, client=JevClient(MockJev(), JsonCache(tmp_path / "cost")))
    m = _manifest(cfg)
    assert m["full_run"]["mode"] == "batched"
    batched_tokens = m["full_run"]["input_tokens"]
    single_tokens = sum(MockJev().ask(s, {q: t}).input_tokens
                        for s, qs in _states_and_questions(cfg) for q, t in qs.items())
    assert batched_tokens < 0.5 * single_tokens


def _states_and_questions(cfg):
    pool = read_table("pool", cfg).set_index("item_id")
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"]
    battery = load_battery(cfg.jev.battery_version)
    return [(render_item(pool.loc[i].to_dict(), "judge"), build_questions(battery, checks[i])) for i in checks.index]


def test_aggregation_against_hand_computed_values():
    p = {("orig", "q1"): 0.9, ("orig", "q2"): 0.6, ("orig", "q3"): 0.8,
         ("para", "q1"): 0.5, ("para", "q2"): 0.2, ("para", "q3"): 0.4}
    rows = [{"item_id": "finqa:a", "battery": b, "question": q, "p_yes": v, "entropy_bits": binary_entropy(v)}
            for (b, q), v in p.items()]
    out = aggregate(pd.DataFrame(rows), set()).iloc[0]
    assert out["R"] == 0.9 and out["I"] == pytest.approx(0.7) and out["V"] == pytest.approx(0.63)
    assert out["indecision"] == pytest.approx(binary_entropy(0.6))   # 0.6 is the least decided
    assert out["R_para"] == 0.5 and out["I_para"] == pytest.approx(0.3)
    assert out["indecision_para"] == pytest.approx(1.0)              # P(yes) = 0.5 is 1 bit
    assert not out["is_negative"]


def test_binary_entropy():
    assert binary_entropy(0.5) == 1.0 and binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0
    assert binary_entropy(0.11) == pytest.approx(-(0.11 * math.log2(0.11) + 0.89 * math.log2(0.89)))


def test_warm_cache_sends_no_requests(scored, tmp_path):
    cfg, _, _ = scored
    backend = MockJev()
    client = JevClient(backend, JsonCache(tmp_path / "warm"))
    jev_stage.run(cfg, client=client)
    before = len(backend.requests)
    jev_stage.run(cfg, client=client)
    assert len(backend.requests) == before


@pytest.mark.spec("S1-10")
def test_rescoring_reproduces_scores(scored, tmp_path):
    cfg, _, _ = scored
    jev_stage.run(cfg, client=JevClient(MockJev(), JsonCache(tmp_path / "a")))
    first = read_table("jev_scores", cfg)
    jev_stage.run(cfg, client=JevClient(MockJev(), JsonCache(tmp_path / "b")))
    again = read_table("jev_scores", cfg)
    num = first.select_dtypes("number").columns
    assert np.abs(first[num].to_numpy() - again[num].to_numpy()).max() <= cfg.jev.determinism.tolerance


@pytest.mark.spec("S1-17")
def test_determinism_check_triggers_triplicate(scored, tmp_path):
    cfg, _, _ = scored
    jev_stage.run(cfg, client=JevClient(NoisyJev(0.05), JsonCache(tmp_path / "noisy")))
    long = read_table("jev_scores_long", cfg)
    reqs = read_table("jev_requests", cfg)
    repeats = cfg.jev.determinism.repeats
    assert (long["n_calls"] == repeats).all()
    assert (reqs.groupby("item_id").size() == repeats).all() and (reqs["n_questions"] == 6).all()
    m = _manifest(cfg)
    assert m["determinism"]["triplicate"] is True and m["full_run"]["repeats"] == repeats
    assert run_checks(cfg) == []
    jev_stage.run(cfg, client=JevClient(NoisyJev(0.004), JsonCache(tmp_path / "quiet")))
    assert (read_table("jev_scores_long", cfg)["n_calls"] == 1).all()


@pytest.fixture(scope="module")
def scored_v2(tmp_path_factory, synth_raw):
    from fqsel.config import load_config

    root = tmp_path_factory.mktemp("jev_v2")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={root / 'cache'}"])
    assert cfg.jev.battery_version == "v2"
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    backend = MockJev()
    client = JevClient(backend, JsonCache(root / "jevcache"))
    jev_stage.run(cfg, client=client)
    return cfg, client, backend


def test_v2_answer_free_questions_never_see_the_stated_answer(scored_v2):
    """Battery v2: Q2 and Q2' are answered against the qa rendering (no stated answer); the rest against
    the judge rendering. Each stored P(yes) matches its own state and wording."""
    cfg, _, backend = scored_v2
    pool = read_table("pool", cfg).set_index("item_id")
    checks = read_table("q3_checks", cfg).set_index("item_id")["check_text"]
    battery = load_battery("v2")
    assert battery["answer_free"] == ["q2"]
    long = read_table("jev_scores_long", cfg)
    for _, row in long.iterrows():
        kind = "qa" if row["question"] == "q2" else "judge"
        state = render_item(pool.loc[row["item_id"]].to_dict(), kind)
        assert ("Stated answer:" in state) == (kind == "judge")
        text = build_questions(battery, checks[row["item_id"]])[QIDS[(row["battery"], row["question"])]]
        assert row["p_yes"] == pytest.approx(backend.answer(state, text))
    for qs in backend.requests:   # batched or single-question, a request never mixes the two states
        assert set(qs) <= {"q2", "q2p"} or set(qs) <= {"q1", "q3", "q1p", "q3p"}


def test_v2_original_and_negative_share_q2_and_it_is_sent_once(scored_v2):
    cfg, _, backend = scored_v2
    long = read_table("jev_scores_long", cfg)
    pool = read_table("pool", cfg).set_index("item_id")
    q2 = long[long["question"] == "q2"].set_index(["item_id", "battery"])["p_yes"]
    negs = [i for i in pool.index if i.endswith("#neg") and i in set(long["item_id"])]
    assert negs
    for n in negs:
        base = pool.loc[n, "base_item_id"]
        for b in ("orig", "para"):
            assert q2[(n, b)] == q2[(base, b)]          # determinacy cannot differ: the answer is not shown
    sent_qa = [qs for qs in backend.requests if set(qs) == {"q2", "q2p"}]
    # one round of scoring plus the preflight's two batched rounds on the sample, each deduplicated
    states = {render_item(pool.loc[i].to_dict(), "qa") for i in long["item_id"].unique()}
    assert len(sent_qa) <= 3 * len(states)
    assert run_checks(cfg) == []
    m = _manifest(cfg)
    assert m["battery_version"] == "v2"


def test_v2_wording_differs_from_v0_only_in_q2():
    v0 = load_battery("v0")
    v2 = load_battery("v2")
    for b in ("orig", "para"):
        assert {q: v0[b][q] for q in ("q1", "q3")} == {q: v2[b][q] for q in ("q1", "q3")}
    assert v2["orig"]["q2"] == ("Do the context and question determine a single answer uniquely, such that no "
                                "materially different answer would also be defensible?")
    assert v2["para"]["q2"] == ("Is there only one reasonable answer to this question given the information, "
                                "rather than several defensible answers?")
