"""(added) Judge comparison: questions, item pairs, states, aggregation and outputs on the synthetic pool."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.judge_llm import JudgeClient, JudgeResponse, MockJudge, request_params
from fqsel.comparators import judges
from fqsel.config import PlaceholderError, root_dir
from fqsel.invariants import run_checks
from fqsel.io import read_table
from fqsel.stages import run_stage
from fqsel.stages.jev import load_battery


class RepeatShiftJudge(MockJudge):
    """Answers 0.1 higher (capped) whenever a prompt is sent a second time: a non-deterministic API."""

    def __init__(self):
        super().__init__("shift")
        self.seen: dict[str, int] = {}

    def ask(self, prompt):
        n = self.seen.get(prompt, 0)
        self.seen[prompt] = n + 1
        resp = super().ask(prompt)
        p = min(resp.p_yes + 0.1 * n, 1.0)
        return JudgeResponse(**{**resp.__dict__, "p_yes_raw": resp.mass * p, "p_no_raw": resp.mass * (1 - p)})


@pytest.fixture(scope="module")
def built(tmp_path_factory, synth_raw):
    from fqsel.config import load_config

    root = tmp_path_factory.mktemp("judges_run")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={root / 'cache'}"])
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    return cfg, root


def client(cfg, root, name, backend=None):
    backend = backend or MockJudge(name)
    return JudgeClient(backend, JsonCache(root / "judgecache" / name), request_params(cfg)), backend


def test_question_set_is_q1_q3_and_q2_in_three_versions():
    qs = judges.question_set(False)
    assert [(q["question"], q["variant"], q["state"]) for q in qs] == [
        ("q1", "all", "judge"), ("q3", "all", "judge"),
        ("q2", "v0", "judge"), ("q2", "v1", "judge"), ("q2", "v2", "qa")]
    for q in qs:   # verbatim from the frozen battery files
        version = "v2" if q["variant"] == "all" else q["variant"]
        assert q["text"] == load_battery(version)["orig"][q["question"]]
    para = judges.question_set(True)
    assert len(para) == 10 and {q["battery"] for q in para} == {"orig", "para"}
    assert [q["text"] for q in para if q["battery"] == "para" and q["variant"] == "v1"] == [load_battery("v1")["para"]["q2"]]


def test_targets_are_complete_original_negative_pairs(built):
    cfg, _ = built
    t = judges.select_targets(cfg)
    neg, orig = t[t["is_derived"]], t[~t["is_derived"]]
    assert len(neg) > 0 and set(neg["base_item_id"]) == set(orig["item_id"])
    assert set(t["item_id"]) <= set(read_table("q3_checks", cfg)["item_id"])
    assert (t["split"] == "pool").all()


def test_judge_prompt_fills_fields_once_and_leaves_braces_in_the_item_alone():
    out = judges.judge_prompt("{state}\n\nQ: {question}\nyes or no", "Table: {x} | {question}", "Is it {so}?")
    assert out == "Table: {x} | {question}\n\nQ: Is it {so}?\nyes or no"


def test_run_writes_one_row_per_item_and_question_with_the_right_state(built):
    cfg, root = built
    c, mock = client(cfg, root, "llama32_3b")
    manifest = judges.run_judge(cfg, "llama32_3b", client=c)
    long = read_table("judge_scores_long", cfg, path="scores/judges/llama32_3b_long.parquet")
    wide = read_table("judge_scores", cfg, path="scores/judges/llama32_3b.parquet")
    targets = judges.select_targets(cfg)
    assert (long.groupby("item_id").size() == 5).all() and set(long["item_id"]) == set(targets["item_id"])
    assert set(wide["item_id"]) == set(targets["item_id"])

    template = judges.load_judge_prompt(cfg)
    reqs = pd.DataFrame(judges.build_requests(cfg, targets, template))
    has_answer = reqs["prompt"].str.contains("\nStated answer: ")
    assert (has_answer == (reqs["state"] == "judge")).all()     # only v2's Q2 is asked without the answer
    for _, r in reqs.iterrows():                                 # every P(yes) is the mock's for that exact prompt
        row = long[(long["item_id"] == r["item_id"]) & (long["question"] == r["question"]) & (long["variant"] == r["variant"])]
        assert row["p_yes"].iloc[0] == pytest.approx(mock.probability(r["prompt"]))

    # an original and its negative share the answer-free Q2 request, so its answer is identical (C1')
    v2 = long[long["variant"] == "v2"].set_index("item_id")
    for _, n in targets[targets["is_derived"]].iterrows():
        assert v2.loc[n["item_id"], "p_yes"] == v2.loc[n["base_item_id"], "p_yes"]
        assert v2.loc[n["item_id"], "request_id"] == v2.loc[n["base_item_id"], "request_id"]
    n_pairs = int(targets["is_derived"].sum())
    assert manifest["n_distinct_requests"] == len(long) - n_pairs

    w = wide.set_index("item_id")
    assert (w["I_v0"] - (w["Q2_v0"] + w["Q3"]) / 2).abs().max() < 1e-12
    assert (w["I_v2"] - (w["Q2_v2"] + w["Q3"]) / 2).abs().max() < 1e-12
    assert (w["min_mass"] - 0.9).abs().max() < 1e-12      # test tolerance: float rounding of p + (1 - p)
    assert manifest["determinism"]["n_items"] == int(cfg.judges.determinism.n_items)
    assert manifest["determinism"]["mean_abs_diff"] == 0.0     # the mock is deterministic
    assert json.loads((root_dir(cfg) / "results" / "judges" / "llama32_3b_manifest.json").read_text()) == manifest
    assert run_checks(cfg) == []


def test_rerun_is_served_from_cache(built):
    cfg, root = built
    c1, _ = client(cfg, root, "gemma3_4b")
    judges.run_judge(cfg, "gemma3_4b", client=c1)
    c2, again = client(cfg, root, "gemma3_4b")
    m = judges.run_judge(cfg, "gemma3_4b", client=c2)
    assert c1.requests_sent > 0 and again.prompts == [] and m["requests_sent_this_run"] == 0


def test_determinism_check_measures_repeat_differences(built):
    cfg, root = built
    c, _ = client(cfg, root, "shift", RepeatShiftJudge())
    det = judges.run_judge(cfg, "llama33_70b", client=c)["determinism"]
    assert det["n_compared"] > 0 and 0 < det["mean_abs_diff"] <= 0.1 + 1e-12


def test_paraphrase_option_doubles_the_questions(built):
    cfg, root = built
    para = cfg.copy()
    para.judges.include_paraphrase = True
    c, _ = client(para, root, "para")
    judges.run_judge(para, "gemma4_31b", client=c)
    long = read_table("judge_scores_long", para, path="scores/judges/gemma4_31b_long.parquet")
    assert (long.groupby("item_id").size() == 10).all()


def test_smoke_scores_whole_pairs_and_writes_nothing(built, tmp_path):
    from fqsel.config import load_config

    cfg, root = built
    fresh = load_config([f"paths.root={tmp_path}", f"paths.raw={cfg.paths.raw}", f"paths.cache={tmp_path / 'c'}"])
    run_stage("pool", fresh)
    run_stage("q3", fresh)
    c, _ = client(fresh, tmp_path, "smoke")
    out = judges.smoke(fresh, "llama32_3b", 2, client=c)
    targets = judges.select_targets(fresh).set_index("item_id")
    items = targets.loc[sorted(set(out["item_id"]))]
    assert (~items["is_derived"]).sum() == 2 and set(items.loc[items["is_derived"], "base_item_id"]) <= set(items.index)
    assert len(out) == 5 * len(items)
    assert not (tmp_path / "scores" / "judges").exists()


def test_real_runs_refuse_a_draft_wrapper_prompt_but_a_smoke_test_may_use_it(cfg, monkeypatch):
    fake = cfg.copy()
    fake._set_flag("struct", False)
    fake.profile = "pilot"
    monkeypatch.setattr(judges, "prompt_status", lambda name: "DRAFT")
    monkeypatch.setattr("fqsel.prompts.prompt_status", lambda name: "DRAFT")
    with pytest.raises(PlaceholderError):
        judges.load_judge_prompt(fake)
    text = judges.load_judge_prompt(fake, allow_draft=True)
    assert "{state}" in text and "{question}" in text and not text.startswith("#")


def test_paraphrases_added_later_give_the_same_tables_as_one_run_with_paraphrases(built, tmp_path):
    """Run without paraphrases, then with: only the paraphrased prompts are sent, and the tables equal those
    of a single run with paraphrases (apart from request IDs and timestamps, which differ between any two runs)."""
    cfg, root = built
    para = cfg.copy()
    para.judges.include_paraphrase = True
    per_call = ["request_id", "timestamp"]

    def tables(c):
        long = read_table("judge_scores_long", c, path="scores/judges/llama32_3b_long.parquet").drop(columns=per_call)
        return long, read_table("judge_scores", c, path="scores/judges/llama32_3b.parquet")

    split, backend = client(cfg, tmp_path, "split")
    judges.run_judge(cfg, "llama32_3b", client=split)
    first = list(backend.prompts)
    split2 = JudgeClient(backend, JsonCache(tmp_path / "judgecache" / "split"), request_params(para))
    m_split = judges.run_judge(para, "llama32_3b", client=split2)
    added = backend.prompts[len(first):]
    long_split, wide_split = tables(para)

    once = JudgeClient(MockJudge("split"), JsonCache(tmp_path / "judgecache" / "once"), request_params(para))
    m_once = judges.run_judge(para, "llama32_3b", client=once)   # the same judge, from an empty cache
    long_once, wide_once = tables(para)

    reqs = pd.DataFrame(judges.build_requests(para, judges.select_targets(para), judges.load_judge_prompt(para)))
    assert set(added) == set(reqs.loc[reqs["battery"] == "para", "prompt"])   # only paraphrases were sent
    pd.testing.assert_frame_equal(long_split, long_once)
    pd.testing.assert_frame_equal(wide_split, wide_once)
    differs = {"requests_sent_this_run"}
    assert {k: v for k, v in m_split.items() if k not in differs} == {k: v for k, v in m_once.items() if k not in differs}
