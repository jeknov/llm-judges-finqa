"""(added) CBI replication with judges: population, paired arms, requests, aggregation and outputs."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.judge_llm import JudgeClient, JudgeResponse, MockJudge, request_params
from fqsel.comparators import judges_cbi
from fqsel.config import CODE_DIR, load_config, root_dir
from fqsel.invariants import run_checks
from fqsel.io import read_table
from fqsel.stages import JUDGES_SCOPE, get_stage, run_stage

SNAP = {"self_consistency": "self_consistency_2026-10-25", "jev_on_qwen": "jev_on_qwen_2026-10-25",
        "pool_and_jev": "full_v2_2026-10-25"}


def _write_snapshots(cfg, root):
    """Synthetic snapshots in the layout notebook 03 reads: every third original is a confident error,
    every third is answered correctly and unanimously, the rest get mixed answers (not unanimous)."""
    pool = read_table("pool", cfg)
    checks = read_table("q3_checks", cfg)
    orig = pool[(pool["split"] == "pool") & ~pool["is_derived"] & pool["item_id"].isin(checks["item_id"])]
    samples, sc, jq, jev = [], [], [], []
    kinds = {}
    for n, (_, it) in enumerate(orig.sort_values("item_id").iterrows()):
        kind = ("cbi", "reliable", "mixed")[n % 3]
        kinds[it["item_id"]] = kind
        gold = it["gold_answer"]
        answers = {"cbi": ["987654"] * 5, "reliable": [gold] * 5, "mixed": [gold, "1", "2", "3", "4"]}[kind]
        for k, a in enumerate(answers):
            samples.append({"item_id": it["item_id"], "k": k, "seed": k, "answer": a, "correct": a == gold,
                            "text": f"reasoning {k}\nAnswer: {a}"})
        sc.append({"item_id": it["item_id"], "modal_answer": answers[0]})
        jq.append({"item_id": it["item_id"], "stated_answer": answers[0], "p_q1": 0.9 if n % 2 else 0.2})
        jev.append({"item_id": it["item_id"], "R": 0.3 if n % 4 == 0 else 0.95})
    base = root / "runs"
    (base / SNAP["self_consistency"] / "scores").mkdir(parents=True)
    (base / SNAP["jev_on_qwen"]).mkdir(parents=True)
    (base / SNAP["pool_and_jev"] / "data").mkdir(parents=True)
    (base / SNAP["pool_and_jev"] / "scores").mkdir(parents=True)
    pd.DataFrame(samples).to_parquet(base / SNAP["self_consistency"] / "scores" / "sc_samples.parquet")
    pd.DataFrame(sc).to_parquet(base / SNAP["self_consistency"] / "scores" / "self_consistency.parquet")
    pd.DataFrame(jq).to_parquet(base / SNAP["jev_on_qwen"] / "jev_on_qwen.parquet")
    pd.DataFrame(jev).to_parquet(base / SNAP["pool_and_jev"] / "scores" / "jev_scores.parquet")
    pool.to_parquet(base / SNAP["pool_and_jev"] / "data" / "pool.parquet")
    checks.to_parquet(base / SNAP["pool_and_jev"] / "data" / "q3_checks.parquet")
    return kinds


@pytest.fixture(scope="module")
def built(tmp_path_factory, synth_raw):
    root = tmp_path_factory.mktemp("judges_cbi")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={root / 'cache'}"])
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    kinds = _write_snapshots(cfg, root)
    return cfg, root, kinds


def client(cfg, root, name, backend=None):
    backend = backend or MockJudge(name)
    return JudgeClient(backend, JsonCache(root / "judgecache" / name), request_params(cfg)), backend


def test_cbi_judges_are_the_owners_choice_on_pinned_endpoints(cfg):
    assert list(cfg.judges.cbi.judges) == ["qwen3_30b", "gemma4_31b", "llama33_70b"]
    m = cfg.judges.models
    assert m.qwen3_30b.name == "qwen/qwen3-30b-a3b-instruct-2507" and m.qwen3_30b.expected_provider == "DekaLLM"
    assert list(m.qwen3_30b.provider.order) == ["dekallm"] and m.qwen3_30b.provider.allow_fallbacks is False
    for j in ("gemma4_31b", "llama33_70b"):
        assert list(m[j].provider.order) == ["parasail/fp8"] and m[j].expected_provider == "Parasail"


def test_population_is_the_unanimous_items_with_cbi_marked(built):
    cfg, _, kinds = built
    items = judges_cbi.population(cfg)
    assert set(items.index) == {i for i, k in kinds.items() if k in ("cbi", "reliable")}
    assert set(items.index[items["cbi"]]) == {i for i, k in kinds.items() if k == "cbi"}
    assert items["check_text"].notna().all()


def test_both_arms_on_the_same_items_with_the_right_stated_answer(built):
    cfg, _, _ = built
    items = judges_cbi.population(cfg)
    reqs = pd.DataFrame(judges_cbi.build_requests(items, judges_cbi.load_judge_prompt(cfg)))
    assert set(reqs.loc[reqs["arm"] == "gold", "item_id"]) == set(reqs.loc[reqs["arm"] == "model", "item_id"]) == set(items.index)
    assert (reqs.groupby("item_id").size() == 4).all()
    for _, r in reqs.iterrows():
        it = items.loc[r["item_id"]]
        expected = it["gold_answer"] if r["arm"] == "gold" else it["model_answer"]
        assert r["stated_answer"] == expected
        if r["state"] == "judge":
            assert f"Stated answer: {expected}\n" in r["prompt"]
        else:                                                            # Q2 v2: no stated answer at all
            assert r["question"] == "q2" and "Stated answer:" not in r["prompt"]
    assert set(reqs.loc[reqs["arm"] == "model", "question"]) == {"q1"}


def test_run_writes_paired_scores_that_match_the_judge(built):
    cfg, root, _ = built
    c, mock = client(cfg, root, "a")
    m = judges_cbi.run_judge(cfg, "gemma4_31b", client=c)
    long = read_table("judge_cbi_long", cfg, path="scores/judges_cbi/gemma4_31b_long.parquet")
    wide = read_table("judge_cbi", cfg, path="scores/judges_cbi/gemma4_31b.parquet").set_index("item_id")
    items = judges_cbi.population(cfg)
    reqs = judges_cbi.build_requests(items, judges_cbi.load_judge_prompt(cfg))
    for r in reqs:                                                       # every P(yes) is the judge's for that prompt
        row = long[(long["item_id"] == r["item_id"]) & (long["arm"] == r["arm"]) & (long["question"] == r["question"])]
        assert row["p_yes"].iloc[0] == pytest.approx(mock.probability(r["prompt"]))
    for item_id, w in wide.iterrows():
        g = long[long["item_id"] == item_id].set_index(["arm", "question"])["p_yes"]
        assert w["R_gold"] == g[("gold", "q1")] and w["R_model"] == g[("model", "q1")]
        assert w["I_v2"] == pytest.approx((g[("gold", "q2")] + g[("gold", "q3")]) / 2)
        assert w["cbi"] == bool(items.loc[item_id, "cbi"])
    # reliable items whose modal answer equals the gold send one shared Q1 request for both arms
    same = items[items["model_answer"] == items["gold_answer"]].index
    assert len(same) and (wide.loc[same, "R_gold"] == wide.loc[same, "R_model"]).all()
    assert m["n_distinct_requests"] == len(long) - len(same)
    assert m["determinism"]["mean_abs_diff"] == 0.0 and set(m["determinism"]["mean_abs_diff_by_arm"]) == {"gold", "model"}
    assert json.loads((root_dir(cfg) / "results" / "judges_cbi" / "gemma4_31b_manifest.json").read_text()) == m
    assert run_checks(cfg) == []


class RepeatShiftJudge(MockJudge):
    def __init__(self):
        super().__init__("shift")
        self.seen: dict[str, int] = {}

    def ask(self, prompt):
        n = self.seen.get(prompt, 0)
        self.seen[prompt] = n + 1
        resp = super().ask(prompt)
        p = min(resp.p_yes + 0.1 * n, 1.0)
        return JudgeResponse(**{**resp.__dict__, "p_yes_raw": resp.mass * p, "p_no_raw": resp.mass * (1 - p)})


def test_determinism_covers_both_arms_of_one_item_sample(built):
    cfg, root, _ = built
    c, backend = client(cfg, root, "shift", RepeatShiftJudge())
    det = judges_cbi.run_judge(cfg, "llama33_70b", client=c)["determinism"]
    assert det["n_compared"] > 0 and 0 < det["mean_abs_diff"] <= 0.1 + 1e-12
    assert all(v is not None and v > 0 for v in det["mean_abs_diff_by_arm"].values())


def test_rerun_is_served_from_cache_and_smoke_writes_nothing(built, tmp_path):
    cfg, root, _ = built
    c1, _ = client(cfg, root, "cache")
    judges_cbi.run_judge(cfg, "qwen3_30b", client=c1)
    first = {n: read_table(n, cfg, path=f"scores/judges_cbi/qwen3_30b{suffix}.parquet")
             for n, suffix in (("judge_cbi_long", "_long"), ("judge_cbi", ""))}
    c2, again = client(cfg, root, "cache")
    m2 = judges_cbi.run_judge(cfg, "qwen3_30b", client=c2)
    assert m2["requests_sent_this_run"] == 0 and again.prompts == []
    for n, suffix in (("judge_cbi_long", "_long"), ("judge_cbi", "")):          # a re-run reproduces every value
        pd.testing.assert_frame_equal(read_table(n, cfg, path=f"scores/judges_cbi/qwen3_30b{suffix}.parquet"), first[n])
    out = judges_cbi.smoke(cfg, "qwen3_30b", 4, client=client(cfg, root, "smoke")[0])
    assert out["item_id"].nunique() == 4 and out.drop_duplicates("item_id")["cbi"].sum() == 2


def test_command_reads_its_own_settings_and_needs_the_final_prompt(monkeypatch):
    from fqsel import cli

    seen = []
    monkeypatch.setattr(cli, "load_config", lambda overrides, ignore=(): seen.append(tuple(ignore)) and None)
    monkeypatch.setattr("fqsel.comparators.judges_cbi.run", lambda cfg, names: None)
    assert cli.main(["judges-cbi", "--judge", "x"]) == 0
    assert set(seen[0]) == set(get_stage("jev").ignores) - set(JUDGES_SCOPE)


SNAPSHOTS = CODE_DIR / "runs"


@pytest.mark.real_data
@pytest.mark.skipif(not (SNAPSHOTS / SNAP["jev_on_qwen"]).exists(), reason="snapshots not present")
def test_real_population_is_notebook_03s_unanimous_set(tmp_path):
    (tmp_path / "runs").symlink_to(SNAPSHOTS)
    cfg = load_config(["profile=full", f"paths.root={tmp_path}"],
                      ignore=tuple(i for i in get_stage("jev").ignores if i not in JUDGES_SCOPE))
    items = judges_cbi.population(cfg)
    assert len(items) == 1730 and int(items["cbi"].sum()) == 281      # notebook 03, sections 9 and 10
    sc = pd.read_parquet(SNAPSHOTS / SNAP["self_consistency"] / "scores" / "self_consistency.parquet").set_index("item_id")
    assert (sc.loc[items.index, "modal_answer"] == items["model_answer"]).all()   # the answer Jev was shown in section 10
    reqs = judges_cbi.build_requests(items, judges_cbi.load_judge_prompt(cfg))
    assert len(reqs) == 4 * 1730
