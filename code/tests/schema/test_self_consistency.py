"""Self-consistency comparator (Spec 3: sc_acc, sc_modal_share on the stratified subsample)."""

from __future__ import annotations

import pandas as pd
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.strong_llm import MockStrongLLM, StrongClient, sc_seed
from fqsel.comparators import self_consistency as sc
from fqsel.io import read_table
from fqsel.match import modal_group
from fqsel.stages import run_stage


class AnsweringLLM(MockStrongLLM):
    """Returns 'Answer: <gold>' for even k-derived seeds and a wrong number otherwise; counts calls."""

    def __init__(self, gold_by_prompt, wrong_every=2):
        super().__init__()
        self.gold_by_prompt, self.wrong_every, self.calls = gold_by_prompt, wrong_every, 0

    def generate_text(self, prompt, sampling, seed):
        self.calls += 1
        gold = self.gold_by_prompt[prompt]
        return f"Some reasoning.\nAnswer: {gold if seed % self.wrong_every else '987654321'}"


@pytest.fixture(scope="module")
def built(tmp_path_factory, synth_raw):
    from fqsel.clients.jev import JevClient, MockJev
    from fqsel.config import load_config
    from fqsel.prompts import build_messages
    from fqsel.stages import jev as jev_stage

    root = tmp_path_factory.mktemp("sc_run")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={root / 'cache'}"])
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    jev_stage.run(cfg, client=JevClient(MockJev(), JsonCache(root / "jevcache")))
    pool = read_table("pool", cfg)
    orig = pool[(pool["split"] == "pool") & ~pool["is_derived"]]
    gold = {build_messages(r.to_dict(), "reason")[-1]["content"]: r["gold_answer"] for _, r in orig.iterrows()}
    return cfg, gold, root


def test_subsample_is_stratified_deterministic_and_originals_only(built):
    cfg, _, _ = built
    a, b = sc.select_subsample(cfg), sc.select_subsample(cfg)
    pd.testing.assert_frame_equal(a, b)
    pool = read_table("pool", cfg).set_index("item_id")
    assert not pool.loc[a["item_id"], "is_derived"].any()
    assert set(a["r_quintile"]) <= {1, 2, 3, 4, 5}
    for ds, g in a.groupby("dataset"):   # tiny pool: every original is taken, so all quintiles are present
        assert len(g) == min(sc.PER_DATASET, int(((pool["dataset"] == ds) & (pool["split"] == "pool") & ~pool["is_derived"]).sum()))


@pytest.mark.spec("S3-02")
def test_scores_match_the_samples_and_the_seeds(built):
    cfg, gold, root = built
    llm = AnsweringLLM(gold)
    sc.run(cfg, llm=StrongClient(llm, JsonCache(root / "strong_a")))
    samples, out = read_table("sc_samples", cfg), read_table("self_consistency", cfg)
    n = int(cfg.model.strong.self_consistency.n_samples)
    assert (samples.groupby("item_id").size() == n).all()
    assert set(out["item_id"]) == set(read_table("sc_subsample", cfg)["item_id"])
    for _, r in samples.iterrows():
        assert r["seed"] == sc_seed(cfg, r["item_id"], r["k"]) % 2**31   # as sent in the request body
    for item_id, g in samples.groupby("item_id"):
        row = out.set_index("item_id").loc[item_id]
        assert row["sc_acc"] == pytest.approx(g["correct"].mean())
        n_right = int(g["correct"].sum())
        assert row["sc_modal_share"] == pytest.approx(max(n_right, n - n_right) / n)
        assert row["modal_correct"] == (n_right > n - n_right or (n_right == n - n_right and g.sort_values("k")["correct"].iloc[0]))
    assert llm.calls == len(samples)


def test_rerun_is_served_from_cache(built):
    cfg, gold, root = built
    first = AnsweringLLM(gold)
    sc.run(cfg, llm=StrongClient(first, JsonCache(root / "strong_b")))
    again = AnsweringLLM(gold)
    sc.run(cfg, llm=StrongClient(again, JsonCache(root / "strong_b")))
    assert first.calls > 0 and again.calls == 0


def test_more_samples_reuse_the_first_ones(built):
    cfg, gold, root = built
    llm = AnsweringLLM(gold)
    sc.run(cfg, llm=StrongClient(llm, JsonCache(root / "strong_c")))
    five = llm.calls
    ten = cfg.copy()
    ten.model.strong.self_consistency.n_samples = 10
    llm2 = AnsweringLLM(gold)
    sc.run(ten, llm=StrongClient(llm2, JsonCache(root / "strong_c")))
    assert llm2.calls == five          # only samples 5..9 are new


def test_modal_group_ties_and_missing_answers():
    assert modal_group(["1", "2", "2", None, "1"], "arith") == ("1", 2)   # tie -> earliest
    assert modal_group([None, None], "arith") == (None, 0)
    assert modal_group(["12.5%", "12.5 percent", "3"], "arith") == ("12.5%", 2)
