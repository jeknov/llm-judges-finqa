"""Stage surprisal (Spec 2; Implementation Plan M4 tests) on the synthetic pool with the toy model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fqsel.io import read_table
from fqsel.stages import run_stage
from fqsel.stages.surprisal import within_dataset_percentile


@pytest.fixture(scope="module")
def built(tmp_path_factory, synth_raw, model_cache):
    from fqsel.config import load_config

    root = tmp_path_factory.mktemp("surprisal_run")
    cfg = load_config([f"paths.root={root}", f"paths.raw={synth_raw}", f"paths.cache={model_cache}"])
    run_stage("pool", cfg)
    run_stage("surprisal", cfg)
    return cfg, read_table("pool", cfg), read_table("surprisal", cfg), read_table("surprisal_samples", cfg)


@pytest.mark.spec("S2-01")
def test_one_row_per_non_derived_item_and_S_for_pool_items_only(built):
    _, pool, out, _ = built
    assert set(out["item_id"]) == set(pool.loc[~pool["is_derived"], "item_id"])   # all splits, no #neg
    assert not out["item_id"].str.contains("#").any()
    split = pool.set_index("item_id").loc[out["item_id"], "split"].to_numpy()
    assert out.loc[split == "pool", "S"].notna().all() and out.loc[split != "pool", "S"].isna().all()


def test_samples_and_per_item_summaries_agree(built):
    cfg, _, out, samples = built
    n = int(cfg.surprisal.n_samples)
    assert (samples.groupby("item_id").size() == n).all()
    agg = samples.groupby("item_id").agg(acc=("correct", "mean"), n_correct=("correct", "sum"),
                                         n_unparsed=("answer", lambda a: int(a.isna().sum())))
    o = out.set_index("item_id").loc[agg.index]
    assert np.allclose(o["acc"], agg["acc"]) and (o["n_correct"] == agg["n_correct"]).all()
    assert (o["n_unparsed"] == agg["n_unparsed"]).all()
    assert set(np.round(out["acc"] * n, 6)) <= set(range(n + 1))
    assert (out["n_answer_tokens"] > 0).all() and (out["nll_sum"] > 0).all()
    assert np.allclose(out["nll_mean"], out["nll_sum"] / out["n_answer_tokens"])


@pytest.mark.spec("S2-06")
def test_same_seeds_reproduce_every_sample(built, tmp_path, synth_raw, model_cache):
    from fqsel.config import load_config

    cfg2 = load_config([f"paths.root={tmp_path}", f"paths.raw={synth_raw}", f"paths.cache={model_cache}"])
    run_stage("pool", cfg2)
    run_stage("surprisal", cfg2)
    _, _, out, samples = built
    pd.testing.assert_frame_equal(read_table("surprisal_samples", cfg2), samples)
    pd.testing.assert_frame_equal(read_table("surprisal", cfg2), out)


@pytest.mark.spec("S2-07")
def test_S_is_within_dataset_and_uniform():
    rng = np.random.default_rng(0)
    nll = pd.Series(np.r_[rng.gamma(2, 1, 400), rng.gamma(5, 3, 300)])
    ds = pd.Series(["finqa"] * 400 + ["tatqa"] * 300)
    s = within_dataset_percentile(nll, ds)
    shuffled = nll.copy()
    shuffled[ds == "tatqa"] = rng.permutation(nll[ds == "tatqa"].to_numpy()) * 10   # change the other dataset
    assert np.allclose(within_dataset_percentile(shuffled, ds)[ds == "finqa"], s[ds == "finqa"])
    for d in ("finqa", "tatqa"):
        assert stats.kstest(s[ds == d], "uniform").pvalue > 0.05
    assert ((s > 0) & (s < 1)).all()
