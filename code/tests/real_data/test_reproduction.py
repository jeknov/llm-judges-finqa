"""(added) Earlier experiments must stay reproducible from their archives as the code grows.

Each test re-runs a finished experiment with today's code and default config into a temporary root, from the
archived inputs and API responses only (a stand-in judge raises on any request that is not cached), and
compares the result with the frozen snapshot. Requires the DVC-tracked runs/ and .cache/ to be checked out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.judge_llm import JudgeClient, request_params
from fqsel.config import CODE_DIR, load_config
from fqsel.io import read_table
from fqsel.stages import JUDGES_SCOPE, get_stage

RUNS, CACHE = CODE_DIR / "runs", CODE_DIR / ".cache"
pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not (RUNS / "judges_2026-10-26").exists() or not (CACHE / "judges").exists(),
                                 reason="archived runs or caches not checked out")]


class NoNetwork:
    """A judge backend that only exists to make cache misses fail loudly."""

    def __init__(self, model_version: str):
        self.model_version = model_version

    def ask(self, prompt: str):
        raise RuntimeError("cache miss: a reproduction must not call the API")


@pytest.mark.parametrize("judge,paraphrase", [("llama32_3b", False), ("llama33_70b", False),
                                              ("gemma3_4b", True), ("gemma4_31b", True)])
def test_judge_comparison_reproduces_from_the_archive(tmp_path, judge, paraphrase):
    from fqsel.comparators import judges

    (tmp_path / "data").symlink_to(CODE_DIR / "data")
    cfg = load_config(["profile=full", f"paths.root={tmp_path}", f"paths.cache={CACHE}",
                       f"judges.include_paraphrase={str(paraphrase).lower()}"],
                      ignore=tuple(i for i in get_stage("jev").ignores if i not in JUDGES_SCOPE))
    m = cfg.judges.models[judge]
    client = JudgeClient(NoNetwork(f"{m.name}@{'/'.join(m.provider.order)}"), JsonCache(CACHE / "judges" / judge),
                         request_params(cfg))
    manifest = judges.run_judge(cfg, judge, client=client)
    snap = RUNS / "judges_2026-10-26" / judge
    for name, f in (("judge_scores_long", f"{judge}_long.parquet"), ("judge_scores", f"{judge}.parquet")):
        pd.testing.assert_frame_equal(read_table(name, cfg, path=f"scores/judges/{f}"), pd.read_parquet(snap / f))
    old = json.loads((snap / f"{judge}_manifest.json").read_text())
    det_old, det_new = old.pop("determinism"), manifest.pop("determinism")
    old.pop("requests_sent_this_run"), manifest.pop("requests_sent_this_run")
    assert manifest == old
    # the archived value was summed in set order (fixed 2026-10-29); it can differ in the last digit only
    assert det_new["mean_abs_diff"] == pytest.approx(det_old["mean_abs_diff"], rel=1e-12)
    assert {k: v for k, v in det_new.items() if k != "mean_abs_diff"} == {k: v for k, v in det_old.items() if k != "mean_abs_diff"}


def test_cbi_audit_export_reproduces_the_sheets_sent_to_annotators(tmp_path):
    from fqsel import cbi_audit

    sent = CODE_DIR / "annotation" / "cbi_audit"
    if not (sent / "_design.csv").exists():
        pytest.skip("audit sheets not exported on this machine")
    (tmp_path / "runs").symlink_to(RUNS)
    cfg = load_config(["profile=full", f"paths.root={tmp_path}"], ignore=get_stage("q3").ignores)
    cbi_audit.export_task(cfg)
    new = Path(tmp_path) / "annotation" / "cbi_audit"
    for f in ("_design.csv", "_key.csv", "INSTRUCTIONS.md"):
        assert (new / f).read_bytes() == (sent / f).read_bytes(), f
    labels = ["category", "note"]                    # annotators fill these in; everything else must match
    for a in cfg.annotation.annotators:
        ours = pd.read_csv(new / f"{a}.csv", dtype=str, keep_default_na=False)
        theirs = pd.read_csv(sent / f"{a}.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(ours.drop(columns=labels), theirs.drop(columns=labels))


@pytest.mark.parametrize(("judge", "snapshot"), [("llama33_70b", "judges_cbi_2026-10-30"),
                                                ("gemma4_31b", "judges_cbi_2026-10-30"),
                                                ("qwen3_235b", "judges_cbi_qwen_2026-10-30")])
def test_cbi_replication_reproduces_from_the_archive(tmp_path, judge, snapshot):
    from fqsel.comparators import judges_cbi

    snap = RUNS / snapshot / judge
    if not snap.exists():
        pytest.skip("CBI replication snapshot not checked out")
    (tmp_path / "runs").symlink_to(RUNS)
    cfg = load_config(["profile=full", f"paths.root={tmp_path}", f"paths.cache={CACHE}"],
                      ignore=tuple(i for i in get_stage("jev").ignores if i not in JUDGES_SCOPE))
    m = cfg.judges.models[judge]
    client = JudgeClient(NoNetwork(f"{m.name}@{'/'.join(m.provider.order)}"), JsonCache(CACHE / "judges" / judge),
                         request_params(cfg))
    manifest = judges_cbi.run_judge(cfg, judge, client=client)
    for name, f in (("judge_cbi_long", f"{judge}_long.parquet"), ("judge_cbi", f"{judge}.parquet")):
        pd.testing.assert_frame_equal(read_table(name, cfg, path=f"scores/judges_cbi/{f}"), pd.read_parquet(snap / f))
    old = json.loads((snap / f"{judge}_manifest.json").read_text())
    old.pop("requests_sent_this_run"), manifest.pop("requests_sent_this_run")
    assert manifest == old


def test_cbi_audit_labels_reproduce_from_the_returned_sheets(tmp_path):
    """Importing the archived returned sheets (both annotators, then the adjudication) gives the frozen labels."""
    import shutil

    from fqsel import cbi_audit

    snap = RUNS / "cbi_audit_2026-10-31"
    if not snap.exists():
        pytest.skip("CBI audit snapshot not checked out")
    src = snap / "annotation" / "cbi_audit"
    dst = tmp_path / "annotation" / "cbi_audit"
    keep = ["_design.csv", "_key.csv", "annotator_1.csv", "annotator_2.csv", "returned",
            "adjudication/annotator_3.csv", "adjudication/returned"]
    for rel in keep:                                       # inputs only; every output is rebuilt
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if (src / rel).is_dir() else shutil.copy2)(src / rel, dst / rel)
    cfg = load_config(["profile=full", f"paths.root={tmp_path}"], ignore=tuple(get_stage("q3").ignores))   # as `annotate`
    assert cbi_audit.import_task(cfg) == json.loads((snap / "results" / "cbi_audit.json").read_text())
    assert cbi_audit.import_adjudication(cfg) == json.loads((snap / "results" / "cbi_audit_final.json").read_text())
    pd.testing.assert_frame_equal(pd.read_parquet(dst / "final_labels.parquet"), pd.read_parquet(src / "final_labels.parquet"))
    pd.testing.assert_frame_equal(pd.read_parquet(dst / "labels.parquet").drop(columns="imported_at"),
                                  pd.read_parquet(src / "labels.parquet").drop(columns="imported_at"))
