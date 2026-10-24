"""update_gates (locking, schema validation) and record_verdict (upsert)."""

from __future__ import annotations

import json
import multiprocessing as mp

import jsonschema
import pandas as pd
import pytest

from fqsel.gates import gates_path, read_gates, update_gates
from fqsel.verdicts import HYPOTHESES_PATH, record_verdict


def _bump(args):
    root, key, n = args
    from fqsel.config import load_config

    cfg = load_config([f"paths.root={root}"])
    for i in range(n):
        update_gates(cfg, lambda g, i=i: g["derived"]["medians"].update({f"{key}{i}": {"I": 0.5, "S": 0.5}}))


@pytest.mark.spec("S3-11")
def test_two_concurrent_writers_lose_no_update(cfg, tmp_path):
    update_gates(cfg, lambda g: None)
    with mp.get_context("spawn").Pool(2) as pool:
        pool.map(_bump, [(str(tmp_path), "a", 15), (str(tmp_path), "b", 15)])
    medians = read_gates(cfg)["derived"]["medians"]
    assert len(medians) == 30


@pytest.mark.spec("S3-05")
def test_invalid_update_is_refused_and_file_kept(cfg):
    update_gates(cfg, lambda g: None)
    before = gates_path(cfg).read_text()
    with pytest.raises(jsonschema.ValidationError):
        update_gates(cfg, lambda g: g["derived"].update(r_threshold="high"))
    assert gates_path(cfg).read_text() == before


def test_default_gates_validate(cfg):
    data = update_gates(cfg, lambda g: None)
    assert data["derived"]["r_threshold"] == 0.9 and data["gates"]["D5"] is None
    assert json.loads(gates_path(cfg).read_text())["battery_version"] == "v0"


@pytest.mark.spec("S5-10")
def test_record_verdict_upserts(cfg, tmp_path):
    record_verdict(cfg, "H1", "none", "not_supported", "rule-a", {"auroc": 0.6})
    record_verdict(cfg, "H2", "G", "supported", "rule-b", {})
    record_verdict(cfg, "H1", "none", "supported", "rule-a", {"auroc": 0.8})
    df = pd.read_csv(tmp_path / HYPOTHESES_PATH)
    assert len(df) == 2
    assert df.set_index("hypothesis").loc["H1", "verdict"] == "supported"
    assert list(df["hypothesis"]) == ["H1", "H2"]
