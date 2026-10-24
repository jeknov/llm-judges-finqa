"""End-to-end smoke test, profile=tiny (Spec 1, Testing strategy layer 4).

Runs every stage in order through the CLI on synthetic raw data with the toy model and mock clients,
twice, and checks that the outputs validate and are identical apart from timestamps. Gated stages
(icl, e1, e2, e3) halt cleanly until the gates stage exists (M5).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from fqsel.cli import main
from fqsel.stages import STAGES

TIMESTAMP_COLUMNS = {"jev_scores_long.parquet": ["timestamp"]}


def _run_all(root: Path, raw: Path, cache: Path) -> None:
    overrides = [f"paths.root={root}", f"paths.raw={raw}", f"paths.cache={cache}"]
    for stage in STAGES:
        code = main(["run", stage.name, *overrides])
        assert code == (3 if stage.gated else 0), stage.name   # 3 = halted: no gates.json before M5
    assert main(["check", *overrides]) == 0


def _outputs(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".parquet", ".csv", ".json") and ".cache" not in p.parts}


@pytest.mark.e2e
def test_tiny_pipeline_twice_identical(tmp_path, synth_raw):
    start = time.monotonic()
    a, b = tmp_path / "a", tmp_path / "b"
    _run_all(a, synth_raw, tmp_path / "cache_a")
    _run_all(b, synth_raw, tmp_path / "cache_b")
    assert time.monotonic() - start < 20 * 60

    out_a, out_b = _outputs(a), _outputs(b)
    assert out_a.keys() == out_b.keys()
    assert {"data/pool.parquet", "data/verified.parquet", "data/q3_checks.parquet",
            "scores/jev_scores_long.parquet", "scores/jev_scores.parquet"} <= out_a.keys()
    for rel in out_a:
        name = Path(rel).name
        if name in TIMESTAMP_COLUMNS:
            fa = pd.read_parquet(out_a[rel]).drop(columns=TIMESTAMP_COLUMNS[name])
            fb = pd.read_parquet(out_b[rel]).drop(columns=TIMESTAMP_COLUMNS[name])
            pd.testing.assert_frame_equal(fa, fb)
        else:
            assert out_a[rel].read_bytes() == out_b[rel].read_bytes(), rel
