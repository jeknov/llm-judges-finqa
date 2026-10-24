"""Shared fixtures: configs rooted in a temp dir, synthetic raw data, a session-wide toy-model cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from fqsel.config import CODE_DIR, load_config
from tests.fixtures.synth_raw import write_synthetic_raw

REAL_RAW = CODE_DIR / "data" / "raw"


@pytest.fixture(scope="session")
def model_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="session")
def synth_raw(tmp_path_factory) -> Path:
    return write_synthetic_raw(tmp_path_factory.mktemp("raw"))


@pytest.fixture
def make_cfg(tmp_path, synth_raw, model_cache):
    """make_cfg(*overrides, root=None, raw=None) -> a tiny-profile config writing under a temp dir."""

    def _make(*overrides: str, root: Path | None = None, raw: Path | None = None):
        root = root or tmp_path
        return load_config([
            f"paths.root={root}",
            f"paths.raw={raw or synth_raw}",
            f"paths.cache={model_cache}",
            *overrides,
        ])

    return _make


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


def requires_real_data():
    return pytest.mark.skipif(not (REAL_RAW / "finqa" / "train.json").exists(), reason="raw data not downloaded")
