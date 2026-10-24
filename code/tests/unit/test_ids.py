"""stable_seed and item IDs (Spec 1, Shared conventions)."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from fqsel.ids import base_id, dataset_of, derived_id, make_item_id, stable_seed


@pytest.mark.spec("S1-13")
def test_stable_seed_same_in_separate_processes():
    code = "from fqsel.ids import stable_seed; print(stable_seed(20261022, 'finqa:x', 'neg', 3))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
            for _ in range(2)}
    assert outs == {f"{stable_seed(20261022, 'finqa:x', 'neg', 3)}\n"}


def test_stable_seed_matches_spec_formula():
    expected = int(hashlib.sha256(b"20261022|finqa:x|neg").hexdigest()[:16], 16)
    assert stable_seed(20261022, "finqa:x", "neg") == expected


def test_stable_seed_depends_on_order():
    assert stable_seed("a", "b") != stable_seed("b", "a")


def test_item_ids():
    assert make_item_id("finqa", "ADI/2009/page_49.pdf-1") == "finqa:ADI/2009/page_49.pdf-1"
    assert derived_id("tatqa:abc", "neg") == "tatqa:abc#neg"
    assert base_id("tatqa:abc#corrupt") == "tatqa:abc"
    assert dataset_of("financebench:financebench_id_1") == "financebench"
    with pytest.raises(ValueError):
        make_item_id("other", "1")
    with pytest.raises(ValueError):
        derived_id("finqa:1", "copy")
