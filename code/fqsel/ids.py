"""Item IDs and process-independent seeds (Spec 1, Shared conventions)."""

from __future__ import annotations

import hashlib

import numpy as np

DATASETS = ("finqa", "tatqa", "financebench")
DERIVED_SUFFIXES = ("neg", "corrupt", "premdel")


def stable_seed(*parts) -> int:
    """int(sha256("|".join(str(p) for p in parts)).hexdigest()[:16], 16). The built-in hash is banned."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def rng(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


def make_item_id(dataset: str, source_id: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    return f"{dataset}:{source_id}"


def derived_id(item_id: str, suffix: str) -> str:
    if suffix not in DERIVED_SUFFIXES:
        raise ValueError(f"unknown derived suffix {suffix!r}")
    return f"{base_id(item_id)}#{suffix}"


def base_id(item_id: str) -> str:
    return item_id.split("#", 1)[0]


def dataset_of(item_id: str) -> str:
    return item_id.split(":", 1)[0]
