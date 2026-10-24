"""results/gates.json: locked, schema-validated read-update-write (Spec 3, gates.json schema).

Every writer goes through update_gates(); gated stages call require_not_halted() at start-up.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from pathlib import Path

import jsonschema
from filelock import FileLock
from omegaconf import DictConfig

from fqsel.config import CONF_DIR, root_dir
from fqsel.provenance import git_commit, now_iso

GATES_PATH = "results/gates.json"
SCHEMA_PATH = CONF_DIR / "gates.schema.json"


class HaltError(SystemExit):
    """Raised (exit code 3) when a gated stage must not run."""

    def __init__(self, message: str):
        super().__init__(3)
        self.message = message

    def __str__(self) -> str:
        return self.message


def default_gates(battery_version: str) -> dict:
    return {
        "battery_version": battery_version,
        "computed_at": now_iso(),
        "git_commit": git_commit(),
        "gates": {g: None for g in ("D1", "D2", "D3", "D4", "D5", "D6")},
        "flags": {"abs_rho_I_pvi": None, "abs_rho_I_icl_reducible": None, "novelty_flag": False},
        "derived": {
            "halt": False,
            "halt_reason": None,
            "r_source_arith": "jev",
            "r_threshold": 0.9,
            "cells_r_filter": True,
            "medians": {},
            "halt_spec6": False,
            "halt_spec6_override": None,
            "single_proxy_claims": True,
        },
    }


def gates_path(cfg: DictConfig) -> Path:
    return root_dir(cfg) / GATES_PATH


def validate_gates(data: dict) -> None:
    jsonschema.validate(data, json.loads(SCHEMA_PATH.read_text()))


def read_gates(cfg: DictConfig) -> dict:
    data = json.loads(gates_path(cfg).read_text())
    validate_gates(data)
    return data


def update_gates(cfg: DictConfig, update: Callable[[dict], None]) -> dict:
    """Apply `update` (which mutates the dict) under a file lock; refuse to save an invalid result."""
    path = gates_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=60):
        data = json.loads(path.read_text()) if path.exists() else default_gates(cfg.jev.battery_version)
        new = copy.deepcopy(data)
        update(new)
        new["computed_at"] = now_iso()
        new["git_commit"] = git_commit()
        validate_gates(new)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    return new


def require_not_halted(cfg: DictConfig, stage: str, spec6: bool = False) -> dict:
    path = gates_path(cfg)
    if not path.exists():
        raise HaltError(f"stage {stage}: {GATES_PATH} is missing; run the gates stage (Spec 3) first")
    data = read_gates(cfg)
    derived = data["derived"]
    if derived["halt"]:
        raise HaltError(f"stage {stage}: halted by gates ({derived['halt_reason']}); a human decision is required")
    if spec6 and derived["halt_spec6"] and not derived["halt_spec6_override"]:
        raise HaltError(f"stage {stage}: halted by gate D5 and no halt_spec6_override reason is set")
    return data
