"""Hydra config loading, config hashing and the placeholder policy (Implementation Plan, Placeholder policy)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

CODE_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = CODE_DIR / "conf"
PROMPTS_DIR = CODE_DIR / "prompts"

PROMPT_STATUSES = ("FINAL", "DRAFT", "PLACEHOLDER")


class PlaceholderError(RuntimeError):
    """A value or prompt the owner has not decided yet was needed by a pilot or full run."""


def load_config(overrides: Iterable[str] = (), ignore: Iterable[str] = ()) -> DictConfig:
    """`ignore` lists config prefixes and prompt files the caller never reads (Stage.ignores); they may
    stay undecided. Reading one anyway still fails: OmegaConf raises on `???` and load_prompt refuses
    non-FINAL prompts outside tiny."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    OmegaConf.set_struct(cfg, True)
    if cfg.profile != "tiny":
        check_real_run_ready(cfg, ignore)
    return cfg


def root_dir(cfg: DictConfig) -> Path:
    root = Path(cfg.paths.root)
    return root if root.is_absolute() else (CODE_DIR / root).resolve()


def raw_dir(cfg: DictConfig) -> Path:
    raw = Path(cfg.paths.raw)
    return raw if raw.is_absolute() else root_dir(cfg) / raw


def cache_dir(cfg: DictConfig) -> Path:
    cache = Path(cfg.paths.cache)
    return cache if cache.is_absolute() else root_dir(cfg) / cache


def config_hash(cfg: DictConfig) -> str:
    """SHA-256 of the resolved config, with paths removed so that relocating the repo changes nothing."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    container.pop("paths", None)
    blob = json.dumps(container, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def find_missing(node, prefix: str = "") -> list[str]:
    """Dotted keys whose value is `???`."""
    missing = []
    if isinstance(node, DictConfig):
        for key in node.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if OmegaConf.is_missing(node, key):
                missing.append(path)
            else:
                missing.extend(find_missing(node[key], path))
    elif isinstance(node, ListConfig):
        for i in range(len(node)):
            missing.extend(find_missing(node[i], f"{prefix}[{i}]"))
    return missing


def prompt_status(name: str) -> str:
    first = (PROMPTS_DIR / name).read_text().splitlines()[0]
    for status in PROMPT_STATUSES:
        if first.startswith(f"# STATUS: {status}"):
            return status
    raise ValueError(f"prompts/{name} has no '# STATUS:' header line")


def unfinished_prompts() -> list[str]:
    return sorted(
        p.name for p in PROMPTS_DIR.iterdir() if p.is_file() and prompt_status(p.name) != "FINAL"
    )


def _ignored(key: str, ignore: tuple[str, ...]) -> bool:
    return any(key == i or key.startswith(f"{i}.") for i in ignore)


def check_real_run_ready(cfg: DictConfig, ignore: Iterable[str] = ()) -> None:
    ignore = tuple(ignore)
    problems = [f"config value {k} is ???" for k in find_missing(cfg) if not _ignored(k, ignore)]
    problems += [f"prompts/{p} is not FINAL" for p in unfinished_prompts() if p not in ignore]
    if problems:
        raise PlaceholderError(
            f"profile={cfg.profile} cannot run with owner decisions pending:\n  - "
            + "\n  - ".join(problems)
        )
