"""Git commit and timestamps stamped into every output (Spec 1, Determinism)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from functools import lru_cache

from fqsel.config import CODE_DIR


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=CODE_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=CODE_DIR, capture_output=True, text=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
