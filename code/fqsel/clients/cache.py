"""On-disk response cache keyed by SHA-256 of the request parts (Spec 1, Client interface)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


def cache_key(*parts) -> str:
    return hashlib.sha256(json.dumps(list(parts), sort_keys=True, default=str).encode()).hexdigest()


class JsonCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str):
        path = self._path(key)
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, value) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")  # unique per writer thread
        tmp.write_text(json.dumps(value, sort_keys=True))
        os.replace(tmp, path)
