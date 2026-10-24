"""Jev client behind one interface, with a seeded mock (Spec 1, Jev battery scoring, Client interface).

jev_decide(prompt, options) -> {option: probability}, probabilities summing to 1. Responses are cached
keyed by SHA-256 of (jev_model_version, prompt, options); repeat > 0 (used only by the determinism check
and triplicate scoring) adds the repeat index to the key so that repeated calls are really re-sent.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from omegaconf import DictConfig

from fqsel.clients.cache import JsonCache, cache_key
from fqsel.config import cache_dir
from fqsel.ids import stable_seed
from fqsel.provenance import now_iso


class JevBackend(Protocol):
    model_version: str

    def decide(self, prompt: str, options: list[str]) -> tuple[dict[str, float], dict]:
        """Return (probabilities per option, call info such as request_id and cost)."""


class MockJev:
    """Seeded, deterministic probabilities through the real interface. No network."""

    model_version = "mock-jev-1"

    def decide(self, prompt: str, options: list[str]) -> tuple[dict[str, float], dict]:
        u = (stable_seed("mockjev", prompt, *options) % 10**6) / 10**6
        first = 0.02 + 0.96 * u
        probs = {options[0]: first}
        rest = options[1:]
        for opt in rest:
            probs[opt] = (1.0 - first) / len(rest)
        return probs, {"request_id": f"mock-{stable_seed(prompt) % 10**8}", "cost_usd": 0.0}


class RealJev:
    """TypeSafe Jev API client. The request format, model version string and how options are passed must
    come from TypeSafe's documentation (Blocking questions: external facts #2); nothing is assumed here."""

    def __init__(self, cfg: DictConfig):
        raise NotImplementedError(
            "Real Jev client pending TypeSafe API documentation (Implementation Plan, external facts #2)"
        )


class JevClient:
    """Caching, logging wrapper used by every stage."""

    def __init__(self, backend: JevBackend, cache: JsonCache, log_path: Path | None = None):
        self.backend = backend
        self.cache = cache
        self.log_path = log_path
        self.requests_sent = 0
        self._lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return self.backend.model_version

    def decide(self, prompt: str, options: list[str], repeat: int = 0) -> dict[str, float]:
        parts = [self.model_version, prompt, list(options)] + ([repeat] if repeat else [])
        key = cache_key(*parts)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        probs, info = self.backend.decide(prompt, list(options))
        if set(probs) != set(options) or abs(sum(probs.values()) - 1.0) > 1e-6:
            raise ValueError(f"Jev returned invalid probabilities: {probs}")
        probs = {o: float(probs[o]) for o in options}
        self.cache.put(key, probs)
        with self._lock:
            self.requests_sent += 1
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"key": key, "at": now_iso(), **info}, sort_keys=True) + "\n")
        return probs


def make_jev(cfg: DictConfig) -> JevClient:
    backend = MockJev() if cfg.clients.jev == "mock" else RealJev(cfg)
    root = cache_dir(cfg)
    return JevClient(backend, JsonCache(root / "jev"), root / "logs" / "jev_calls.jsonl")
