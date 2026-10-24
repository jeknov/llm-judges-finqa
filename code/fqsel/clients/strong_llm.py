"""Strong LLM (model.strong) behind one interface, with a seeded mock (Specs 1 and 3).

generate_json(prompt, schema, temperature, seed) -> dict conforming to `schema`, using the API's
structured-output mode. Responses are cached by SHA-256 of (model version, prompt, schema, temperature, seed).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

import jsonschema
from omegaconf import DictConfig

from fqsel.clients.cache import JsonCache, cache_key
from fqsel.config import cache_dir
from fqsel.ids import stable_seed


class StrongBackend(Protocol):
    model_version: str

    def generate_json(self, prompt: str, schema: dict, temperature: float, seed: int) -> dict: ...


class MockStrongLLM:
    """Deterministic structured outputs through the real interface. `responder` overrides the default
    answer in tests (for example to return holds_if_answer_true = false)."""

    model_version = "mock-strong-1"

    def __init__(self, responder: Callable[[str, dict, float, int], dict] | None = None):
        self.responder = responder
        self.calls: list[dict] = []

    def generate_json(self, prompt: str, schema: dict, temperature: float, seed: int) -> dict:
        self.calls.append({"prompt": prompt, "temperature": temperature, "seed": seed})
        if self.responder is not None:
            return self.responder(prompt, schema, temperature, seed)
        n = stable_seed("mockstrong", prompt) % 1000
        return {
            "check_text": f"The stated answer is consistent with the figures given in the context (ref {n}).",
            "check_type": "unit",
            "holds_if_answer_true": True,
        }


class RealStrongLLM:
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError(
            "Real strong-LLM client pending the owner's choice of model.strong (external facts #1)"
        )


class StrongClient:
    def __init__(self, backend: StrongBackend, cache: JsonCache):
        self.backend = backend
        self.cache = cache

    @property
    def model_version(self) -> str:
        return self.backend.model_version

    def generate_json(self, prompt: str, schema: dict, temperature: float, seed: int) -> dict:
        key = cache_key(self.model_version, prompt, json.dumps(schema, sort_keys=True), temperature, seed)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = self.backend.generate_json(prompt, schema, temperature, seed)
        jsonschema.validate(out, schema)
        self.cache.put(key, out)
        return out


def make_strong(cfg: DictConfig, backend: StrongBackend | None = None) -> StrongClient:
    if backend is None:
        backend = MockStrongLLM() if cfg.clients.strong == "mock" else RealStrongLLM(cfg)
    return StrongClient(backend, JsonCache(cache_dir(cfg) / "strong"))
