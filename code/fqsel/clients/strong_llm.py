"""Strong LLM (model.strong) behind one interface, with a seeded mock (Specs 1 and 3).

generate_json(prompt, schema, temperature, seed) -> dict conforming to `schema`, using the API's
structured-output mode. Responses are cached by SHA-256 of (model version, prompt, schema, temperature, seed).

The real backend is Qwen3-235B-A22B-Instruct-2507 via OpenRouter (owner decision 2026-10-24), with one
provider endpoint pinned (no fallbacks), JSON-schema-constrained output, explicit temperature and seed.
Every response's provider is checked against the pinned one and logged with its request ID and usage.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import jsonschema
from omegaconf import DictConfig, OmegaConf

from fqsel.clients.cache import JsonCache, cache_key
from fqsel.clients.retry import RateLimited, TransientError, with_retries
from fqsel.config import cache_dir
from fqsel.ids import stable_seed
from fqsel.prompts import single_user_message
from fqsel.provenance import now_iso


class StrongBackend(Protocol):
    model_version: str

    def generate_json(self, prompt: str, schema: dict, temperature: float, seed: int) -> dict: ...

    def generate_text(self, prompt: str, sampling: dict, seed: int) -> str:
        """Free-text sample; `sampling` holds temperature, top_p, top_k, min_p and max_tokens."""


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens")


def sc_sampling(cfg: DictConfig) -> dict:
    """Self-consistency sampling parameters (Spec 3; P5, owner decision 2026-10-24)."""
    sc = cfg.model.strong.self_consistency
    return {k: (int(sc[k]) if k in ("top_k", "max_tokens") else float(sc[k])) for k in SAMPLING_KEYS}


def sc_seed(cfg: DictConfig, item_id: str, k: int) -> int:
    """Seed of self-consistency sample k for an item: stable_seed(global_seed, item_id, "sc", k)."""
    return stable_seed(cfg.global_seed, item_id, cfg.model.strong.self_consistency.seed_label, k)


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

    def generate_text(self, prompt: str, sampling: dict, seed: int) -> str:
        self.calls.append({"prompt": prompt, "sampling": dict(sampling), "seed": seed})
        return f"Reasoning briefly.\nAnswer: {stable_seed('mockstrong-text', prompt, seed) % 100}"


class ProviderMismatch(RuntimeError):
    """A response came from a provider other than the pinned one."""


class RealStrongLLM:
    """OpenRouter chat-completions client for model.strong, pinned to one provider endpoint."""

    def __init__(self, cfg: DictConfig, log_path: Path | None = None, sleep=None):
        s = cfg.model.strong
        self.name = s.name
        self.endpoint = s.endpoint
        self.provider = OmegaConf.to_container(s.provider, resolve=True)
        self.expected_provider = s.expected_provider
        self.timeout = s.timeout_s
        self.retry = {k: s.retry[k] for k in ("max_attempts", "base_delay_s", "max_delay_s")}
        self.sleep = sleep
        self.log_path = log_path
        self._lock = threading.Lock()
        self.key = os.environ.get(s.api_key_env)
        if not self.key:
            raise RuntimeError(f"environment variable {s.api_key_env} is not set")
        self.model_version = f"{s.name}@{'/'.join(self.provider['order'])}"

    def request_body(self, prompt: str, temperature: float, seed: int, schema: dict | None = None,
                     sampling: dict | None = None) -> dict:
        body = {
            "model": self.name,
            "messages": single_user_message(prompt),
            "temperature": temperature,
            "seed": int(seed % 2**31),
            "provider": self.provider,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "strict": True, "schema": schema}}
        for k, v in (sampling or {}).items():
            if k != "temperature":
                body[k] = v
        return body

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("retry-after")
                raise RateLimited(float(ra) if ra else None) from e
            if e.code >= 500:
                raise TransientError(f"HTTP {e.code}") from e
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            raise TransientError(str(e)) from e

    def generate_json(self, prompt: str, schema: dict, temperature: float, seed: int) -> dict:
        return json.loads(self._complete(self.request_body(prompt, temperature, seed, schema=schema)))

    def generate_text(self, prompt: str, sampling: dict, seed: int) -> str:
        return self._complete(self.request_body(prompt, sampling["temperature"], seed, sampling=sampling))

    def _complete(self, body: dict) -> str:
        kwargs = {} if self.sleep is None else {"sleep": self.sleep}
        data = with_retries(lambda: self._post(body), **self.retry, **kwargs)
        provider = data.get("provider")
        if provider != self.expected_provider:
            raise ProviderMismatch(f"response served by {provider!r}, expected {self.expected_provider!r}")
        if self.log_path is not None:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"id": data.get("id"), "provider": provider, "model": data.get("model"),
                                        "usage": data.get("usage"), "temperature": body["temperature"],
                                        "seed": body["seed"], "at": now_iso()}, sort_keys=True) + "\n")
        return data["choices"][0]["message"]["content"]


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

    def generate_text(self, prompt: str, sampling: dict, seed: int) -> str:
        key = cache_key(self.model_version, prompt, json.dumps(sampling, sort_keys=True), seed, "text")
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = self.backend.generate_text(prompt, sampling, seed)
        self.cache.put(key, out)
        return out


def make_strong(cfg: DictConfig, backend: StrongBackend | None = None) -> StrongClient:
    if backend is None:
        if cfg.clients.strong == "mock":
            backend = MockStrongLLM()
        else:
            backend = RealStrongLLM(cfg, log_path=cache_dir(cfg) / "logs" / "strong_calls.jsonl")
    return StrongClient(backend, JsonCache(cache_dir(cfg) / "strong"))
