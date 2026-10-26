"""LLM judges for the judge comparison (owner decision 2026-10-25), behind one interface with a seeded mock.

ask(prompt) -> JudgeResponse: one OpenRouter chat request that generates a single token at temperature 0
and returns that token's top log-probabilities. P(yes) = p_yes / (p_yes + p_no), where p_yes and p_no sum
the probabilities of every top token reading "yes" or "no" after stripping whitespace, tokenizer word
markers and case; `mass` = p_yes + p_no shows how far the judge kept to the yes/no format. When neither
appears among the top tokens, P(yes) is None.

Each judge is pinned to one provider endpoint and precision (conf/judges), with no fallbacks; a response
from any other provider, without log-probabilities, or carrying reasoning text is rejected. Responses are
cached by SHA-256 of (judge model version, prompt, request parameters); repeat > 0, used only by the
determinism check, adds the repeat index so that the request is really re-sent.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from omegaconf import DictConfig, OmegaConf

from fqsel.clients.cache import JsonCache, cache_key
from fqsel.clients.openrouter import ProviderMismatch, post_chat
from fqsel.clients.retry import with_retries
from fqsel.config import cache_dir
from fqsel.ids import stable_seed
from fqsel.prompts import single_user_message
from fqsel.provenance import now_iso

_WORD_MARKERS = "▁Ġ"   # SentencePiece "▁" and byte-level BPE "Ġ" word-start markers


@dataclass(frozen=True)
class JudgeResponse:
    p_yes_raw: float            # summed probability of the "yes" tokens among the top log-probabilities
    p_no_raw: float
    top_token: str              # the token the judge generated
    provider: str
    served_model: str           # the model ID the response reports
    request_id: str
    input_tokens: int
    cost: float | None          # USD, as OpenRouter reports it
    timestamp: str

    @property
    def mass(self) -> float:
        return self.p_yes_raw + self.p_no_raw

    @property
    def p_yes(self) -> float | None:
        return self.p_yes_raw / self.mass if self.mass > 0 else None


def normalise_token(token: str) -> str:
    return token.strip().strip(_WORD_MARKERS).strip().lower()


def yes_no_mass(top_logprobs: list[dict]) -> tuple[float, float]:
    """(p_yes, p_no) summed over the top tokens that read "yes" or "no"."""
    p = {"yes": 0.0, "no": 0.0}
    for entry in top_logprobs:
        word = normalise_token(entry["token"])
        if word in p:
            p[word] += math.exp(entry["logprob"])
    return p["yes"], p["no"]


class JudgeBackend(Protocol):
    model_version: str

    def ask(self, prompt: str) -> JudgeResponse:
        """One request; raises RateLimited on HTTP 429 and TransientError on other retryable failures."""


class MockJudge:
    """Seeded, deterministic answers through the real interface. No network.

    P(yes) depends only on the prompt; `mass` (default 0.9) is split between "Yes" and " yes" variants
    and "no", so that summing variants is exercised.
    """

    def __init__(self, name: str = "mock-judge", mass: float = 0.9):
        self.model_version = f"{name}@mock"
        self.mass = mass
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def probability(self, prompt: str) -> float:
        u = (stable_seed("mockjudge", self.model_version, prompt) % 10**6) / 10**6
        return 0.02 + 0.96 * u

    def ask(self, prompt: str) -> JudgeResponse:
        with self._lock:
            self.prompts.append(prompt)
            n = len(self.prompts)
        p = self.probability(prompt)
        return JudgeResponse(p_yes_raw=self.mass * p, p_no_raw=self.mass * (1 - p),
                             top_token="Yes" if p >= 0.5 else "No", provider="mock", served_model=self.model_version,
                             request_id=f"mock-{n}", input_tokens=len(prompt) // 4, cost=None, timestamp=now_iso())


class RealJudge:
    """OpenRouter chat-completions client for one judge, pinned to one provider endpoint."""

    def __init__(self, cfg: DictConfig, judge: str, log_path: Path | None = None, sleep=None):
        j, m = cfg.judges, cfg.judges.models[judge]
        self.name = m.name
        self.endpoint = j.endpoint
        self.provider = OmegaConf.to_container(m.provider, resolve=True)
        self.expected_provider = m.expected_provider
        self.reasoning = m.reasoning
        self.params = request_params(cfg)
        self.seed = int(stable_seed(cfg.global_seed, j.seed_label) % 2**31)
        self.timeout = j.timeout_s
        self.retry = {k: j.retry[k] for k in ("max_attempts", "base_delay_s", "max_delay_s")}
        self.sleep = sleep
        self.log_path = log_path
        self._lock = threading.Lock()
        self.key = os.environ.get(j.api_key_env)
        if not self.key:
            raise RuntimeError(f"environment variable {j.api_key_env} is not set")
        self.model_version = f"{m.name}@{'/'.join(self.provider['order'])}"

    def request_body(self, prompt: str) -> dict:
        body = {"model": self.name, "messages": single_user_message(prompt), **self.params, "logprobs": True,
                "seed": self.seed, "provider": self.provider, "usage": {"include": True}}
        if self.reasoning is not None:
            body["reasoning"] = {"enabled": bool(self.reasoning)}
        return body

    def ask(self, prompt: str) -> JudgeResponse:
        body = self.request_body(prompt)
        kwargs = {} if self.sleep is None else {"sleep": self.sleep}
        data = with_retries(lambda: post_chat(self.endpoint, self.key, body, self.timeout), **self.retry, **kwargs)
        provider = data.get("provider")
        if provider != self.expected_provider:
            raise ProviderMismatch(f"response served by {provider!r}, expected {self.expected_provider!r}")
        choice = data["choices"][0]
        if choice["message"].get("reasoning"):
            raise ValueError(f"{self.name} returned reasoning text; the answer token is not the first token")
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content or not content[0].get("top_logprobs"):
            raise ValueError(f"{self.name} via {provider} returned no top log-probabilities")
        p_yes, p_no = yes_no_mass(content[0]["top_logprobs"])
        usage = data.get("usage") or {}
        resp = JudgeResponse(p_yes_raw=p_yes, p_no_raw=p_no, top_token=content[0]["token"], provider=provider,
                             served_model=str(data.get("model")), request_id=str(data.get("id")),
                             input_tokens=int(usage.get("prompt_tokens", 0)),
                             cost=float(usage["cost"]) if usage.get("cost") is not None else None,
                             timestamp=now_iso())
        if self.log_path is not None:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"id": resp.request_id, "provider": provider, "model": resp.served_model,
                                        "usage": usage, "at": resp.timestamp}, sort_keys=True) + "\n")
        return resp


def request_params(cfg: DictConfig) -> dict:
    """The generation parameters every judge request carries (they are part of the cache key)."""
    j = cfg.judges
    return {"temperature": float(j.temperature), "max_tokens": int(j.max_tokens), "top_logprobs": int(j.top_logprobs)}


class JudgeClient:
    """Caching wrapper used by the judges comparator."""

    def __init__(self, backend: JudgeBackend, cache: JsonCache, params: dict):
        self.backend = backend
        self.cache = cache
        self.params = params
        self.requests_sent = 0
        self._lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return self.backend.model_version

    def ask(self, prompt: str, repeat: int = 0) -> JudgeResponse:
        parts = [self.model_version, prompt, json.dumps(self.params, sort_keys=True)] + ([repeat] if repeat else [])
        key = cache_key(*parts)
        hit = self.cache.get(key)
        if hit is not None:
            return JudgeResponse(**hit)
        resp = self.backend.ask(prompt)
        if not (0.0 <= resp.p_yes_raw <= 1.0 + 1e-6 and 0.0 <= resp.p_no_raw <= 1.0 + 1e-6):
            raise ValueError(f"{self.model_version} returned invalid probabilities: {resp}")
        self.cache.put(key, asdict(resp))
        with self._lock:
            self.requests_sent += 1
        return resp


def make_judge(cfg: DictConfig, judge: str, backend: JudgeBackend | None = None) -> JudgeClient:
    if judge not in cfg.judges.models:
        raise KeyError(f"unknown judge {judge!r}; configured: {', '.join(cfg.judges.models)}")
    if backend is None:
        if cfg.clients.judges == "mock":
            backend = MockJudge(judge)
        else:
            backend = RealJudge(cfg, judge, log_path=cache_dir(cfg) / "logs" / f"judge_{judge}_calls.jsonl")
    return JudgeClient(backend, JsonCache(cache_dir(cfg) / "judges" / judge), request_params(cfg))
