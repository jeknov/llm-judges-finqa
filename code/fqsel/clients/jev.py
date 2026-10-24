"""Jev client behind one interface, with a seeded mock (Spec 1, Jev battery scoring, Client interface).

jev_ask(state, questions) -> {question_id: P(yes)}: one request to POST /v1/systemone carrying the state
once and every question as a noul question; Jev answers the questions in parallel and they cannot see one
another's answers, so batching does not couple them. Billing is per input token, so a batched request
costs roughly the state once instead of once per question.

Responses are cached keyed by SHA-256 of (model ID, state, canonical JSON of questions); repeat > 0 (used
only by the determinism check and triplicate scoring) adds the repeat index to the key so that repeated
calls are really re-sent. 429 responses are retried with exponential backoff honouring retry-after.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from omegaconf import DictConfig, OmegaConf

from fqsel.clients.cache import JsonCache, cache_key
from fqsel.clients.retry import RateLimited, TransientError, with_retries
from fqsel.config import PlaceholderError, cache_dir
from fqsel.ids import stable_seed
from fqsel.provenance import now_iso


@dataclass(frozen=True)
class JevResponse:
    probs: dict[str, float]     # P(yes) per question ID
    request_id: str
    model_version: str          # the versioned model ID the response reports
    input_tokens: int
    timestamp: str


class JevBackend(Protocol):
    model: str

    def ask(self, state: str, questions: dict[str, str]) -> JevResponse:
        """One request; raises RateLimited on HTTP 429 and TransientError on other retryable failures."""


def canonical_questions(questions: dict[str, str]) -> str:
    return json.dumps(questions, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def request_body(model: str, state: str, questions: dict[str, str], qtype: str) -> dict:
    """The request body of Spec 1: {"model", "state", "questions": {qid: {"type", "instructions"}}}."""
    return {"model": model, "state": state,
            "questions": {qid: {"type": qtype, "instructions": text} for qid, text in questions.items()}}


class MockJev:
    """Seeded, deterministic answers through the real interface. No network.

    Each question's P(yes) depends only on (state, its instructions), never on the other questions in
    the request, mirroring Jev's independence of batched questions. `rate_limit_first` simulates that
    many 429 responses (with `retry_after`) before answering.
    """

    def __init__(self, model: str = "mock-jev-1", rate_limit_first: int = 0, retry_after: float | None = 2.0):
        self.model = model
        self.rate_limit_first = rate_limit_first
        self.retry_after = retry_after
        self.requests: list[dict[str, str]] = []
        self._calls: dict[str, int] = {}
        self._lock = threading.Lock()

    def answer(self, state: str, text: str) -> float:
        u = (stable_seed("mockjev", state, text) % 10**6) / 10**6
        return 0.02 + 0.96 * u

    def ask(self, state: str, questions: dict[str, str]) -> JevResponse:
        if self.rate_limit_first > 0:
            self.rate_limit_first -= 1
            raise RateLimited(self.retry_after)
        base = f"{stable_seed(state, canonical_questions(questions)) % 10**10}"
        with self._lock:
            self.requests.append(dict(questions))
            n = self._calls.get(base, 0)
            self._calls[base] = n + 1
        tokens = (len(state) + sum(len(t) for t in questions.values())) // 4 + 260
        rid = f"mock-{base}-{n}"   # repeated identical requests get distinct IDs, like a real API
        return JevResponse({q: self.answer(state, t) for q, t in questions.items()}, rid, self.model,
                           tokens, now_iso())


class RealJev:
    """TypeSafe Jev API client (POST /v1/systemone). The response fields and the authentication header
    are set in conf/jev from TypeSafe's API reference (Spec 1, Client interface)."""

    def __init__(self, cfg: DictConfig):
        api = cfg.jev.api
        missing = [k for k in ("auth_header", "auth_scheme", "answers_field", "probability_field", "request_id_header",
                             "model_field")
                   if OmegaConf.is_missing(api, k)]
        if missing:
            raise PlaceholderError(f"jev.api.{', jev.api.'.join(missing)} must be taken from TypeSafe's API reference")
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", str(cfg.jev.model)):
            # aliases (jev-latest, jev-preview) move between versions; Spec 1 pins a versioned ID
            raise ValueError(f"jev.model must be a versioned ID such as jev-1.14.0, not {cfg.jev.model!r}")
        self.model = cfg.jev.model
        self.endpoint = cfg.jev.endpoint
        self.qtype = cfg.jev.question_type
        self.api = api
        self.key = os.environ.get(api.key_env)
        if not self.key:
            raise RuntimeError(f"environment variable {api.key_env} is not set")

    def ask(self, state: str, questions: dict[str, str]) -> JevResponse:
        body = json.dumps(request_body(self.model, state, questions, self.qtype)).encode()
        req = urllib.request.Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", self.api.auth_header: f"{self.api.auth_scheme} {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.api.timeout_s) as resp:
                data = json.loads(resp.read())
                request_id = resp.headers.get(self.api.request_id_header)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("retry-after")
                raise RateLimited(float(ra) if ra else None) from e
            if e.code >= 500:
                raise TransientError(f"HTTP {e.code}") from e
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            raise TransientError(str(e)) from e
        answers = data[self.api.answers_field]
        probs = {qid: float(answers[qid][self.api.probability_field]) for qid in questions}
        if not request_id:
            raise ValueError(f"Jev response has no {self.api.request_id_header} header")
        if str(data[self.api.model_field]) != self.model:  # Spec 1: the pinned version, never another
            raise ValueError(f"Jev answered with {data[self.api.model_field]!r}, not the pinned {self.model!r}")
        return JevResponse(probs, str(request_id), str(data[self.api.model_field]),
                           int(data["usage"]["input_tokens"]), now_iso())


class JevClient:
    """Caching, retrying, logging wrapper used by every stage."""

    def __init__(self, backend: JevBackend, cache: JsonCache, log_path: Path | None = None,
                 retry: dict | None = None, sleep=None):
        self.backend = backend
        self.cache = cache
        self.log_path = log_path
        self.retry = retry or {"max_attempts": 6, "base_delay_s": 1.0, "max_delay_s": 60.0}
        self.sleep = sleep
        self.requests_sent = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.backend.model

    def ask(self, state: str, questions: dict[str, str], repeat: int = 0) -> JevResponse:
        parts = [self.model, state, canonical_questions(questions)] + ([repeat] if repeat else [])
        key = cache_key(*parts)
        hit = self.cache.get(key)
        if hit is not None:
            return JevResponse(**hit)
        kwargs = {} if self.sleep is None else {"sleep": self.sleep}
        resp = with_retries(lambda: self.backend.ask(state, questions), **self.retry, **kwargs)
        if set(resp.probs) != set(questions) or not all(0.0 <= p <= 1.0 for p in resp.probs.values()):
            raise ValueError(f"Jev returned invalid answers for questions {sorted(questions)}: {resp.probs}")
        self.cache.put(key, asdict(resp))
        with self._lock:
            self.requests_sent += 1
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"key": key, "request_id": resp.request_id, "model": resp.model_version,
                                        "input_tokens": resp.input_tokens, "n_questions": len(questions),
                                        "at": resp.timestamp}, sort_keys=True) + "\n")
        return resp


def make_jev(cfg: DictConfig) -> JevClient:
    backend = MockJev() if cfg.clients.jev == "mock" else RealJev(cfg)
    root = cache_dir(cfg)
    retry = {k: cfg.jev.retry[k] for k in ("max_attempts", "base_delay_s", "max_delay_s")}
    return JevClient(backend, JsonCache(root / "jev"), root / "logs" / "jev_calls.jsonl", retry)


def jev_ask(client: JevClient, state: str, questions: dict[str, str]) -> dict[str, float]:
    """Spec 1's jev_ask: P(yes) per question ID for one batched request."""
    return dict(client.ask(state, questions).probs)
