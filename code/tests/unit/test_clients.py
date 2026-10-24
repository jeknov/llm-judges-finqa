"""Jev and strong-LLM clients: shared interface, batching, caching, 429 handling (Spec 1, Client interface)."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import jsonschema
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.jev import JevClient, JevResponse, MockJev, RealJev, jev_ask, request_body
from fqsel.clients.retry import RateLimited, TransientError, with_retries
from fqsel.clients.strong_llm import MockStrongLLM, StrongClient
from fqsel.config import PlaceholderError

QUESTIONS = {"q1": "Is it supported?", "q2": "Is it determined?", "q3": "Does it entail X?",
             "q1p": "Backed up?", "q2p": "Only answer?", "q3p": "Must X hold?"}

# Test-only stand-ins for the response mapping that must come from TypeSafe's API reference.
FAKE_API = ["jev.api.auth_header=X-Test-Key", "jev.api.answers_field=answers",
            "jev.api.probability_field=p", "jev.api.request_id_field=id", "jev.api.model_field=model"]


class FakeHTTP:
    """Replaces urllib.request.urlopen: `n_429` rate-limit responses, then a valid batched answer."""

    def __init__(self, n_429=0, retry_after="3"):
        self.n_429, self.retry_after, self.bodies, self.headers = n_429, retry_after, [], []

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data))
        self.headers.append(dict(req.header_items()))
        if self.n_429:
            self.n_429 -= 1
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                         {"retry-after": self.retry_after}, io.BytesIO(b""))
        qs = self.bodies[-1]["questions"]
        payload = {"id": f"req-{len(self.bodies)}", "model": "jev-1.13.0",
                   "answers": {q: {"p": 0.25 + 0.05 * i} for i, q in enumerate(sorted(qs))},
                   "usage": {"input_tokens": 512}}
        return io.BytesIO(json.dumps(payload).encode())


@pytest.fixture
def real_backend(make_cfg, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    http = FakeHTTP(n_429=1)
    monkeypatch.setattr(urllib.request, "urlopen", http)
    return RealJev(make_cfg(*FAKE_API)), http


def _client(backend, tmp_path, sleeps=None):
    return JevClient(backend, JsonCache(tmp_path / "c"), tmp_path / "log.jsonl",
                     sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


@pytest.mark.spec("S1-16")
@pytest.mark.parametrize("which", ["mock", "real"])
def test_jev_interface_and_429_handling(which, tmp_path, real_backend):
    """MockJev and the real client: same arguments, one P(yes) in [0, 1] per question ID, and the same
    handling of a 429 carrying retry-after (wait at least retry-after, then succeed)."""
    if which == "mock":
        backend = MockJev(rate_limit_first=1, retry_after=3.0)
    else:
        backend, _ = real_backend
    sleeps: list[float] = []
    client = _client(backend, tmp_path, sleeps)
    probs = jev_ask(client, "STATE TEXT", QUESTIONS)
    assert set(probs) == set(QUESTIONS)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in probs.values())
    assert sleeps == [3.0]          # retry-after (3 s) exceeds the first backoff step (1 s)
    assert client.requests_sent == 1


def test_real_client_sends_one_batched_request_in_spec_format(real_backend, tmp_path):
    backend, http = real_backend
    resp = _client(backend, tmp_path).ask("STATE", QUESTIONS)
    body = http.bodies[-1]
    assert body == request_body("jev-1.13.0", "STATE", QUESTIONS, "noul")
    assert body["questions"]["q3p"] == {"type": "noul", "instructions": "Must X hold?"}
    assert http.headers[-1]["X-test-key"] == "secret"
    assert resp.model_version == "jev-1.13.0" and resp.input_tokens == 512 and resp.request_id == "req-2"


def test_real_client_refuses_until_api_reference_fields_are_set(cfg, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    with pytest.raises(PlaceholderError):
        RealJev(cfg)


def test_mock_answers_do_not_depend_on_batching():
    m = MockJev()
    batched = m.ask("S", QUESTIONS).probs
    single = {q: m.ask("S", {q: t}).probs[q] for q, t in QUESTIONS.items()}
    assert batched == single
    assert len(set(batched.values())) == len(QUESTIONS)


def test_cache_hit_sends_no_request_and_key_covers_model_state_questions(tmp_path):
    backend = MockJev()
    client = _client(backend, tmp_path)
    client.ask("S", QUESTIONS)
    client.ask("S", dict(reversed(list(QUESTIONS.items()))))   # same questions, other order: canonical key
    assert len(backend.requests) == 1
    client.ask("S", QUESTIONS, repeat=1)                         # repeats are re-sent on purpose
    client.ask("S2", QUESTIONS)
    client.ask("S", {**QUESTIONS, "q1": "changed"})
    assert len(backend.requests) == 4
    other = MockJev(model="jev-1.14.0")
    _client(other, tmp_path).ask("S", QUESTIONS)
    assert len(other.requests) == 1


def test_cached_response_keeps_request_id_and_tokens(tmp_path):
    client = _client(MockJev(), tmp_path)
    first = client.ask("S", QUESTIONS)
    again = _client(MockJev(), tmp_path).ask("S", QUESTIONS)
    assert again == first


def test_invalid_answers_rejected(tmp_path):
    class Bad(MockJev):
        def ask(self, state, questions):
            r = super().ask(state, questions)
            return JevResponse({**r.probs, "q1": 1.5}, r.request_id, r.model_version, r.input_tokens, r.timestamp)

    with pytest.raises(ValueError):
        _client(Bad(), tmp_path).ask("S", QUESTIONS)


def test_retry_backoff():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientError("server error")
        return "ok"

    assert with_retries(flaky, 5, 1.0, 60.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]
    with pytest.raises(TransientError):
        with_retries(lambda: (_ for _ in ()).throw(TransientError("x")), 2, 1.0, 60.0, sleep=lambda s: None)


def test_retry_after_honoured_and_backoff_kept_when_larger():
    def fail_once(retry_after):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise RateLimited(retry_after)
            return "ok"

        return fn

    for retry_after, expected in [(10.0, [10.0]), (0.5, [1.0]), (None, [1.0])]:
        sleeps: list[float] = []
        with_retries(fail_once(retry_after), 3, 1.0, 60.0, sleep=sleeps.append)
        assert sleeps == expected


def test_strong_client_validates_schema_and_caches(tmp_path):
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    mock = MockStrongLLM(responder=lambda p, s, t, seed: {"x": 1})
    client = StrongClient(mock, JsonCache(tmp_path / "s"))
    assert client.generate_json("p", schema, 0.0, 1) == {"x": 1}
    client.generate_json("p", schema, 0.0, 1)
    assert len(mock.calls) == 1
    bad = StrongClient(MockStrongLLM(responder=lambda p, s, t, seed: {"x": "one"}), JsonCache(tmp_path / "b"))
    with pytest.raises(jsonschema.ValidationError):
        bad.generate_json("p", schema, 0.0, 1)
