"""Jev and strong-LLM clients: shared interface, batching, caching, 429 handling (Spec 1, Client interface)."""

from __future__ import annotations

import http.client
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
# A live response from POST /v1/systemone (2026-10-24), with its request-ID header.
LIVE_RESPONSE = {"model": "jev-1.14.0",
                 "answers": {"q1": {"type": "noul", "noul": 0.99}, "q2": {"type": "noul", "noul": 0.01}},
                 "usage": {"input_tokens": 326, "output_tokens": 38}}
LIVE_REQUEST_ID = "req_01a16f2d056d7caa8399f76507b79875"
LIVE_QUESTIONS = {"q1": "Did revenue increase from 2018 to 2019?", "q2": "Was revenue in 2019 below 50?"}


class _Response(io.BytesIO):
    def __init__(self, payload: dict, headers: dict):
        super().__init__(json.dumps(payload).encode())
        self.headers = headers


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
        payload = {"model": "jev-1.14.0",
                   "answers": {q: {"type": "noul", "noul": 0.25 + 0.05 * i} for i, q in enumerate(sorted(qs))},
                   "usage": {"input_tokens": 512, "output_tokens": 40}}
        return _Response(payload, {"x-typesafe-request-id": f"req-{len(self.bodies)}"})


@pytest.fixture
def real_backend(make_cfg, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    http = FakeHTTP(n_429=1)
    monkeypatch.setattr(urllib.request, "urlopen", http)
    return RealJev(make_cfg()), http


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
    assert body == request_body("jev-1.14.0", "STATE", QUESTIONS, "noul")
    assert body["questions"]["q3p"] == {"type": "noul", "instructions": "Must X hold?"}
    assert http.headers[-1]["Authorization"] == "Bearer secret"
    assert resp.model_version == "jev-1.14.0" and resp.input_tokens == 512 and resp.request_id == "req-2"


def test_real_client_refuses_without_api_reference_fields(make_cfg, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    with pytest.raises(PlaceholderError):
        RealJev(make_cfg("jev.api.probability_field=???"))


def test_real_client_parses_a_live_response(make_cfg, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Response(
        LIVE_RESPONSE, {"x-typesafe-request-id": LIVE_REQUEST_ID}))
    resp = RealJev(make_cfg()).ask("Revenue was 120 in 2019 and 100 in 2018.", LIVE_QUESTIONS)
    assert resp.probs == {"q1": 0.99, "q2": 0.01} and resp.model_version == "jev-1.14.0"
    assert resp.request_id == LIVE_REQUEST_ID and resp.input_tokens == 326


def test_real_client_requires_request_id_header(make_cfg, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Response(LIVE_RESPONSE, {}))
    with pytest.raises(ValueError, match="x-typesafe-request-id"):
        RealJev(make_cfg()).ask("STATE", {"q1": "Did revenue increase?"})


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
    other = MockJev(model="jev-1.15.0")
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


class FakeOpenRouter:
    """Replaces urllib.request.urlopen for the strong-LLM client: `n_429` rate limits, then a JSON answer."""

    def __init__(self, n_429=0, provider="DeepInfra", content=None):
        self.n_429, self.provider, self.bodies = n_429, provider, []
        self.content = content or {"check_text": "The ratio is above 1.", "check_type": "ordering",
                                   "holds_if_answer_true": True}

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data))
        self.auth = req.get_header("Authorization")
        if self.n_429:
            self.n_429 -= 1
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"retry-after": "4"}, io.BytesIO(b""))
        payload = {"id": f"gen-{len(self.bodies)}", "provider": self.provider, "model": "qwen/qwen3-235b-a22b-07-25",
                   "choices": [{"message": {"role": "assistant", "content": self.content if isinstance(
                       self.content, str) else json.dumps(self.content)}}],
                   "usage": {"prompt_tokens": 900, "completion_tokens": 60}}
        return io.BytesIO(json.dumps(payload).encode())


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["check_text", "check_type", "holds_if_answer_true"],
          "properties": {"check_text": {"type": "string"}, "check_type": {"type": "string"},
                         "holds_if_answer_true": {"type": "boolean"}}}


@pytest.fixture
def openrouter(cfg, monkeypatch, tmp_path):
    from fqsel.clients.strong_llm import RealStrongLLM

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    fake = FakeOpenRouter(n_429=1)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    sleeps: list[float] = []
    backend = RealStrongLLM(cfg, log_path=tmp_path / "strong.jsonl", sleep=sleeps.append)
    return backend, fake, sleeps


def test_openrouter_request_pins_provider_and_sampling(openrouter):
    backend, fake, sleeps = openrouter
    out = backend.generate_json("PROMPT", SCHEMA, 0.7, 12345)
    assert out["holds_if_answer_true"] is True
    body = fake.bodies[-1]
    assert body["model"] == "qwen/qwen3-235b-a22b-2507"
    assert body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert body["temperature"] == 0.7 and body["seed"] == 12345
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "output", "strict": True, "schema": SCHEMA}}
    assert body["provider"] == {"order": ["deepinfra/fp8"], "quantizations": ["fp8"],
                                "allow_fallbacks": False, "require_parameters": True}
    assert fake.auth == "Bearer or-secret"
    assert sleeps == [4.0]                      # 429 retried after retry-after
    assert backend.model_version == "qwen/qwen3-235b-a22b-2507@deepinfra/fp8"


def test_openrouter_logs_provider_and_usage(openrouter, tmp_path):
    backend, _, _ = openrouter
    backend.generate_json("PROMPT", SCHEMA, 0.0, 1)
    rec = json.loads((tmp_path / "strong.jsonl").read_text().splitlines()[-1])
    assert rec["provider"] == "DeepInfra" and rec["usage"]["prompt_tokens"] == 900 and rec["id"] == "gen-2"


def test_openrouter_rejects_other_providers(cfg, monkeypatch):
    from fqsel.clients.strong_llm import ProviderMismatch, RealStrongLLM

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setattr(urllib.request, "urlopen", FakeOpenRouter(provider="SomeOtherHost"))
    with pytest.raises(ProviderMismatch):
        RealStrongLLM(cfg).generate_json("PROMPT", SCHEMA, 0.0, 1)


def test_openrouter_needs_api_key(cfg, monkeypatch):
    from fqsel.clients.strong_llm import RealStrongLLM

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        RealStrongLLM(cfg)


def test_strong_config_is_qwen_via_openrouter(cfg):
    s = cfg.model.strong
    assert s.name == "qwen/qwen3-235b-a22b-2507" and s.version == "Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert list(s.provider.order) == ["deepinfra/fp8"] and s.provider.allow_fallbacks is False


def test_self_consistency_settings_p5(cfg):
    """P5 (owner decision 2026-10-24): temperature 0.7 with Qwen's recommended top_p/top_k/min_p."""
    from fqsel.clients.strong_llm import sc_sampling

    assert cfg.model.strong.self_consistency.n_samples == 5
    assert sc_sampling(cfg) == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "max_tokens": 512}
    # Same temperature as every other sampled measure in the study (Spec 2 accuracy, Spec 4 CBI).
    assert sc_sampling(cfg)["temperature"] == cfg.model.strong.q3_retry_temperature


def test_self_consistency_seeds(cfg):
    from fqsel.clients.strong_llm import sc_seed
    from fqsel.ids import stable_seed

    seeds = [sc_seed(cfg, "finqa:x", k) for k in range(5)]
    assert seeds[0] == stable_seed(20261022, "finqa:x", "sc", 0)
    assert len(set(seeds)) == 5 and sc_seed(cfg, "finqa:y", 0) != seeds[0]


def test_openrouter_text_sample_carries_all_sampling_parameters(openrouter, cfg):
    from fqsel.clients.strong_llm import sc_sampling, sc_seed

    backend, fake, _ = openrouter
    fake.content = "Reasoning.\nAnswer: 5"
    out = backend.generate_text("Q", sc_sampling(cfg), sc_seed(cfg, "finqa:x", 2))
    body = fake.bodies[-1]
    assert out == "Reasoning.\nAnswer: 5"
    assert {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens")} == sc_sampling(cfg)
    assert body["seed"] == sc_seed(cfg, "finqa:x", 2) % 2**31
    assert "response_format" not in body and body["provider"]["allow_fallbacks"] is False


def test_strong_client_caches_text_samples_per_seed(tmp_path, cfg):
    from fqsel.clients.strong_llm import sc_sampling

    mock = MockStrongLLM()
    client = StrongClient(mock, JsonCache(tmp_path / "t"))
    s = sc_sampling(cfg)
    a = client.generate_text("Q", s, 1)
    assert client.generate_text("Q", s, 1) == a and len(mock.calls) == 1
    client.generate_text("Q", s, 2)
    assert len(mock.calls) == 2


def test_real_client_rejects_a_different_model_version(make_cfg, monkeypatch):
    """A response from any version other than the pinned one (here a hypothetical later release) is an error."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    other = {**LIVE_RESPONSE, "model": "jev-1.15.0"}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Response(
        other, {"x-typesafe-request-id": LIVE_REQUEST_ID}))
    with pytest.raises(ValueError, match="pinned"):
        RealJev(make_cfg()).ask("STATE", {"q1": "Did revenue increase?"})


@pytest.mark.parametrize("alias", ["jev-latest", "jev-preview", "jev-1.14"])
def test_real_client_refuses_aliases(make_cfg, monkeypatch, alias):
    """Aliases move to new releases (both resolved to jev-1.14.0 on 2026-10-24); only a versioned ID may be set."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    with pytest.raises(ValueError, match="versioned ID"):
        RealJev(make_cfg(f"jev.model={alias}"))


def _drop_first(real_urlopen_fake):
    """Wraps a fake urlopen so that its first call fails with a connection dropped mid-response."""
    state = {"n": 0}

    def call(req, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise http.client.IncompleteRead(b"partial")
        return real_urlopen_fake(req, timeout)
    return call


def test_openrouter_retries_a_dropped_connection(cfg, monkeypatch):
    from fqsel.clients.strong_llm import RealStrongLLM

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    fake = FakeOpenRouter()
    monkeypatch.setattr(urllib.request, "urlopen", _drop_first(fake))
    sleeps: list[float] = []
    out = RealStrongLLM(cfg, sleep=sleeps.append).generate_json("PROMPT", SCHEMA, 0.0, 1)
    assert out["holds_if_answer_true"] is True and len(sleeps) == 1 and len(fake.bodies) == 1


def test_jev_retries_a_dropped_connection(make_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    ok = lambda req, timeout=None: _Response(LIVE_RESPONSE, {"x-typesafe-request-id": LIVE_REQUEST_ID})  # noqa: E731
    monkeypatch.setattr(urllib.request, "urlopen", _drop_first(ok))
    sleeps: list[float] = []
    resp = _client(RealJev(make_cfg()), tmp_path, sleeps).ask("Revenue was 120 in 2019 and 100 in 2018.", LIVE_QUESTIONS)
    assert resp.probs == {"q1": 0.99, "q2": 0.01} and len(sleeps) == 1
