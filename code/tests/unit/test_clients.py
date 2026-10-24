"""Jev and strong-LLM clients: shared interface, caching, retries (Spec 1, Client interface)."""

from __future__ import annotations

import jsonschema
import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.jev import JevClient, MockJev, RealJev
from fqsel.clients.retry import TransientError, with_retries
from fqsel.clients.strong_llm import MockStrongLLM, StrongClient


class FakeRealJev:
    """Stands in for the real backend until TypeSafe's documentation is available: same signature,
    probabilities returned through the same (probs, info) contract."""

    model_version = "fake-real-1"

    def __init__(self):
        self.calls = 0

    def decide(self, prompt, options):
        self.calls += 1
        return {options[0]: 0.7, options[1]: 0.3}, {"request_id": "r1", "cost_usd": 0.000044}


@pytest.mark.spec("S1-16")
@pytest.mark.parametrize("backend_cls", [MockJev, FakeRealJev])
def test_jev_interface(tmp_path, backend_cls):
    client = JevClient(backend_cls(), JsonCache(tmp_path / "c"), tmp_path / "log.jsonl")
    probs = client.decide("prompt text", ["yes", "no"])
    assert list(probs) == ["yes", "no"]
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in probs.values())
    assert abs(sum(probs.values()) - 1.0) <= 1e-6
    assert isinstance(client.model_version, str) and client.model_version


@pytest.mark.spec("S1-16")
def test_real_jev_refuses_until_documented(cfg):
    with pytest.raises(NotImplementedError):
        RealJev(cfg)


def test_mock_jev_is_deterministic():
    assert MockJev().decide("p", ["yes", "no"]) == MockJev().decide("p", ["yes", "no"])
    assert MockJev().decide("p", ["yes", "no"])[0] != MockJev().decide("q", ["yes", "no"])[0]


def test_cache_hit_sends_no_request(tmp_path):
    backend = FakeRealJev()
    client = JevClient(backend, JsonCache(tmp_path / "c"))
    client.decide("p", ["yes", "no"])
    client.decide("p", ["yes", "no"])
    assert backend.calls == 1 and client.requests_sent == 1
    client.decide("p", ["yes", "no"], repeat=1)  # repeats are re-sent on purpose
    assert backend.calls == 2


def test_cache_key_includes_model_version(tmp_path):
    cache = JsonCache(tmp_path / "c")
    a = FakeRealJev()
    JevClient(a, cache).decide("p", ["yes", "no"])
    b = FakeRealJev()
    b.model_version = "fake-real-2"
    JevClient(b, cache).decide("p", ["yes", "no"])
    assert b.calls == 1


def test_invalid_probabilities_rejected(tmp_path):
    class Bad(FakeRealJev):
        def decide(self, prompt, options):
            return {options[0]: 0.7, options[1]: 0.7}, {}

    with pytest.raises(ValueError):
        JevClient(Bad(), JsonCache(tmp_path / "c")).decide("p", ["yes", "no"])


def test_retry_backoff():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientError("rate limited")
        return "ok"

    assert with_retries(flaky, 5, 1.0, 60.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]
    with pytest.raises(TransientError):
        with_retries(lambda: (_ for _ in ()).throw(TransientError("x")), 2, 1.0, 60.0, sleep=lambda s: None)


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
