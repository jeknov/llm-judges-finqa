"""(added) Judge comparison client: P(yes) from first-token log-probabilities, provider pinning, caching."""

from __future__ import annotations

import io
import json
import math
import urllib.error
import urllib.request

import pytest

from fqsel.clients.cache import JsonCache
from fqsel.clients.judge_llm import (
    JudgeClient,
    MockJudge,
    RealJudge,
    make_judge,
    normalise_token,
    request_params,
    yes_no_mass,
)
from fqsel.clients.openrouter import ProviderMismatch
from fqsel.ids import stable_seed

OWNER_JUDGES = {  # owner decision 2026-10-25: model, provider endpoint, reported provider
    "llama32_3b": ("meta-llama/llama-3.2-3b-instruct", "parasail/bf16", "Parasail"),
    "llama33_70b": ("meta-llama/llama-3.3-70b-instruct", "novita/bf16", "Novita"),
    "gemma3_4b": ("google/gemma-3-4b-it", "parasail/bf16", "Parasail"),
    "gemma4_31b": ("google/gemma-4-31b-it", "novita/bf16", "Novita"),
}

# Hand-built top log-probabilities: "Yes" 0.6, " yes" 0.1, "No" 0.2, "Maybe" 0.05.
TOP = [{"token": "Yes", "logprob": math.log(0.6)}, {"token": " yes", "logprob": math.log(0.1)},
       {"token": "No", "logprob": math.log(0.2)}, {"token": "Maybe", "logprob": math.log(0.05)}]


class FakeOpenRouter:
    """Replaces urllib.request.urlopen: `n_429` rate limits, then a one-token answer with log-probabilities."""

    def __init__(self, n_429=0, provider="Parasail", top=TOP, reasoning=None, logprobs=True):
        self.n_429, self.provider, self.top, self.reasoning, self.logprobs = n_429, provider, top, reasoning, logprobs
        self.bodies = []

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data))
        if self.n_429:
            self.n_429 -= 1
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"retry-after": "3"}, io.BytesIO(b""))
        message = {"role": "assistant", "content": self.top[0]["token"]}
        if self.reasoning:
            message["reasoning"] = self.reasoning
        choice = {"message": message}
        if self.logprobs:
            choice["logprobs"] = {"content": [{"token": self.top[0]["token"], "logprob": self.top[0]["logprob"],
                                               "top_logprobs": self.top}]}
        payload = {"id": f"gen-{len(self.bodies)}", "provider": self.provider, "model": "meta-llama/llama-3.2-3b-instruct",
                   "choices": [choice], "usage": {"prompt_tokens": 812, "completion_tokens": 1, "cost": 0.0000406}}
        return io.BytesIO(json.dumps(payload).encode())


@pytest.fixture
def real(cfg, monkeypatch, tmp_path):
    def _make(judge="llama32_3b", **fake_kwargs):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
        fake = FakeOpenRouter(**fake_kwargs)
        monkeypatch.setattr(urllib.request, "urlopen", fake)
        sleeps: list[float] = []
        return RealJudge(cfg, judge, log_path=tmp_path / "judge.jsonl", sleep=sleeps.append), fake, sleeps
    return _make


def test_owner_judges_are_pinned_to_one_bf16_endpoint_without_fallbacks(cfg):
    models = cfg.judges.models
    assert set(models) == set(OWNER_JUDGES)
    for key, (name, tag, provider) in OWNER_JUDGES.items():
        m = models[key]
        assert m.name == name and m.expected_provider == provider
        assert list(m.provider.order) == [tag] and list(m.provider.quantizations) == ["bf16"]
        assert m.provider.allow_fallbacks is False and m.provider.require_parameters is True
    assert {models[k].family for k in models} == {"llama", "gemma"}


def test_token_normalisation_and_yes_no_mass():
    assert [normalise_token(t) for t in ("Yes", " yes", "▁Yes", "Ġno", "NO\n", "yes.")] == \
        ["yes", "yes", "yes", "no", "no", "yes."]
    p_yes, p_no = yes_no_mass(TOP)
    assert p_yes == pytest.approx(0.7) and p_no == pytest.approx(0.2)   # variants summed, "Maybe" ignored
    assert yes_no_mass([{"token": "Maybe", "logprob": 0.0}]) == (0.0, 0.0)


def test_request_body_pins_provider_and_reads_one_token(real, cfg):
    backend, fake, sleeps = real(n_429=1)
    resp = backend.ask("PROMPT")
    body = fake.bodies[-1]
    assert body == {"model": "meta-llama/llama-3.2-3b-instruct", "messages": [{"role": "user", "content": "PROMPT"}],
                    "temperature": 0.0, "max_tokens": 1, "top_logprobs": 20, "logprobs": True,
                    "seed": stable_seed(cfg.global_seed, "judges") % 2**31,
                    "provider": {"order": ["parasail/bf16"], "quantizations": ["bf16"],
                                 "allow_fallbacks": False, "require_parameters": True},
                    "usage": {"include": True}}
    assert sleeps == [3.0]                      # 429 retried after retry-after
    assert resp.p_yes == pytest.approx(0.7 / 0.9) and resp.mass == pytest.approx(0.9)
    assert resp.top_token == "Yes" and resp.input_tokens == 812 and resp.cost == pytest.approx(0.0000406)
    assert backend.model_version == "meta-llama/llama-3.2-3b-instruct@parasail/bf16"
    assert json.loads(backend.log_path.read_text().splitlines()[-1])["provider"] == "Parasail"


def test_gemma4_switches_reasoning_off_and_others_send_nothing(real):
    backend, fake, _ = real("gemma4_31b", provider="Novita")
    backend.ask("PROMPT")
    assert fake.bodies[-1]["reasoning"] == {"enabled": False}
    for judge in ("llama32_3b", "llama33_70b", "gemma3_4b"):
        backend, fake, _ = real(judge, provider=OWNER_JUDGES[judge][2])
        backend.ask("PROMPT")
        assert "reasoning" not in fake.bodies[-1]


@pytest.mark.parametrize("fake_kwargs,error", [
    ({"provider": "SomeOtherHost"}, ProviderMismatch),
    ({"logprobs": False}, ValueError),
    ({"reasoning": "Let me think about the table first."}, ValueError),
])
def test_responses_that_cannot_give_p_yes_are_rejected(real, fake_kwargs, error):
    backend, _, _ = real(**fake_kwargs)
    with pytest.raises(error):
        backend.ask("PROMPT")


def test_no_yes_or_no_among_top_tokens_gives_missing_p_yes(real):
    backend, _, _ = real(top=[{"token": "The", "logprob": math.log(0.9)}])
    resp = backend.ask("PROMPT")
    assert resp.mass == 0.0 and resp.p_yes is None


def test_api_key_required(cfg, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        RealJudge(cfg, "llama32_3b")


def test_cache_key_covers_judge_prompt_parameters_and_repeat(cfg, tmp_path):
    mock = MockJudge("a")
    client = JudgeClient(mock, JsonCache(tmp_path), request_params(cfg))
    first = client.ask("P")
    assert client.ask("P") == first and len(mock.prompts) == 1          # served from the cache
    client.ask("P", repeat=1)
    assert len(mock.prompts) == 2                                       # the determinism repeat is re-sent
    JudgeClient(MockJudge("b"), JsonCache(tmp_path), request_params(cfg)).ask("P")
    other_params = JudgeClient(mock, JsonCache(tmp_path), {**request_params(cfg), "top_logprobs": 5})
    other_params.ask("P")
    assert len(mock.prompts) == 3                                       # different parameters, new request


def test_make_judge_uses_the_mock_on_tiny_and_refuses_unknown_judges(cfg):
    client = make_judge(cfg, "gemma3_4b")
    assert isinstance(client.backend, MockJudge) and client.model_version == "gemma3_4b@mock"
    with pytest.raises(KeyError):
        make_judge(cfg, "gpt-5")
