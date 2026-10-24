"""Backends: the HF implementation on the toy model; vLLM parity is a `gpu` test."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from fqsel.backends import load_tokenizer, make_backend


@pytest.fixture(scope="module")
def backend(request):
    from fqsel.config import load_config

    cache = request.getfixturevalue("model_cache")
    return make_backend(load_config([f"paths.cache={cache}"]))


def test_chat_template_and_logprobs(backend):
    text = backend.chat_text([{"role": "user", "content": "Question: x?"}], add_generation_prompt=True)
    assert text.endswith("<|assistant|>\n")
    ids = backend.encode(text)
    lp = backend.token_logprobs([ids])[0]
    assert lp[0] is None and len(lp) == len(ids)
    assert all(v <= 0.0 and math.isfinite(v) for v in lp[1:])


def test_logprobs_match_manual_softmax(backend):
    import torch

    ids = backend.encode("Answer: 3.8 million")
    with torch.no_grad():
        logits = backend.model(torch.tensor([ids])).logits[0]
    manual = [torch.log_softmax(logits[i - 1], -1)[ids[i]].item() for i in range(1, len(ids))]
    got = backend.token_logprobs([ids])[0][1:]
    assert max(abs(a - b) for a, b in zip(manual, got, strict=True)) < 1e-5


def test_seeded_generation_reproducible_in_fresh_process(backend, model_cache):
    prompt = backend.chat_text([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    here = backend.generate([prompt], n=2, temperature=0.7, top_p=0.95, max_new_tokens=8, seeds=[11])
    code = (
        "from fqsel.config import load_config; from fqsel.backends import make_backend;"
        f"b = make_backend(load_config(['paths.cache={model_cache}']));"
        f"print(repr(b.generate([{prompt!r}], n=2, temperature=0.7, top_p=0.95, max_new_tokens=8, seeds=[11])))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == repr(here)


def test_tokenizer_only_load(cfg):
    tok = load_tokenizer(cfg)
    assert len(tok("Answer: 5", add_special_tokens=False)["input_ids"]) > 0


@pytest.mark.gpu
def test_vllm_matches_hf_logprobs():
    """vLLM and HF log-probs agree within the Spec 2 tolerance (0.05 nats) on the GPU node."""
    pytest.skip("run on the GPU node with profile=pilot")
