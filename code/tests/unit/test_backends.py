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
@pytest.mark.spec("S2-03")
def test_vllm_matches_fp32_transformers_nll():
    """Spec 2 cross-check on the GPU node: answer_nll from vLLM (bf16) and from fp32 Transformers agree within
    0.05 nats on 200 random pool items of the current pool. Run from code/ with `pytest -m gpu` after the pool
    stage; needs room for one 7B model at a time (vLLM is released before the fp32 copy loads)."""
    import gc

    import torch

    from fqsel.backends.hf_backend import HFBackend
    from fqsel.backends.vllm_backend import VLLMBackend
    from fqsel.config import load_config
    from fqsel.ids import rng
    from fqsel.io import read_table
    from fqsel.nll import answer_nll
    from fqsel.prompts import build_messages
    from fqsel.stages import get_stage

    cfg = load_config(["profile=full"], ignore=get_stage("surprisal").ignores)
    pool = read_table("pool", cfg)
    items = pool[(pool["split"] == "pool") & ~pool["is_derived"]].sort_values("item_id")
    n = int(cfg.surprisal.crosscheck_n)
    pick = items.iloc[sorted(rng(cfg.global_seed, "nll_crosscheck").permutation(len(items))[:n])]
    msgs = [build_messages(r.to_dict(), "direct", with_answer=True) for _, r in pick.iterrows()]
    base = cfg.model.base
    vllm = VLLMBackend(base.name, base.revision, base.dtype, base.min_context)
    a = answer_nll(vllm, msgs)["nll_sum"].to_numpy()
    del vllm
    gc.collect()
    torch.cuda.empty_cache()
    hf = HFBackend(base.name, revision=base.revision, dtype="float32", device="cuda")
    b = answer_nll(hf, msgs)["nll_sum"].to_numpy()
    worst = float(abs(a - b).max())
    assert worst <= float(cfg.surprisal.crosscheck_tol_nats), f"max |vLLM - fp32| = {worst:.4f} nats"


@pytest.mark.spec("S2-10")
def test_sampling_in_a_fresh_process_reproduces_identical_text(model_cache):
    """20 prompts sampled with their stable seeds give the same text in a new Python process."""
    from fqsel.backends.hf_backend import HFBackend
    from fqsel.backends.toy import build_toy_model

    path = str(build_toy_model(model_cache / "toy_model"))
    prompts = [f"<|user|>\nQuestion {i}: what was the change?<|end|>\n<|assistant|>\n" for i in range(20)]
    seeds = [1000 + i for i in range(20)]
    here = HFBackend(path).generate(prompts, 2, 0.7, 0.95, 12, seeds)
    code = (f"from fqsel.backends.hf_backend import HFBackend; "
            f"print(repr(HFBackend({path!r}).generate({prompts!r}, 2, 0.7, 0.95, 12, {seeds!r})))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == repr(here)
