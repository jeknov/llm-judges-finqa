"""vLLM backend: used on the GPU node for pilot and full runs (owner decision 2026-10-23).

Not importable on machines without CUDA; its tests are marked `gpu`.
"""

from __future__ import annotations

from collections.abc import Sequence

from fqsel.backends.base import LMBackend


class VLLMBackend(LMBackend):
    def __init__(self, path: str, revision: str, dtype: str, max_model_len: int):
        from vllm import LLM

        self.name = path
        self.llm = LLM(
            model=path,
            revision=revision,
            tokenizer_revision=revision,
            dtype=dtype,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
        )

    @property
    def tokenizer(self):
        return self.llm.get_tokenizer()

    def token_logprobs(self, sequences: Sequence[Sequence[int]]) -> list[list[float | None]]:
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=list(s)) for s in sequences], params)
        result = []
        for seq, out in zip(sequences, outputs, strict=True):
            row: list[float | None] = [None]
            for pos, entry in enumerate(out.prompt_logprobs[1:], start=1):
                row.append(entry[seq[pos]].logprob)
            result.append(row)
        return result

    def generate(self, prompts, n, temperature, top_p, max_new_tokens, seeds) -> list[list[str]]:
        from vllm import SamplingParams, TokensPrompt

        params = [
            SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=s % (2**31))
            for s in seeds
        ]
        reqs = [TokensPrompt(prompt_token_ids=self.encode(p)) for p in prompts]
        outputs = self.llm.generate(reqs, params)
        return [[c.text for c in out.outputs] for out in outputs]
