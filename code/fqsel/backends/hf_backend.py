"""Hugging Face Transformers backend: used for profile=tiny and local tests (owner decision 2026-10-23)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fqsel.backends.base import LMBackend

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class HFBackend(LMBackend):
    def __init__(self, path: str, revision: str | None = None, dtype: str = "float32", device: str = "cpu"):
        from transformers import AutoTokenizer

        self.name = path
        self.revision = revision
        self.dtype = DTYPES[dtype]
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        self._model = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                self.name, revision=self.revision, dtype=self.dtype
            ).to(self.device)
            self._model.eval()
        return self._model

    @torch.no_grad()
    def token_logprobs(self, sequences: Sequence[Sequence[int]]) -> list[list[float | None]]:
        out = []
        for seq in sequences:
            ids = torch.tensor([list(seq)], device=self.device)
            logits = self.model(ids).logits.float()
            logp = torch.log_softmax(logits[0, :-1], dim=-1)
            picked = logp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1).tolist()
            out.append([None] + picked)
        return out

    @torch.no_grad()
    def generate(self, prompts, n, temperature, top_p, max_new_tokens, seeds) -> list[list[str]]:
        results = []
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        for prompt, seed in zip(prompts, seeds, strict=True):
            torch.manual_seed(seed % (2**63 - 1))
            ids = torch.tensor([self.encode(prompt)], device=self.device)
            sample = temperature > 0
            gen = self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=sample,
                temperature=temperature if sample else None,
                top_p=top_p if sample else None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=pad,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new = gen[:, ids.shape[1]:]
            results.append([self.tokenizer.decode(row, skip_special_tokens=True) for row in new])
        return results
