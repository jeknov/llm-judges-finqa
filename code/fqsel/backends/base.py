"""The single language-model interface shared by the HF (tiny/local) and vLLM (GPU) backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class LMBackend(ABC):
    name: str

    @property
    @abstractmethod
    def tokenizer(self): ...

    def chat_text(self, messages: Sequence[dict], add_generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def encode(self, text: str) -> list[int]:
        """Token IDs of already-templated text; special tokens come from the template text itself."""
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @abstractmethod
    def token_logprobs(self, sequences: Sequence[Sequence[int]]) -> list[list[float | None]]:
        """Per position i, log P(token_i | tokens_<i); position 0 is None."""

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        n: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seeds: Sequence[int],
    ) -> list[list[str]]:
        """n completions per templated prompt, each prompt sampled with its own seed."""
