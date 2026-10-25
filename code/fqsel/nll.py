"""Teacher-forced answer NLL (Spec 2), `answer_nll`: the one place the answer span is located.

The messages end with the assistant turn `Answer: {answer}`. The full conversation is chat-templated and
tokenized once; the span is located by tokenizing the templated text up to and including "Answer:" and up
to the end of the answer. The boundary sits before the space after the colon, because BPE tokenizers
attach a leading space to the next token (" 12"); a boundary after the space would split that token for
almost every answer. The answer span therefore starts with the space-prefixed first answer token, and
decoding it gives the answer with one leading space. If the tokenizer merges tokens across a boundary, so that the shorter
tokenization is not an exact prefix of the full one, the boundary moves earlier one token at a time until
it is, and the item is flagged `boundary_adjusted`. Only answer tokens count; the prompt, the "Answer:"
prefix and the end-of-turn token do not. Spec 3 and Spec 6 call this with different messages and never
change the span logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from fqsel.backends.base import LMBackend

ANSWER_PREFIX = "Answer: "     # the assistant turn's prefix; the span boundary is after "Answer:"
BOUNDARY = "Answer:"


@dataclass(frozen=True)
class AnswerSpan:
    ids: list[int]
    start: int                 # first answer token
    end: int                   # one past the last answer token
    boundary_adjusted: bool


def _prefix_len(ids: list[int], part: list[int]) -> tuple[int, bool]:
    """Length of the longest prefix of `part` that is also a prefix of `ids`, and whether it was shortened."""
    n = len(part)
    shortened = False
    while n > 0 and ids[:n] != part[:n]:
        n -= 1
        shortened = True
    return n, shortened


def answer_span(backend: LMBackend, messages: Sequence[dict]) -> AnswerSpan:
    last = messages[-1]
    if last["role"] != "assistant" or not last["content"].startswith(ANSWER_PREFIX):
        raise ValueError("answer_nll needs messages ending with the assistant turn 'Answer: <answer>'")
    content = last["content"]
    full = backend.chat_text(messages)
    at = full.rfind(content)
    if at < 0:
        raise ValueError("the chat template altered the assistant turn; cannot locate the answer")
    ids = backend.encode(full)
    start, adj_start = _prefix_len(ids, backend.encode(full[: at + len(BOUNDARY)]))
    end, adj_end = _prefix_len(ids, backend.encode(full[: at + len(content)]))
    if end <= start:
        raise ValueError("empty answer span")
    return AnswerSpan(ids, start, end, adj_start or adj_end)


def answer_nll(backend: LMBackend, message_lists: Sequence[Sequence[dict]]) -> pd.DataFrame:
    """One row per message list: nll_sum (nats), n_answer_tokens, nll_mean, boundary_adjusted."""
    spans = [answer_span(backend, m) for m in message_lists]
    logprobs = backend.token_logprobs([s.ids for s in spans])
    rows = []
    for span, lp in zip(spans, logprobs, strict=True):
        vals = lp[span.start: span.end]
        if any(v is None for v in vals):
            raise ValueError("missing log-probability inside the answer span")
        nll = -float(sum(vals))
        n = span.end - span.start
        rows.append({"nll_sum": nll, "n_answer_tokens": n, "nll_mean": nll / n,
                     "boundary_adjusted": span.boundary_adjusted})
    return pd.DataFrame(rows, columns=["nll_sum", "n_answer_tokens", "nll_mean", "boundary_adjusted"])
