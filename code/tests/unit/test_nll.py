"""answer_nll and the answer span (Spec 2, Teacher-forced answer NLL; Implementation Plan M4 tests)."""

from __future__ import annotations

import pytest

from fqsel.backends.base import LMBackend
from fqsel.backends.toy import build_toy_model
from fqsel.nll import answer_nll, answer_span
from fqsel.prompts import build_messages
from tests.fixtures.golden_items import golden_items

# Hand-picked answers with leading spaces, percent signs, negatives, accounting parentheses, units and spans.
ANSWERS = ["12.5%", "-3.2", "$1,234", "(5.6)", "12.76 percent", "yes", "no", "0.5", "-0.41", "1,200",
           "3.8 million", "Europe", "Europe; Asia", "$ 1,200", "2019", "−7", "45%", "0", "1.25", "N/A"]


class KnownLogprobs(LMBackend):
    """Wraps a real tokenizer; position i has log-probability -(i + 1) / 1000, so sums are known exactly."""

    name = "known"

    def __init__(self, tokenizer):
        self._tok = tokenizer

    @property
    def tokenizer(self):
        return self._tok

    def token_logprobs(self, sequences):
        return [[None] + [-(i + 1) / 1000 for i in range(1, len(s))] for s in sequences]

    def generate(self, *a, **k):
        raise NotImplementedError


class MergingTokenizer:
    """Character tokenizer that merges ': 4' into one token, so the 'Answer:' boundary is crossed."""

    chat_template = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</>{% endfor %}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}</>" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text.startswith(": 4", i):
                ids.append(100000)
                i += 3
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def toy_tok(tmp_path_factory):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(build_toy_model(tmp_path_factory.mktemp("toy") / "m")))


def _messages(answer):
    item = dict(golden_items()["tatqa:q-arith"], gold_answer=answer)
    return build_messages(item, "direct", with_answer=True)


@pytest.mark.spec("S2-08")
def test_known_logprobs_give_the_exact_sum(toy_tok):
    backend = KnownLogprobs(toy_tok)
    msgs = _messages("12.5%")
    span = answer_span(backend, msgs)
    expected = sum((i + 1) / 1000 for i in range(span.start, span.end))
    row = answer_nll(backend, [msgs]).iloc[0]
    assert row["nll_sum"] == pytest.approx(expected, abs=1e-12)
    assert row["n_answer_tokens"] == span.end - span.start
    assert row["nll_mean"] == pytest.approx(expected / row["n_answer_tokens"])


@pytest.mark.spec("S2-08")
@pytest.mark.spec("S2-02")
@pytest.mark.parametrize("answer", ANSWERS)
def test_span_covers_exactly_the_answer_tokens(toy_tok, answer):
    span = answer_span(KnownLogprobs(toy_tok), _messages(answer))
    assert not span.boundary_adjusted
    assert toy_tok.decode(span.ids[span.start: span.end]) == " " + answer   # leading space belongs to the answer
    assert toy_tok.decode(span.ids[: span.start]).endswith("Answer:")


def test_boundary_merge_moves_the_start_earlier_and_is_flagged():
    backend = KnownLogprobs(MergingTokenizer())
    span = answer_span(backend, _messages("42"))
    assert span.boundary_adjusted
    assert span.ids[span.start] == 100000          # the merged ': 4' token is now inside the span
    assert answer_nll(backend, [_messages("42")]).iloc[0]["boundary_adjusted"]


def test_rejects_messages_without_an_answer_turn(toy_tok):
    with pytest.raises(ValueError):
        answer_span(KnownLogprobs(toy_tok), build_messages(golden_items()["tatqa:q-arith"], "direct"))
