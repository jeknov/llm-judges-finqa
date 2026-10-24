"""make_wrong_answer(item, tag): wrong answers for Spec 1 validation negatives (tag "neg") and Spec 6
corrupted items (tag "corrupt"); the tag enters the seed.

Rules (Spec 1, Validation negatives):
- Numeric: multiply the gold value by a factor from `factors`, drawn with stable_seed(item_id, tag), or
  flip its sign for direction-type items (final operation subtract), keeping the gold's precision, units
  and format. If the gold value is 0, use another numeric value from the same table instead.
- Boolean answers from `greater`: flip yes and no.
- Span: substitute a different span of the same word count from the same context — non-numeric table
  cells first, else word n-grams of the paragraphs (owner decision 2026-10-23). For multi-span answers one
  span, chosen with the same seed, is replaced.
- The new answer must not match the gold under answers_match; otherwise redraw (attempt k uses
  stable_seed(item_id, tag, k)) up to `max_redraws` times, then return None so the caller draws a
  replacement item.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from fqsel.data.common import format_number
from fqsel.ids import rng
from fqsel.io import decode_json
from fqsel.match import _NUM, answers_match, normalize_bool, normalize_span, parse_number
from fqsel.ops import final_operation


def gold_for_match(item: Mapping):
    """What answers_match compares against: the span list for multi-span items, else gold_answer."""
    value = decode_json(item["gold_value"])
    if item["answer_type"] == "lookup" and isinstance(value, list) and len(value) > 1:
        return value
    return item["gold_answer"]


def format_like(gold_text: str, new_value: float) -> str:
    """Replace the first number in gold_text with new_value at the same precision and separators."""
    m = _NUM.search(gold_text)
    raw = m.group(0)
    decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
    prefix = re.sub(r"[-−–](\s*\$?\s*)$", r"\1", gold_text[: m.start()])
    digits = format_number(abs(new_value), decimals, thousands="," in raw)
    sign = "-" if new_value < 0 and digits.strip("0.,") != "" else ""
    return prefix + sign + digits + gold_text[m.end():]


def _table_numbers(table: Sequence[Sequence[str]]) -> list[float]:
    values = []
    for row in table[1:] if len(table) > 1 else table:
        for cell in row[1:]:
            p = parse_number(cell, strict=True)
            if p is not None and p.value != 0:
                values.append(p.value)
    return sorted(set(values))


def _numeric(item: Mapping, tag: str, attempt: int, factors: Sequence[float], seed_parts: list):
    gold_text = item["gold_answer"]
    gold = parse_number(gold_text, strict=True)
    gold_value = decode_json(item["gold_value"])
    op = final_operation(item)
    generator = rng(*seed_parts)
    if gold.value == 0:
        choices = _table_numbers(item["table"])
        if not choices:
            return None
        new = float(choices[int(generator.integers(len(choices)))])
        return format_like(gold_text, new), new, "table_value"
    if op is not None and op.op == "subtract":
        new_display, strategy, scale = -gold.value, "sign_flip", -1.0
    else:
        factor = float(factors[int(generator.integers(len(factors)))])
        new_display, strategy, scale = gold.value * factor, f"factor:{factor:g}", factor
    new_text = format_like(gold_text, new_display)
    if isinstance(gold_value, list):
        new_value = [format_like(gold_value[0], new_display)] if gold_value else [new_text]
    elif isinstance(gold_value, (int, float)):
        new_value = float(gold_value) * scale
    else:
        new_value = new_display
    return new_text, new_value, strategy


def _candidate_spans(item: Mapping, n_words: int, exclude: set[str]) -> list[str]:
    cells = sorted({
        c.strip() for row in item["table"] for c in row
        if c.strip() and parse_number(c) is None and len(c.split()) == n_words
    })
    cells = [c for c in cells if normalize_span(c) not in exclude and normalize_span(c)]
    if cells:
        return cells
    grams = set()
    for text in item["context_text"]:
        words = text.split()
        for i in range(len(words) - n_words + 1):
            gram = " ".join(words[i : i + n_words]).strip(" .,;:")
            if gram and len(gram.split()) == n_words and normalize_span(gram) not in exclude:
                grams.add(gram)
    return sorted(grams)


def _span(item: Mapping, seed_parts: list):
    spans = list(decode_json(item["gold_value"]))
    if not spans:
        return None
    generator = rng(*seed_parts)
    idx = int(generator.integers(len(spans))) if len(spans) > 1 else 0
    exclude = {normalize_span(s) for s in spans}
    candidates = _candidate_spans(item, max(1, len(spans[idx].split())), exclude)
    if not candidates:
        return None
    new_spans = list(spans)
    new_spans[idx] = candidates[int(generator.integers(len(candidates)))]
    text = "; ".join(new_spans) + (f" {item['scale']}" if item["scale"] else "")
    return text, new_spans, "span_swap"


def make_wrong_answer(
    item: Mapping, tag: str, factors: Sequence[float], max_redraws: int
) -> dict | None:
    """{"gold_answer", "gold_value", "perturbation"} for a wrong answer, or None if none was found."""
    gold_bool = normalize_bool(item["gold_answer"])
    if gold_bool is not None:
        return {"gold_answer": "no" if gold_bool == "yes" else "yes",
                "gold_value": "no" if gold_bool == "yes" else "yes", "perturbation": "bool_flip"}
    is_numeric = parse_number(item["gold_answer"], strict=True) is not None
    reference = gold_for_match(item)
    for attempt in range(max_redraws + 1):
        seed_parts = [item["item_id"], tag] + ([attempt] if attempt else [])
        out = _numeric(item, tag, attempt, factors, seed_parts) if is_numeric else _span(item, seed_parts)
        if out is None:
            return None
        text, value, strategy = out
        candidate = value if isinstance(reference, list) else text
        if not answers_match(candidate, reference, item["answer_type"]):
            return {"gold_answer": text, "gold_value": value, "perturbation": strategy}
    return None
