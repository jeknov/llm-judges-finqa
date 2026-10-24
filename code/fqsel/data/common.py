"""Shared pieces of the dataset loaders: the pool-schema row builder and number formatting."""

from __future__ import annotations

from fqsel.io import encode_json

MAX_DECIMALS = 5  # FinQA's executor rounds to 5 decimals


def format_number(value: float, decimals: int, thousands: bool = False) -> str:
    text = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"
    if text.lstrip("-").replace(",", "").replace(".", "").strip("0") == "":
        text = text.lstrip("-")  # never print "-0.00"
    return text


def minimal_decimals(value: float) -> int:
    for d in range(MAX_DECIMALS + 1):
        if round(value, d) == round(value, MAX_DECIMALS):
            return d
    return MAX_DECIMALS


def faithful_decimals(value: float, decimals: int | None) -> int:
    """`decimals`, raised until the printed value stays within the answers_match tolerance of `value`."""
    from fqsel.match import ABS_TOL, REL_TOL

    d = minimal_decimals(value) if decimals is None else decimals
    while d < MAX_DECIMALS:
        diff = abs(round(value, d) - value)
        if diff <= ABS_TOL or diff <= REL_TOL * abs(value):
            break
        d += 1
    return d


def context_chars(context_text: list[str], table: list[list[str]]) -> int:
    return sum(len(s) for s in context_text) + sum(len(c) for row in table for c in row)


def make_item(
    *,
    item_id: str,
    dataset: str,
    source_split: str,
    question: str,
    context_text: list[str],
    context_pos: list[str],
    table: list[list[str]],
    program: str | None,
    derivation: str | None,
    gold_answer: str,
    gold_value,
    answer_type: str,
    source_answer_type: str,
    scale: str,
    support_ids: list[str],
) -> dict:
    """A pool-schema row; `split` and `qa_tokens` are filled in by the pool stage."""
    return {
        "item_id": item_id,
        "dataset": dataset,
        "split": None,
        "source_split": source_split,
        "question": question,
        "context_text": list(context_text),
        "context_pos": list(context_pos),
        "table": [list(map(str, row)) for row in table],
        "program": program,
        "derivation": derivation,
        "gold_answer": gold_answer,
        "gold_value": encode_json(gold_value),
        "answer_type": answer_type,
        "source_answer_type": source_answer_type,
        "scale": scale,
        "support_ids": list(support_ids),
        "context_chars": context_chars(context_text, table),
        "qa_tokens": -1,
        "is_derived": False,
        "base_item_id": item_id,
        "perturbation": None,
    }
