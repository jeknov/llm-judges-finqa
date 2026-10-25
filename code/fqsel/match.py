"""answers_match(pred, gold, answer_type) — Spec 1, Verified correctness, Comparison function.

Numbers: parse both sides, accepting percent and scale words; a percentage matches either its fraction or
its percent form, and a scaled number matches either its raw or its scaled form. Match if, after rounding
both to the gold precision, the relative error is <= 0.5% or the absolute error is <= 0.005.
Spans: case-insensitive exact match after removing punctuation and articles; multi-span needs set equality.
Booleans (FinQA `greater`): yes/true and no/false map to one form.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass

REL_TOL = 0.005
ABS_TOL = 0.005

SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
BOOLS = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
# Words allowed around a "single number with optional units" (strict parsing, Spec 1 FinanceBench filter).
UNIT_WORDS = {
    "usd", "us", "dollar", "dollars", "eur", "euro", "euros", "gbp", "percent", "per", "cent",
    "x", "times", "shares", "share", "in", "thousand", "thousands", "million", "millions",
    "billion", "billions", "trillion", "trillions", "mn", "bn",
}

_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+")
_MINUS = "-−–"
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"−", "–", "—", "’", "“", "”"}


@dataclass(frozen=True)
class ParsedNumber:
    value: float      # signed, before applying scale or percent
    decimals: int     # digits after the decimal point in the source text
    pct: bool
    scale: float      # 1.0 when no scale word follows


def _parse_at(text: str, m: re.Match) -> ParsedNumber:
    raw = m.group(0)
    value = float(raw.replace(",", ""))
    decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
    before = text[: m.start()].rstrip(" $€£")
    if before.endswith(tuple(_MINUS)):
        value = -value
    after = text[m.end():].lower()
    pct = bool(re.match(r"\s*(%|percent\b|per cent\b)", after))
    scale_m = re.match(r"\s*(thousand|million|billion|trillion)s?\b", after)
    scale = SCALES[scale_m.group(1)] if scale_m else 1.0
    if re.fullmatch(r"\s*\(\s*\$?\s*" + re.escape(raw) + r"\s*\)\s*", text):
        value = -abs(value)  # accounting negative, e.g. "(1,234)"
    return ParsedNumber(value, decimals, pct, scale)


def parse_number(text: str, strict: bool = False) -> ParsedNumber | None:
    """First number in `text`; with strict=True the text must be one number with optional units only."""
    if text is None:
        return None
    text = str(text).strip()
    matches = list(_NUM.finditer(text))
    if not matches:
        return None
    if strict:
        if len(matches) != 1:
            return None
        rest = (text[: matches[0].start()] + " " + text[matches[0].end():]).lower()
        words = re.findall(r"[a-z$]+", rest.replace("$", " "))
        if any(w not in UNIT_WORDS for w in words):
            return None
    return _parse_at(text, matches[0])


def _candidates(n: ParsedNumber) -> list[tuple[float, int, float]]:
    """(value, decimals, unit) forms of a number. `unit` converts the absolute tolerance, which is
    defined in the units the number is printed in, to the candidate's units: 0.005 on "12.76 percent"
    is 0.00005 on its fraction form 0.1276, not 0.005 (half a percentage point)."""
    out = [(n.value, n.decimals, 1.0)]
    if n.scale != 1.0:
        out.append((n.value * n.scale, n.decimals - round(math.log10(n.scale)), n.scale))
    if n.pct:
        out.append((n.value / 100.0, n.decimals + 2, 0.01))
    return out


def values_close(pred: float, gold: float, decimals: int, unit: float = 1.0) -> bool:
    rp, rg = round(pred, decimals), round(gold, decimals)
    diff = abs(rp - rg)
    return diff <= ABS_TOL * unit * (1 + 1e-9) or diff <= REL_TOL * abs(rg) * (1 + 1e-9)


def numbers_match(pred: ParsedNumber, gold: ParsedNumber) -> bool:
    return any(
        values_close(pv, gv, gd, unit) for gv, gd, unit in _candidates(gold) for pv, _, _ in _candidates(pred)
    )


def normalize_bool(text) -> str | None:
    if text is None:
        return None
    words = normalize_span(str(text)).split()
    return BOOLS.get(words[0]) if len(words) == 1 else None


def normalize_span(text: str) -> str:
    text = "".join(ch for ch in str(text).lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _multi_span_match(pred, gold: list) -> bool:
    gold_set = {normalize_span(g) for g in gold}
    if isinstance(pred, (list, tuple)):
        return {normalize_span(p) for p in pred} == gold_set
    for pattern in (r";", r"[;,]", r"[;,\n]|\band\b"):
        parts = {normalize_span(p) for p in re.split(pattern, str(pred))}
        if parts - {""} == gold_set:
            return True
    return False


def answers_match(pred, gold, answer_type: str) -> bool:
    if answer_type not in ("lookup", "arith"):
        raise ValueError(f"unknown answer_type {answer_type!r}")
    if pred is None:
        return False
    if isinstance(gold, (list, tuple)):
        if len(gold) != 1:
            return _multi_span_match(pred, list(gold))
        gold = gold[0]
    gold = str(gold)
    gold_bool = normalize_bool(gold)
    if gold_bool is not None:
        return normalize_bool(pred) == gold_bool
    g = parse_number(gold, strict=True)
    if g is not None:
        p = parse_number(str(pred))
        return p is not None and numbers_match(p, g)
    return normalize_span(str(pred)) == normalize_span(gold)


def modal_group(answers, answer_type: str) -> tuple[str | None, int]:
    """(first member of the largest group of mutually matching answers, its size); ties go to the earliest
    answer; None (unparsed) answers form no group. Shared by Spec 2 (modal_share) and Spec 3 (sc_modal_share)."""
    best, size = None, 0
    for a in answers:
        if a is None:
            continue
        n = sum(b is not None and answers_match(b, a, answer_type) for b in answers)
        if n > size:
            best, size = a, n
    return best, size
