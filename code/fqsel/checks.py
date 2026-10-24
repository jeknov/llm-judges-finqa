"""generate_check(item, answer=None): one Q3 consequence check per item (Spec 1, Q3 check generation).

Deterministic templates keyed on the final operation, with numbers from executed values printed at the
gold precision; a strong-LLM fallback under a JSON schema for lookup items and template failures.

Every template's direction is decided by the given answer, so a check regenerated for a wrong answer
(negatives, Spec 6 corruptions) is still true whenever that answer is true. For `multiply` this is the
owner's rule P2 (|second factor| vs 1): since answer = a x b, |answer| > |a| exactly when |b| > 1, and
|b| = 1 or a zero factor sends the item to the fallback.
"""

from __future__ import annotations

from collections.abc import Mapping

from omegaconf import DictConfig

from fqsel.clients.strong_llm import StrongClient
from fqsel.data.common import faithful_decimals, format_number, minimal_decimals
from fqsel.ids import stable_seed
from fqsel.match import normalize_bool, parse_number
from fqsel.ops import final_operation
from fqsel.prompts import load_prompt, render_item

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "check_text": {"type": "string", "minLength": 1},
        "check_type": {"enum": ["direction", "total", "ordering", "magnitude", "unit"]},
        "holds_if_answer_true": {"type": "boolean"},
    },
    "required": ["check_text", "check_type", "holds_if_answer_true"],
    "additionalProperties": False,
}

TEMPLATE_TYPES = {
    "subtract": "direction", "divide": "ordering", "add": "total", "table_sum": "total",
    "multiply": "magnitude", "greater": "ordering", "table_average": "total",
    "table_max": "ordering", "table_min": "ordering",
}


def _fmt(value: float, decimals: int) -> str:
    """Print at the gold precision, raised only where that precision would distort the executed value
    (e.g. fraction operands 1.19 and 0.52 of a "67%" answer would both print as "1")."""
    return format_number(value, faithful_decimals(value, decimals))


def template_check(item: Mapping, answer: str) -> tuple[str, str, str] | None:
    """(check_text, check_type, template_op), or None when the item must go to the LLM fallback.

    Any item whose program or derivation ends in an operation with a template uses it, including FinQA
    table-op programs, which count as lookup items (owner decision 2026-10-23); items without a program
    (TAT-QA spans and counts, FinanceBench) go to the fallback.
    """
    op = final_operation(item)
    if op is None or op.op not in TEMPLATE_TYPES:
        return None
    if op.op == "greater":
        flag = normalize_bool(answer)
        if flag is None:
            return None
        a, b = op.args
        # A yes/no answer has no precision, so each argument is printed with the fewest exact decimals.
        verb = "larger" if flag == "yes" else "not larger"
        text = f"{_fmt(a, minimal_decimals(a))} is {verb} than {_fmt(b, minimal_decimals(b))}."
        return text, TEMPLATE_TYPES["greater"], "greater"

    parsed = parse_number(answer, strict=True)
    if parsed is None or parsed.value == 0:
        return None
    value, d, pct = parsed.value, parsed.decimals, parsed.pct

    if op.op == "subtract":
        a, b = op.args
        change = "an increase" if value > 0 else "a decrease"
        return f"The change from {_fmt(b, d)} to {_fmt(a, d)} is {change}.", "direction", "subtract"
    if op.op == "divide":
        threshold = 100.0 if pct else 1.0
        if value == threshold:
            return None
        side = "above" if value > threshold else "below"
        return (f"The ratio is {side} 100%." if pct else f"The ratio is {side} 1."), "ordering", "divide"
    if op.op in ("add", "table_sum"):
        addends = list(op.args)
        if not addends or any(x <= 0 for x in addends):
            return None
        return f"The total is larger than {_fmt(max(addends), d)}.", "total", op.op
    if op.op == "multiply":
        a, b = op.args
        if a == 0 or b == 0 or abs(b) == 1:
            return None
        size = "larger" if abs(value) > abs(a) else "smaller"
        return f"The result is {size} in absolute value than {_fmt(a, d)}.", "magnitude", "multiply"
    if op.op == "table_average":
        row = list(op.args)
        return f"The average lies between {_fmt(min(row), d)} and {_fmt(max(row), d)}.", "total", "table_average"
    if op.op == "table_max":
        return "No other value in the row is larger than the answer.", "ordering", "table_max"
    if op.op == "table_min":
        return "No other value in the row is smaller than the answer.", "ordering", "table_min"
    return None


def generate_check(
    item: Mapping, cfg: DictConfig, llm: StrongClient, answer: str | None = None
) -> dict | None:
    """The Q3 check for `answer` (default: the gold answer), or None if the LLM fallback failed twice."""
    answer = item["gold_answer"] if answer is None else answer
    shown = dict(item)
    shown["gold_answer"] = answer
    tpl = template_check(shown, answer)
    if tpl is not None:
        text, ctype, op = tpl
        return {"check_text": text, "check_type": ctype, "source": "template",
                "template_op": op, "generator_model": "template"}
    prompt = load_prompt("q3_fallback.txt", cfg).replace("{item}", render_item(shown, "judge"))
    attempts = [cfg.model.strong.q3_first_temperature, cfg.model.strong.q3_retry_temperature]
    for i, temperature in enumerate(attempts):
        seed = stable_seed(cfg.global_seed, item["item_id"], "q3", i)
        out = llm.generate_json(prompt, CHECK_SCHEMA, float(temperature), seed)
        if out["holds_if_answer_true"] and out["check_text"].strip():
            return {"check_text": out["check_text"].strip(), "check_type": out["check_type"],
                    "source": "llm", "template_op": None, "generator_model": llm.model_version}
    return None
