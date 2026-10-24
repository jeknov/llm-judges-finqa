"""Q3 check generation: templates, truth conditions and the LLM fallback (Spec 1, Q3 check generation)."""

from __future__ import annotations

import jsonschema
import pytest

from fqsel.checks import generate_check, template_check
from fqsel.clients.cache import JsonCache
from fqsel.clients.strong_llm import MockStrongLLM, StrongClient
from fqsel.data.finqa import record_to_item
from tests.fixtures.golden_items import golden_items

ITEMS = golden_items()


def finqa(program, exe, answer, table=None):
    rec = {"id": "T/1/page_1.pdf-1", "pre_text": ["x ."], "post_text": [],
           "table": table or [["", "2019"], ["sales", "1", "5", "3"]],
           "qa": {"question": "q?", "answer": answer, "program": program, "exe_ans": exe, "gold_inds": {}}}
    return record_to_item(rec, "train")


@pytest.mark.parametrize("program,exe,answer,expected", [
    ("subtract(1200, 1000)", 200.0, "200", ("The change from 1000 to 1200 is an increase.", "direction")),
    ("subtract(900, 1000)", -100.0, "-100", ("The change from 1000 to 900 is a decrease.", "direction")),
    ("divide(150, 100)", 1.5, "1.5", ("The ratio is above 1.", "ordering")),
    ("divide(50, 100)", 0.5, "50%", ("The ratio is below 100%.", "ordering")),
    ("add(2.4, 2.5)", 4.9, "4.9", ("The total is larger than 2.5.", "total")),
    ("multiply(4, 2.5)", 10.0, "10", ("The result is larger in absolute value than 4.", "magnitude")),
    ("multiply(4, 0.5)", 2.0, "2", ("The result is smaller in absolute value than 4.", "magnitude")),
    ("greater(3.5, 3.2)", "yes", "yes", ("3.5 is larger than 3.2.", "ordering")),
    ("greater(3.2, 3.5)", "no", "no", ("3.2 is not larger than 3.5.", "ordering")),
    ("table_average(sales, none)", 3.0, "3", ("The average lies between 1 and 5.", "total")),
    ("table_max(sales, none)", 5.0, "5", ("No other value in the row is larger than the answer.", "ordering")),
    ("table_min(sales, none)", 1.0, "1", ("No other value in the row is smaller than the answer.", "ordering")),
    ("table_sum(sales, none)", 9.0, "9", ("The total is larger than 5.", "total")),
])
def test_templates(program, exe, answer, expected):
    item = finqa(program, exe, answer)
    text, ctype, _ = template_check(item, item["gold_answer"])
    assert (text, ctype) == expected


@pytest.mark.parametrize("program,exe,answer", [
    ("subtract(5, 5)", 0.0, "0"),          # answer exactly zero
    ("divide(5, 5)", 1.0, "1"),            # ratio exactly at the threshold
    ("add(-2, 5)", 3.0, "3"),              # a non-positive addend
    ("multiply(4, 1)", 4.0, "4"),          # P2: |second factor| = 1
    ("multiply(4, -1)", -4.0, "-4"),
    ("exp(2, 3)", 8.0, "8"),               # no template
])
def test_template_preconditions_route_to_fallback(program, exe, answer):
    item = finqa(program, exe, answer)
    assert template_check(item, item["gold_answer"]) is None


def test_operands_keep_their_value_at_low_gold_precision():
    item = finqa("subtract(1.19, 0.52)", 0.67, "67%")
    text, _, _ = template_check(item, item["gold_answer"])
    assert text == "The change from 0.52 to 1.19 is an increase."


def test_multiply_rule_p2_matches_answer_comparison():
    """For gold answers, |answer| > |a| exactly when |b| > 1 (owner rule P2)."""
    for a, b in [(4, 2.5), (4, 0.5), (-3, 2), (-3, -0.25), (0.08, 1000), (5, -3)]:
        item = finqa(f"multiply({a}, {b})", round(a * b, 5), f"{a * b:g}")
        text, _, _ = template_check(item, item["gold_answer"])
        assert ("larger" in text) == (abs(b) > 1)


def test_direction_follows_the_given_answer():
    """A check regenerated for a wrong answer is true whenever that answer is true."""
    item = finqa("subtract(1200, 1000)", 200.0, "200")
    text, _, _ = template_check(item, "-200")
    assert text.endswith("a decrease.")


def test_tatqa_templates_from_derivation():
    item = ITEMS["tatqa:q-arith"]          # (1,617-1,434)/1,434 = 12.76 percent: top-level divide
    assert template_check(item, item["gold_answer"])[0] == "The ratio is below 100%."
    neg = ITEMS["tatqa:q-neg"]             # 1,617 - 1,434
    assert template_check(neg, neg["gold_answer"])[0] == "The change from 1434 to 1617 is an increase."
    assert template_check(ITEMS["tatqa:q-count"], "2") is None   # count has no expression
    assert template_check(ITEMS["tatqa:q-span"], "Europe") is None


def _client(tmp_path, responder=None):
    mock = MockStrongLLM(responder=responder)
    return StrongClient(mock, JsonCache(tmp_path / "s")), mock


def test_llm_fallback_for_lookup(cfg, tmp_path):
    client, mock = _client(tmp_path)
    check = generate_check(ITEMS["tatqa:q-span"], cfg, client)
    assert check["source"] == "llm" and check["template_op"] is None
    assert check["generator_model"] == "mock-strong-1"
    assert "Stated answer: Europe" in mock.calls[0]["prompt"] and mock.calls[0]["temperature"] == 0.0


def test_llm_fallback_retries_once_then_drops(cfg, tmp_path):
    def never_holds(prompt, schema, temperature, seed):
        return {"check_text": "x", "check_type": "unit", "holds_if_answer_true": False}

    client, mock = _client(tmp_path, never_holds)
    assert generate_check(ITEMS["tatqa:q-span"], cfg, client) is None
    assert [c["temperature"] for c in mock.calls] == [0.0, 0.7]


def test_llm_fallback_second_attempt_succeeds(cfg, tmp_path):
    def second(prompt, schema, temperature, seed):
        return {"check_text": "Europe is a region.", "check_type": "unit", "holds_if_answer_true": temperature > 0}

    client, _ = _client(tmp_path, second)
    assert generate_check(ITEMS["tatqa:q-span"], cfg, client)["check_text"] == "Europe is a region."


def test_llm_output_schema_enforced(cfg, tmp_path):
    def bad(prompt, schema, temperature, seed):
        return {"check_text": "x", "check_type": "colour", "holds_if_answer_true": True}

    client, _ = _client(tmp_path, bad)
    with pytest.raises(jsonschema.ValidationError):
        generate_check(ITEMS["tatqa:q-span"], cfg, client)


def test_check_for_wrong_answer_uses_that_answer(cfg, tmp_path):
    client, mock = _client(tmp_path)
    generate_check(ITEMS["tatqa:q-span"], cfg, client, answer="North America")
    assert "Stated answer: North America" in mock.calls[-1]["prompt"]
