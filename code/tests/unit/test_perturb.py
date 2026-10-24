"""make_wrong_answer for validation negatives (Spec 1) and, later, Spec 6 corruptions."""

from __future__ import annotations

import json

import pytest

from fqsel.data.finqa import record_to_item
from fqsel.match import answers_match
from fqsel.perturb import format_like, gold_for_match, make_wrong_answer
from tests.fixtures.golden_items import golden_items

ITEMS = golden_items()
FACTORS = [0.5, 0.8, 1.25, 2.0]


def wrong(item, tag="neg"):
    return make_wrong_answer(item, tag, FACTORS, 5)


def finqa(program, exe, answer, table=None):
    rec = {"id": "T/1/page_9.pdf-1", "pre_text": ["x ."], "post_text": [],
           "table": table or [["", "2019", "2018"], ["sales", "$ 1,200", "$ 300"]],
           "qa": {"question": "q?", "answer": answer, "program": program, "exe_ans": exe, "gold_inds": {}}}
    return record_to_item(rec, "train")


@pytest.mark.spec("S1-15")
@pytest.mark.parametrize("item_id", sorted(k for k in ITEMS if not k.startswith("financebench")))
def test_negative_never_matches_gold(item_id):
    item = ITEMS[item_id]
    out = wrong(item)
    assert out is not None
    candidate = out["gold_value"] if isinstance(gold_for_match(item), list) else out["gold_answer"]
    assert not answers_match(candidate, gold_for_match(item), item["answer_type"])


@pytest.mark.spec("S1-15")
def test_zero_gold_uses_table_value():
    item = finqa("subtract(300, 300)", 0.0, "0")
    out = wrong(item)
    assert out["perturbation"] == "table_value"
    assert not answers_match(out["gold_answer"], "0", "arith")
    assert out["gold_answer"] in {"1200", "300"}


@pytest.mark.spec("S1-15")
def test_boolean_flip():
    out = wrong(ITEMS["finqa:GOLD/2019/page_2.pdf-3"])
    assert out == {"gold_answer": "no", "gold_value": "no", "perturbation": "bool_flip"}


def test_direction_items_flip_sign():
    out = wrong(ITEMS["tatqa:q-neg"])      # final operation subtract
    assert out["perturbation"] == "sign_flip" and out["gold_answer"] == "-183 thousand"
    assert out["gold_value"] == -183000.0


def test_factor_keeps_precision_units_and_value_consistency():
    out = wrong(ITEMS["finqa:GOLD/2019/page_1.pdf-1"])   # "20%", gold_value 0.2, final op divide
    factor = float(out["perturbation"].split(":")[1])
    assert factor in FACTORS
    assert out["gold_answer"].endswith("%") and "." not in out["gold_answer"]
    assert out["gold_value"] == pytest.approx(0.2 * factor)


def test_seed_depends_on_tag_and_is_deterministic():
    item = ITEMS["finqa:GOLD/2019/page_1.pdf-1"]
    assert wrong(item) == wrong(item)
    picks = {wrong(dict(item, item_id=f"finqa:x{i}"), tag)["perturbation"] for i in range(20) for tag in ("neg", "corrupt")}
    assert len(picks) > 1


def test_span_swap_same_word_count_table_first():
    out = wrong(ITEMS["tatqa:q-span"])     # gold "Europe": one word; table cells hold "Revenue", "Europe", ...
    assert out["perturbation"] == "span_swap" and out["gold_value"] == ["Revenue"]
    multi = wrong(ITEMS["tatqa:q-multi"])
    assert len(multi["gold_value"]) == 2 and set(multi["gold_value"]) != {"Europe", "North America"}


def test_format_like():
    assert format_like("$ 1,234.5 million", 2469.0) == "$ 2,469.0 million"
    assert format_like("-11.5%", 11.5) == "11.5%"
    assert format_like("20%", -10) == "-10%"
    assert format_like("3", 1.5) == "2"


def test_redraw_exhaustion_returns_none():
    # Every factor rounds back to the gold at 0 decimals when the gold is 1 and values are tiny? Use a
    # one-word span with no other candidate in the context instead.
    rec_item = dict(ITEMS["tatqa:q-span"], table=[["Europe"]], context_text=["Europe"])
    assert wrong(rec_item) is None


def test_negative_gold_value_is_json_serialisable():
    for item_id in ITEMS:
        if item_id.startswith("financebench"):
            continue
        json.dumps(wrong(ITEMS[item_id])["gold_value"])
