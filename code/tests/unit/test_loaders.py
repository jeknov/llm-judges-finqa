"""Dataset loaders against hand-written expected rows (Spec 1, Field mapping to pool.parquet)."""

from __future__ import annotations

import pytest

from fqsel.data.finqa import answer_type_of, final_op, format_gold
from fqsel.io import decode_json
from fqsel.verify import verify_item
from tests.fixtures.golden_items import golden_items

ITEMS = golden_items()


def test_finqa_percent_item_row():
    it = ITEMS["finqa:GOLD/2019/page_1.pdf-1"]
    assert it["dataset"] == "finqa" and it["source_split"] == "train"
    assert it["question"] == "what was the percentage change in net sales from 2018 to 2019?"
    assert it["context_text"] == ["revenue grew strongly in 2019 .", "the company expanded abroad .",
                                  "management expects further growth ."]
    assert it["context_pos"] == ["pre", "pre", "post"]
    assert it["table"][1] == ["net sales", "$ 1,200", "$ 1,000"]
    assert it["program"] == "subtract(1200, 1000), divide(#0, 1000)" and it["derivation"] is None
    assert it["gold_answer"] == "20%"
    assert decode_json(it["gold_value"]) == 0.2
    assert it["answer_type"] == "arith" and it["source_answer_type"] == "divide"
    assert it["support_ids"] == ["table_1", "text_0"]
    assert it["context_chars"] == 149  # 31 + 29 + 35 sentence chars + 54 table chars, counted by hand
    assert it["is_derived"] is False and it["base_item_id"] == it["item_id"] and it["perturbation"] is None


def test_finqa_boolean_lookup_and_noisy_rows():
    greater = ITEMS["finqa:GOLD/2019/page_2.pdf-3"]
    assert greater["gold_answer"] == "yes" and decode_json(greater["gold_value"]) == "yes"
    assert greater["answer_type"] == "arith" and greater["context_pos"] == ["pre"]
    lookup = ITEMS["finqa:GOLD/2019/page_3.pdf-2"]
    assert lookup["gold_answer"] == "1200" and lookup["answer_type"] == "lookup"
    assert lookup["source_answer_type"] == "table_max"
    noisy = ITEMS["finqa:GOLD/2019/page_4.pdf-1"]
    assert noisy["gold_answer"] == "3.8"  # qa.answer "380" disagrees with exe_ans 3.8: precision raised
    assert noisy["table"] == []


@pytest.mark.parametrize("exe,answer,expected", [
    (0.2, "20%", "20%"),
    (0.97656, "56.25%", "97.66%"),
    (0.96667, "0.96", "0.97"),
    (93670.33333, "93670", "93670"),
    (3.8, "380", "3.8"),
    (22929.0, "", "22929"),
    (-0.5425, "", "-0.5425"),
    (7.78443, "7.7%", "7.8%"),
    ("no", "no", "no"),
])
def test_format_gold(exe, answer, expected):
    assert format_gold(exe, answer) == expected


def test_answer_type_rule():
    assert answer_type_of("table_sum(revenue, none)") == "lookup"
    assert answer_type_of("table_max(a, none), divide(#0, 2)") == "arith"
    assert answer_type_of("greater(1, 2)") == "arith"
    assert final_op("subtract(1, 2), divide(#0, 2)") == "divide"


def test_tatqa_rows():
    arith = ITEMS["tatqa:q-arith"]
    assert arith["gold_answer"] == "12.76 percent"
    assert decode_json(arith["gold_value"]) == pytest.approx(0.1276)
    assert arith["context_text"] == ["The Company operates in two regions.", "Revenue grew in Europe and North America."]
    assert arith["context_pos"] == ["post", "post"]
    assert arith["support_ids"] == ["table"] and arith["answer_type"] == "arith"
    assert arith["derivation"] == "(1,617-1,434)/1,434" and arith["program"] is None
    count = ITEMS["tatqa:q-count"]
    assert count["gold_answer"] == "2" and decode_json(count["gold_value"]) == 2.0
    assert count["support_ids"] == ["para_1", "table"] and count["source_answer_type"] == "count"
    span = ITEMS["tatqa:q-span"]
    assert span["gold_answer"] == "Europe" and decode_json(span["gold_value"]) == ["Europe"]
    assert span["answer_type"] == "lookup" and span["support_ids"] == ["para_2"]
    multi = ITEMS["tatqa:q-multi"]
    assert multi["gold_answer"] == "Europe; North America"
    assert ITEMS["tatqa:q-neg"]["gold_answer"] == "183 thousand"
    assert decode_json(ITEMS["tatqa:q-neg"]["gold_value"]) == 183000.0


def test_financebench_filter_and_rows():
    fb = {k: v for k, v in ITEMS.items() if k.startswith("financebench:")}
    assert sorted(fb) == ["financebench:financebench_id_00001", "financebench:financebench_id_00002"]
    first = fb["financebench:financebench_id_00001"]
    assert first["gold_answer"] == "$1577.00" and decode_json(first["gold_value"]) == 1577.0
    assert first["context_text"] == ["Purchases of property, plant and equipment (1,577)"]
    assert first["table"] == [] and first["answer_type"] == "arith"
    assert fb["financebench:financebench_id_00002"]["context_pos"] == ["pre", "pre"]


@pytest.mark.parametrize("item_id", sorted(k for k in ITEMS if not k.startswith("financebench")))
def test_golden_items_verify(item_id):
    v = verify_item(ITEMS[item_id])
    assert v.verified is True, (item_id, v)


def test_financebench_verified_is_null():
    v = verify_item(ITEMS["financebench:financebench_id_00001"])
    assert v.verified is None and v.failure_reason is None
