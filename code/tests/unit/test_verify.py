"""FinQA executor (ported reference logic) and TAT-QA evaluator (Spec 1, Verified correctness)."""

from __future__ import annotations

import pytest

from fqsel.verify.finqa_executor import execute, program_tokenization, str_to_num
from fqsel.verify.tatqa_eval import DerivationError, parse_derivation, verify_tatqa

TABLE = [["", "2019", "2018"], ["net sales", "$ 1,200", "$ 1,000 ( 1,000 )"], ["margin", "10%", "20%"]]


@pytest.mark.parametrize("program,expected", [
    ("add(1, 2)", 3.0),
    ("subtract(1200, 1000)", 200.0),
    ("multiply(2.5, const_100)", 250.0),
    ("divide(3, 4)", 0.75),
    ("exp(2, 3)", 8.0),
    ("greater(5, 3)", "yes"),
    ("greater(3, 5)", "no"),
    ("subtract(1200, 1000), divide(#0, 1000)", 0.2),
    ("add(const_m1, const_1000)", 999.0),
    ("divide(10%, 2)", 0.05),                       # "%" numbers are divided by 100
    ("table_sum(net sales, none)", 2200.0),         # "$" stripped, cell cut at "("
    ("table_average(net sales, none)", 1100.0),
    ("table_max(margin, none)", 0.2),
    ("table_min(margin, none)", 0.1),
    ("divide(1, 3)", 0.33333),                      # rounded to 5 decimals like the reference
])
def test_executor_operations(program, expected):
    res = execute(program, TABLE)
    assert res.ok and res.value == expected


@pytest.mark.parametrize("program,failure", [
    ("divide(1, 0)", "exec_error"),
    ("table_max(unknown row, none)", "exec_error"),
    ("table_max(#0, none)", "exec_error"),          # reference code fails on a step reference here
    ("modulo(1, 2)", "parse"),
    ("add(one, 2)", "parse"),
])
def test_executor_failures(program, failure):
    res = execute(program, TABLE)
    assert not res.ok and res.failure == failure


def test_executor_records_resolved_steps():
    res = execute("subtract(1200, 1000), divide(#0, 1000)", TABLE)
    assert [s.op for s in res.steps] == ["subtract", "divide"]
    assert res.steps[1].args == [200.0, 1000.0]


def test_reference_helpers():
    assert program_tokenization("add(1, 2), divide(#0, 3)") == ["add(", "1", "2", ")", "divide(", "#0", "3", ")", "EOF"]
    assert str_to_num("1,234") == 1234.0 and str_to_num("const_m1") == -1.0 and str_to_num("x") == "n/a"


@pytest.mark.parametrize("derivation,value", [
    ("(2.9+2.9)/2", 2.9),
    ("1,617 - 1,434", 183.0),
    ("[(61+9)/2] / [(87+9)/2]", 35 / 48),
    ("$1,552,691+$1,470,797", 3023488.0),
    ("7,572 - (6,919)", 14491.0),                   # accounting negative
    ("-5 + 3", -2.0),
    ("12%", 12.0),                                  # percent sign stripped
])
def test_derivation_values(derivation, value):
    assert parse_derivation(derivation).value == pytest.approx(value)


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')",
    "open('x')",
    "a.b + 1",
    "2 ** 3",
    "20.1 million - 16.8 million",
    "[1, 2]",
])
def test_ast_whitelist_rejects(bad):
    with pytest.raises((DerivationError, SyntaxError)):
        parse_derivation(bad)


def test_tatqa_arithmetic_percent_and_mismatch():
    ok = verify_tatqa("arithmetic", "(1,617-1,434)/1,434", "12.76 percent", 0.1276, [], [])
    assert ok.ok
    bad = verify_tatqa("arithmetic", "(1,617-1,434)/1,434", "13.5 percent", 0.135, [], [])
    assert bad.failure == "mismatch"
    assert verify_tatqa("arithmetic", "20 million - 1", "19", 19.0, [], []).failure == "parse"
    assert verify_tatqa("arithmetic", "1/0", "1", 1.0, [], []).failure == "exec_error"


def test_tatqa_count_rule_p7():
    paras, table = ["We sell in Europe."], [["Region"], ["North America"]]
    assert verify_tatqa("count", "Europe ## North America", "2", 2.0, paras, table).ok
    assert verify_tatqa("count", "Europe ## North America", "3", 3.0, paras, table).failure == "mismatch"
    assert verify_tatqa("count", "Europe ## Asia", "2", 2.0, paras, table).failure == "span_missing"
    assert verify_tatqa("count", "europe##NORTH AMERICA", "2", 2.0, paras, table).ok  # case-insensitive


def test_tatqa_spans():
    paras, table = ["The Company’s strategy  is unchanged."], [["", "2019"], ["Rate of inflation", "2.9"]]
    assert verify_tatqa("span", "", "x", ["The company's strategy is unchanged"], paras, table).ok
    assert verify_tatqa("multi-span", "", "x", ["Rate of inflation", "2.9"], paras, table).ok
    assert verify_tatqa("span", "", "x", ["Rate of deflation"], paras, table).failure == "span_missing"
    assert verify_tatqa("multi-span", "", "x", ["2.9", ""], paras, table).failure == "span_missing"
