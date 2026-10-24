"""answers_match: the Spec 1 hand-written suite of 100 cases, plus boundary and property tests."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fqsel.match import answers_match, normalize_span, parse_number

# (pred, gold, answer_type, expected)
CASES = [
    # --- percent forms (20) ---
    ("25%", "25%", "arith", True),
    ("25", "25%", "arith", True),
    ("0.25", "25%", "arith", True),
    ("25 percent", "25%", "arith", True),
    ("25.0%", "25%", "arith", True),
    ("24%", "25%", "arith", False),
    ("2.5%", "25%", "arith", False),
    ("25%", "0.25", "arith", True),
    ("0.25", "0.25", "arith", True),
    ("26%", "0.25", "arith", False),
    ("150.18", "150.18 percent", "arith", True),
    ("1.5018", "150.18 percent", "arith", True),
    ("150.2%", "150.18 percent", "arith", True),
    ("-11.5%", "-11.5%", "arith", True),
    ("11.5%", "-11.5%", "arith", False),
    ("-0.115", "-11.5%", "arith", True),
    ("12.76%", "12.76 percent", "arith", True),
    ("12.9%", "12.76 percent", "arith", False),
    ("The margin was 34.7%", "34.65%", "arith", True),
    ("0%", "0%", "arith", True),
    # --- scale words (15) ---
    ("3.8 million", "3.8", "arith", True),
    ("3,800,000", "3.8 million", "arith", True),
    ("3.8", "3.8 million", "arith", True),
    ("3800 thousand", "3.8 million", "arith", True),
    ("$3.8 million", "3.8 million", "arith", True),
    ("4.2 million", "3.8 million", "arith", False),
    ("1,234", "1234 thousand", "arith", True),
    ("1234000", "1234 thousand", "arith", True),
    ("1.234 million", "1234 thousand", "arith", True),
    ("2.9 billion", "2.9 billion", "arith", True),
    ("2900000000", "2.9 billion", "arith", True),
    ("2.9 million", "2.9 billion", "arith", True),  # raw numbers agree: the scale word is optional
    ("29 billion", "2.9 billion", "arith", False),
    ("38 million", "-38 million", "arith", False),
    ("-38", "-38 million", "arith", True),
    # --- rounding and tolerance (20) ---
    ("380", "380", "arith", True),
    ("380.4", "380", "arith", True),
    ("381", "380", "arith", True),    # within 0.5% relative
    ("382", "380", "arith", False),   # 0.53% off
    ("0.97", "0.96667", "arith", True),
    ("0.967", "0.96667", "arith", True),
    ("0.9", "0.96667", "arith", False),
    ("93670", "93670.33333", "arith", True),
    ("93670.3", "93670.33", "arith", True),
    ("1.0", "1", "arith", True),
    ("1.4", "1", "arith", True),      # rounds to the gold precision (0 decimals) first
    ("1.6", "1", "arith", False),
    ("0.004", "0", "arith", True),    # absolute error <= 0.005 after rounding to gold precision
    ("0.01", "0.00", "arith", False),
    ("0.004", "0.00", "arith", True),
    ("100.4", "100", "arith", True),
    ("13225.8", "13225.8", "arith", True),
    ("13290", "13225.8", "arith", True),   # 0.49% off
    ("13300", "13225.8", "arith", False),  # 0.56% off
    ("16532.2", "13225.8", "arith", False),
    # --- signs, currency and separators (15) ---
    ("$1,577.00", "$1577.00", "arith", True),
    ("1577", "$1577.00", "arith", True),
    ("USD 1,577 million", "1577", "arith", True),
    ("-2.44", "-2.44", "arith", True),
    ("2.44", "-2.44", "arith", False),
    ("−2.44", "-2.44", "arith", True),   # unicode minus
    ("(2.44)", "-2.44", "arith", True),       # accounting negative
    ("-$5.2", "-5.2", "arith", True),
    ("$-5.2", "-5.2", "arith", True),
    ("5.2", "-5.2", "arith", False),
    ("1,496", "1496", "arith", True),
    ("Answer is 7.4", "7.4", "arith", True),
    ("7.40", "7.4", "arith", True),
    ("74", "7.4", "arith", False),
    ("no number here", "7.4", "arith", False),
    # --- booleans (10) ---
    ("yes", "yes", "arith", True),
    ("Yes.", "yes", "arith", True),
    ("true", "yes", "arith", True),
    ("TRUE", "yes", "arith", True),
    ("no", "yes", "arith", False),
    ("false", "no", "arith", True),
    ("No", "no", "arith", True),
    ("yes", "no", "arith", False),
    ("maybe", "yes", "arith", False),
    ("yes it is", "yes", "arith", False),
    # --- single spans (10) ---
    ("Rate of inflation", "rate of inflation", "lookup", True),
    ("the rate of inflation", "Rate of inflation", "lookup", True),
    ("Rate of inflation.", "Rate of inflation", "lookup", True),
    ("Rate of  inflation", "Rate of inflation", "lookup", True),
    ("Rate of deflation", "Rate of inflation", "lookup", False),
    ("Europe", "Europe", "lookup", True),
    ("North America", "Europe", "lookup", False),
    ("A decline in pre-tax earnings", "a decline in pretax earnings", "lookup", True),
    ("2.9", ["2.9"], "lookup", True),
    ("2.9%", ["2.9"], "lookup", True),
    # --- multi-span (10) ---
    ("Europe; North America", ["Europe", "North America"], "lookup", True),
    ("North America, Europe", ["Europe", "North America"], "lookup", True),
    ("Europe and North America", ["Europe", "North America"], "lookup", True),
    ("Europe", ["Europe", "North America"], "lookup", False),
    ("Europe; North America; Asia", ["Europe", "North America"], "lookup", False),
    (["North America", "Europe"], ["Europe", "North America"], "lookup", True),
    ("Rate of inflation; Discount rate; Rate of increase in salaries",
     ["Rate of inflation", "Rate of increase in salaries", "Discount rate"], "lookup", True),
    ("Rate of inflation; Discount rate",
     ["Rate of inflation", "Rate of increase in salaries", "Discount rate"], "lookup", False),
    ("the Netherlands; United States", ["the Netherlands", "United States"], "lookup", True),
    (None, ["Europe", "North America"], "lookup", False),
]


def test_suite_has_100_cases():
    assert len(CASES) == 100


@pytest.mark.spec("S1-06")
@pytest.mark.parametrize("pred,gold,answer_type,expected", CASES)
def test_answers_match_suite(pred, gold, answer_type, expected):
    assert answers_match(pred, gold, answer_type) is expected


@pytest.mark.parametrize("pred,gold,expected", [
    ("100.5", "100", True),       # exactly 0.5% relative
    ("100.51", "100.00", False),  # just above 0.5% relative
    ("0.005", "0.000", True),     # exactly 0.005 absolute
    ("0.006", "0.000", False),
])
def test_tolerance_boundaries(pred, gold, expected):
    assert answers_match(pred, gold, "arith") is expected


def test_unknown_answer_type_rejected():
    with pytest.raises(ValueError):
        answers_match("1", "1", "number")


def test_strict_parse_single_number_with_units():
    assert parse_number("$1577.00", strict=True) is not None
    assert parse_number("1,577 USD millions", strict=True) is not None
    assert parse_number("The consumer segment shrunk by 0.9% organically.", strict=True) is None
    assert parse_number("2019 and 2018", strict=True) is None


@given(st.floats(min_value=-1e9, max_value=1e9, allow_nan=False).map(lambda x: round(x, 3)))
def test_number_matches_itself(x):
    text = f"{x:.3f}"
    assert answers_match(text, text, "arith")


@given(st.text(alphabet=st.characters(categories=["L", "Zs"]), min_size=1, max_size=30))
def test_span_matches_itself(s):
    if normalize_span(s):
        assert answers_match(s, s, "lookup")
