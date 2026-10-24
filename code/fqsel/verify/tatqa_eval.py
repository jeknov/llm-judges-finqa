"""TAT-QA verification (Spec 1, Verified correctness). Never calls eval.

- arithmetic: parse `derivation` with an AST whitelist (numbers, + - * /, parentheses, unary minus) after
  stripping thousands separators, percent signs and currency symbols; square brackets are parentheses,
  and a parenthesised bare number is an accounting negative, "(6,919)" -> -6919 (owner decision 2026-10-23).
- count (P7, owner decision 2026-10-23): verified if the number of "##"-separated items in `derivation`
  equals the gold count and every item appears in the paragraphs or table.
- span / multi-span: every gold span appears in the paragraphs or a cell.
"Appears" means verbatim after normalising whitespace, letter case and curly quotes (owner decision
2026-10-23: strict case-sensitive matching verified only 98.0% of train spans against the 99% target).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_ALLOWED_BINOPS = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}


class DerivationError(ValueError):
    pass


@dataclass
class Node:
    """A parsed derivation: op is a binary op name, "neg" or "num"; value is the evaluated result."""

    op: str
    value: float
    children: list[Node] = field(default_factory=list)


def clean_derivation(text: str) -> str:
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = text.replace("%", "").replace("$", "")
    text = re.sub(r"\(\s*(\d+(?:\.\d+)?)\s*\)", r"(-\1)", text)
    return text.strip()


def parse_derivation(text: str) -> Node:
    try:
        tree = ast.parse(clean_derivation(text), mode="eval")
    except SyntaxError as e:
        raise DerivationError(f"unparsable derivation: {text!r}") from e
    return _build(tree.body)


def _build(node: ast.AST) -> Node:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Node("num", float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        child = _build(node.operand)
        return Node("neg", -child.value, [child]) if isinstance(node.op, ast.USub) else child
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _build(node.left), _build(node.right)
        op = _ALLOWED_BINOPS[type(node.op)]
        if op == "add":
            value = left.value + right.value
        elif op == "subtract":
            value = left.value - right.value
        elif op == "multiply":
            value = left.value * right.value
        else:
            if right.value == 0:
                raise ZeroDivisionError("division by zero in derivation")
            value = left.value / right.value
        return Node(op, value, [left, right])
    raise DerivationError(f"disallowed syntax: {type(node).__name__}")


def flatten_addends(node: Node) -> list[float]:
    if node.op == "add":
        return flatten_addends(node.children[0]) + flatten_addends(node.children[1])
    return [node.value]


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_for_presence(text: str) -> str:
    return " ".join(str(text).split()).lower().translate(_QUOTES)


def appears_in_context(span: str, paragraphs: list[str], table: list[list[str]]) -> bool:
    needle = normalize_for_presence(span)
    if not needle:
        return False
    if any(needle in normalize_for_presence(p) for p in paragraphs):
        return True
    return any(needle in normalize_for_presence(cell) for row in table for cell in row)


@dataclass
class TatResult:
    value: float | list | None
    failure: str | None   # None, "parse", "exec_error", "mismatch", "span_missing"

    @property
    def ok(self) -> bool:
        return self.failure is None


def verify_tatqa(source_answer_type: str, derivation: str | None, gold_answer: str, gold_value,
                 paragraphs: list[str], table: list[list[str]]) -> TatResult:
    from fqsel.match import answers_match

    if source_answer_type == "arithmetic":
        try:
            value = parse_derivation(derivation or "").value
        except DerivationError:
            return TatResult(None, "parse")
        except (ZeroDivisionError, OverflowError):
            return TatResult(None, "exec_error")
        ok = answers_match(repr(value), gold_answer, "arith")
        return TatResult(value, None if ok else "mismatch")
    if source_answer_type == "count":
        items = [s.strip() for s in (derivation or "").split("##") if s.strip()]
        if not items:
            return TatResult(None, "parse")
        if float(len(items)) != float(gold_value):
            return TatResult(float(len(items)), "mismatch")
        if not all(appears_in_context(s, paragraphs, table) for s in items):
            return TatResult(float(len(items)), "span_missing")
        return TatResult(float(len(items)), None)
    if source_answer_type in ("span", "multi-span"):
        spans = list(gold_value)
        if all(appears_in_context(s, paragraphs, table) for s in spans):
            return TatResult(spans, None)
        return TatResult(spans, "span_missing")
    raise ValueError(f"unknown TAT-QA answer_type {source_answer_type!r}")
