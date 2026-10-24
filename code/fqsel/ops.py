"""The final operation of an item's program or derivation, with its executed arguments.

Shared by the Q3 templates (fqsel.checks) and the wrong-answer generator (fqsel.perturb): the template is
chosen by the final operation of the FinQA program or the top-level operator of the TAT-QA derivation,
and numbers come from executed values (Spec 1, Q3 check generation).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FinalOp:
    op: str            # add, subtract, multiply, divide, exp, greater, table_sum/average/max/min
    args: list         # binary ops: [a, b]; add: all addends; table ops: the numeric row


def final_operation(item: Mapping) -> FinalOp | None:
    if item["dataset"] == "finqa" and item["program"]:
        from fqsel.verify.finqa_executor import execute

        res = execute(item["program"], item["table"])
        if not res.ok or not res.steps:
            return None
        last = res.steps[-1]
        return FinalOp(last.op, list(last.args))
    if item["dataset"] == "tatqa" and item["source_answer_type"] == "arithmetic":
        from fqsel.verify.tatqa_eval import DerivationError, flatten_addends, parse_derivation

        try:
            node = parse_derivation(item["derivation"] or "")
        except (DerivationError, ZeroDivisionError, OverflowError):
            return None
        if node.op in ("num", "neg"):
            return None
        if node.op == "add":
            return FinalOp("add", flatten_addends(node))
        return FinalOp(node.op, [node.children[0].value, node.children[1].value])
    return None
