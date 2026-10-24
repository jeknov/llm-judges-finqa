"""FinQA loader (Spec 1, Field mapping to pool.parquet). Field names confirmed against the release files.

gold_answer: qa.exe_ans formatted to the precision shown in qa.answer. When qa.answer is a percentage,
exe_ans is printed in percent form (x100) if that is closer to the stated number, with a "%" sign. If
printing at the stated precision would move the value outside the answers_match tolerance of exe_ans
(e.g. qa.answer "380" with exe_ans 3.8), decimals are added until it fits; if qa.answer has no number,
the fewest decimals that reproduce exe_ans are used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fqsel.data.common import faithful_decimals, format_number, make_item
from fqsel.ids import make_item_id
from fqsel.match import parse_number

ARITH_OPS = ("add", "subtract", "multiply", "divide", "exp", "greater")
_OP = re.compile(r"([a-z_]+)\(")


def final_op(program: str) -> str:
    return _OP.findall(program)[-1]


def answer_type_of(program: str) -> str:
    return "arith" if any(op in ARITH_OPS for op in _OP.findall(program)) else "lookup"


def format_gold(exe_ans, answer_text: str) -> str:
    if isinstance(exe_ans, str):
        return exe_ans
    stated = parse_number(answer_text)
    pct = "%" in answer_text
    value = float(exe_ans)
    if pct and stated is not None and abs(value * 100 - stated.value) < abs(value - stated.value):
        value *= 100
    decimals = faithful_decimals(value, stated.decimals if stated is not None else None)
    return format_number(value, decimals) + ("%" if pct else "")


def _support_sort_key(key: str) -> tuple[str, int]:
    kind, _, idx = key.partition("_")
    return kind, int(idx) if idx.isdigit() else -1


def record_to_item(record: dict, source_split: str) -> dict:
    qa = record["qa"]
    pre, post = list(record["pre_text"]), list(record["post_text"])
    return make_item(
        item_id=make_item_id("finqa", record["id"]),
        dataset="finqa",
        source_split=source_split,
        question=qa["question"],
        context_text=pre + post,
        context_pos=["pre"] * len(pre) + ["post"] * len(post),
        table=record["table"],
        program=qa["program"],
        derivation=None,
        gold_answer=format_gold(qa["exe_ans"], str(qa.get("answer", ""))),
        gold_value=qa["exe_ans"],
        answer_type=answer_type_of(qa["program"]),
        source_answer_type=final_op(qa["program"]),
        scale="",
        support_ids=sorted(qa.get("gold_inds", {}).keys(), key=_support_sort_key),
    )


def load_finqa(raw_dir: Path, files: dict) -> list[dict]:
    items = []
    for source_split, rel in files.items():
        for record in json.loads((Path(raw_dir) / rel).read_text()):
            items.append(record_to_item(record, source_split))
    return items
