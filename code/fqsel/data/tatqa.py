"""TAT-QA loader (Spec 1, Field mapping to pool.parquet). Field names confirmed against the release files.

One TAT-QA record holds a table, paragraphs and several questions; each question becomes one item with
item_id "tatqa:<question uid>". Paragraphs are kept in `order` and placed after the table (context_pos
"post"), as in the TAT-QA layout. support_ids are "para_<order>" for each rel_paragraphs entry, plus
"table" when answer_from includes the table.

gold_answer: the answer joined with its scale word ("150.18 percent"); spans joined with "; ".
gold_value: numeric answer x scale (percent = 0.01) for arithmetic and count; the span list otherwise.
The one arithmetic item whose answer is not a number keeps its text as gold_value and fails verification.
"""

from __future__ import annotations

import json
from pathlib import Path

from fqsel.data.common import make_item
from fqsel.ids import make_item_id

SCALE_FACTORS = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 0.01}
ARITH_TYPES = ("arithmetic", "count")


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def question_to_item(record: dict, q: dict, source_split: str) -> dict:
    paragraphs = sorted(record["paragraphs"], key=lambda p: p["order"])
    texts = [p["text"] for p in paragraphs]
    scale = q.get("scale", "") or ""
    atype = q["answer_type"]
    if atype in ARITH_TYPES:
        answer = q["answer"]
        gold_answer = f"{answer} {scale}".strip() if scale else str(answer)
        number = _number(answer)
        if atype == "count":
            gold_value = number
        else:
            gold_value = number * SCALE_FACTORS[scale] if number is not None else str(answer)
    else:
        spans = [str(s) for s in q["answer"]]
        gold_answer = "; ".join(spans)
        if scale:
            gold_answer = f"{gold_answer} {scale}"
        gold_value = spans
    support = [f"para_{o}" for o in q.get("rel_paragraphs", [])]
    if "table" in (q.get("answer_from") or ""):
        support.append("table")
    return make_item(
        item_id=make_item_id("tatqa", q["uid"]),
        dataset="tatqa",
        source_split=source_split,
        question=q["question"],
        context_text=texts,
        context_pos=["post"] * len(texts),
        table=record["table"]["table"],
        program=None,
        derivation=q.get("derivation") or "",
        gold_answer=gold_answer,
        gold_value=gold_value,
        answer_type="arith" if atype in ARITH_TYPES else "lookup",
        source_answer_type=atype,
        scale=scale,
        support_ids=support,
    )


def load_tatqa(raw_dir: Path, files: dict) -> list[dict]:
    items = []
    for source_split, rel in files.items():
        for record in json.loads((Path(raw_dir) / rel).read_text()):
            for q in record["questions"]:
                items.append(question_to_item(record, q, source_split))
    return items
