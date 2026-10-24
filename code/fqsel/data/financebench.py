"""FinanceBench loader (Spec 1, Probe and test sets). Field names confirmed against the open-source release.

Keeps questions whose gold answer parses as a single number with optional units (strict parse_number)
and which come with evidence text; the evidence texts, in release order, become the context.
Items get split = test, answer_type = arith, dataset = financebench.
"""

from __future__ import annotations

import json
from pathlib import Path

from fqsel.data.common import make_item
from fqsel.ids import make_item_id
from fqsel.match import parse_number


def row_to_item(row: dict) -> dict | None:
    parsed = parse_number(row.get("answer", ""), strict=True)
    evidence = [e["evidence_text"] for e in row.get("evidence") or [] if (e.get("evidence_text") or "").strip()]
    if parsed is None or not evidence:
        return None
    value = parsed.value * parsed.scale / (100.0 if parsed.pct else 1.0)
    return make_item(
        item_id=make_item_id("financebench", row["financebench_id"]),
        dataset="financebench",
        source_split="test",
        question=row["question"],
        context_text=evidence,
        context_pos=["pre"] * len(evidence),
        table=[],
        program=None,
        derivation=None,
        gold_answer=str(row["answer"]).strip(),
        gold_value=value,
        answer_type="arith",
        source_answer_type=row.get("question_type") or "",
        scale="",
        support_ids=[],
    )


def load_financebench(raw_dir: Path, files: dict) -> tuple[list[dict], int]:
    """Returns (kept items, number filtered out as non-numeric or without evidence)."""
    kept, dropped = [], 0
    for line in (Path(raw_dir) / files["test"]).read_text().splitlines():
        if not line.strip():
            continue
        item = row_to_item(json.loads(line))
        if item is None:
            dropped += 1
        else:
            kept.append(item)
    return kept, dropped
