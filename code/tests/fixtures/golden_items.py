"""Loaded golden fixture items, keyed by item_id (used by loader, prompt, check and perturbation tests)."""

from __future__ import annotations

from pathlib import Path

from fqsel.data.financebench import load_financebench
from fqsel.data.finqa import load_finqa
from fqsel.data.tatqa import load_tatqa

GOLDEN = Path(__file__).parent / "golden"


def golden_items() -> dict[str, dict]:
    items = load_finqa(GOLDEN, {"train": "finqa/train.json"})
    items += load_tatqa(GOLDEN, {"train": "tatqa/train.json"})
    fb, _ = load_financebench(GOLDEN, {"test": "financebench/financebench_open_source.jsonl"})
    out = {}
    for it in items + fb:
        it = dict(it)
        it["split"] = "test" if it["dataset"] == "financebench" else "pool"
        it["qa_tokens"] = 1
        out[it["item_id"]] = it
    return out
