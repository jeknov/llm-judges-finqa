"""verify_item: execution-based verification of one item (Spec 1, Verified correctness)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fqsel.io import decode_json


@dataclass(frozen=True)
class Verification:
    exec_value: object          # executed value (number, "yes"/"no", span list) or None
    verified: bool | None       # None for FinanceBench, which has no program
    failure_reason: str | None


def verify_item(item: Mapping) -> Verification:
    dataset = item["dataset"]
    gold_value = decode_json(item["gold_value"])
    if dataset == "financebench":
        return Verification(None, None, None)
    if dataset == "finqa":
        from fqsel.verify.finqa_executor import execute

        res = execute(item["program"], item["table"])
        if not res.ok:
            return Verification(None, False, res.failure)
        return Verification(res.value, res.value == gold_value, None if res.value == gold_value else "mismatch")
    if dataset == "tatqa":
        from fqsel.verify.tatqa_eval import verify_tatqa

        paragraphs = list(item["context_text"])
        res = verify_tatqa(item["source_answer_type"], item["derivation"], item["gold_answer"],
                           gold_value, paragraphs, item["table"])
        return Verification(res.value, res.ok, res.failure)
    raise ValueError(f"unknown dataset {dataset!r}")
