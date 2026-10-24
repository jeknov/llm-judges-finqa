"""render_item, build_messages, extract_answer and prompt files (Spec 1 Shared helpers, Spec 2).

Snapshots freeze the exact text build_messages produces for 10 items in each format; any prompt change
shows up in review. Regenerate deliberately with UPDATE_SNAPSHOTS=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from fqsel.config import PROMPTS_DIR, PlaceholderError
from fqsel.prompts import (
    DIRECT_INSTRUCTION,
    REASON_INSTRUCTION,
    build_messages,
    extract_answer,
    load_prompt,
    render_item,
)
from tests.fixtures.golden_items import golden_items

SNAPSHOT = Path(__file__).parent.parent / "snapshots" / "build_messages.json"
ITEMS = golden_items()
SNAPSHOT_IDS = sorted(ITEMS)[:10]


def _current() -> dict:
    return {i: {fmt: build_messages(ITEMS[i], fmt) for fmt in ("direct", "reason")} for i in SNAPSHOT_IDS}


@pytest.mark.spec("S1-14")
def test_build_messages_snapshots():
    assert len(SNAPSHOT_IDS) == 10
    current = _current()
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    assert SNAPSHOT.exists(), "snapshot missing; create it deliberately with UPDATE_SNAPSHOTS=1"
    assert current == json.loads(SNAPSHOT.read_text())


def test_instruction_wording_is_verbatim_from_spec2():
    assert DIRECT_INSTRUCTION == "Give only the final answer, with units if any."
    assert REASON_INSTRUCTION == "Reason briefly, then give the final answer on its own last line as: Answer: <value>"


def test_render_layout_finqa():
    text = render_item(ITEMS["finqa:GOLD/2019/page_1.pdf-1"], "qa")
    assert text == (
        "Context:\nrevenue grew strongly in 2019 .\nthe company expanded abroad .\n\n"
        "Table:\n|  | 2019 | 2018 |\n| net sales | $ 1,200 | $ 1,000 |\n| cost of sales | $ 700 | $ 650 |\n\n"
        "management expects further growth .\n\n"
        "Question: what was the percentage change in net sales from 2018 to 2019?"
    )


def test_render_layout_tatqa_and_judge():
    item = ITEMS["tatqa:q-span"]
    text = render_item(item, "judge")
    assert text.startswith("Table:\n|  | 2019 | 2018 |")
    assert "\n\nContext:\nThe Company operates in two regions.\nRevenue grew" in text
    assert text.endswith("Question: Which region had revenue of 300?\n\nStated answer: Europe")
    assert "Stated answer" not in render_item(item, "qa")


def test_demos_are_prior_direct_turns_in_order():
    a, b, target = (ITEMS[i] for i in ("finqa:GOLD/2019/page_2.pdf-3", "tatqa:q-count", "tatqa:q-arith"))
    msgs = build_messages(target, "reason", demos=[a, b])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"].endswith(DIRECT_INSTRUCTION) and msgs[1]["content"] == "Answer: yes"
    assert msgs[3]["content"] == "Answer: 2"
    assert msgs[4]["content"].endswith(REASON_INSTRUCTION)


def test_with_answer_appends_teacher_forced_turn():
    msgs = build_messages(ITEMS["tatqa:q-arith"], "direct", with_answer=True)
    assert msgs[-1] == {"role": "assistant", "content": "Answer: 12.76 percent"}
    with pytest.raises(ValueError):
        build_messages(ITEMS["tatqa:q-arith"], "reason", with_answer=True)


@pytest.mark.parametrize("text,atype,expected", [
    ("Reasoning...\nAnswer: 3.8 million", "arith", "3.8 million"),
    ("Answer: 1\nmore thought\nAnswer: 2.", "arith", "2"),
    ("answer:   Europe ", "lookup", "Europe"),
    ("the result is 12.5% overall", "arith", "12.5%"),
    ("revenue was $1,234 then -5.6", "arith", "-5.6"),
    ("no explicit answer", "lookup", None),
    ("nothing numeric", "arith", None),
    ("Answer: yes!", "arith", "yes"),
])
def test_extract_answer(text, atype, expected):
    assert extract_answer(text, atype) == expected


def test_extract_answer_fallback_only_looks_at_last_200_chars():
    text = "42 " + "x" * 300
    assert extract_answer(text, "arith") is None


def test_jev_battery_matches_spec1_wording():
    battery = yaml.safe_load((PROMPTS_DIR / "jev_battery.yaml").read_text())
    assert battery["orig"] == {
        "q1": "Given the context and question, is the stated answer supported by the context?",
        "q2": "Do the context and question determine this answer uniquely, such that no materially different answer would also be defensible?",
        "q3": "Assume the stated answer is true. Does it entail the following check? Check: {check_text}",
    }
    assert battery["para"] == {
        "q1": "Does the information provided back up the answer given to this question?",
        "q2": "Is this the only reasonable answer to the question given the information, rather than one of several defensible answers?",
        "q3": "If the answer given is correct, must the following statement also be correct? Statement: {check_text}",
    }


def test_non_final_prompt_refused_outside_tiny(cfg):
    load_prompt("judge_quality.txt", cfg)  # a PLACEHOLDER is usable under profile=tiny
    fake = cfg.copy()
    fake._set_flag("struct", False)
    fake.profile = "pilot"
    with pytest.raises(PlaceholderError):
        load_prompt("judge_quality.txt", fake)


def test_approved_q3_fallback_prompt_loads_for_real_runs(cfg):
    """P4 approved by the owner on 2026-10-24: the Q3 fallback prompt is FINAL and usable by pilot/full."""
    from fqsel.config import prompt_status

    assert prompt_status("q3_fallback.txt") == "FINAL"
    fake = cfg.copy()
    fake._set_flag("struct", False)
    fake.profile = "pilot"
    text = load_prompt("q3_fallback.txt", fake)
    assert "{item}" in text and not text.startswith("#")
