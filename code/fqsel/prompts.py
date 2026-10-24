"""render_item, build_messages, extract_answer and prompt-file loading (Spec 1 Shared helpers, Spec 2).

build_messages is the only place prompts are assembled. The render layout (labels "Context:", "Table:",
"Question:", "Stated answer:") was approved by the owner on 2026-10-23 and is frozen by snapshot tests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from omegaconf import DictConfig

from fqsel.config import PROMPTS_DIR, PlaceholderError, prompt_status

DIRECT_INSTRUCTION = "Give only the final answer, with units if any."
REASON_INSTRUCTION = (
    "Reason briefly, then give the final answer on its own last line as: Answer: <value>"
)
INSTRUCTIONS = {"direct": DIRECT_INSTRUCTION, "reason": REASON_INSTRUCTION}
ANSWER_PREFIX = "Answer: "

_ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def render_table(table: Sequence[Sequence[str]]) -> str:
    """Pipe-delimited rows, the first row being the header; cells exactly as in the source."""
    return "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in table)


def render_item(item: Mapping, mode: str) -> str:
    """`qa`: context and question. `judge`: context, question and stated answer.

    Context sentences marked `pre` come before the table and those marked `post` after it. The first
    text block is labelled "Context:"; a second block (post-text after a table) follows unlabelled.
    """
    if mode not in ("qa", "judge"):
        raise ValueError(f"unknown render mode {mode!r}")
    sentences, positions = list(item["context_text"]), list(item["context_pos"])
    pre = [s for s, p in zip(sentences, positions, strict=True) if p == "pre"]
    post = [s for s, p in zip(sentences, positions, strict=True) if p == "post"]
    blocks: list[str] = []
    if pre:
        blocks.append("Context:\n" + "\n".join(pre))
    if len(item["table"]) > 0:
        blocks.append("Table:\n" + render_table(item["table"]))
    if post:
        blocks.append(("" if pre else "Context:\n") + "\n".join(post))
    blocks.append("Question: " + item["question"])
    if mode == "judge":
        blocks.append("Stated answer: " + item["gold_answer"])
    return "\n\n".join(blocks)


def build_messages(
    item: Mapping, fmt: str, demos: Sequence[Mapping] = (), with_answer: bool = False
) -> list[dict[str, str]]:
    """Chat messages for fmt in {direct, reason}.

    Demonstrations are prior user/assistant turns in the `direct` format, in the order given.
    with_answer=True appends the teacher-forced assistant turn `Answer: {gold_answer}` (direct only),
    as used for answer NLL (Spec 2) and gradients (Spec 4).
    """
    if fmt not in INSTRUCTIONS:
        raise ValueError(f"unknown format {fmt!r}")
    if with_answer and fmt != "direct":
        raise ValueError("with_answer is only defined for the direct format")
    messages: list[dict[str, str]] = []
    for demo in demos:
        messages.append({"role": "user", "content": _user_turn(demo, "direct")})
        messages.append({"role": "assistant", "content": ANSWER_PREFIX + demo["gold_answer"]})
    messages.append({"role": "user", "content": _user_turn(item, fmt)})
    if with_answer:
        messages.append({"role": "assistant", "content": ANSWER_PREFIX + item["gold_answer"]})
    return messages


def _user_turn(item: Mapping, fmt: str) -> str:
    return render_item(item, "qa") + "\n\n" + INSTRUCTIONS[fmt]


def extract_answer(text: str, answer_type: str) -> str | None:
    """Spec 2, Answer extraction. Returns None for an unparsed sample."""
    lines = _ANSWER_LINE.findall(text or "")
    if lines:
        answer = lines[-1]
    elif answer_type == "arith":
        numbers = _LAST_NUMBER.findall((text or "")[-200:])
        if not numbers:
            return None
        answer = numbers[-1]
    else:
        return None
    answer = answer.strip().rstrip(".,;:!").strip()
    return answer or None


def load_prompt(name: str, cfg: DictConfig) -> str:
    """Prompt text without its '# ' header lines. Non-FINAL prompts are refused outside profile=tiny."""
    status = prompt_status(name)
    if status != "FINAL" and cfg.profile != "tiny":
        raise PlaceholderError(f"prompts/{name} is {status}; the owner has not approved it")
    lines = (PROMPTS_DIR / name).read_text().splitlines()
    while lines and lines[0].startswith("# "):
        lines.pop(0)
    return "\n".join(lines).strip() + "\n"
