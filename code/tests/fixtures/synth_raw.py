"""Deterministic synthetic raw data in the FinQA, TAT-QA and FinanceBench release formats.

Used by the tiny end-to-end test so that CI needs no downloaded data. Gold answers are computed here with
plain Python arithmetic, independently of fqsel's executor and evaluator. Contexts are random word
sequences, so no synthetic item leaks into another.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

WORDS = ("revenue income growth margin segment cost asset liability cash equity debt interest tax "
         "dividend share market price volume capital reserve loss profit expense operating net gross "
         "quarter fiscal annual budget forecast credit lease pension goodwill inventory").split()
ROW_NAMES = ["net sales", "operating income", "total assets", "capital spending", "cash flow",
             "interest expense", "gross margin", "tax expense"]


def _sentence(r: random.Random, n: int = 10) -> str:
    return " ".join(r.choice(WORDS) for _ in range(n)) + " ."


def _table(r: random.Random) -> list[list[str]]:
    names = r.sample(ROW_NAMES, 3)
    return [["", "2019", "2018"]] + [[n, str(r.randint(100, 999)), str(r.randint(100, 999))] for n in names]


def _finqa_record(r: random.Random, idx: int, split: str, lookup: bool) -> dict:
    table = _table(r)
    row = table[1 + r.randrange(3)]
    a, b = float(row[1]), float(row[2])
    if lookup:
        op = r.choice(["table_max", "table_min", "table_average"])
        program = f"{op}({row[0]}, none)"
        exe = {"table_max": max(a, b), "table_min": min(a, b), "table_average": (a + b) / 2}[op]
        answer = f"{exe:.1f}"
    else:
        kind = r.choice(["subtract", "divide", "add", "greater", "change"])
        if kind == "subtract":
            program, exe = f"subtract({a:g}, {b:g})", a - b
        elif kind == "divide":
            program, exe = f"divide({a:g}, {b:g})", a / b
        elif kind == "add":
            program, exe = f"add({a:g}, {b:g})", a + b
        elif kind == "greater":
            program, exe = f"greater({a:g}, {b:g})", "yes" if a > b else "no"
        else:
            program, exe = f"subtract({a:g}, {b:g}), divide(#0, {b:g})", (a - b) / b
        if isinstance(exe, float):
            exe = round(exe, 5)
        answer = exe if isinstance(exe, str) else (f"{exe * 100:.1f}%" if kind == "change" else f"{exe:.2f}")
    return {
        "id": f"SYN/{split}/page_{idx}.pdf-1",
        "pre_text": [_sentence(r) for _ in range(2)],
        "post_text": [_sentence(r)],
        "table": table,
        "qa": {"question": f"what is the value {idx} ?", "answer": answer, "program": program,
               "exe_ans": exe, "gold_inds": {"text_0": "x", "table_1": "y"}},
    }


def _tatqa_record(r: random.Random, idx: int, split: str, n_arith: int, n_lookup: int) -> dict:
    table = _table(r)
    paragraphs = [{"uid": f"p{idx}-{k}", "order": k + 1, "text": _sentence(r, 14)} for k in range(2)]
    questions = []
    for k in range(n_arith + n_lookup):
        uid = f"syn-{split}-{idx}-{k}"
        row = table[1 + r.randrange(3)]
        a, b = int(row[1]), int(row[2])
        base = {"uid": uid, "order": k + 1, "question": f"question {idx}-{k} ?", "rel_paragraphs": ["1"],
                "req_comparison": False}
        if k < n_arith:
            if r.random() < 0.8:
                scale = r.choice(["", "thousand", "percent"])
                if scale == "percent":
                    q = {"answer": round((a - b) / b * 100, 2), "derivation": f"({a} - {b}) / {b}"}
                else:
                    q = {"answer": a - b, "derivation": f"{a:,} - {b:,}"}
                q.update(answer_type="arithmetic", answer_from="table", scale=scale)
            else:
                names = [table[1][0], table[2][0]]
                q = {"answer": "2", "derivation": " ## ".join(names), "answer_type": "count",
                     "answer_from": "table", "scale": ""}
        else:
            if r.random() < 0.7:
                q = {"answer": [row[0]], "derivation": "", "answer_type": "span", "answer_from": "table",
                     "scale": ""}
            else:
                q = {"answer": [table[1][0], table[2][0]], "derivation": "", "answer_type": "multi-span",
                     "answer_from": "table", "scale": ""}
        questions.append({**base, **q})
    return {"table": {"uid": f"t{idx}", "table": table}, "paragraphs": paragraphs, "questions": questions}


def write_synthetic_raw(directory: Path, seed: int = 7) -> Path:
    r = random.Random(seed)
    directory = Path(directory)
    (directory / "finqa").mkdir(parents=True, exist_ok=True)
    (directory / "tatqa").mkdir(parents=True, exist_ok=True)
    (directory / "financebench").mkdir(parents=True, exist_ok=True)
    plan = {"train": (70, 20), "dev": (30, 10), "test": (22, 8)}
    for split, (n_arith, n_lookup) in plan.items():
        records = [_finqa_record(r, i, split, lookup=i >= n_arith) for i in range(n_arith + n_lookup)]
        (directory / "finqa" / f"{split}.json").write_text(json.dumps(records))
    tat_files = {"train": "tatqa_dataset_train.json", "dev": "tatqa_dataset_dev.json",
                 "test": "tatqa_dataset_test_gold.json"}
    tat_plan = {"train": 18, "dev": 8, "test": 6}  # records of 3 arithmetic + 2 lookup questions
    for split, n_rec in tat_plan.items():
        records = [_tatqa_record(r, i, split, 3, 2) for i in range(n_rec)]
        (directory / "tatqa" / tat_files[split]).write_text(json.dumps(records))
    rows = []
    for i in range(25):
        numeric = i < 22
        rows.append({
            "financebench_id": f"financebench_id_{i:05d}",
            "question": f"What is metric {i}?",
            "answer": f"${r.randint(100, 9999)}.00" if numeric else "It improved because of pricing.",
            "question_type": "metrics-generated",
            "evidence": [{"evidence_text": _sentence(r, 20)}],
        })
    (directory / "financebench" / "financebench_open_source.jsonl").write_text(
        "\n".join(json.dumps(x) for x in rows) + "\n")
    return directory
