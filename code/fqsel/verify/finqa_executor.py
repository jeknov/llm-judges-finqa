"""FinQA program executor, ported from the official repository (czyssrs/FinQA, code/evaluate/evaluate.py:
str_to_num, process_row, eval_program, program_tokenization).

The arithmetic and parsing rules are the reference ones, including its quirks: numbers in "%" are divided
by 100, table cells are cut at the first "(", the final result is rounded to 5 decimals, and comparison
with `exe_ans` is exact equality. The port is restructured to return every step's resolved arguments,
which the Q3 templates (Spec 1) need, and to name the failure instead of returning a bare flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALL_OPS = ["add", "subtract", "multiply", "divide", "exp", "greater",
           "table_max", "table_min", "table_sum", "table_average"]
BINARY_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}


@dataclass
class Step:
    op: str
    raw_args: tuple[str, str]
    args: list            # resolved: two numbers for binary ops, the numeric row for table ops
    result: float | str


@dataclass
class ExecResult:
    value: float | str | None
    failure: str | None   # None, "parse" or "exec_error"
    steps: list[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failure is None


def str_to_num(text: str):
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            try:
                return float(text.replace("%", "")) / 100.0
            except ValueError:
                return "n/a"
        if "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            return float(text)
        return "n/a"


def process_row(row_in: list[str]):
    row_out = []
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        num = str_to_num(num)
        if num == "n/a":
            return "n/a"
        row_out.append(num)
    return row_out


def program_tokenization(original_program: str) -> list[str]:
    program = []
    for tok in original_program.split(", "):
        cur_tok = ""
        for c in tok:
            if c == ")":
                if cur_tok != "":
                    program.append(cur_tok)
                    cur_tok = ""
            cur_tok += c
            if c in ["(", ")"]:
                program.append(cur_tok)
                cur_tok = ""
        if cur_tok != "":
            program.append(cur_tok)
    program.append("EOF")
    return program


def execute(program_text: str, table: list[list[str]]) -> ExecResult:
    """Execute a FinQA program string against its table."""
    program = program_tokenization(program_text)[:-1]
    for ind, token in enumerate(program):
        if ind % 4 == 0 and token.strip("(") not in ALL_OPS:
            return ExecResult(None, "parse")
        if (ind + 1) % 4 == 0 and token != ")":
            return ExecResult(None, "parse")

    steps_text = "|".join(program).split(")")[:-1]
    res_dict: dict[int, float | str] = {}
    steps: list[Step] = []
    this_res: float | str = "n/a"
    try:
        for ind, step_text in enumerate(steps_text):
            step_text = step_text.strip()
            if len(step_text.split("(")) > 2:
                return ExecResult(None, "parse", steps)
            op = step_text.split("(")[0].strip("|").strip()
            args = step_text.split("(")[1].strip("|").strip()
            raw1 = args.split("|")[0].strip()
            raw2 = args.split("|")[1].strip()

            if op in BINARY_OPS:
                arg1 = res_dict[int(raw1.replace("#", ""))] if "#" in raw1 else str_to_num(raw1)
                arg2 = res_dict[int(raw2.replace("#", ""))] if "#" in raw2 else str_to_num(raw2)
                if arg1 == "n/a" or arg2 == "n/a":
                    return ExecResult(None, "parse", steps)
                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1**arg2
                else:
                    this_res = "yes" if arg1 > arg2 else "no"
                steps.append(Step(op, (raw1, raw2), [arg1, arg2], this_res))
                res_dict[ind] = this_res
            elif "table" in op:
                table_dict = {row[0]: row[1:] for row in table}
                if "#" in raw1:
                    # The reference code leaves the row undefined here and fails; so do we.
                    return ExecResult(None, "exec_error", steps)
                if raw1 not in table_dict:
                    return ExecResult(None, "exec_error", steps)
                num_row = process_row(table_dict[raw1])
                if num_row == "n/a":
                    return ExecResult(None, "exec_error", steps)
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                else:
                    return ExecResult(None, "parse", steps)
                steps.append(Step(op, (raw1, raw2), list(num_row), this_res))
                res_dict[ind] = this_res
            else:
                return ExecResult(None, "parse", steps)
        if this_res not in ("yes", "no", "n/a"):
            this_res = round(this_res, 5)
    except (ZeroDivisionError, OverflowError, KeyError, IndexError, TypeError, ValueError):
        return ExecResult(None, "exec_error", steps)
    if this_res == "n/a":
        return ExecResult(None, "exec_error", steps)
    return ExecResult(this_res, None, steps)
