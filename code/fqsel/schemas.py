"""Table schemas, one per file in the Spec 1 file layout.

Each table is declared once as a mapping of column name to `Col`; the pandera validator (run on every
read and write) and the Arrow schema (the physical Parquet types) are both generated from it.
Schemas for tables produced in later milestones are added with those milestones.

`gold_value` holds a float, a list of spans or a string depending on the item, which one Parquet column
cannot store, so it is JSON-encoded; use `fqsel.io.decode_json` to read it back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pyarrow as pa_arrow

from fqsel.ids import DATASETS

SPLITS = ("pool", "probe", "test", "fixed_demo")
SOURCE_SPLITS = ("train", "dev", "test")
ANSWER_TYPES = ("lookup", "arith")
CHECK_TYPES = ("direction", "total", "ordering", "magnitude", "unit")
FAILURE_REASONS = ("parse", "exec_error", "mismatch", "span_missing")
# Labels of the CBI hand audit (conf/annotation, task cbi_audit); the other tasks label yes/no.
CBI_AUDIT_LABELS = ("model_error", "scoring_artefact", "ambiguous_question", "questionable_gold")


@dataclass(frozen=True)
class Col:
    kind: str  # str | int | float | bool | json | list_str | list_list_str
    nullable: bool = False
    isin: tuple | None = None
    unit_interval: bool = False
    nonempty: bool = False


ARROW_TYPES = {
    "str": pa_arrow.string(),
    "json": pa_arrow.string(),
    "int": pa_arrow.int64(),
    "float": pa_arrow.float64(),
    "bool": pa_arrow.bool_(),
    "list_str": pa_arrow.list_(pa_arrow.string()),
    "list_list_str": pa_arrow.list_(pa_arrow.list_(pa_arrow.string())),
}


def _is_null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _type_ok(kind: str, v) -> bool:
    if kind in ("str", "json"):
        return isinstance(v, str)
    if kind == "int":
        return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
    if kind == "float":
        return isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_))
    if kind == "bool":
        return isinstance(v, (bool, np.bool_))
    if kind == "list_str":
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if kind == "list_list_str":
        return isinstance(v, list) and all(
            isinstance(r, list) and all(isinstance(x, str) for x in r) for r in v
        )
    raise ValueError(kind)


def _column(name: str, col: Col) -> pa.Column:
    def check(series: pd.Series) -> pd.Series:
        def ok(v) -> bool:
            if _is_null(v):
                return col.nullable
            if not _type_ok(col.kind, v):
                return False
            if col.isin is not None and v not in col.isin:
                return False
            if col.unit_interval and not (0.0 <= float(v) <= 1.0):
                return False
            if col.nonempty and len(v) == 0:
                return False
            return True

        return series.map(ok).astype(bool)

    return pa.Column(
        dtype=None, nullable=True, checks=pa.Check(check, error=f"{name}: invalid value"), name=name
    )


@dataclass(frozen=True)
class TableDef:
    path: str
    columns: dict[str, Col]
    unique: tuple[str, ...] = ("item_id",)

    def pandera(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            {n: _column(n, c) for n, c in self.columns.items()},
            strict=True,
            ordered=True,
            unique=list(self.unique) if self.unique else None,
        )

    def arrow(self) -> pa_arrow.Schema:
        return pa_arrow.schema(
            [pa_arrow.field(n, ARROW_TYPES[c.kind], nullable=c.nullable) for n, c in self.columns.items()]
        )

    def list_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.kind.startswith("list")]


ITEM_COLUMNS: dict[str, Col] = {
    "item_id": Col("str", nonempty=True),
    "dataset": Col("str", isin=DATASETS),
    "split": Col("str", isin=SPLITS),
    "source_split": Col("str", isin=SOURCE_SPLITS),
    "question": Col("str", nonempty=True),
    "context_text": Col("list_str"),
    "context_pos": Col("list_str"),
    "table": Col("list_list_str"),
    "program": Col("str", nullable=True),
    "derivation": Col("str", nullable=True),
    "gold_answer": Col("str", nonempty=True),
    "gold_value": Col("json"),
    "answer_type": Col("str", isin=ANSWER_TYPES),
    "source_answer_type": Col("str"),
    "scale": Col("str"),
    "support_ids": Col("list_str"),
    "context_chars": Col("int"),
    "qa_tokens": Col("int"),
    "is_derived": Col("bool"),
    "base_item_id": Col("str", nonempty=True),
    "perturbation": Col("str", nullable=True),
}

TABLES: dict[str, TableDef] = {
    "pool": TableDef("data/pool.parquet", ITEM_COLUMNS),
    "verified": TableDef(
        "data/verified.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "exec_value": Col("json", nullable=True),
            "verified": Col("bool", nullable=True),
            "failure_reason": Col("str", nullable=True, isin=FAILURE_REASONS),
        },
    ),
    "q3_checks": TableDef(
        "data/q3_checks.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "check_text": Col("str", nonempty=True),
            "check_type": Col("str", isin=CHECK_TYPES),
            "source": Col("str", isin=("template", "llm")),
            "template_op": Col("str", nullable=True),
            "generator_model": Col("str"),
            "validated": Col("bool", nullable=True),
        },
    ),
    "jev_scores_long": TableDef(
        "scores/jev_scores_long.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "battery": Col("str", isin=("orig", "para")),
            "question": Col("str", isin=("q1", "q2", "q3")),
            "p_yes": Col("float", unit_interval=True),
            "p_no": Col("float", unit_interval=True),
            "entropy_bits": Col("float", unit_interval=True),
            "n_calls": Col("int"),
            "batched": Col("bool"),
            "jev_model_version": Col("str", nonempty=True),
            "request_id": Col("str", nonempty=True),   # ";"-joined when n_calls > 1
            "timestamp": Col("str"),
        },
        unique=("item_id", "battery", "question"),
    ),
    # One row per request that produced the stored scores, for cost tracking. Requests made only by the
    # determinism and batching checks are totalled in results/jev_manifest.json instead.
    "jev_requests": TableDef(
        "scores/jev_requests.parquet",
        {
            "request_id": Col("str", nonempty=True),
            "item_id": Col("str", nonempty=True),
            "n_questions": Col("int"),
            "input_tokens": Col("int"),
            "jev_model_version": Col("str", nonempty=True),
            "timestamp": Col("str"),
        },
        unique=("request_id",),
    ),
    "jev_scores": TableDef(
        "scores/jev_scores.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "is_negative": Col("bool"),
            "R": Col("float", unit_interval=True),
            "I": Col("float", unit_interval=True),
            "V": Col("float", unit_interval=True),
            "indecision": Col("float", unit_interval=True),
            "R_para": Col("float", unit_interval=True),
            "I_para": Col("float", unit_interval=True),
            "V_para": Col("float", unit_interval=True),
            "indecision_para": Col("float", unit_interval=True),
        },
    ),
    "surprisal": TableDef(
        "scores/surprisal.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "nll_sum": Col("float"),
            "n_answer_tokens": Col("int"),
            "nll_mean": Col("float"),
            "S": Col("float", nullable=True, unit_interval=True),
            "acc": Col("float", unit_interval=True),
            "n_correct": Col("int"),
            "n_unparsed": Col("int"),
            "modal_answer": Col("str", nullable=True),
            "modal_share": Col("float", unit_interval=True),
            "boundary_adjusted": Col("bool"),
        },
    ),
    "surprisal_samples": TableDef(
        "scores/surprisal_samples.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "k": Col("int"),
            "seed": Col("int"),
            "text": Col("str"),
            "answer": Col("str", nullable=True),
            "correct": Col("bool"),
        },
        unique=("item_id", "k"),
    ),
    "sc_subsample": TableDef(
        "data/sc_subsample.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "dataset": Col("str", isin=("finqa", "tatqa")),
            "answer_type": Col("str", isin=("arith", "lookup")),
            "r_quintile": Col("int"),
        },
    ),
    "sc_samples": TableDef(
        "scores/sc_samples.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "k": Col("int"),
            "seed": Col("int"),
            "answer": Col("str", nullable=True),
            "correct": Col("bool"),
            "text": Col("str"),
        },
        unique=("item_id", "k"),
    ),
    "self_consistency": TableDef(
        "scores/self_consistency.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "n_samples": Col("int"),
            "n_extracted": Col("int"),
            "sc_acc": Col("float", unit_interval=True),
            "sc_modal_share": Col("float", unit_interval=True),
            "modal_answer": Col("str", nullable=True),
            "modal_correct": Col("bool"),
            "generator_model": Col("str"),
        },
    ),
    # Judge comparison: path is per judge, scores/judges/<judge>_long.parquet (pass path=). One row per item
    # and question; `variant` is the battery version of Q2 (v0, v1, v2) and "all" for Q1 and Q3, whose
    # wording every version shares. p_yes is None when neither yes nor no is among the top tokens.
    "judge_scores_long": TableDef(
        "scores/judges/{judge}_long.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "judge": Col("str", nonempty=True),
            "battery": Col("str", isin=("orig", "para")),
            "question": Col("str", isin=("q1", "q2", "q3")),
            "variant": Col("str", isin=("v0", "v1", "v2", "all")),
            "state": Col("str", isin=("judge", "qa")),
            "p_yes": Col("float", nullable=True, unit_interval=True),
            "mass": Col("float"),
            "top_token": Col("str"),
            "provider": Col("str", nonempty=True),
            "served_model": Col("str", nonempty=True),
            "request_id": Col("str", nonempty=True),
            "input_tokens": Col("int"),
            "cost": Col("float", nullable=True),
            "timestamp": Col("str"),
        },
        unique=("item_id", "battery", "question", "variant"),
    ),
    # Path is per judge, scores/judges/<judge>.parquet. Original battery only; R = Q1, I_<v> = mean(Q2_<v>, Q3).
    "judge_scores": TableDef(
        "scores/judges/{judge}.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "judge": Col("str", nonempty=True),
            "is_negative": Col("bool"),
            "base_item_id": Col("str", nonempty=True),
            "R": Col("float", nullable=True, unit_interval=True),
            "Q2_v0": Col("float", nullable=True, unit_interval=True),
            "Q2_v1": Col("float", nullable=True, unit_interval=True),
            "Q2_v2": Col("float", nullable=True, unit_interval=True),
            "Q3": Col("float", nullable=True, unit_interval=True),
            "I_v0": Col("float", nullable=True, unit_interval=True),
            "I_v1": Col("float", nullable=True, unit_interval=True),
            "I_v2": Col("float", nullable=True, unit_interval=True),
            "min_mass": Col("float"),
        },
        unique=("item_id",),
    ),
    # CBI replication: path is per judge, scores/judges_cbi/<judge>_long.parquet. One row per unanimous item and
    # request: the gold arm (Q1, Q2 v2 answer-free, Q3) and the model arm (Q1 on the strong model's modal answer).
    "judge_cbi_long": TableDef(
        "scores/judges_cbi/{judge}_long.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "judge": Col("str", nonempty=True),
            "arm": Col("str", isin=("gold", "model")),
            "question": Col("str", isin=("q1", "q2", "q3")),
            "state": Col("str", isin=("judge", "qa")),
            "stated_answer": Col("str"),
            "p_yes": Col("float", nullable=True, unit_interval=True),
            "mass": Col("float"),
            "top_token": Col("str"),
            "provider": Col("str", nonempty=True),
            "served_model": Col("str", nonempty=True),
            "request_id": Col("str", nonempty=True),
            "input_tokens": Col("int"),
            "cost": Col("float", nullable=True),
            "timestamp": Col("str"),
        },
        unique=("item_id", "arm", "question"),
    ),
    # Path is per judge, scores/judges_cbi/<judge>.parquet. R_gold / R_model: Q1 with the gold / the model's answer.
    "judge_cbi": TableDef(
        "scores/judges_cbi/{judge}.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "judge": Col("str", nonempty=True),
            "cbi": Col("bool"),
            "jev_cell": Col("str", isin=("jev_sides_with_model", "both_supported", "gold_only", "neither")),
            "R_gold": Col("float", nullable=True, unit_interval=True),
            "Q2_v2": Col("float", nullable=True, unit_interval=True),
            "Q3": Col("float", nullable=True, unit_interval=True),
            "I_v2": Col("float", nullable=True, unit_interval=True),
            "R_model": Col("float", nullable=True, unit_interval=True),
            "min_mass": Col("float"),
        },
        unique=("item_id",),
    ),
    # Path is per task: annotation/<task>/labels.parquet (pass path= to write_table/read_table).
    "annotation_labels": TableDef(
        "annotation/{task}/labels.parquet",
        {
            "item_id": Col("str", nonempty=True),
            "annotator": Col("str", nonempty=True),
            "label_column": Col("str", nonempty=True),
            "label": Col("str", isin=("yes", "no", *CBI_AUDIT_LABELS)),
            "note": Col("str", nullable=True),
            "imported_at": Col("str"),
        },
        unique=("item_id", "annotator", "label_column"),
    ),
}
