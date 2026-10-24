"""Human labeling through CSV files, never an annotation platform (Spec 1, Annotation sheets).

export_sheets(task, items, cfg) writes annotation/<task>/<annotator>.csv per annotator plus a private key
file mapping opaque row IDs to item IDs (item IDs can reveal validation status, e.g. "#neg"). Annotators
fill in the label columns with yes or no and return the same file, which is placed in
annotation/<task>/returned/<annotator>.csv. import_sheets(task, cfg) enforces what a CSV cannot: labels
are yes/no after trimming and lowercasing; no row is added, removed or reordered; no item column is
edited (checked against the row_hash written at export). Labels go to annotation/<task>/labels.parquet.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from fqsel.config import root_dir
from fqsel.ids import rng
from fqsel.io import write_table
from fqsel.provenance import now_iso

KEY_FILE = "_key.csv"


class AnnotationError(ValueError):
    pass


def task_dir(cfg: DictConfig, task: str) -> Path:
    return root_dir(cfg) / "annotation" / task


def row_id(task: str, item_id: str) -> str:
    return hashlib.sha256(f"{task}|{item_id}".encode()).hexdigest()[:12]


def row_hash(values: list[str]) -> str:
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:16]


def _task(cfg: DictConfig, task: str):
    if task not in cfg.annotation.tasks:
        raise AnnotationError(f"unknown annotation task {task!r}")
    spec = cfg.annotation.tasks[task]
    return list(spec.item_columns), list(spec.label_columns)


def export_sheets(task: str, items: pd.DataFrame, cfg: DictConfig) -> list[Path]:
    """`items` has an item_id column plus the task's item columns."""
    item_cols, label_cols = _task(cfg, task)
    missing = [c for c in ["item_id", *item_cols] if c not in items.columns]
    if missing:
        raise AnnotationError(f"items lack columns {missing}")
    out_dir = task_dir(cfg, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, it in items.sort_values("item_id").iterrows():
        values = [str(it[c]) for c in item_cols]
        rows.append({"row_id": row_id(task, it["item_id"]), **dict(zip(item_cols, values, strict=True)),
                     "row_hash": row_hash(values), **{c: "" for c in label_cols}, "note": ""})
    pd.DataFrame(
        [{"row_id": r["row_id"], "item_id": it} for r, it in zip(rows, sorted(items["item_id"]), strict=True)]
    ).to_csv(out_dir / KEY_FILE, index=False, quoting=csv.QUOTE_ALL)
    header = ["row_id", *item_cols, "row_hash", *label_cols, "note"]
    paths = []
    for annotator in cfg.annotation.annotators:
        order = rng(task, annotator).permutation(len(rows))
        sheet = pd.DataFrame([rows[i] for i in order], columns=header)
        path = out_dir / f"{annotator}.csv"
        sheet.to_csv(path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
        paths.append(path)
    return paths


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def import_sheets(task: str, cfg: DictConfig) -> pd.DataFrame:
    item_cols, label_cols = _task(cfg, task)
    base = task_dir(cfg, task)
    key = dict(_read_csv(base / KEY_FILE)[["row_id", "item_id"]].itertuples(index=False))
    allowed = {str(v).lower() for v in cfg.annotation.labels}
    records = []
    imported_at = now_iso()
    for annotator in cfg.annotation.annotators:
        sent = _read_csv(base / f"{annotator}.csv")
        path = base / "returned" / f"{annotator}.csv"
        if not path.exists():
            raise AnnotationError(f"{task}: no returned file for {annotator} at {path}")
        back = _read_csv(path)
        where = f"{task}/{annotator}"
        if list(back.columns) != list(sent.columns):
            raise AnnotationError(f"{where}: columns changed from {list(sent.columns)} to {list(back.columns)}")
        if list(back["row_id"]) != list(sent["row_id"]):
            raise AnnotationError(f"{where}: rows were added, removed or reordered")
        for i, row in back.iterrows():
            if row_hash([row[c] for c in item_cols]) != row["row_hash"] or row["row_hash"] != sent.loc[i, "row_hash"]:
                raise AnnotationError(f"{where}: item columns of row {row['row_id']} were edited")
            for col in label_cols:
                label = row[col].strip().lower()
                if label not in allowed:
                    raise AnnotationError(f"{where}: row {row['row_id']} has invalid label {row[col]!r} in {col}")
                records.append({"item_id": key[row["row_id"]], "annotator": annotator, "label_column": col,
                                "label": label, "note": row["note"].strip() or None,
                                "imported_at": imported_at})
    df = pd.DataFrame(records).sort_values(["item_id", "annotator", "label_column"]).reset_index(drop=True)
    write_table("annotation_labels", df, cfg, path=base / "labels.parquet")
    return df
