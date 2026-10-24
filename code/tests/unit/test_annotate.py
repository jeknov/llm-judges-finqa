"""CSV annotation sheets: export, return, strict import (Spec 1, Annotation sheets; never Argilla)."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from fqsel.annotate import KEY_FILE, AnnotationError, export_sheets, import_sheets, task_dir


@pytest.fixture
def items():
    return pd.DataFrame({
        "item_id": ["finqa:a", "finqa:b#neg", "tatqa:c"],
        "item_text": ['Context:\nrevenue, "net", up\n\nQuestion: x?', "Table:\n| a | b |", "plain"],
        "check_text": ["The ratio is above 1.", "The total is larger than 4.", "Europe is listed."],
        "secret_score": [0.9, 0.1, 0.5],
    })


def _return(cfg, annotator, edit=None):
    sent = pd.read_csv(task_dir(cfg, "q3_validation") / f"{annotator}.csv", dtype=str, keep_default_na=False)
    sent["follows_from_answer"] = ["yes", " YES ", "no"][: len(sent)]
    if edit:
        sent = edit(sent)
    out = task_dir(cfg, "q3_validation") / "returned"
    out.mkdir(exist_ok=True)
    sent.to_csv(out / f"{annotator}.csv", index=False, quoting=csv.QUOTE_ALL)


def test_round_trip(cfg, items):
    paths = export_sheets("q3_validation", items, cfg)
    assert [p.name for p in paths] == ["annotator_1.csv", "annotator_2.csv"]
    for a in ("annotator_1", "annotator_2"):
        _return(cfg, a)
    labels = import_sheets("q3_validation", cfg)
    assert len(labels) == 6 and set(labels["label"]) <= {"yes", "no"}
    assert set(labels["item_id"]) == {"finqa:a", "finqa:b#neg", "tatqa:c"}
    assert (task_dir(cfg, "q3_validation") / "labels.parquet").exists()


def test_sheets_leak_nothing(cfg, items):
    export_sheets("q3_validation", items, cfg)
    text = (task_dir(cfg, "q3_validation") / "annotator_1.csv").read_text()
    header = text.splitlines()[0]
    assert header == '"row_id","item_text","check_text","row_hash","follows_from_answer","note"'
    assert "#neg" not in text and "finqa:" not in text and "0.9" not in text
    assert "finqa:b#neg" in (task_dir(cfg, "q3_validation") / KEY_FILE).read_text()


def test_awkward_text_survives(cfg, items):
    export_sheets("q3_validation", items, cfg)
    sheet = pd.read_csv(task_dir(cfg, "q3_validation") / "annotator_1.csv", dtype=str, keep_default_na=False)
    assert set(sheet["item_text"]) == set(items["item_text"])


def test_per_annotator_order_is_seeded(cfg, items):
    export_sheets("q3_validation", items, cfg)
    first = (task_dir(cfg, "q3_validation") / "annotator_1.csv").read_text()
    export_sheets("q3_validation", items, cfg)
    assert (task_dir(cfg, "q3_validation") / "annotator_1.csv").read_text() == first


@pytest.mark.parametrize("edit,message", [
    (lambda s: s.assign(follows_from_answer=["maybe", "yes", "no"]), "invalid label"),
    (lambda s: s.assign(follows_from_answer=["", "yes", "no"]), "invalid label"),
    (lambda s: s.iloc[::-1], "reordered"),
    (lambda s: s.iloc[:2], "removed"),
    (lambda s: s.assign(check_text=["edited", s.loc[1, "check_text"], s.loc[2, "check_text"]]), "edited"),
    (lambda s: s.drop(columns=["note"]), "columns changed"),
])
def test_import_rejects(cfg, items, edit, message):
    export_sheets("q3_validation", items, cfg)
    _return(cfg, "annotator_1", edit)
    _return(cfg, "annotator_2")
    with pytest.raises(AnnotationError, match=message):
        import_sheets("q3_validation", cfg)


def test_missing_return_file(cfg, items):
    export_sheets("q3_validation", items, cfg)
    with pytest.raises(AnnotationError, match="no returned file"):
        import_sheets("q3_validation", cfg)
