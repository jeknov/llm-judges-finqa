"""(added) CBI hand audit: CBI definitions, sample design, per-annotator sheets with task labels, blinding."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from fqsel import cbi_audit
from fqsel.annotate import KEY_FILE, AnnotationError, export_sheets, import_sheets, task_dir
from fqsel.config import CODE_DIR, load_config
from fqsel.stages import get_stage

LABELS = ["model_error", "scoring_artefact", "ambiguous_question", "questionable_gold"]


def test_numeric_equivalence_and_consistency_follow_notebook_03():
    assert cbi_audit.numeric_equivalent("3,815,000,000", "NT$3,815 million", "")
    assert cbi_audit.numeric_equivalent("$239 million", "$239", "million")
    assert not cbi_audit.numeric_equivalent("240", "239", "")
    assert not cbi_audit.numeric_equivalent(None, "239", "")
    assert cbi_audit.entropy_consistency(["12", "12 ", "12", "12", "12"]) == pytest.approx(1.0)
    assert cbi_audit.entropy_consistency(["12", "12", "12", "12", None]) < 0.8        # 4 of 5 is not consistent at 0.8
    assert cbi_audit.entropy_consistency([None] * 5) == pytest.approx(0.0)           # unparsed answers never agree


def _synthetic_cbi():
    rows = [(f"finqa:a{i}", "jev_sides_with_model") for i in range(40)]
    rows += [(f"tatqa:b{i}", "both_supported") for i in range(80)]
    rows += [(f"tatqa:c{i}", "gold_only") for i in range(10)] + [(f"finqa:d{i}", "neither") for i in range(10)]
    df = pd.DataFrame(rows, columns=["item_id", "cell"]).set_index("item_id")
    df["stratum"] = (df["cell"] == "jev_sides_with_model").map({True: "A", False: "B"})
    return df


def test_design_counts_overlap_and_balance(cfg):
    d = cbi_audit.design(_synthetic_cbi(), cfg)
    assert len(d) == 60 and d["item_id"].is_unique
    for _stratum, g in d.groupby("stratum"):
        assert len(g) == 30 and g["double"].sum() == 10
        single = g[~g["double"]]["annotators"].value_counts().to_dict()
        assert single == {"annotator_1": 10, "annotator_2": 10}
    b = d[d["stratum"] == "B"]["cell"].value_counts().to_dict()
    assert b == {"both_supported": 24, "gold_only": 3, "neither": 3}          # proportional to 80/10/10
    assert (d[d["double"]]["annotators"] == "annotator_1;annotator_2").all()
    pd.testing.assert_frame_equal(d, cbi_audit.design(_synthetic_cbi(), cfg))   # seeded


def test_design_refuses_an_impossible_quota(cfg):
    small = _synthetic_cbi().iloc[:20]                                          # stratum A has only 20 items
    with pytest.raises(ValueError):
        cbi_audit.design(small, cfg)


@pytest.fixture
def items():
    return pd.DataFrame({"item_id": ["finqa:x", "finqa:y", "tatqa:z"],
                         "item_text": ["Question: a?", "Question: b?", "Question: c?"],
                         "gold_answer": ["1", "2", "3"], "gold_derivation": ["add(1, 0)", "", ""],
                         "model_answer": ["10", "20", "30"], "model_reasoning": ["r1", "r2", "r3"]})


def _return(cfg, annotator, labels):
    sent = pd.read_csv(task_dir(cfg, "cbi_audit") / f"{annotator}.csv", dtype=str, keep_default_na=False)
    sent["category"] = labels[: len(sent)]
    out = task_dir(cfg, "cbi_audit") / "returned"
    out.mkdir(exist_ok=True)
    sent.to_csv(out / f"{annotator}.csv", index=False, quoting=csv.QUOTE_ALL)


def test_per_annotator_sheets_round_trip_with_task_labels(cfg, items):
    assignment = {"annotator_1": ["finqa:x", "finqa:y"], "annotator_2": ["finqa:x", "tatqa:z"]}
    export_sheets("cbi_audit", items, cfg, assignment=assignment)
    sheets = {a: pd.read_csv(task_dir(cfg, "cbi_audit") / f"{a}.csv", dtype=str) for a in assignment}
    assert len(sheets["annotator_1"]) == 2 and len(sheets["annotator_2"]) == 2
    assert len(pd.read_csv(task_dir(cfg, "cbi_audit") / KEY_FILE)) == 3
    _return(cfg, "annotator_1", [" Model_Error ", "questionable_gold"])
    _return(cfg, "annotator_2", ["model_error", "scoring_artefact"])
    labels = import_sheets("cbi_audit", cfg)
    assert len(labels) == 4 and set(labels["label"]) <= set(LABELS)
    assert labels.groupby("item_id").size().to_dict() == {"finqa:x": 2, "finqa:y": 1, "tatqa:z": 1}


def test_labels_outside_the_task_set_are_rejected(cfg, items):
    export_sheets("cbi_audit", items, cfg, assignment={"annotator_1": ["finqa:x"], "annotator_2": ["finqa:y", "tatqa:z"]})
    _return(cfg, "annotator_1", ["yes"])                                       # a Q3 label is not an audit label
    _return(cfg, "annotator_2", ["model_error", "model_error"])
    with pytest.raises(AnnotationError, match="invalid label"):
        import_sheets("cbi_audit", cfg)


def test_assignment_must_cover_every_item_and_every_annotator(cfg, items):
    with pytest.raises(AnnotationError):
        export_sheets("cbi_audit", items, cfg, assignment={"annotator_1": ["finqa:x"], "annotator_2": ["finqa:y"]})
    with pytest.raises(AnnotationError):
        export_sheets("cbi_audit", items, cfg, assignment={"annotator_1": list(items["item_id"])})


def test_q3_tasks_still_label_yes_no(cfg):
    from fqsel.annotate import _task

    assert _task(cfg, "q3_validation")[2] == ["yes", "no"]
    assert _task(cfg, "cbi_audit")[2] == LABELS


SNAPSHOTS = CODE_DIR / "runs"
needs_snapshots = pytest.mark.skipif(not (SNAPSHOTS / "jev_on_qwen_2026-10-25").exists(), reason="snapshots not present")


@pytest.mark.real_data
@needs_snapshots
def test_real_cbi_population_reproduces_notebook_03_and_sheets_are_blind(tmp_path):
    (tmp_path / "runs").symlink_to(SNAPSHOTS)
    cfg = load_config(["profile=full", f"paths.root={tmp_path}"], ignore=get_stage("q3").ignores)
    cbi = cbi_audit.cbi_items(cfg)
    assert len(cbi) == 281                                                      # notebook 03, section 10
    assert cbi["cell"].value_counts().to_dict() == {"both_supported": 163, "jev_sides_with_model": 90,
                                                    "gold_only": 14, "neither": 14}
    cbi_audit.export_task(cfg)
    out = task_dir(cfg, "cbi_audit")
    plan = pd.read_csv(out / "_design.csv")
    assert len(plan) == 60 and plan["double"].sum() == 20
    for a in ("annotator_1", "annotator_2"):
        text = (out / f"{a}.csv").read_text()
        sheet = pd.read_csv(out / f"{a}.csv", dtype=str)
        assert len(sheet) == 40
        assert "jev_sides_with_model" not in text and "finqa:" not in text and "tatqa:" not in text
        assert list(sheet.columns) == ["row_id", "item_text", "gold_answer", "gold_derivation", "model_answer",
                                       "model_reasoning", "row_hash", "category", "note"]
    assert (out / "INSTRUCTIONS.md").exists()
