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


def test_export_never_overwrites_entered_labels_and_reexport_is_identical(cfg, items):
    assignment = {"annotator_1": ["finqa:x", "finqa:y"], "annotator_2": ["tatqa:z"]}
    export_sheets("cbi_audit", items, cfg, assignment=assignment)
    sheets = {p.name: p.read_bytes() for p in task_dir(cfg, "cbi_audit").glob("*.csv")}
    export_sheets("cbi_audit", items, cfg, assignment=assignment)          # unlabelled: same bytes again
    assert {p.name: p.read_bytes() for p in task_dir(cfg, "cbi_audit").glob("*.csv")} == sheets
    path = task_dir(cfg, "cbi_audit") / "annotator_1.csv"
    sent = pd.read_csv(path, dtype=str, keep_default_na=False)
    sent.loc[0, "category"] = "model_error"                                 # the annotator has started
    sent.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    with pytest.raises(AnnotationError, match="already has entries"):
        export_sheets("cbi_audit", items, cfg, assignment=assignment)
    assert pd.read_csv(path, dtype=str, keep_default_na=False).loc[0, "category"] == "model_error"


def test_adjudication_sheet_has_only_the_disagreements_with_both_labels(cfg, items):
    assignment = {"annotator_1": ["finqa:x", "finqa:y", "tatqa:z"], "annotator_2": ["finqa:x", "finqa:y"]}
    export_sheets("cbi_audit", items, cfg, assignment=assignment)
    rid = dict(pd.read_csv(task_dir(cfg, "cbi_audit") / KEY_FILE, dtype=str)[["item_id", "row_id"]].itertuples(index=False))
    by_row = {"annotator_1": {rid["finqa:x"]: "model_error", rid["finqa:y"]: "questionable_gold", rid["tatqa:z"]: "scoring_artefact"},
              "annotator_2": {rid["finqa:x"]: "model_error", rid["finqa:y"]: "ambiguous_question"}}
    for a, lab in by_row.items():                                         # labels in each sheet's own row order
        sent = pd.read_csv(task_dir(cfg, "cbi_audit") / f"{a}.csv", dtype=str, keep_default_na=False)
        _return(cfg, a, [lab[r] for r in sent["row_id"]])
    import_sheets("cbi_audit", cfg)                                       # validated labels, as the import writes them
    path = cbi_audit.export_adjudication(cfg)[0]
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(sheet["row_id"]) == [rid["finqa:y"]]                       # agreed and single-annotated items left out
    row = sheet.iloc[0]
    assert (row["annotator_1_label"], row["annotator_2_label"]) == ("questionable_gold", "ambiguous_question")
    assert row["category"] == "" and row["note"] == ""
    original = pd.read_csv(task_dir(cfg, "cbi_audit") / "annotator_1.csv", dtype=str, keep_default_na=False).set_index("row_id")
    for c in ["item_text", "gold_answer", "gold_derivation", "model_answer", "model_reasoning", "row_hash"]:
        assert row[c] == original.loc[row["row_id"], c]                    # item shown exactly as before
    first = path.read_bytes()
    cbi_audit.export_adjudication(cfg)                                     # unlabelled: same bytes again
    assert path.read_bytes() == first
    sheet.loc[0, "category"] = "questionable_gold"                         # the third annotator has started
    sheet.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    with pytest.raises(AnnotationError, match="already has entries"):
        cbi_audit.export_adjudication(cfg)


def test_adjudication_import_gives_one_final_label_per_item(cfg, items):
    assignment = {"annotator_1": ["finqa:x", "finqa:y", "tatqa:z"], "annotator_2": ["finqa:x", "finqa:y"]}
    export_sheets("cbi_audit", items, cfg, assignment=assignment)
    base = task_dir(cfg, "cbi_audit")
    pd.DataFrame({"item_id": ["finqa:x", "finqa:y", "tatqa:z"], "stratum": ["A", "A", "B"],
                  "cell": ["jev_sides_with_model", "jev_sides_with_model", "both_supported"]}).to_csv(base / "_design.csv", index=False)
    rid = dict(pd.read_csv(base / KEY_FILE, dtype=str)[["item_id", "row_id"]].itertuples(index=False))
    by_row = {"annotator_1": {rid["finqa:x"]: "model_error", rid["finqa:y"]: "questionable_gold", rid["tatqa:z"]: "scoring_artefact"},
              "annotator_2": {rid["finqa:x"]: "model_error", rid["finqa:y"]: "ambiguous_question"}}
    for a, lab in by_row.items():
        sent = pd.read_csv(base / f"{a}.csv", dtype=str, keep_default_na=False)
        _return(cfg, a, [lab[r] for r in sent["row_id"]])
    import_sheets("cbi_audit", cfg)
    path = cbi_audit.export_adjudication(cfg)[0]
    with pytest.raises(AnnotationError, match="no returned adjudication file"):
        cbi_audit.import_adjudication(cfg)
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    (path.parent / "returned").mkdir()
    edited = sheet.assign(annotator_1_label="model_error", category="ambiguous_question")
    edited.to_csv(path.parent / "returned" / path.name, index=False, quoting=csv.QUOTE_ALL)
    with pytest.raises(AnnotationError, match="edited outside category and note"):   # earlier labels must stay as shown
        cbi_audit.import_adjudication(cfg)
    sheet.assign(category=" Ambiguous_Question ").to_csv(path.parent / "returned" / path.name, index=False, quoting=csv.QUOTE_ALL)
    summary = cbi_audit.import_adjudication(cfg)
    final = pd.read_parquet(base / "final_labels.parquet").set_index("item_id")
    assert final["label"].to_dict() == {"finqa:x": "model_error", "finqa:y": "ambiguous_question", "tatqa:z": "scoring_artefact"}
    assert final["source"].to_dict() == {"finqa:x": "agreed", "finqa:y": "adjudicated", "tatqa:z": "single"}
    a = summary["by_stratum"]["A"]
    assert a["n_items"] == 2 and a["labels"]["item_problem"]["n"] == 1 and a["labels"]["model_error"]["share"] == 0.5
    lo, hi = a["labels"]["item_problem"]["wilson_95"]
    assert lo < 0.5 < hi
