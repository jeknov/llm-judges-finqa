"""Human validation of Q3 checks through CSV sheets (Spec 1, Validation)."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from fqsel.annotate import task_dir
from fqsel.io import read_table
from fqsel.q3_validation import export_task, import_task
from fqsel.stages import run_stage


@pytest.fixture
def built(make_cfg, tmp_path):
    cfg = make_cfg(root=tmp_path / "run")
    for stage in ("pool", "q3"):
        run_stage(stage, cfg)
    return cfg


def _answer_all(cfg, task, label_for):
    base = task_dir(cfg, task)
    key = pd.read_csv(base / "_key.csv", dtype=str).set_index("row_id")["item_id"]
    (base / "returned").mkdir(exist_ok=True)
    for annotator in cfg.annotation.annotators:
        sheet = pd.read_csv(base / f"{annotator}.csv", dtype=str, keep_default_na=False)
        sheet["follows_from_answer"] = [label_for(key[r], annotator) for r in sheet["row_id"]]
        sheet.to_csv(base / "returned" / f"{annotator}.csv", index=False, quoting=csv.QUOTE_ALL)


@pytest.mark.spec("S1-07")
def test_llm_check_validation_accepts_at_90_percent(built):
    cfg = built
    export_task("q3_validation", cfg)
    q3 = read_table("q3_checks", cfg)
    sampled = pd.read_csv(task_dir(cfg, "q3_validation") / "_key.csv")["item_id"]
    assert len(sampled) == cfg.data.q3.validation_n
    assert set(q3.set_index("item_id").loc[sampled, "source"]) == {"llm"}
    first = sorted(sampled)[0]
    _answer_all(cfg, "q3_validation", lambda item, a: "no" if (item == first and a == "annotator_2") else "yes")
    summary = import_task("q3_validation", cfg)
    assert summary["share_both_yes"] == pytest.approx(0.9) and summary["accepted"] is True
    validated = read_table("q3_checks", cfg).set_index("item_id")["validated"]
    assert validated[first] is False or validated[first] == False  # noqa: E712
    assert validated[sorted(sampled)[1]] == True  # noqa: E712
    assert validated.drop(list(sampled)).isna().all()


@pytest.mark.spec("S1-07")
def test_llm_check_validation_rejects_below_90_percent(built):
    cfg = built
    export_task("q3_validation", cfg)
    sampled = sorted(pd.read_csv(task_dir(cfg, "q3_validation") / "_key.csv")["item_id"])
    _answer_all(cfg, "q3_validation", lambda item, a: "no" if item in sampled[:2] else "yes")
    assert import_task("q3_validation", cfg)["accepted"] is False


@pytest.mark.spec("S1-07")
def test_template_hand_check_sheets(built):
    cfg = built
    export_task("q3_templates", cfg)
    q3 = read_table("q3_checks", cfg).set_index("item_id")
    sampled = pd.read_csv(task_dir(cfg, "q3_templates") / "_key.csv")["item_id"]
    ops = q3.loc[sampled, "template_op"]
    assert set(q3.loc[sampled, "source"]) == {"template"}
    for op, n_available in q3[q3["source"] == "template"]["template_op"].value_counts().items():
        assert (ops == op).sum() == min(n_available, cfg.data.q3.template_check_n)
    _answer_all(cfg, "q3_templates", lambda item, a: "yes")
    summary = import_task("q3_templates", cfg)
    assert all(v == 1.0 for v in summary["share_both_yes_by_template"].values())
    assert json.loads((cfg_root(cfg) / "results" / "q3_templates.json").read_text())["task"] == "q3_templates"


def cfg_root(cfg):
    from fqsel.config import root_dir

    return root_dir(cfg)
