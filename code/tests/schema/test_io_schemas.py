"""read_table / write_table and the pandera schemas (Spec 1; Implementation Plan M1 tests)."""

from __future__ import annotations

import json

import pandas as pd
import pandera.errors
import pytest

from fqsel.io import decode_json, encode_json, read_metadata, read_table, table_path, write_table


def _item(item_id="finqa:a", **kw):
    row = {
        "item_id": item_id, "dataset": "finqa", "split": "pool", "source_split": "train",
        "question": "what is x?", "context_text": ["s1", "s2"], "context_pos": ["pre", "post"],
        "table": [["", "2019"], ["sales", "$ 5"]], "program": "add(1, 2)", "derivation": None,
        "gold_answer": "3", "gold_value": encode_json(3.0), "answer_type": "arith",
        "source_answer_type": "add", "scale": "", "support_ids": ["text_0"], "context_chars": 10,
        "qa_tokens": 20, "is_derived": False, "base_item_id": item_id, "perturbation": None,
    }
    row.update(kw)
    return row


def test_round_trip_preserves_values_and_metadata(cfg):
    df = pd.DataFrame([_item("finqa:a"), _item("tatqa:b", dataset="tatqa", program=None, derivation="1+2",
                                               table=[], gold_value=encode_json(["x", "y"]))])
    write_table("pool", df, cfg)
    back = read_table("pool", cfg)
    assert back.to_dict("records") == df.to_dict("records")
    assert decode_json(back.loc[1, "gold_value"]) == ["x", "y"]
    meta = read_metadata("pool", cfg)
    assert meta["fqsel.table"] == "pool"
    assert set(json.loads(meta["fqsel.model_ids"])) == {"model.base", "model.strong", "jev"}
    assert len(meta["fqsel.config_hash"]) == 64 and meta["fqsel.git_commit"]


def test_write_is_byte_deterministic(cfg, tmp_path):
    df = pd.DataFrame([_item("finqa:a")])
    write_table("pool", df, cfg)
    first = table_path(cfg, "pool").read_bytes()
    write_table("pool", df, cfg)
    assert table_path(cfg, "pool").read_bytes() == first


@pytest.mark.parametrize("bad", [
    {"item_id": "finqa:a"},                          # duplicate of the first row
    {"split": "train"},                              # not a split
    {"answer_type": "number"},
    {"gold_answer": ""},                             # empty gold
    {"context_text": "not a list"},
    {"qa_tokens": 1.5},
])
def test_pool_schema_rejects_bad_rows(cfg, bad):
    rows = [_item("finqa:a"), {**_item("finqa:z"), **bad}]
    with pytest.raises((pandera.errors.SchemaErrors, pandera.errors.SchemaError)):
        write_table("pool", pd.DataFrame(rows), cfg)


def test_probability_schema_rejects_out_of_range(cfg):
    row = {"item_id": "finqa:a", "battery": "orig", "question": "q1", "p_yes": 1.2, "p_no": -0.2,
           "entropy_bits": 0.0, "n_calls": 1, "jev_model_version": "m", "timestamp": "t"}
    with pytest.raises((pandera.errors.SchemaErrors, pandera.errors.SchemaError)):
        write_table("jev_scores_long", pd.DataFrame([row]), cfg)


def test_missing_column_rejected(cfg):
    df = pd.DataFrame([_item()]).drop(columns=["scale"])
    with pytest.raises((pandera.errors.SchemaErrors, pandera.errors.SchemaError)):
        write_table("pool", df, cfg)


def test_items_union_without_e3(cfg):
    write_table("pool", pd.DataFrame([_item("finqa:a")]), cfg)
    assert list(read_table("items", cfg)["item_id"]) == ["finqa:a"]
