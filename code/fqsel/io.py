"""read_table / write_table: schema validation and provenance metadata on every read and write (Spec 1).

Metadata holds the git commit, config hash and model identifiers but no timestamp, so that rebuilding a
table from the same inputs reproduces the file byte for byte (Spec 1 acceptance).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa_arrow
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

from fqsel.config import config_hash, root_dir
from fqsel.provenance import git_commit
from fqsel.schemas import TABLES

META_PREFIX = "fqsel."


def table_path(cfg: DictConfig, name: str, path: str | Path | None = None) -> Path:
    rel = Path(path) if path is not None else Path(TABLES[name].path)
    return rel if rel.is_absolute() else root_dir(cfg) / rel


def model_ids(cfg: DictConfig) -> dict[str, str]:
    c = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    base, strong = c["model"]["base"], c["model"]["strong"]
    return {
        "model.base": f"{base['name']}@{base['revision']}",
        "model.strong": f"{strong['name']}@{strong['version']}",
        "jev": str(c["jev"]["model_version"]),
    }


def _to_python(v):
    if isinstance(v, np.ndarray):
        return [_to_python(x) for x in v.tolist()]
    if isinstance(v, list):
        return [_to_python(x) for x in v]
    return v


def write_table(
    name: str,
    df: pd.DataFrame,
    cfg: DictConfig,
    path: str | Path | None = None,
    extra_meta: dict | None = None,
) -> Path:
    tdef = TABLES[name]
    df = df.reset_index(drop=True)
    tdef.pandera().validate(df, lazy=True)
    table = pa_arrow.Table.from_pandas(df, schema=tdef.arrow(), preserve_index=False)
    meta = {
        "fqsel.table": name,
        "fqsel.git_commit": git_commit(),
        "fqsel.config_hash": config_hash(cfg),
        "fqsel.model_ids": json.dumps(model_ids(cfg), sort_keys=True),
        "fqsel.extra": json.dumps(extra_meta or {}, sort_keys=True, default=str),
    }
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in meta.items()})
    out = table_path(cfg, name, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    return out


def read_table(name: str, cfg: DictConfig, path: str | Path | None = None) -> pd.DataFrame:
    if name == "items":
        # Spec 1: the union of pool.parquet and e3_items.parquet (added by Spec 6 at M8).
        frames = [read_table("pool", cfg)]
        e3 = table_path(cfg, "pool", "data/e3_items.parquet")
        if e3.exists() and "e3_items" in TABLES:
            frames.append(read_table("e3_items", cfg))
        out = pd.concat(frames, ignore_index=True)
        if out["item_id"].duplicated().any():
            raise ValueError("items: duplicate item_id across pool and e3_items")
        return out
    tdef = TABLES[name]
    df = pq.read_table(table_path(cfg, name, path)).to_pandas()
    for col in tdef.list_columns():
        df[col] = df[col].map(_to_python)
    tdef.pandera().validate(df, lazy=True)
    return df


def read_metadata(name: str, cfg: DictConfig, path: str | Path | None = None) -> dict[str, str]:
    raw = pq.read_schema(table_path(cfg, name, path)).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items() if k.decode().startswith(META_PREFIX)}


def encode_json(value) -> str:
    return json.dumps(value, sort_keys=True)


def decode_json(text: str | None):
    return None if text is None else json.loads(text)
