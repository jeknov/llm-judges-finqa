"""Stage `verify` (Spec 1, Verified correctness): execution result for every item in pool.parquet,
including the #neg copies (whose wrong answers fail verification by construction)."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from fqsel.io import encode_json, read_table, write_table
from fqsel.verify import verify_item


def run(cfg: DictConfig) -> None:
    pool = read_table("pool", cfg)
    rows = []
    for _, item in pool.iterrows():
        v = verify_item(item)
        rows.append({
            "item_id": item["item_id"],
            "exec_value": None if v.exec_value is None else encode_json(v.exec_value),
            "verified": v.verified,
            "failure_reason": v.failure_reason,
        })
    df = pd.DataFrame(rows, columns=["item_id", "exec_value", "verified", "failure_reason"])
    write_table("verified", df.sort_values("item_id"), cfg)
    print(f"verify: {int((df['verified'] == True).sum())} verified of {len(df)}")  # noqa: E712
