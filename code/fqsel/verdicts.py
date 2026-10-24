"""record_verdict: upsert one row of results/hypotheses.csv keyed by (hypothesis, proxy) (Spec 1)."""

from __future__ import annotations

import json
import os

import pandas as pd
from filelock import FileLock
from omegaconf import DictConfig

from fqsel.config import root_dir
from fqsel.provenance import git_commit, now_iso

HYPOTHESES_PATH = "results/hypotheses.csv"
COLUMNS = ["hypothesis", "proxy", "verdict", "rule", "evidence", "recorded_at", "git_commit"]


def record_verdict(
    cfg: DictConfig, hypothesis: str, proxy: str, verdict: str, rule: str, evidence: dict | str
) -> pd.DataFrame:
    path = root_dir(cfg) / HYPOTHESES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=60):
        df = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=COLUMNS)
        df = df[~((df["hypothesis"] == hypothesis) & (df["proxy"] == proxy))]
        row = {
            "hypothesis": hypothesis,
            "proxy": proxy,
            "verdict": verdict,
            "rule": rule,
            "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True),
            "recorded_at": now_iso(),
            "git_commit": git_commit(),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values(["hypothesis", "proxy"]).reset_index(drop=True)[COLUMNS]
        tmp = path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return df
