"""Stage registry: `python -m fqsel run <stage>` (Spec 1, Code layout; e3 from Spec 6).

Stages not yet implemented are placeholders that report their milestone. Gated stages check
results/gates.json at start-up, even as placeholders, so the halt guard exists from day one.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from omegaconf import DictConfig

from fqsel.gates import require_not_halted


@dataclass(frozen=True)
class Stage:
    name: str
    spec: int
    milestone: str
    module: str | None = None   # fqsel.stages.<module> with run(cfg); None = placeholder
    gated: bool = False
    spec6_gate: bool = False
    ignores: tuple[str, ...] = ()  # config prefixes / prompt files this stage never reads (may stay ???)


# Spec 1 stages run before the proxies (P1), the statistics (P8) and the judge (P3) are needed.
_EARLY = ("proxy", "stats", "judge_quality.txt")
# The judge comparison (python -m fqsel judges) is outside the pipeline; no stage reads its settings.
JUDGES_SCOPE = ("judges", "judge_yesno.txt")


STAGES: list[Stage] = [
    Stage("pool", 1, "M2", "pool", ignores=(*_EARLY, "jev", *JUDGES_SCOPE)),
    Stage("verify", 1, "M2", "verify", ignores=(*_EARLY, "jev", *JUDGES_SCOPE)),
    Stage("q3", 1, "M2", "q3", ignores=(*_EARLY, "jev", *JUDGES_SCOPE)),
    Stage("jev", 1, "M3", "jev", ignores=(*_EARLY, *JUDGES_SCOPE)),
    Stage("surprisal", 2, "M4", "surprisal", ignores=(*_EARLY, "jev", *JUDGES_SCOPE)),
    Stage("comparators", 3, "M5", ignores=JUDGES_SCOPE),
    Stage("e0", 3, "M5", ignores=JUDGES_SCOPE),
    Stage("gates", 3, "M5", ignores=JUDGES_SCOPE),
    Stage("frame", 3, "M5", ignores=JUDGES_SCOPE),
    Stage("influence", 4, "M6", ignores=JUDGES_SCOPE),
    Stage("icl", 4, "M6", gated=True, ignores=JUDGES_SCOPE),
    Stage("e1", 5, "M7", gated=True, ignores=JUDGES_SCOPE),
    Stage("e2", 5, "M7", gated=True, ignores=JUDGES_SCOPE),
    Stage("e3", 6, "M8", gated=True, spec6_gate=True, ignores=JUDGES_SCOPE),
    Stage("figures", 5, "M7", ignores=JUDGES_SCOPE),
]
STAGE_NAMES = [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(f"unknown stage {name!r}; known: {', '.join(STAGE_NAMES)}")


def run_stage(name: str, cfg: DictConfig) -> int:
    stage = get_stage(name)
    if stage.gated:
        require_not_halted(cfg, name, spec6=stage.spec6_gate)
    if stage.module is None:
        print(f"stage {name}: placeholder, implemented in milestone {stage.milestone} (Spec {stage.spec})")
        return 0
    module = importlib.import_module(f"fqsel.stages.{stage.module}")
    module.run(cfg)
    return 0
