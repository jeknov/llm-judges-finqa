"""M0 guard-rail tests: repository-wide rules that must hold from day one."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml
from omegaconf.errors import ConfigAttributeError

from fqsel.config import CODE_DIR, PlaceholderError, find_missing, load_config, unfinished_prompts
from fqsel.gates import HaltError, update_gates
from fqsel.stages import STAGE_NAMES, STAGES, run_stage

PKG = CODE_DIR / "fqsel"
SPEC_STAGE_ORDER = ["pool", "verify", "q3", "jev", "surprisal", "comparators", "e0", "gates", "frame",
                    "influence", "icl", "e1", "e2", "e3", "figures"]


def _py_files(*roots: Path) -> list[Path]:
    return [p for r in roots for p in r.rglob("*.py") if ".venv" not in p.parts]


def _calls_named(tree: ast.AST, names: set[str]) -> list[int]:
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names]


@pytest.mark.spec("S1-13")
def test_no_builtin_hash():
    offenders = [f"{p}:{line}" for p in _py_files(PKG) for line in _calls_named(ast.parse(p.read_text()), {"hash"})]
    assert not offenders, f"built-in hash() is banned; use fqsel.ids.stable_seed: {offenders}"


def test_no_eval_or_exec():
    offenders = [f"{p}:{line}" for p in _py_files(PKG)
                 for line in _calls_named(ast.parse(p.read_text()), {"eval", "exec"})]
    assert not offenders, f"eval/exec are banned: {offenders}"


def test_python_only():
    r_files = [p for p in CODE_DIR.rglob("*.R") if ".venv" not in p.parts]
    rpy2 = [p for p in _py_files(PKG, CODE_DIR / "tests", CODE_DIR / "scripts")
            if re.search(r"^\s*(import|from)\s+rpy2", p.read_text(), re.MULTILINE)]
    assert not r_files and not rpy2, "the pipeline is Python-only (owner decision 2026-10-23)"


def test_stage_registry_matches_spec_and_dvc():
    assert STAGE_NAMES == SPEC_STAGE_ORDER
    dvc = yaml.safe_load((CODE_DIR / "dvc.yaml").read_text())
    assert list(dvc["stages"]) == SPEC_STAGE_ORDER
    for name, stage in dvc["stages"].items():
        assert stage["cmd"].startswith(f"python -m fqsel run {name} ")


def test_single_prompt_assembler():
    """build_messages in fqsel/prompts.py is the only place chat messages are built."""
    offenders = []
    for p in _py_files(PKG):
        if p.name == "prompts.py":
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value == "role" for k in node.keys
            ):
                offenders.append(f"{p}:{node.lineno}")
    assert not offenders, f"chat messages built outside build_messages: {offenders}"


def test_acceptance_traceability():
    table = (CODE_DIR / "tests" / "ACCEPTANCE.md").read_text()
    rows = re.findall(r"^\| (S\d-\d\d) \| .*? \| (implemented|partial|pending) \|", table, re.MULTILINE)
    status = dict(rows)
    assert len(status) == len(rows), "duplicate IDs in ACCEPTANCE.md"
    marked: set[str] = set()
    for p in _py_files(CODE_DIR / "tests"):
        marked |= set(re.findall(r'mark\.spec\("(S\d-\d\d)"\)', p.read_text()))
    unknown = marked - set(status)
    untested = {i for i, s in status.items() if s in ("implemented", "partial")} - marked
    premature = {i for i in marked if status.get(i) == "pending"}
    assert not unknown, f"tests reference IDs missing from ACCEPTANCE.md: {sorted(unknown)}"
    assert not untested, f"implemented/partial IDs without a test: {sorted(untested)}"
    assert not premature, f"IDs with tests still marked pending: {sorted(premature)}"


def test_config_is_strict(cfg):
    with pytest.raises(ConfigAttributeError):
        _ = cfg.data.no_such_key


def test_tiny_profile_has_no_missing_values_it_uses(cfg):
    """Tiny runs with mocks; only values not needed by tiny (real models, later milestones) may be ???."""
    allowed = {"stats.top_rounding", "stats.top_tie_break"}
    assert set(find_missing(cfg)) <= allowed


@pytest.mark.parametrize("profile", ["pilot", "full"])
def test_no_placeholders_for_real_runs(profile):
    """Pilot/full refuse to load while any ??? or non-FINAL prompt remains (Placeholder policy)."""
    assert unfinished_prompts(), "if every prompt is FINAL this test needs the config-only branch"
    with pytest.raises(PlaceholderError) as err:
        load_config([f"profile={profile}"])
    message = str(err.value)
    assert "stats.top_rounding" in message and "prompts/judge_quality.txt" in message
    assert "proxy.lora.seed" not in message  # P1 decided 2026-10-24
    assert "model.base" not in message  # decided 2026-10-23: OLMo 3 7B Instruct


def test_spec1_stages_ignore_only_later_decisions():
    """Spec 1 stages may run before P1/P3/P8 are decided, but never skip their own sections."""
    from fqsel.stages import get_stage

    for name in ("pool", "verify", "q3", "jev"):
        ignores = set(get_stage(name).ignores)
        assert {"proxy", "stats", "judge_quality.txt"} <= ignores
        assert not ignores & {"data", "model", "q3_fallback.txt", "jev_battery_v0.yaml", "jev_battery_v1.yaml"}
    assert "jev" not in get_stage("jev").ignores
    assert set(get_stage("surprisal").ignores) == {"proxy", "stats", "judge_quality.txt", "jev"}  # Spec 2 needs only the pool
    assert all(not s.ignores for s in STAGES if s.spec > 2)


def test_jev_stage_loads_for_real_runs_but_refuses_its_own_placeholders():
    from fqsel.stages import get_stage

    load_config(["profile=pilot"], ignore=get_stage("jev").ignores)  # every Spec 1 value is decided
    with pytest.raises(PlaceholderError) as err:
        load_config(["profile=pilot", "jev.api.answers_field=???"], ignore=get_stage("jev").ignores)
    message = str(err.value)
    assert "jev.api.answers_field" in message
    assert "stats.top_rounding" not in message and "judge_quality.txt" not in message


def test_ignored_value_still_fails_when_read():
    """Ignoring a section only skips the start-up check; reading a ??? there still raises."""
    from omegaconf.errors import MissingMandatoryValue

    from fqsel.config import check_real_run_ready

    cfg = load_config()
    check_real_run_ready(cfg, ignore=find_missing(cfg) + unfinished_prompts())
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.stats.top_rounding


def test_model_base_is_pinned_to_a_commit():
    """model.base must name one checkpoint at an exact commit, never a branch or tag (Spec 2)."""
    from omegaconf import OmegaConf

    base = OmegaConf.load(CODE_DIR / "conf" / "model" / "base.yaml")
    assert base.name == "allenai/Olmo-3-7B-Instruct"
    assert re.fullmatch(r"[0-9a-f]{40}", base.revision)
    assert base.min_context >= 16384


@pytest.mark.spec("S3-07")
@pytest.mark.spec("S6-01")
@pytest.mark.parametrize("stage", [s.name for s in STAGES if s.gated])
def test_halt_guard_on_gated_stages(cfg, stage):
    with pytest.raises(HaltError):  # no gates.json yet
        run_stage(stage, cfg)

    def halt(g):
        g["derived"]["halt"] = True
        g["derived"]["halt_reason"] = "difficulty_proxy"

    update_gates(cfg, halt)
    with pytest.raises(HaltError):
        run_stage(stage, cfg)

    update_gates(cfg, lambda g: g["derived"].update(halt=False, halt_reason=None))
    assert run_stage(stage, cfg) == 0


@pytest.mark.spec("S6-01")
def test_spec6_gate_needs_override(cfg):
    update_gates(cfg, lambda g: g["derived"].update(halt_spec6=True))
    with pytest.raises(HaltError):
        run_stage("e3", cfg)
    assert run_stage("e1", cfg) == 0  # D5 halts only Spec 6
    update_gates(cfg, lambda g: g["derived"].update(halt_spec6_override="owner: run E3 as exploratory"))
    assert run_stage("e3", cfg) == 0


def test_gates_schema_file_is_valid_json():
    json.loads((CODE_DIR / "conf" / "gates.schema.json").read_text())


def test_annotate_uses_q3_placeholder_scope(monkeypatch):
    """`annotate` reads Q3 data only, so it may run once Spec 1's own values are decided."""
    from fqsel import cli
    from fqsel.stages import get_stage

    seen = {}
    monkeypatch.setattr(cli, "load_config", lambda overrides, ignore=(): seen.setdefault("ignore", ignore) and None)
    monkeypatch.setattr("fqsel.q3_validation.export_task", lambda task, cfg: [])
    cli.main(["annotate", "export", "q3_validation"])
    assert tuple(seen["ignore"]) == get_stage("q3").ignores
