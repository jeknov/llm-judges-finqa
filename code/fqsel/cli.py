"""Command line: python -m fqsel {run <stage> | check | annotate export|import <task> | self-consistency}
[hydra overrides]."""

from __future__ import annotations

import argparse
import sys

from fqsel.config import load_config
from fqsel.gates import HaltError
from fqsel.stages import STAGE_NAMES, get_stage, run_stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fqsel")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run one pipeline stage")
    p_run.add_argument("stage", choices=STAGE_NAMES)
    p_run.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. profile=full")
    p_check = sub.add_parser("check", help="re-validate all outputs and cross-table invariants")
    p_check.add_argument("overrides", nargs="*")
    p_ann = sub.add_parser("annotate", help="export or import human-label CSV files")
    p_ann.add_argument("action", choices=["export", "import"])
    p_ann.add_argument("task")
    p_ann.add_argument("overrides", nargs="*")
    p_sc = sub.add_parser("self-consistency",
                          help="Spec 3 comparator sc_acc/sc_modal_share on the 3,000-item subsample (strong LLM only)")
    p_sc.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    # a command skips only the placeholders of the stage whose data it reads; annotation tasks are all Q3's,
    # and self-consistency reads only Spec 1 outputs and the strong-LLM settings, like the jev stage
    stage = {"run": getattr(args, "stage", None), "annotate": "q3", "self-consistency": "jev"}.get(args.command)
    cfg = load_config(args.overrides, ignore=get_stage(stage).ignores if stage else ())
    if args.command == "run":
        try:
            return run_stage(args.stage, cfg)
        except HaltError as e:
            print(f"HALT: {e}", file=sys.stderr)
            return e.code
    if args.command == "check":
        from fqsel.invariants import run_checks

        problems = run_checks(cfg)
        for p in problems:
            print(f"FAIL: {p}")
        print("check: OK" if not problems else f"check: {len(problems)} problem(s)")
        return 1 if problems else 0
    if args.command == "self-consistency":
        from fqsel.comparators import self_consistency

        self_consistency.run(cfg)
        return 0
    from fqsel import q3_validation

    if args.action == "export":
        for path in q3_validation.export_task(args.task, cfg):
            print(f"wrote {path}")
    else:
        print(q3_validation.import_task(args.task, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
