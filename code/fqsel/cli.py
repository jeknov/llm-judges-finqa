"""Command line: python -m fqsel {run <stage> | check | annotate export|import <task> | self-consistency |
judges [--judge NAME]... [--smoke N_PAIRS] | judges-cbi [--judge NAME]... [--smoke N_ITEMS]} [hydra overrides]."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from fqsel.config import load_config
from fqsel.gates import HaltError
from fqsel.stages import JUDGES_SCOPE, STAGE_NAMES, get_stage, run_stage


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
    p_j = sub.add_parser("judges", help="judge comparison: the Jev battery asked of the OpenRouter judges (conf/judges)")
    p_j.add_argument("--judge", action="append", help="judge key in conf/judges (repeatable); default: judges.comparison")
    p_j.add_argument("--smoke", type=int, metavar="N_PAIRS",
                     help="score N_PAIRS originals and their negatives, print the answers, write no table")
    p_j.add_argument("overrides", nargs="*")
    p_jc = sub.add_parser("judges-cbi", help="CBI replication: judges score the gold and the strong model's answer "
                                             "on notebook 03's unanimous items (conf/judges, cbi)")
    p_jc.add_argument("--judge", action="append", help="judge key in conf/judges (repeatable); default: judges.cbi.judges")
    p_jc.add_argument("--smoke", type=int, metavar="N_ITEMS",
                      help="score N_ITEMS unanimous items (half CBI), print the answers, write no table")
    p_jc.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    # a command skips only the placeholders of the stage whose data it reads; annotation tasks are all Q3's,
    # and self-consistency reads only Spec 1 outputs and the strong-LLM settings, like the jev stage
    stage = {"run": getattr(args, "stage", None), "annotate": "q3", "self-consistency": "jev",
             "judges": "jev", "judges-cbi": "jev"}.get(args.command)
    ignore = get_stage(stage).ignores if stage else ()
    if args.command == "judges-cbi":
        # reads the frozen snapshots, the judge settings and the CBI definition (annotation config); FINAL prompt only
        ignore = tuple(i for i in ignore if i not in JUDGES_SCOPE)
    if args.command == "judges":
        # the judges read Spec 1 outputs and their own settings; a smoke test may use the DRAFT wrapper prompt
        ignore = tuple(i for i in ignore if i not in JUDGES_SCOPE) + (("judge_yesno.txt",) if args.smoke else ())
    cfg = load_config(args.overrides, ignore=ignore)
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
    if args.command == "judges":
        from fqsel.comparators import judges

        names = args.judge or list(cfg.judges.comparison)
        if args.smoke:
            with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
                for name in names:
                    print(f"== {name}\n{judges.smoke(cfg, name, args.smoke)}")
            return 0
        judges.run(cfg, names)
        return 0
    if args.command == "judges-cbi":
        from fqsel.comparators import judges_cbi

        names = args.judge or list(cfg.judges.cbi.judges)
        if args.smoke:
            with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_rows", None,
                                   "display.max_colwidth", 30):
                for name in names:
                    print(f"== {name}\n{judges_cbi.smoke(cfg, name, args.smoke)}")
            return 0
        judges_cbi.run(cfg, names)
        return 0
    if args.task == "cbi_audit":
        from fqsel import cbi_audit

        if args.action == "export":
            for path in cbi_audit.export_task(cfg):
                print(f"wrote {path}")
        else:
            print(cbi_audit.import_task(cfg))
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
