"""The `eval` command group: replay the corpus into a system under test and score it."""

import argparse
import csv
import sys
from collections import Counter
from typing import Any

from pm_traitbench.cli_args import add_common_args
from pm_traitbench.config import load_config
from pm_traitbench.harness.baselines import BASELINES, baseline_factory
from pm_traitbench.harness.judge import judge_run
from pm_traitbench.harness.runner import (
    check_run_name,
    default_run_name,
    load_factory,
    run_dir,
    run_sut,
)
from pm_traitbench.harness.sample import write_sample
from pm_traitbench.harness.score import score_run
from pm_traitbench.tables.store import DataStore

_COLUMNS = ("probe_type", "form", "scorer", "n", "accuracy", "chance", "parse_errors")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def add_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `eval` with its `run`, `judge`, `sample` and `score` subcommands."""
    eval_parser = subparsers.add_parser(
        "eval", help="replay the corpus into a system under test and score its answers"
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    run = eval_sub.add_parser("run", help="replay every PM into a system under test")
    add_common_args(run)
    run.add_argument(
        "--sut",
        required=True,
        metavar="NAME",
        help=f"a baseline ({', '.join(BASELINES)}) or a factory as package.module:factory",
    )
    run.add_argument(
        "--run-name", default=None, metavar="NAME", help="output name (default: from --sut)"
    )
    run.add_argument("--workers", type=_positive_int, default=1, metavar="N", help="parallel PMs")
    run.add_argument("--force", action="store_true", help="discard an existing run first")

    judge = eval_sub.add_parser("judge", help="grade a finished run's open replies with the judges")
    add_common_args(judge)
    judge.add_argument("--run-name", required=True, metavar="NAME", help="the run to judge")
    judge.add_argument(
        "--force", action="store_true", help="discard existing judgements and judge again"
    )

    sample = eval_sub.add_parser("sample", help="export a blind sample for human rating")
    add_common_args(sample)
    sample.add_argument("--run-name", required=True, metavar="NAME", help="the judged run")
    sample.add_argument("--force", action="store_true", help="replace an existing sample file")
    sample.add_argument(
        "--size", type=_positive_int, default=None, metavar="N", help="rows (default: config)"
    )

    score = eval_sub.add_parser("score", help="score a finished run")
    add_common_args(score)
    score.add_argument("--run-name", required=True, metavar="NAME", help="the run to score")


def _table(entries: list[dict[str, Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    rows = [list(_COLUMNS)] + [[cell(e[c]) for c in _COLUMNS] for e in entries]
    widths = [max(len(row[i]) for row in rows) for i in range(len(_COLUMNS))]
    return "\n".join(
        "  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)).rstrip() for row in rows
    )


def run_eval(args: argparse.Namespace) -> int:
    """Run the chosen `eval` subcommand; return 1 when a PM failed, else 0."""
    config = load_config(args.config)
    store = DataStore(args.data_dir, config.output)
    if args.eval_command == "score":
        summary = score_run(config, store, check_run_name(args.run_name))
        print(_table(summary["by_type"]))
        awaiting = ", ".join(f"{k} {v}" for k, v in summary["awaiting_judge"].items())
        print(f"awaiting judge: {awaiting}")
        for judge, stats in sorted(summary["agreement"].items()):
            kappa = "-" if stats["kappa"] is None else f"{stats['kappa']:.3f}"
            print(f"agreement {judge}: n={stats['n']} rate={stats['agreement']:.3f} kappa={kappa}")
        return 0
    if args.eval_command == "judge":
        judged = judge_run(config, store, check_run_name(args.run_name), force=args.force)
        counts = Counter(str(row.judge) for row in judged.judgements)
        by_judge = ", ".join(f"{name}: {n}" for name, n in sorted(counts.items()))
        print(
            f"judged {len(judged.judgements)} items ({by_judge}), "
            f"skipped {sum(judged.skipped.values())}"
        )
        return 0
    if args.eval_command == "sample":
        path = write_sample(
            config, store, check_run_name(args.run_name), args.size, force=args.force
        )
        with path.open(newline="", encoding="utf-8") as f:
            count = sum(1 for _ in csv.DictReader(f))
        print(f"wrote {count} rows to {path}")
        return 0

    run_name = check_run_name(args.run_name or default_run_name(args.sut))
    if args.sut in BASELINES:
        factory = baseline_factory(args.sut, config, run_dir(args.data_dir, run_name), run_name)
    else:
        factory = load_factory(args.sut)
    result = run_sut(
        config,
        store,
        factory,
        sut_name=args.sut,
        run_name=run_name,
        workers=args.workers,
        force=args.force,
    )
    print(
        f"completed {len(result.completed)}, skipped {len(result.skipped)}, "
        f"failed {len(result.failed)}"
    )
    for pm_id, trace in result.failed.items():
        last = trace.strip().splitlines()[-1:] or ["no traceback"]
        print(f"{pm_id}: {last[0]}", file=sys.stderr)
    return 1 if result.failed else 0
