"""Command-line entry point; pipeline stages register as subcommands."""

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from pm_traitbench import pipeline
from pm_traitbench.config import load_config
from pm_traitbench.errors import PmTraitbenchError
from pm_traitbench.market.real import fetch as real_fetch
from pm_traitbench.stages import Stage, run_stage
from pm_traitbench.tables.store import DataStore


def build_parser(stages: Sequence[Stage]) -> argparse.ArgumentParser:
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate stage name: {name}")
            seen.add(name)
    numbers = [stage.number for stage in stages]
    if len(numbers) != len(set(numbers)):
        seen_numbers: set[int] = set()
        for number in numbers:
            if number in seen_numbers:
                raise ValueError(f"duplicate stage number: {number}")
            seen_numbers.add(number)

    parser = argparse.ArgumentParser(
        prog="pm-traitbench",
        description="Pipeline for generating the PM-TraitBench dataset.",
    )
    parser.add_argument("--version", action="version", version=version("pm-traitbench"))
    subparsers = parser.add_subparsers(dest="command")

    for stage in sorted(stages, key=lambda s: s.number):
        subparser = subparsers.add_parser(stage.name, help=f"stage {stage.number}: {stage.help}")
        subparser.add_argument(
            "--config",
            type=Path,
            default=None,
            metavar="PATH",
            help="YAML file overriding default settings",
        )
        subparser.add_argument(
            "--data-dir",
            type=Path,
            default=Path("data"),
            metavar="PATH",
            help="directory for pipeline tables (default: data)",
        )
        subparser.add_argument(
            "--force", action="store_true", help="overwrite existing output tables"
        )
        subparser.set_defaults(stage=stage)

    fetch_parser = subparsers.add_parser(
        "fetch-market", help="fetch real market raw data into the raw cache"
    )
    fetch_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="YAML file overriding default settings",
    )
    fetch_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        metavar="PATH",
        help="directory for pipeline tables (default: data)",
    )
    fetch_parser.add_argument("--force", action="store_true", help="refetch every cached raw file")

    return parser


def main(argv: list[str] | None = None, stages: Sequence[Stage] | None = None) -> int:
    if stages is None:
        stages = pipeline.STAGES
    parser = build_parser(stages)
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        config = load_config(args.config)
        if args.command == "fetch-market":
            manifest = real_fetch.fetch_all(
                config, args.data_dir, force=args.force, opener=real_fetch.urlopen_bytes
            )
            if manifest.entries:
                print(f"fetched {len(manifest.entries)} files into {args.data_dir}")
            else:
                print("no real market seeds configured")
            return 0
        store = DataStore(args.data_dir, config.output)
        run_stage(args.stage, config, store, force=args.force)
    except PmTraitbenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code

    return 0
