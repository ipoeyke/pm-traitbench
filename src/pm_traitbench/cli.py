"""Command-line entry point; pipeline stages register as subcommands."""

import argparse
from importlib.metadata import version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pm-traitbench",
        description="Pipeline for generating the PM-TraitBench dataset.",
    )
    parser.add_argument("--version", action="version", version=version("pm-traitbench"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
