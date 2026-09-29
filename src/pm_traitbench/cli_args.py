"""Argument definitions shared by every command that reads a config and a data directory."""

import argparse
from pathlib import Path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add `--config` and `--data-dir`."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="YAML file overriding default settings",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        metavar="PATH",
        help="directory for pipeline tables (default: data)",
    )
