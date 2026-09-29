"""Enforces that nothing outside the harness package imports it, except the CLI entry point."""

import ast

from tests.dialogue.test_imports import _SRC_ROOT, _has_prefix, _imported_names, _package_for

_HARNESS = "pm_traitbench.harness"
_HARNESS_ROOT = _SRC_ROOT / "pm_traitbench" / "harness"
_CLI_PATH = _SRC_ROOT / "pm_traitbench" / "cli.py"


def test_harness_is_imported_only_by_the_cli_entry_point() -> None:
    violations = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if _HARNESS_ROOT in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in sorted(_imported_names(tree, _package_for(path))):
            if not _has_prefix(name, _HARNESS):
                continue
            if path == _CLI_PATH and _has_prefix(name, f"{_HARNESS}.cli"):
                continue
            violations.append(f"{path.relative_to(_SRC_ROOT)}: imports '{name}'")

    assert not violations, "\n".join(violations)
