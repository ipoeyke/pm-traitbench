"""Enforces the probes package's boundaries: it makes no model call and only the pipeline
imports it.
"""

import ast
from pathlib import Path

from tests.dialogue.test_imports import _SRC_ROOT, _has_prefix, _imported_names, _package_for

_PROBES_PREFIX = "pm_traitbench.probes"
_PROBES_ROOT = _SRC_ROOT / "pm_traitbench" / "probes"
_PIPELINE_PATH = _SRC_ROOT / "pm_traitbench" / "pipeline.py"
_FORBIDDEN_IN_PROBES = ("anthropic", "pm_traitbench.dialogue")


def _names_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _imported_names(tree, _package_for(path))


def test_probes_package_never_imports_a_model_client_or_dialogue() -> None:
    violations = []
    for path in sorted(_PROBES_ROOT.rglob("*.py")):
        for name in sorted(_names_in(path)):
            if any(_has_prefix(name, prefix) for prefix in _FORBIDDEN_IN_PROBES):
                violations.append(f"{path.relative_to(_PROBES_ROOT)}: imports '{name}'")

    assert not violations, "\n".join(violations)


def test_probes_package_is_imported_only_by_pipeline() -> None:
    violations = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if _PROBES_ROOT in path.parents or path == _PIPELINE_PATH:
            continue
        for name in sorted(_names_in(path)):
            if _has_prefix(name, _PROBES_PREFIX):
                violations.append(f"{path.relative_to(_SRC_ROOT)}: imports '{name}'")

    assert not violations, "\n".join(violations)
