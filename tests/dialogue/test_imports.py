"""Enforces the dialogue package's dependency boundary in both directions: no module under
it may import the behaviour engine, signal plan, gate 1 or market packages, and no module
outside it may import the dialogue package, except pipeline.py and gate 2, which may reach
only its client, prompts and stage plumbing.
"""

import ast
from pathlib import Path

_FORBIDDEN_PREFIXES = (
    "pm_traitbench.engine",
    "pm_traitbench.signals",
    "pm_traitbench.gates",
    "pm_traitbench.market",
)
_DIALOGUE_PREFIX = "pm_traitbench.dialogue"
_GATE2_ALLOWED_PREFIXES = (
    "pm_traitbench.dialogue.client",
    "pm_traitbench.dialogue.prompts",
    "pm_traitbench.dialogue.stage",
)
_GATE2_FORBIDDEN_PREFIXES = (
    "pm_traitbench.dialogue.validate",
    "pm_traitbench.signals",
    "pm_traitbench.engine",
)

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_DIALOGUE_ROOT = _SRC_ROOT / "pm_traitbench" / "dialogue"
_PIPELINE_PATH = _SRC_ROOT / "pm_traitbench" / "pipeline.py"
_GATE2_ROOT = _SRC_ROOT / "pm_traitbench" / "gates" / "gate2"
_BASELINES_PATH = _SRC_ROOT / "pm_traitbench" / "harness" / "baselines.py"


def _package_for(path: Path) -> str:
    """The `__package__` a module at `path` would have, for resolving a relative import."""
    return ".".join(path.parent.relative_to(_SRC_ROOT).parts)


def _resolve_from_module(node: ast.ImportFrom, package: str) -> str | None:
    """The absolute module `from ... import ...` names, resolving `node.level` dots against
    the importing file's own `package`.
    """
    if node.level == 0:
        return node.module
    bits = package.rsplit(".", node.level - 1)
    base = bits[0]
    return f"{base}.{node.module}" if node.module else base


def _imported_names(tree: ast.Module, package: str) -> set[str]:
    """Every absolute dotted name a file's imports could refer to.

    Covers `import x.y`, `from x.y import z` and `from x import y` (which
    imports `x.y` as a side effect), resolving relative imports against
    `package`.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_module(node, package)
            if base:
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _has_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def test_imported_names_catches_attribute_style_and_relative_imports():
    """`from pkg import submodule` and a relative import climbing out of the package are both
    real ways to reach a forbidden module, so the scan below must resolve them.
    """
    tree = ast.parse("from pm_traitbench import engine\nfrom ..market import stage\n")

    names = _imported_names(tree, "pm_traitbench.dialogue")

    assert "pm_traitbench.engine" in names
    assert "pm_traitbench.market" in names
    assert "pm_traitbench.market.stage" in names


def test_dialogue_package_never_imports_engine_signals_gates_or_market():
    violations = []
    for path in sorted(_DIALOGUE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = _package_for(path)
        for name in sorted(_imported_names(tree, package)):
            if any(_has_prefix(name, prefix) for prefix in _FORBIDDEN_PREFIXES):
                violations.append(f"{path.relative_to(_DIALOGUE_ROOT)}: imports '{name}'")

    assert not violations, "\n".join(violations)


def _violations_under(root: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Every import under `root` whose name starts with one of `forbidden`."""
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = _package_for(path)
        for name in sorted(_imported_names(tree, package)):
            if any(_has_prefix(name, prefix) for prefix in forbidden):
                violations.append(f"{path.relative_to(_SRC_ROOT)}: imports '{name}'")
    return violations


def test_dialogue_package_is_imported_only_by_pipeline_gate2_plumbing_and_baselines():
    violations = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if _DIALOGUE_ROOT in path.parents or path in (_PIPELINE_PATH, _BASELINES_PATH):
            continue  # dialogue/ imports itself; pipeline.py and the model-backed baselines use it
        in_gate2 = _GATE2_ROOT in path.parents
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = _package_for(path)
        for name in sorted(_imported_names(tree, package)):
            if not _has_prefix(name, _DIALOGUE_PREFIX):
                continue
            if in_gate2 and any(_has_prefix(name, p) for p in _GATE2_ALLOWED_PREFIXES):
                continue
            violations.append(f"{path.relative_to(_SRC_ROOT)}: imports '{name}'")

    assert not violations, "\n".join(violations)


def test_gate2_never_imports_validate_signals_or_engine():
    violations = _violations_under(_GATE2_ROOT, _GATE2_FORBIDDEN_PREFIXES)

    assert not violations, "\n".join(violations)
