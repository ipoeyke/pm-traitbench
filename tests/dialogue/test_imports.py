"""Enforces the dialogue package's dependency boundary: no module under it may import the
behaviour engine, signal plan, gate 1 or market packages -- only their read-only row tables.
"""

import ast
from pathlib import Path

_FORBIDDEN_PREFIXES = (
    "pm_traitbench.engine",
    "pm_traitbench.signals",
    "pm_traitbench.gates",
    "pm_traitbench.market",
)

_DIALOGUE_ROOT = Path(__file__).resolve().parents[2] / "src" / "pm_traitbench" / "dialogue"


def _imported_modules(tree: ast.Module) -> set[str]:
    """Every module name a file imports, via `import x` or `from x import y`."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES
    )


def test_dialogue_package_never_imports_engine_signals_gates_or_market():
    violations = []
    for path in sorted(_DIALOGUE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in sorted(_imported_modules(tree)):
            if _is_forbidden(module):
                violations.append(f"{path.relative_to(_DIALOGUE_ROOT)}: imports '{module}'")

    assert not violations, "\n".join(violations)
