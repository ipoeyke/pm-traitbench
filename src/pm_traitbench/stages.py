"""Pipeline stage declarations and the runner that enforces their I/O contract."""

from collections.abc import Callable
from dataclasses import dataclass

from pm_traitbench.config import Config
from pm_traitbench.errors import StageIOError
from pm_traitbench.tables.specs import TableSpec
from pm_traitbench.tables.store import DataStore


@dataclass(frozen=True)
class Stage:
    """A pipeline step: the tables it reads and writes, and its run function."""

    number: int
    name: str
    help: str
    run: Callable[[Config, DataStore], None]
    reads: tuple[TableSpec, ...] = ()
    writes: tuple[TableSpec, ...] = ()


def run_stage(stage: Stage, config: Config, store: DataStore, *, force: bool = False) -> None:
    """Check preconditions, run the stage, verify outputs, then record metadata."""
    missing = [spec.name for spec in stage.reads if not store.exists(spec)]
    if missing:
        raise StageIOError(
            f"stage '{stage.name}' is missing required table(s): {', '.join(missing)}"
        )

    existing = [spec.name for spec in stage.writes if store.exists(spec)]
    if existing and not force:
        raise StageIOError(
            f"stage '{stage.name}' output table(s) already exist: {', '.join(existing)}; "
            "use --force to overwrite"
        )

    stage.run(config, store)

    unwritten = [spec.name for spec in stage.writes if not store.exists(spec)]
    if unwritten:
        raise StageIOError(
            f"stage '{stage.name}' did not write expected table(s): {', '.join(unwritten)}"
        )

    store.write_run_metadata(stage.name, config)
