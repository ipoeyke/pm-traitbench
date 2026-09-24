"""Pipeline stage declarations and the runner that enforces their I/O contract."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pm_traitbench.config import Config
from pm_traitbench.errors import StageIOError
from pm_traitbench.tables.schema import to_record
from pm_traitbench.tables.specs import TableSpec
from pm_traitbench.tables.store import DataStore


@dataclass(frozen=True)
class Stage:
    """A pipeline step: the tables it reads, writes and appends to, and its run function."""

    number: int
    name: str
    help: str
    run: Callable[[Config, DataStore], dict[str, Any] | None]
    reads: tuple[TableSpec, ...] = ()
    writes: tuple[TableSpec, ...] = ()
    # A table an earlier stage wrote, that this stage rewrites with extra rows
    # while leaving the pre-existing rows untouched.
    appends: tuple[TableSpec, ...] = ()


def run_stage(stage: Stage, config: Config, store: DataStore, *, force: bool = False) -> None:
    """Check preconditions, run the stage, verify outputs, then record metadata."""
    missing = [spec.name for spec in (*stage.reads, *stage.appends) if not store.exists(spec)]
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

    pre_existing = {spec: [to_record(row) for row in store.read(spec)] for spec in stage.appends}

    extra = stage.run(config, store)

    # Existence is not enough: under force an older file would pass for a fresh one.
    unwritten = [
        spec.name for spec in (*stage.writes, *stage.appends) if not store.was_written(spec)
    ]
    if unwritten:
        raise StageIOError(
            f"stage '{stage.name}' did not write expected table(s): {', '.join(unwritten)}"
        )

    altered = []
    for spec, records in pre_existing.items():
        new_records = [to_record(row) for row in store.read(spec)]
        if any(record not in new_records for record in records):
            altered.append(spec.name)
    if altered:
        raise StageIOError(
            f"stage '{stage.name}' altered pre-existing rows of table(s): {', '.join(altered)}"
        )

    store.write_run_metadata(stage.name, config, extra)
