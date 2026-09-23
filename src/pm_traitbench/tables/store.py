"""The data store: the sole read/write path pipeline stages use for tables.

Resolves each table's on-disk format, validates and sorts rows, and writes
atomically so a failed write never corrupts or partially replaces a table.
"""

import datetime
import importlib.metadata
import json
import os
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from pm_traitbench.config import Config, OutputConfig
from pm_traitbench.errors import StageIOError, TableValidationError
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import to_record
from pm_traitbench.tables.specs import TableSpec

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _row_key(spec: TableSpec, row: BaseModel) -> tuple:
    return tuple(getattr(row, field) for field in spec.key)


def _sort_key(spec: TableSpec, row: BaseModel) -> tuple:
    # A null key component (for example a market-wide event's instrument_id)
    # sorts first and never collides with a non-null component of another type.
    return tuple((0, "") if v is None else (1, v) for v in _row_key(spec, row))


def _check_duplicate_keys(spec: TableSpec, rows: Sequence[BaseModel]) -> None:
    counts = Counter(_row_key(spec, row) for row in rows)
    duplicate = next((key for key, count in counts.items() if count > 1), None)
    if duplicate is not None:
        raise TableValidationError(f"table '{spec.name}' has duplicate key {duplicate}")


class DataStore:
    """Reads and writes a pipeline's tables, in whatever format is configured."""

    def __init__(self, data_dir: Path, output: OutputConfig) -> None:
        self._data_dir = data_dir
        self._output = output
        self._written: set[str] = set()

    def format_name(self, spec: TableSpec) -> str:
        """Resolve the format name for a table: its own override, else the global format."""
        return self._output.tables.get(spec.name, self._output.format)

    def path(self, spec: TableSpec) -> Path:
        extension = FORMATS[self.format_name(spec)].extension
        return self._data_dir / f"{spec.name}.{extension}"

    def exists(self, spec: TableSpec) -> bool:
        return self.path(spec).exists()

    def write(self, spec: TableSpec, rows: Sequence[BaseModel]) -> Path:
        for row in rows:
            if not isinstance(row, spec.model):
                raise TableValidationError(
                    f"table '{spec.name}' expects rows of type {spec.model.__name__}, "
                    f"got {type(row).__name__}"
                )

        _check_duplicate_keys(spec, rows)
        keyed = sorted(((_sort_key(spec, row), row) for row in rows), key=lambda item: item[0])

        target = self.path(spec)
        tmp_path = target.with_name(target.name + ".tmp")
        fmt = FORMATS[self.format_name(spec)]
        records = [to_record(row) for _, row in keyed]

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fmt.write(records, spec.model, tmp_path)
            os.replace(tmp_path, target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        self._written.add(spec.name)
        return target

    def was_written(self, spec: TableSpec) -> bool:
        """Whether this store instance has written the table, as opposed to finding it on disk."""
        return spec.name in self._written

    def read(self, spec: TableSpec) -> list[BaseModel]:
        fmt_name = self.format_name(spec)
        fmt = FORMATS[fmt_name]
        target = self.path(spec)
        if not target.exists():
            raise StageIOError(self._missing_table_message(spec, target))

        records = fmt.read(target, spec.model)
        rows: list[BaseModel] = []
        for row_number, record in enumerate(records, start=1):
            try:
                rows.append(spec.model.model_validate(record))
            except ValidationError as e:
                fields = ", ".join(sorted({str(err["loc"][0]) for err in e.errors() if err["loc"]}))
                raise TableValidationError(
                    f"table '{spec.name}' row {row_number} failed validation for "
                    f"field(s) {fields}: {e}"
                ) from e

        _check_duplicate_keys(spec, rows)
        return rows

    def _missing_table_message(self, spec: TableSpec, target: Path) -> str:
        alternates = [
            self._data_dir / f"{spec.name}.{fmt.extension}"
            for fmt in FORMATS.values()
            if fmt.extension != target.suffix.lstrip(".")
            and (self._data_dir / f"{spec.name}.{fmt.extension}").exists()
        ]
        message = f"table '{spec.name}' not found at '{target}'"
        if alternates:
            found = ", ".join(str(path) for path in alternates)
            message += f"; found {found} instead"
        return message

    def write_run_metadata(
        self, stage_name: str, config: Config, extra: dict[str, Any] | None = None
    ) -> Path:
        metadata = {
            "stage": stage_name,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "package_version": importlib.metadata.version("pm-traitbench"),
            "git_commit": _git_commit(),
            "root_seed": config.seed.root,
            "config": config.model_dump(mode="json"),
        }
        if extra:
            collisions = sorted(set(extra) & set(metadata))
            if collisions:
                raise ValueError(f"run metadata extra key(s) collide with fixed keys: {collisions}")
            metadata.update(extra)
        metadata_dir = self._data_dir / "run_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / f"{stage_name}.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path
