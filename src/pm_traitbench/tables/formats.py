"""JSONL and parquet readers/writers for row model tables.

Formats work only with plain records (dicts as produced by ``to_record``);
they never validate rows and never sort them.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from pm_traitbench.errors import TableValidationError
from pm_traitbench.tables.introspect import ColumnInfo, columns


class TableFormat(Protocol):
    extension: str

    def write(self, records: list[dict[str, Any]], model: type[BaseModel], path: Path) -> None: ...

    def read(self, path: Path, model: type[BaseModel]) -> list[dict[str, Any]]: ...


class JsonlFormat:
    extension = "jsonl"

    def write(self, records: list[dict[str, Any]], model: type[BaseModel], path: Path) -> None:
        cols = columns(model)
        with open(path, "w", encoding="utf-8", newline="") as f:
            for row_number, record in enumerate(records, start=1):
                try:
                    ordered = {col.name: record[col.name] for col in cols}
                except KeyError as e:
                    raise TableValidationError(
                        f"JSONL write to '{path}' row {row_number} is missing column {e}"
                    ) from e
                f.write(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")

    def read(self, path: Path, model: type[BaseModel]) -> list[dict[str, Any]]:
        rows = []
        with open(path, encoding="utf-8", newline="") as f:
            for row_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    raise TableValidationError(f"JSONL file '{path}' row {row_number} is blank")
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    raise TableValidationError(
                        f"JSONL file '{path}' row {row_number} is malformed: {e}"
                    ) from e
        return rows


def _pyarrow_type(col: ColumnInfo) -> pa.DataType:
    if col.kind == "str" or col.kind == "mixed":
        return pa.string()
    if col.kind == "int":
        return pa.int64()
    if col.kind == "float":
        return pa.float64()
    if col.kind == "bool":
        return pa.bool_()
    if col.kind == "date":
        return pa.date32()
    return _pyarrow_struct(col.struct)


def _pyarrow_struct(struct_model: type[BaseModel]) -> pa.DataType:
    return pa.struct(
        [
            pa.field(col.name, _pyarrow_type(col), nullable=col.nullable)
            for col in columns(struct_model)
        ]
    )


def _pyarrow_schema(model: type[BaseModel]) -> pa.Schema:
    return pa.schema(
        [pa.field(col.name, _pyarrow_type(col), nullable=col.nullable) for col in columns(model)]
    )


def _to_arrow_value(col: ColumnInfo, value: Any) -> Any:
    if value is None:
        return None
    if col.kind == "date":
        return datetime.date.fromisoformat(value)
    if col.kind == "mixed" and isinstance(value, float):
        return repr(value)
    if col.kind == "struct":
        return {c.name: _to_arrow_value(c, value[c.name]) for c in columns(col.struct)}
    return value


def _record_for_arrow(cols: list[ColumnInfo], record: dict[str, Any]) -> dict[str, Any]:
    return {col.name: _to_arrow_value(col, record[col.name]) for col in cols}


class ParquetFormat:
    extension = "parquet"

    def write(self, records: list[dict[str, Any]], model: type[BaseModel], path: Path) -> None:
        cols = columns(model)
        schema = _pyarrow_schema(model)
        rows = [_record_for_arrow(cols, record) for record in records]
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path, compression="zstd")

    def read(self, path: Path, model: type[BaseModel]) -> list[dict[str, Any]]:
        return pq.read_table(path).to_pylist()


FORMATS: dict[str, TableFormat] = {
    "jsonl": JsonlFormat(),
    "parquet": ParquetFormat(),
}
