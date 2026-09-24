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


_NUMBER_SUFFIX = "_num"
_TEXT_SUFFIX = "_text"


def _pyarrow_type(col: ColumnInfo) -> pa.DataType:
    if col.kind == "str":
        return pa.string()
    if col.kind == "int":
        return pa.int64()
    if col.kind == "float":
        return pa.float64()
    if col.kind == "bool":
        return pa.bool_()
    if col.kind == "date":
        return pa.date32()
    if col.kind == "list_struct":
        return pa.list_(pa.struct(_pyarrow_fields(columns(col.struct))))
    return pa.struct(_pyarrow_fields(columns(col.struct)))


def _pyarrow_fields(cols: list[ColumnInfo]) -> list[pa.Field]:
    fields = []
    for col in cols:
        if col.kind == "mixed":
            # A parquet column has one type, so a number-or-text column is stored
            # as a typed pair with exactly one side filled per non-null value.
            fields.append(pa.field(col.name + _NUMBER_SUFFIX, pa.float64(), nullable=True))
            fields.append(pa.field(col.name + _TEXT_SUFFIX, pa.string(), nullable=True))
        else:
            fields.append(pa.field(col.name, _pyarrow_type(col), nullable=col.nullable))
    return fields


def _to_arrow_row(cols: list[ColumnInfo], record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for col in cols:
        value = record[col.name]
        if col.kind == "mixed":
            is_text = isinstance(value, str)
            row[col.name + _NUMBER_SUFFIX] = None if is_text else value
            row[col.name + _TEXT_SUFFIX] = value if is_text else None
        elif value is None:
            row[col.name] = None
        elif col.kind == "date":
            row[col.name] = datetime.date.fromisoformat(value)
        elif col.kind == "struct":
            row[col.name] = _to_arrow_row(columns(col.struct), value)
        elif col.kind == "list_struct":
            row[col.name] = [_to_arrow_row(columns(col.struct), item) for item in value]
        else:
            row[col.name] = value
    return row


def _from_arrow_row(cols: list[ColumnInfo], row: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for col in cols:
        if col.kind == "mixed":
            number = row[col.name + _NUMBER_SUFFIX]
            text = row[col.name + _TEXT_SUFFIX]
            if number is not None and text is not None:
                raise ValueError(f"column '{col.name}' has both a number and a text value")
            record[col.name] = text if number is None else number
        elif col.kind == "struct" and row[col.name] is not None:
            record[col.name] = _from_arrow_row(columns(col.struct), row[col.name])
        elif col.kind == "list_struct" and row[col.name] is not None:
            struct_cols = columns(col.struct)
            record[col.name] = [_from_arrow_row(struct_cols, item) for item in row[col.name]]
        else:
            record[col.name] = row[col.name]
    return record


class ParquetFormat:
    extension = "parquet"

    def write(self, records: list[dict[str, Any]], model: type[BaseModel], path: Path) -> None:
        cols = columns(model)
        rows = [_to_arrow_row(cols, record) for record in records]
        table = pa.Table.from_pylist(rows, schema=pa.schema(_pyarrow_fields(cols)))
        pq.write_table(table, path, compression="zstd")

    def read(self, path: Path, model: type[BaseModel]) -> list[dict[str, Any]]:
        cols = columns(model)
        records = []
        for row_number, row in enumerate(pq.read_table(path).to_pylist(), start=1):
            try:
                records.append(_from_arrow_row(cols, row))
            except KeyError as e:
                raise TableValidationError(f"parquet file '{path}' is missing column {e}") from e
            except ValueError as e:
                raise TableValidationError(f"parquet file '{path}' row {row_number}: {e}") from e
        return records


FORMATS: dict[str, TableFormat] = {
    "jsonl": JsonlFormat(),
    "parquet": ParquetFormat(),
}
