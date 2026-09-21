"""Column-level introspection of row models, shared by every table format."""

import datetime
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

ColumnKind = Literal["str", "int", "float", "bool", "date", "mixed", "struct"]

_UNION_ORIGINS = (typing.Union, types.UnionType)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: ColumnKind
    nullable: bool
    struct: type[BaseModel] | None


def columns(model: type[BaseModel]) -> list[ColumnInfo]:
    """Describe a row model's columns in field order, using aliases as names."""
    return [_column_info(model, name, field) for name, field in model.model_fields.items()]


def _column_info(model: type[BaseModel], field_name: str, field: FieldInfo) -> ColumnInfo:
    column_name = field.alias if field.alias else field_name
    kind, nullable, struct = _resolve(model, field_name, field.annotation)
    return ColumnInfo(name=column_name, kind=kind, nullable=nullable, struct=struct)


def _resolve(
    model: type[BaseModel], field_name: str, annotation: Any
) -> tuple[ColumnKind, bool, type[BaseModel] | None]:
    annotation = _unwrap_annotated(annotation)
    if get_origin(annotation) in _UNION_ORIGINS:
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not type(None)]
        nullable = len(non_none) < len(args)
        if len(non_none) == 1:
            kind, struct = _kind_of(model, field_name, non_none[0])
        elif set(non_none) == {float, str}:
            kind, struct = "mixed", None
        else:
            raise TypeError(f"unsupported annotation for {model.__name__}.{field_name}")
        return kind, nullable, struct
    kind, struct = _kind_of(model, field_name, annotation)
    return kind, False, struct


def _kind_of(
    model: type[BaseModel], field_name: str, annotation: Any
) -> tuple[ColumnKind, type[BaseModel] | None]:
    annotation = _unwrap_annotated(annotation)
    if isinstance(annotation, type):
        if annotation is bool:
            return "bool", None
        if issubclass(annotation, Enum):
            return "str", None
        if issubclass(annotation, BaseModel):
            return "struct", annotation
        if annotation is str:
            return "str", None
        if annotation is int:
            return "int", None
        if annotation is float:
            return "float", None
        if annotation is datetime.date:
            return "date", None
    raise TypeError(f"unsupported annotation for {model.__name__}.{field_name}")


def _unwrap_annotated(annotation: Any) -> Any:
    if get_origin(annotation) is typing.Annotated:
        return get_args(annotation)[0]
    return annotation
