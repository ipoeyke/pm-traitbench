import datetime

import pytest
from pydantic import BaseModel

from pm_traitbench.tables.introspect import ColumnInfo, columns
from pm_traitbench.tables.schema import DriftEvent, Idea, Leg, Mandate, Persona, Trait


def _by_name(infos: list[ColumnInfo]) -> dict[str, ColumnInfo]:
    return {info.name: info for info in infos}


def test_drift_event_columns() -> None:
    infos = columns(DriftEvent)
    assert [info.name for info in infos] == ["pm_id", "date", "event", "trait_id", "from", "to"]
    assert [info.kind for info in infos] == ["str", "date", "str", "str", "mixed", "mixed"]
    by_name = _by_name(infos)
    assert by_name["from"].nullable is True
    assert by_name["to"].nullable is True
    assert by_name["pm_id"].nullable is False
    assert by_name["date"].nullable is False


def test_persona_mandate_is_struct() -> None:
    by_name = _by_name(columns(Persona))
    assert by_name["mandate"].kind == "struct"
    assert by_name["mandate"].struct is Mandate
    assert by_name["mandate"].nullable is False


def test_trait_value_is_mixed_non_nullable() -> None:
    by_name = _by_name(columns(Trait))
    assert by_name["value"].kind == "mixed"
    assert by_name["value"].nullable is False


def test_trait_multipliers_are_nullable_float() -> None:
    by_name = _by_name(columns(Trait))
    for field_name in ("mult_range", "mult_risk_off", "mult_risk_on"):
        assert by_name[field_name].kind == "float"
        assert by_name[field_name].nullable is True


def test_unsupported_annotation_raises_type_error() -> None:
    class _Unsupported(BaseModel):
        bad: list[int]

    with pytest.raises(TypeError):
        columns(_Unsupported)


def test_idea_legs_is_list_struct_of_leg() -> None:
    by_name = _by_name(columns(Idea))
    assert by_name["legs"].kind == "list_struct"
    assert by_name["legs"].struct is Leg
    assert by_name["legs"].nullable is False


def test_tuple_of_non_model_raises_type_error() -> None:
    class _Unsupported(BaseModel):
        bad: tuple[int, ...]

    with pytest.raises(TypeError):
        columns(_Unsupported)


def test_supported_scalar_and_date_kinds() -> None:
    class _Scalars(BaseModel):
        a_str: str
        a_int: int
        a_float: float
        a_bool: bool
        a_date: datetime.date

    by_name = _by_name(columns(_Scalars))
    assert by_name["a_str"].kind == "str"
    assert by_name["a_int"].kind == "int"
    assert by_name["a_float"].kind == "float"
    assert by_name["a_bool"].kind == "bool"
    assert by_name["a_date"].kind == "date"
    assert all(info.nullable is False for info in by_name.values())
