import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pm_traitbench.enums import (
    Action,
    AssetClass,
    DriftEventType,
    Expression,
    Kind,
    Op,
    RuleScope,
    RuleSource,
    Side,
    Split,
    StreetView,
    Typicality,
)
from pm_traitbench.errors import TableValidationError
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import (
    DriftEvent,
    Gate1CellRow,
    Gate1PmRow,
    Idea,
    Leg,
    Mandate,
    Persona,
    Rule,
    StatedProfile,
    Trait,
    to_record,
)


def _bias_trait() -> Trait:
    return Trait(
        pm_id="pm_001",
        trait_id="t_01",
        kind=Kind.BIAS,
        param="loss_aversion_lambda",
        value=2.6,
        active=True,
        mult_range=1.1,
        mult_risk_off=1.3,
        mult_risk_on=0.9,
    )


def _preference_trait_with_special_chars() -> Trait:
    return Trait(
        pm_id="pm_001",
        trait_id="t_02",
        kind=Kind.PREFERENCE,
        param="sector_focus",
        value='technology, "growth"',
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )


def _rule_text_level() -> Rule:
    return Rule(
        pm_id="pm_001",
        rule_id="r_01",
        source=RuleSource.MANDATE,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="max_drawdown",
        field="drawdown_pct",
        op=Op.LE,
        level="energy",
        unit=None,
        window=1,
        action=Action.CAP,
        text="Cap drawdown.",
    )


def _rule_numeric_level() -> Rule:
    return Rule(
        pm_id="pm_001",
        rule_id="r_02",
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id="idea_001",
        param="stop_loss",
        field="price",
        op=Op.LT,
        level=-15.0,
        unit="usd",
        window=5,
        action=Action.EXIT,
        text="Exit if price falls below level.",
    )


def _update_event() -> DriftEvent:
    return DriftEvent(
        pm_id="pm_001",
        date=datetime.date(2026, 3, 2),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        from_value=1.1,
        to_value=1.4,
    )


def _dormant_event() -> DriftEvent:
    return DriftEvent(
        pm_id="pm_001",
        date=datetime.date(2026, 3, 5),
        event=DriftEventType.DORMANT,
        trait_id="t_01",
        from_value=None,
        to_value=None,
    )


def _persona() -> Persona:
    return Persona(
        pm_id="pm_001",
        market_seed="A",
        split=Split.PILOT,
        mandate=Mandate(
            asset_class=AssetClass.EQUITIES,
            sub_style="value",
            book_size=250_000_000.0,
            risk_unit="pct_nav",
            benchmark="MSCI World",
        ),
        stated_profile=StatedProfile(self_description="Disciplined value investor."),
        typicality=Typicality.TYPICAL,
    )


def _idea_with_two_legs() -> Idea:
    return Idea(
        pm_id="pm_001",
        trade_idea_id="ti_001",
        instrument_id="EQ-AAPL",
        expression=Expression.PAIR,
        side=Side.BUY,
        legs=(
            Leg(instrument_id="EQ-AAPL", tenor=None, side=Side.BUY, weight=1.0),
            Leg(instrument_id="EQ-MSFT", tenor=None, side=Side.SELL, weight=1.0),
        ),
        entry_date=datetime.date(2026, 1, 5),
        exit_date=None,
        entry_level=100.0,
        target_level=110.0,
        stop_level=95.0,
        thesis="Long AAPL versus short MSFT on relative earnings momentum.",
        outcome=None,
        own_signal=0.6,
        forecast=108.0,
        interval_lo=100.0,
        interval_hi=115.0,
        street_view_at_entry=StreetView.NEUTRAL,
        conflict=False,
        followed_street=None,
        conviction=3,
        size_rank=2,
    )


NON_STRUCT_ROWS = [
    _bias_trait(),
    _preference_trait_with_special_chars(),
    _rule_text_level(),
    _rule_numeric_level(),
    _update_event(),
    _dormant_event(),
]


def test_formats_keys_are_exactly_jsonl_parquet() -> None:
    assert set(FORMATS) == {"jsonl", "parquet"}


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
@pytest.mark.parametrize("row", NON_STRUCT_ROWS, ids=lambda r: f"{type(r).__name__}-{r!r}")
def test_round_trip_non_struct_rows(
    tmp_path: Path, format_name: str, row: Trait | Rule | DriftEvent
) -> None:
    fmt = FORMATS[format_name]
    model = type(row)
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], model, path)
    records = fmt.read(path, model)
    assert [model.model_validate(record) for record in records] == [row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_text_that_looks_like_a_number_round_trips_as_text(
    tmp_path: Path, format_name: str
) -> None:
    fmt = FORMATS[format_name]
    row = _preference_trait_with_special_chars().model_copy(update={"value": "12.5"})
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], Trait, path)
    assert [Trait.model_validate(record) for record in fmt.read(path, Trait)] == [row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_persona(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    row = _persona()
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], Persona, path)
    records = fmt.read(path, Persona)
    assert [Persona.model_validate(record) for record in records] == [row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_idea_with_list_struct_legs(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    row = _idea_with_two_legs()
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], Idea, path)
    records = fmt.read(path, Idea)
    assert [Idea.model_validate(record) for record in records] == [row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_gate1_pm_row(
    tmp_path: Path, format_name: str, gate1_pm_row: Gate1PmRow
) -> None:
    fmt = FORMATS[format_name]
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(gate1_pm_row)], Gate1PmRow, path)
    records = fmt.read(path, Gate1PmRow)
    assert [Gate1PmRow.model_validate(record) for record in records] == [gate1_pm_row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_gate1_cell_row(
    tmp_path: Path, format_name: str, gate1_cell_row: Gate1CellRow
) -> None:
    fmt = FORMATS[format_name]
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(gate1_cell_row)], Gate1CellRow, path)
    records = fmt.read(path, Gate1CellRow)
    assert [Gate1CellRow.model_validate(record) for record in records] == [gate1_cell_row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_gate1_cell_row_population(
    tmp_path: Path, format_name: str, gate1_cell_row_population: Gate1CellRow
) -> None:
    fmt = FORMATS[format_name]
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(gate1_cell_row_population)], Gate1CellRow, path)
    records = fmt.read(path, Gate1CellRow)
    assert [Gate1CellRow.model_validate(record) for record in records] == [
        gate1_cell_row_population
    ]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_gate1_cell_row_pooled_across_asset_classes(
    tmp_path: Path, format_name: str, gate1_cell_row_pooled: Gate1CellRow
) -> None:
    fmt = FORMATS[format_name]
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(gate1_cell_row_pooled)], Gate1CellRow, path)
    records = fmt.read(path, Gate1CellRow)
    assert [Gate1CellRow.model_validate(record) for record in records] == [gate1_cell_row_pooled]
    assert records[0]["asset_class"] is None


def test_jsonl_write_missing_record_column_raises_table_validation_error(tmp_path: Path) -> None:
    fmt = FORMATS["jsonl"]
    path = tmp_path / "table.jsonl"
    record = to_record(_bias_trait())
    del record["value"]
    with pytest.raises(TableValidationError, match=r"row 1\b"):
        fmt.write([record], Trait, path)


def test_jsonl_read_blank_line_raises_table_validation_error_naming_the_row(
    tmp_path: Path,
) -> None:
    fmt = FORMATS["jsonl"]
    path = tmp_path / "table.jsonl"
    path.write_text('{"a":1}\n\n', encoding="utf-8")
    with pytest.raises(TableValidationError, match=r"row 2\b"):
        fmt.read(path, Trait)


def test_jsonl_read_malformed_line_raises_table_validation_error_naming_the_row(
    tmp_path: Path,
) -> None:
    fmt = FORMATS["jsonl"]
    path = tmp_path / "table.jsonl"
    path.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(TableValidationError, match=r"row 2\b"):
        fmt.read(path, Trait)


def test_jsonl_bytes_contain_no_carriage_return(tmp_path: Path) -> None:
    fmt = FORMATS["jsonl"]
    path = tmp_path / "table.jsonl"
    rows = [to_record(_bias_trait()), to_record(_preference_trait_with_special_chars())]
    fmt.write(rows, Trait, path)
    assert b"\r" not in path.read_bytes()


def test_two_jsonl_writes_are_byte_identical(tmp_path: Path) -> None:
    fmt = FORMATS["jsonl"]
    rows = [to_record(_bias_trait()), to_record(_preference_trait_with_special_chars())]
    path_a = tmp_path / f"a.{fmt.extension}"
    path_b = tmp_path / f"b.{fmt.extension}"
    fmt.write(rows, Trait, path_a)
    fmt.write(rows, Trait, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_parquet_schema_kinds(tmp_path: Path) -> None:
    fmt = FORMATS["parquet"]

    trait_path = tmp_path / "trait.parquet"
    fmt.write([to_record(_bias_trait())], Trait, trait_path)
    trait_schema = pq.read_schema(trait_path)
    assert "value" not in trait_schema.names
    assert trait_schema.field("value_num").type == pa.float64()
    assert trait_schema.field("value_text").type == pa.string()

    event_path = tmp_path / "event.parquet"
    fmt.write([to_record(_update_event())], DriftEvent, event_path)
    event_schema = pq.read_schema(event_path)
    assert event_schema.field("date").type == pa.date32()

    persona_path = tmp_path / "persona.parquet"
    fmt.write([to_record(_persona())], Persona, persona_path)
    persona_schema = pq.read_schema(persona_path)
    assert pa.types.is_struct(persona_schema.field("mandate").type)

    idea_path = tmp_path / "idea.parquet"
    fmt.write([to_record(_idea_with_two_legs())], Idea, idea_path)
    idea_schema = pq.read_schema(idea_path)
    legs_type = idea_schema.field("legs").type
    assert pa.types.is_list(legs_type)
    assert pa.types.is_struct(legs_type.value_type)


def test_parquet_fills_exactly_one_side_of_a_number_or_text_column(tmp_path: Path) -> None:
    fmt = FORMATS["parquet"]
    path = tmp_path / "trait.parquet"
    rows = [to_record(_bias_trait()), to_record(_preference_trait_with_special_chars())]
    fmt.write(rows, Trait, path)
    stored = pq.read_table(path).to_pylist()
    assert [(row["value_num"] is None, row["value_text"] is None) for row in stored] == [
        (False, True),
        (True, False),
    ]


def test_parquet_drift_endpoints_use_aliased_column_pairs(tmp_path: Path) -> None:
    fmt = FORMATS["parquet"]
    path = tmp_path / "event.parquet"
    fmt.write([to_record(_dormant_event())], DriftEvent, path)
    assert pq.read_schema(path).names[-4:] == ["from_num", "from_text", "to_num", "to_text"]
    assert fmt.read(path, DriftEvent)[0]["from"] is None


def test_parquet_read_rejects_a_row_with_both_sides_filled(tmp_path: Path) -> None:
    fmt = FORMATS["parquet"]
    path = tmp_path / "trait.parquet"
    fmt.write([to_record(_bias_trait())], Trait, path)
    table = pq.read_table(path)
    index = table.schema.get_field_index("value_text")
    table = table.set_column(index, table.schema.field(index), pa.array(["x"], pa.string()))
    pq.write_table(table, path)
    with pytest.raises(TableValidationError, match=r"row 1: column 'value'"):
        fmt.read(path, Trait)


def test_parquet_read_of_a_file_missing_a_column_raises(tmp_path: Path) -> None:
    fmt = FORMATS["parquet"]
    path = tmp_path / "trait.parquet"
    fmt.write([to_record(_bias_trait())], Trait, path)
    pq.write_table(pq.read_table(path).drop_columns(["value_num"]), path)
    with pytest.raises(TableValidationError, match="value_num"):
        fmt.read(path, Trait)


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_zero_record_round_trip(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    path = tmp_path / f"empty.{fmt.extension}"
    fmt.write([], Trait, path)
    assert fmt.read(path, Trait) == []


def test_jsonl_zero_records_gives_empty_file(tmp_path: Path) -> None:
    fmt = FORMATS["jsonl"]
    path = tmp_path / "empty.jsonl"
    fmt.write([], Trait, path)
    assert path.read_bytes() == b""
