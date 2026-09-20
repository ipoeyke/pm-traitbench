import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pm_traitbench.enums import (
    Action,
    AssetClass,
    DriftEventType,
    Kind,
    Op,
    RuleScope,
    RuleSource,
    Split,
    Typicality,
)
from pm_traitbench.errors import TableValidationError
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import (
    DriftEvent,
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


NON_STRUCT_ROWS = [
    _bias_trait(),
    _preference_trait_with_special_chars(),
    _rule_text_level(),
    _rule_numeric_level(),
    _update_event(),
    _dormant_event(),
]


def test_formats_keys_are_exactly_csv_jsonl_parquet() -> None:
    assert set(FORMATS) == {"csv", "jsonl", "parquet"}


@pytest.mark.parametrize("format_name", ["csv", "jsonl", "parquet"])
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
def test_round_trip_persona(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    row = _persona()
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], Persona, path)
    records = fmt.read(path, Persona)
    assert [Persona.model_validate(record) for record in records] == [row]


def test_csv_rejects_struct_column(tmp_path: Path) -> None:
    fmt = FORMATS["csv"]
    path = tmp_path / "table.csv"
    with pytest.raises(TableValidationError, match="Mandate"):
        fmt.write([to_record(_persona())], Persona, path)


def test_csv_header_mismatch_raises(tmp_path: Path) -> None:
    fmt = FORMATS["csv"]
    path = tmp_path / "table.csv"
    path.write_text("wrong,header\n", encoding="utf-8", newline="")
    with pytest.raises(TableValidationError):
        fmt.read(path, Trait)


def test_csv_bytes_contain_no_carriage_return(tmp_path: Path) -> None:
    fmt = FORMATS["csv"]
    path = tmp_path / "table.csv"
    rows = [to_record(_bias_trait()), to_record(_preference_trait_with_special_chars())]
    fmt.write(rows, Trait, path)
    assert b"\r" not in path.read_bytes()


@pytest.mark.parametrize("format_name", ["csv", "jsonl"])
def test_two_writes_are_byte_identical(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
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
    assert trait_schema.field("value").type == pa.string()

    event_path = tmp_path / "event.parquet"
    fmt.write([to_record(_update_event())], DriftEvent, event_path)
    event_schema = pq.read_schema(event_path)
    assert event_schema.field("date").type == pa.date32()

    persona_path = tmp_path / "persona.parquet"
    fmt.write([to_record(_persona())], Persona, persona_path)
    persona_schema = pq.read_schema(persona_path)
    assert pa.types.is_struct(persona_schema.field("mandate").type)


@pytest.mark.parametrize("format_name", ["csv", "jsonl", "parquet"])
def test_zero_record_round_trip(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    path = tmp_path / f"empty.{fmt.extension}"
    fmt.write([], Trait, path)
    assert fmt.read(path, Trait) == []


def test_csv_writes_header_even_for_zero_records(tmp_path: Path) -> None:
    fmt = FORMATS["csv"]
    path = tmp_path / "empty.csv"
    fmt.write([], Trait, path)
    content = path.read_text(encoding="utf-8")
    expected_header = "pm_id,trait_id,kind,param,value,active,mult_range,mult_risk_off,mult_risk_on"
    assert content == expected_header + "\n"


def test_jsonl_zero_records_gives_empty_file(tmp_path: Path) -> None:
    fmt = FORMATS["jsonl"]
    path = tmp_path / "empty.jsonl"
    fmt.write([], Trait, path)
    assert path.read_bytes() == b""
