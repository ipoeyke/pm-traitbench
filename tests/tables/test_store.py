import datetime
import json
from pathlib import Path

import pytest

from pm_traitbench.config import Config, OutputConfig
from pm_traitbench.enums import (
    Action,
    AssetClass,
    DriftEventType,
    EventType,
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
from pm_traitbench.errors import StageIOError, TableValidationError
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import (
    CalendarEvent,
    DriftEvent,
    Idea,
    Leg,
    Mandate,
    Persona,
    Rule,
    StatedProfile,
    Trait,
    to_record,
)
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    IDEAS,
    MARKET_CALENDAR,
    PERSONAS,
    RULES,
    TRAITS,
    TableSpec,
)
from pm_traitbench.tables.store import DataStore


def _persona(pm_id: str) -> Persona:
    return Persona(
        pm_id=pm_id,
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


def _trait(pm_id: str, trait_id: str) -> Trait:
    return Trait(
        pm_id=pm_id,
        trait_id=trait_id,
        kind=Kind.BIAS,
        param="loss_aversion_lambda",
        value=2.6,
        active=True,
        mult_range=1.1,
        mult_risk_off=1.3,
        mult_risk_on=0.9,
    )


def _rule(pm_id: str, rule_id: str) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=RuleSource.MANDATE,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="max_drawdown",
        field="drawdown_pct",
        op=Op.LE,
        level=-15.0,
        unit="usd",
        window=1,
        action=Action.CAP,
        text="Cap drawdown.",
    )


def _drift_event(pm_id: str, date: datetime.date, trait_id: str) -> DriftEvent:
    return DriftEvent(
        pm_id=pm_id,
        date=date,
        event=DriftEventType.UPDATE,
        trait_id=trait_id,
        from_value=1.1,
        to_value=1.4,
    )


def _key_of(spec: TableSpec, row) -> tuple:
    return tuple(getattr(row, field) for field in spec.key)


def _calendar_events(date: datetime.date) -> list[CalendarEvent]:
    return [
        CalendarEvent(
            seed="A",
            date=date,
            instrument_id="AAPL",
            event=EventType.EARNINGS,
            surprise=0.1,
            affected="equities",
        ),
        CalendarEvent(
            seed="A",
            date=date,
            instrument_id=None,
            event=EventType.MACRO_PRINT,
            surprise=0.2,
            affected="all",
        ),
    ]


def _idea_with_legs(pm_id: str, trade_idea_id: str) -> Idea:
    return Idea(
        pm_id=pm_id,
        trade_idea_id=trade_idea_id,
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
        chased_trend=False,
    )


def test_default_format_is_jsonl_for_every_table(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    for spec in (PERSONAS, TRAITS, RULES, DRIFT_EVENTS):
        assert store.format_name(spec) == "jsonl"
        assert store.path(spec) == tmp_path / f"{spec.name}.jsonl"


def test_global_format_parquet_applies_to_all_tables(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig(format="parquet"))
    for spec in (PERSONAS, TRAITS, RULES, DRIFT_EVENTS):
        assert store.format_name(spec) == "parquet"


def test_per_table_override_wins_over_global_format(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig(format="parquet", tables={"traits": "jsonl"}))
    assert store.format_name(TRAITS) == "jsonl"
    assert store.format_name(RULES) == "parquet"


def test_write_returns_path_and_file_exists(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    path = store.write(TRAITS, [_trait("pm_001", "t_01")])
    assert path == tmp_path / "traits.jsonl"
    assert path.exists()
    assert store.exists(TRAITS) is True


def test_write_creates_missing_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    store = DataStore(data_dir, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01")])
    assert (data_dir / "traits.jsonl").exists()


def test_write_sorts_rows_by_key(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    rows = [_trait("pm_002", "t_01"), _trait("pm_001", "t_02"), _trait("pm_001", "t_01")]
    store.write(TRAITS, rows)
    read_back = store.read(TRAITS)
    keys = [_key_of(TRAITS, row) for row in read_back]
    assert keys == sorted(keys)
    assert keys == [("pm_001", "t_01"), ("pm_001", "t_02"), ("pm_002", "t_01")]


def test_write_duplicate_key_raises_and_writes_nothing(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    rows = [_trait("pm_001", "t_01"), _trait("pm_001", "t_01")]
    with pytest.raises(TableValidationError, match="traits"):
        store.write(TRAITS, rows)
    assert not store.exists(TRAITS)
    assert list(tmp_path.glob("*")) == []


def test_write_wrong_model_instance_raises(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    with pytest.raises(TableValidationError):
        store.write(TRAITS, [_persona("pm_001")])


def test_rejected_write_leaves_no_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    store = DataStore(data_dir, OutputConfig())
    with pytest.raises(TableValidationError):
        store.write(TRAITS, [_persona("pm_001")])
    assert not data_dir.exists()


def test_write_replace_failure_removes_tmp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DataStore(tmp_path, OutputConfig())

    def _boom(src: object, dst: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr("pm_traitbench.tables.store.os.replace", _boom)
    with pytest.raises(OSError, match="boom"):
        store.write(TRAITS, [_trait("pm_001", "t_01")])
    assert not (tmp_path / "traits.jsonl.tmp").exists()
    assert not (tmp_path / "traits.jsonl").exists()


def test_write_failure_midway_removes_tmp_and_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DataStore(tmp_path, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01")])
    target = tmp_path / "traits.jsonl"
    original_bytes = target.read_bytes()

    jsonl_format = FORMATS["jsonl"]

    def _write_then_raise(records, model, path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(jsonl_format, "write", _write_then_raise)

    with pytest.raises(RuntimeError, match="boom"):
        store.write(TRAITS, [_trait("pm_002", "t_01")])

    assert target.read_bytes() == original_bytes
    assert not (tmp_path / "traits.jsonl.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_read_missing_table_raises_stage_io_error(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    with pytest.raises(StageIOError, match="traits"):
        store.read(TRAITS)


def test_read_missing_table_mentions_other_extension_present(tmp_path: Path) -> None:
    store_parquet = DataStore(tmp_path, OutputConfig(tables={"traits": "parquet"}))
    store_parquet.write(TRAITS, [_trait("pm_001", "t_01")])
    store_jsonl = DataStore(tmp_path, OutputConfig(tables={"traits": "jsonl"}))
    with pytest.raises(StageIOError, match="traits.parquet"):
        store_jsonl.read(TRAITS)


def _corrupt_active_field(path: Path, row_number: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[row_number - 1])
    record["active"] = "maybe"
    lines[row_number - 1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_corrupted_field_raises_table_validation_error(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01")])
    _corrupt_active_field(tmp_path / "traits.jsonl", row_number=1)

    with pytest.raises(TableValidationError, match=r"row 1\b") as exc_info:
        store.read(TRAITS)
    assert "active" in str(exc_info.value)


def test_read_corrupted_field_on_the_second_row_names_row_two(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01"), _trait("pm_002", "t_01")])
    _corrupt_active_field(tmp_path / "traits.jsonl", row_number=2)

    with pytest.raises(TableValidationError, match=r"row 2\b") as exc_info:
        store.read(TRAITS)
    assert "active" in str(exc_info.value)


def test_read_duplicate_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "traits.jsonl"
    store_jsonl = DataStore(tmp_path, OutputConfig(tables={"traits": "jsonl"}))
    row = _trait("pm_001", "t_01")
    fmt = FORMATS["jsonl"]
    fmt.write([to_record(row), to_record(row)], Trait, path)
    with pytest.raises(TableValidationError, match="traits"):
        store_jsonl.read(TRAITS)


def test_write_twice_gives_identical_bytes(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    rows = [_trait("pm_001", "t_01"), _trait("pm_002", "t_01")]
    path_a = store.write(TRAITS, rows)
    first_bytes = path_a.read_bytes()
    path_b = store.write(TRAITS, rows)
    second_bytes = path_b.read_bytes()
    assert first_bytes == second_bytes


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
@pytest.mark.parametrize(
    "spec", [PERSONAS, TRAITS, RULES, DRIFT_EVENTS], ids=lambda spec: spec.name
)
def test_round_trip_equality_for_every_format(
    tmp_path: Path, spec: TableSpec, format_name: str
) -> None:
    if spec is PERSONAS:
        rows = [_persona("pm_002"), _persona("pm_001")]
    elif spec is TRAITS:
        rows = [_trait("pm_002", "t_01"), _trait("pm_001", "t_01")]
    elif spec is RULES:
        rows = [_rule("pm_002", "r_01"), _rule("pm_001", "r_01")]
    else:
        rows = [
            _drift_event("pm_002", datetime.date(2026, 3, 2), "t_01"),
            _drift_event("pm_001", datetime.date(2026, 3, 2), "t_01"),
        ]
    expected = sorted(rows, key=lambda row: _key_of(spec, row))
    store = DataStore(tmp_path, OutputConfig(tables={spec.name: format_name}))
    store.write(spec, rows)
    assert store.read(spec) == expected


def test_write_run_metadata_has_expected_keys(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    path = store.write_run_metadata("sampling", config)
    assert path == tmp_path / "run_metadata" / "sampling.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "stage",
        "created_at",
        "package_version",
        "git_commit",
        "root_seed",
        "config",
    }
    assert data["stage"] == "sampling"
    assert data["root_seed"] == config.seed.root
    assert data["config"] == config.model_dump(mode="json")
    git_commit = data["git_commit"]
    assert git_commit is None or (isinstance(git_commit, str) and len(git_commit) == 40)
    datetime.datetime.fromisoformat(data["created_at"])


def test_store_records_which_tables_it_wrote(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    assert not store.was_written(TRAITS)
    store.write(TRAITS, [])
    assert store.was_written(TRAITS)
    assert not store.was_written(RULES)
    assert not DataStore(tmp_path, OutputConfig()).was_written(TRAITS)


def test_write_creates_subdirectory_for_market_table(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    rows = _calendar_events(datetime.date(2026, 3, 2))
    path = store.write(MARKET_CALENDAR, rows)
    assert path == tmp_path / "market" / "calendar.jsonl"
    assert path.exists()


def test_write_null_instrument_id_sorts_first_within_same_date(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    rows = _calendar_events(datetime.date(2026, 3, 2))
    store.write(MARKET_CALENDAR, rows)
    read_back = store.read(MARKET_CALENDAR)
    assert [row.instrument_id for row in read_back] == [None, "AAPL"]


def test_write_subdirectory_table_with_parquet_override(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig(tables={"market/calendar": "parquet"}))
    rows = _calendar_events(datetime.date(2026, 3, 2))
    path = store.write(MARKET_CALENDAR, rows)
    assert path == tmp_path / "market" / "calendar.parquet"
    read_back = store.read(MARKET_CALENDAR)
    assert [row.instrument_id for row in read_back] == [None, "AAPL"]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_idea_with_legs(tmp_path: Path, format_name: str) -> None:
    store = DataStore(tmp_path, OutputConfig(tables={"ideas": format_name}))
    rows = [_idea_with_legs("pm_001", "ti_001")]
    store.write(IDEAS, rows)
    assert store.read(IDEAS) == rows


def test_write_run_metadata_merges_extra_keys(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    path = store.write_run_metadata("sampling", config, extra={"check": {"A": 1}})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["check"] == {"A": 1}


def test_write_run_metadata_extra_colliding_key_raises(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    with pytest.raises(ValueError, match="stage"):
        store.write_run_metadata("sampling", config, extra={"stage": "oops"})


def test_read_run_metadata_round_trips_and_returns_none_when_absent(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    config = Config()
    store.write_run_metadata("sampling", config, extra={"check": {"A": 1}})

    data = store.read_run_metadata("sampling")

    assert data is not None
    assert data["check"] == {"A": 1}
    assert data["stage"] == "sampling"
    assert store.read_run_metadata("unknown_stage") is None
