import datetime
import json
from pathlib import Path

import pytest

from pm_traitbench.config import Config, OutputConfig
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
from pm_traitbench.errors import ConfigError, StageIOError, TableValidationError
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
from pm_traitbench.tables.specs import DRIFT_EVENTS, PERSONAS, RULES, TRAITS, TableSpec
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


def test_default_resolution_personas_jsonl_traits_csv(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    assert store.format_name(PERSONAS) == "jsonl"
    assert store.format_name(TRAITS) == "csv"
    assert store.path(PERSONAS) == tmp_path / "personas.jsonl"
    assert store.path(TRAITS) == tmp_path / "traits.csv"


def test_global_format_parquet_applies_to_all_tables(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig(format="parquet"))
    for spec in (PERSONAS, TRAITS, RULES, DRIFT_EVENTS):
        assert store.format_name(spec) == "parquet"


def test_per_table_override_wins_over_global_format(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig(format="parquet", tables={"traits": "jsonl"}))
    assert store.format_name(TRAITS) == "jsonl"
    assert store.format_name(RULES) == "parquet"


def test_csv_for_nested_table_raises_config_error(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig(tables={"personas": "csv"}))
    with pytest.raises(ConfigError, match="personas"):
        store.format_name(PERSONAS)


def test_write_returns_path_and_file_exists(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    path = store.write(TRAITS, [_trait("pm_001", "t_01")])
    assert path == tmp_path / "traits.csv"
    assert path.exists()
    assert store.exists(TRAITS) is True


def test_write_creates_missing_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    store = DataStore(data_dir, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01")])
    assert (data_dir / "traits.csv").exists()


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


def test_write_config_error_before_mkdir_leaves_no_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    store = DataStore(data_dir, OutputConfig(tables={"personas": "csv"}))
    with pytest.raises(ConfigError):
        store.write(PERSONAS, [_persona("pm_001")])
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
    assert not (tmp_path / "traits.csv.tmp").exists()
    assert not (tmp_path / "traits.csv").exists()


def test_write_failure_midway_removes_tmp_and_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DataStore(tmp_path, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01")])
    target = tmp_path / "traits.csv"
    original_bytes = target.read_bytes()

    csv_format = FORMATS["csv"]

    def _write_then_raise(records, model, path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(csv_format, "write", _write_then_raise)

    with pytest.raises(RuntimeError, match="boom"):
        store.write(TRAITS, [_trait("pm_002", "t_01")])

    assert target.read_bytes() == original_bytes
    assert not (tmp_path / "traits.csv.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_read_missing_table_raises_stage_io_error(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    with pytest.raises(StageIOError, match="traits"):
        store.read(TRAITS)


def test_read_missing_table_mentions_other_extension_present(tmp_path: Path) -> None:
    store_parquet = DataStore(tmp_path, OutputConfig(tables={"traits": "parquet"}))
    store_parquet.write(TRAITS, [_trait("pm_001", "t_01")])
    store_csv = DataStore(tmp_path, OutputConfig(tables={"traits": "csv"}))
    with pytest.raises(StageIOError, match="traits.parquet"):
        store_csv.read(TRAITS)


def test_read_corrupted_csv_cell_raises_table_validation_error(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01")])
    path = tmp_path / "traits.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    active_index = header.index("active")
    cells = lines[1].split(",")
    cells[active_index] = "maybe"
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TableValidationError, match=r"row 1\b") as exc_info:
        store.read(TRAITS)
    assert "active" in str(exc_info.value)


def test_read_corrupted_csv_cell_on_the_second_row_names_row_two(tmp_path: Path) -> None:
    store = DataStore(tmp_path, OutputConfig())
    store.write(TRAITS, [_trait("pm_001", "t_01"), _trait("pm_002", "t_01")])
    path = tmp_path / "traits.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    active_index = header.index("active")
    cells = lines[2].split(",")
    cells[active_index] = "maybe"
    lines[2] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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


@pytest.mark.parametrize("format_name", ["csv", "jsonl"])
def test_write_twice_gives_identical_bytes(tmp_path: Path, format_name: str) -> None:
    store = DataStore(tmp_path, OutputConfig(tables={"traits": format_name}))
    rows = [_trait("pm_001", "t_01"), _trait("pm_002", "t_01")]
    path_a = store.write(TRAITS, rows)
    first_bytes = path_a.read_bytes()
    path_b = store.write(TRAITS, rows)
    second_bytes = path_b.read_bytes()
    assert first_bytes == second_bytes


@pytest.mark.parametrize(
    ("spec", "formats"),
    [
        (PERSONAS, ["jsonl", "parquet"]),
        (TRAITS, ["csv", "jsonl", "parquet"]),
        (RULES, ["csv", "jsonl", "parquet"]),
        (DRIFT_EVENTS, ["csv", "jsonl", "parquet"]),
    ],
    ids=["personas", "traits", "rules", "drift_events"],
)
def test_round_trip_equality_for_legal_formats(
    tmp_path: Path, spec: TableSpec, formats: list[str]
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
    for format_name in formats:
        table_dir = tmp_path / format_name
        store = DataStore(table_dir, OutputConfig(tables={spec.name: format_name}))
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
