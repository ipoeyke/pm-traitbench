"""Tests for the gate 1 stage: table I/O, run metadata and the blocking verdict."""

import json

import pytest

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.enums import Gate1Split
from pm_traitbench.errors import Gate1Error, StageIOError
from pm_traitbench.gates.gate1.stage import GATE1_STAGE
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import GATE1_CELLS, GATE1_PM
from pm_traitbench.tables.store import DataStore
from tests.engine.conftest import MULTI_ASSET_PM_ID, NEUTRAL_PMS, stage_config, write_stage_inputs


def test_gate1_stage_writes_both_tables_and_raises_on_the_insufficient_default_thresholds(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)

    with pytest.raises(Gate1Error) as excinfo:
        run_stage(GATE1_STAGE, config, store)

    # The pooled cells are written and readable even though the run then raised.
    for spec in (GATE1_PM, GATE1_CELLS):
        assert store.path(spec).exists()
        store.read(spec)

    pm_rows = store.read(GATE1_PM)
    all_params_by_pm: dict[str, set[str]] = {}
    for row in pm_rows:
        if row.split == Gate1Split.ALL:
            all_params_by_pm.setdefault(row.pm_id, set()).add(row.param)
    expected_pm_ids = {f"pm_{i:03d}" for i in range(1, len(NEUTRAL_PMS) + 1)}
    assert set(all_params_by_pm) == expected_pm_ids
    for pm_id in expected_pm_ids:
        assert all_params_by_pm[pm_id] == set(BIAS_PARAMS)

    metadata_path = tmp_path / "run_metadata" / "gate1.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "failed" in metadata
    assert "warnings" in metadata
    assert "thresholds" in metadata

    # Only five direct-asset PMs total, well under min_pms=5 per cell: every
    # pooled synthetic cell is insufficient, so every cell it covers fails.
    message = str(excinfo.value)
    assert message.startswith("gate 1 failed for: ")
    entries = message.removeprefix("gate 1 failed for: ").split(", ")
    assert entries == metadata["failed"]
    assert entries
    for entry in entries:
        asset_class, param = entry.split("/")
        assert param in BIAS_PARAMS


def test_forced_second_run_gives_byte_identical_gate1_tables(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store, force=True)

    with pytest.raises(Gate1Error):
        run_stage(GATE1_STAGE, config, store, force=True)
    first_pm = store.path(GATE1_PM).read_bytes()
    first_cells = store.path(GATE1_CELLS).read_bytes()

    with pytest.raises(Gate1Error):
        run_stage(GATE1_STAGE, config, store, force=True)
    second_pm = store.path(GATE1_PM).read_bytes()
    second_cells = store.path(GATE1_CELLS).read_bytes()

    assert first_pm == second_pm
    assert first_cells == second_cells


def test_missing_engine_tables_raises_stage_io_error_before_run_is_called(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    # ENGINE_STAGE never ran: ideas, ledger, rule_events and position_days are missing.

    with pytest.raises(StageIOError):
        run_stage(GATE1_STAGE, config, store)

    assert not store.path(GATE1_PM).exists()
    assert not store.path(GATE1_CELLS).exists()


def test_missing_engine_run_metadata_raises_gate1_error(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)
    (tmp_path / "run_metadata" / "engine.json").unlink()

    with pytest.raises(Gate1Error, match="engine run metadata is missing"):
        run_stage(GATE1_STAGE, config, store)


def test_tampered_opportunities_count_raises_gate1_error_naming_pm_and_count(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)

    metadata_path = tmp_path / "run_metadata" / "engine.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pm_id = "pm_001"
    assert MULTI_ASSET_PM_ID not in metadata["opportunities"]
    metadata["opportunities"][pm_id]["ideas"] += 1
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(Gate1Error, match=f"PM '{pm_id}'") as excinfo:
        run_stage(GATE1_STAGE, config, store)
    assert "ideas" in str(excinfo.value)
