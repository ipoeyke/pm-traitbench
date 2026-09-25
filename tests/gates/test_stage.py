"""Tests for the gate 1 stage: table I/O, run metadata and the blocking verdict."""

import json

import pytest

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.enums import AssetClass, Gate1Split, Gate1Verdict, SeedGroupKind
from pm_traitbench.errors import Gate1Error, StageIOError
from pm_traitbench.gates.gate1.stage import GATE1_STAGE
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import GATE1_CELLS, GATE1_PM
from pm_traitbench.tables.store import DataStore
from tests.engine.conftest import MULTI_ASSET_PM_ID, NEUTRAL_PMS, stage_config, write_stage_inputs

# From NEUTRAL_PMS: one equities PM, two rates_credit PMs, two commodities PMs.
_DIRECT_ASSET_CLASSES = (AssetClass.EQUITIES, AssetClass.RATES_CREDIT, AssetClass.COMMODITIES)


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

    # One equities PM, two rates_credit PMs and two commodities PMs: every
    # per-class pooled cell falls short of min_pms=5, so every one of them
    # (one per asset class and bias parameter) is insufficient and none is
    # blocking. Summed across classes there are 5 neutral PMs and 0 active
    # ones, so the cross-class row is insufficient too, on n_active.
    cell_rows = store.read(GATE1_CELLS)
    pooled_all_rows = [
        row
        for row in cell_rows
        if row.seed_group_kind == SeedGroupKind.SYNTHETIC_POOL and row.split == Gate1Split.ALL
    ]
    per_class_rows = [row for row in pooled_all_rows if row.asset_class is not None]
    cross_class_rows = [row for row in pooled_all_rows if row.asset_class is None]

    assert {row.asset_class for row in per_class_rows} == set(_DIRECT_ASSET_CLASSES)
    assert {row.param for row in per_class_rows} == set(BIAS_PARAMS)
    assert len(per_class_rows) == len(_DIRECT_ASSET_CLASSES) * len(BIAS_PARAMS)
    assert {row.param for row in cross_class_rows} == set(BIAS_PARAMS)
    assert len(cross_class_rows) == len(BIAS_PARAMS)
    for row in pooled_all_rows:
        assert row.verdict == Gate1Verdict.INSUFFICIENT
    for row in per_class_rows:
        assert row.blocking is False

    metadata_path = tmp_path / "run_metadata" / "gate1.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "failed" in metadata
    assert "warnings" in metadata
    assert "thresholds" in metadata

    message = str(excinfo.value)
    assert message.startswith("gate 1 failed for: ")
    entries = message.removeprefix("gate 1 failed for: ").split(", ")
    assert entries == metadata["failed"]
    # herding_weight, disposition_ratio and anchoring_rho are report-only by
    # default: insufficient like every other parameter here, but never blocking.
    blocking_params = set(BIAS_PARAMS) - set(config.gate1.report_only_params)
    assert set(entries) == {f"all/{param}" for param in blocking_params}


def test_gate1_stage_with_every_param_report_only_does_not_raise_on_insufficient_cells(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    config = config.model_copy(
        update={"gate1": config.gate1.model_copy(update={"report_only_params": BIAS_PARAMS})}
    )
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)

    run_stage(GATE1_STAGE, config, store)  # every cell is insufficient, but none blocks

    pooled_all_rows = [
        row
        for row in store.read(GATE1_CELLS)
        if row.seed_group_kind == SeedGroupKind.SYNTHETIC_POOL and row.split == Gate1Split.ALL
    ]
    assert pooled_all_rows
    assert all(row.verdict == Gate1Verdict.INSUFFICIENT for row in pooled_all_rows)
    assert all(row.blocking is False for row in pooled_all_rows)


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
