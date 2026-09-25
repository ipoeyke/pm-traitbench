"""Tests for the plan stage: table I/O, run metadata and its link to the skeletons and
signals it writes.
"""

import pytest

from pm_traitbench.catalogues.models import PreferenceGroup
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.enums import Kind
from pm_traitbench.errors import PlanError
from pm_traitbench.signals.stage import PLAN_STAGE
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.schema import Trait
from pm_traitbench.tables.specs import PERSONAS, PLAN_TABLES, SIGNALS, SKELETONS, TRAITS
from pm_traitbench.tables.store import DataStore
from tests.engine.conftest import (  # noqa: F401
    MULTI_ASSET_PM_ID,
    fixture_market,
    neutral_pm,
    stage_config,
    write_stage_inputs,
)


def _plant_carriers_and_preferences(store, catalogue) -> None:
    """Give the first PM (`pm_001`) two active biases and one preference per group, so
    the engine leaves carriers and the plan has preferences to place.
    """
    traits = store.read(TRAITS)
    pm_id = "pm_001"
    pm_traits = [t for t in traits if t.pm_id == pm_id]
    other_traits = [t for t in traits if t.pm_id != pm_id]

    updated = []
    for trait in pm_traits:
        if trait.param == "loss_aversion_lambda":
            updated.append(trait.model_copy(update={"active": True, "value": 2.0}))
        elif trait.param == "overconfidence_coverage":
            updated.append(trait.model_copy(update={"active": True, "value": 0.40}))
        else:
            updated.append(trait)

    next_index = len(updated) + 1
    seen_groups: set[PreferenceGroup] = set()
    for entry in catalogue.preferences_for(_asset_class_of(store, pm_id)):
        if entry.group in seen_groups:
            continue
        seen_groups.add(entry.group)
        updated.append(
            Trait(
                pm_id=pm_id,
                trait_id=f"t_{next_index:02d}",
                kind=Kind.PREFERENCE,
                param=entry.param,
                value=entry.values[0],
                active=True,
                mult_range=None,
                mult_risk_off=None,
                mult_risk_on=None,
            )
        )
        next_index += 1

    store.write(TRAITS, [*other_traits, *updated])


def _asset_class_of(store, pm_id: str):
    persona = next(p for p in store.read(PERSONAS) if p.pm_id == pm_id)
    return persona.mandate.asset_class


def _run_full(tmp_path, fixture_market, neutral_pm, catalogue):
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    _plant_carriers_and_preferences(store, catalogue)

    run_stage(ENGINE_STAGE, config, store)
    run_stage(PLAN_STAGE, config, store)
    return config, store


def test_plan_stage_writes_signals_and_skeletons_that_link_to_each_other(
    tmp_path, fixture_market, neutral_pm, catalogue
) -> None:
    _, store = _run_full(tmp_path, fixture_market, neutral_pm, catalogue)

    for spec in PLAN_TABLES:
        assert store.path(spec).exists()

    signals = store.read(SIGNALS)
    skeletons = store.read(SKELETONS)
    assert signals
    assert skeletons

    session_ids = {(s.pm_id, s.session_id) for s in skeletons}
    for signal in signals:
        assert (signal.pm_id, signal.session_id) in session_ids

    signal_ids = {(s.pm_id, sig.signal_id) for s in skeletons for sig in s.stances}
    all_signal_ids = {(sig.pm_id, sig.signal_id) for sig in signals}
    assert signal_ids <= all_signal_ids

    multi_asset_rows = [s for s in signals if s.pm_id == MULTI_ASSET_PM_ID]
    multi_asset_sessions = [s for s in skeletons if s.pm_id == MULTI_ASSET_PM_ID]
    assert not multi_asset_rows
    assert not multi_asset_sessions


def test_plan_stage_raises_without_engine_metadata(tmp_path, fixture_market, neutral_pm) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)
    (tmp_path / "run_metadata" / "engine.json").unlink()

    with pytest.raises(PlanError, match="engine run metadata is missing"):
        run_stage(PLAN_STAGE, config, store)


def test_plan_stage_metadata_skips_the_multi_asset_pm(
    tmp_path, fixture_market, neutral_pm, catalogue
) -> None:
    _, store = _run_full(tmp_path, fixture_market, neutral_pm, catalogue)

    metadata = store.read_run_metadata("plan")
    assert metadata is not None
    assert metadata["skipped"] == [MULTI_ASSET_PM_ID]
    assert MULTI_ASSET_PM_ID not in metadata["pms"]


def test_plan_stage_writes_byte_identical_tables_on_a_second_run(
    tmp_path, fixture_market, neutral_pm, catalogue
) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    _, store_a = _run_full(dir_a, fixture_market, neutral_pm, catalogue)
    _, store_b = _run_full(dir_b, fixture_market, neutral_pm, catalogue)

    for spec in PLAN_TABLES:
        assert store_a.path(spec).read_bytes() == store_b.path(spec).read_bytes()
