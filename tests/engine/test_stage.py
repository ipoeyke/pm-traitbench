"""Tests for the engine stage: table I/O, run metadata, and failure-before-write."""

import json

import pytest

from pm_traitbench.engine.stage import ENGINE_STAGE, build_views
from pm_traitbench.enums import RuleScope
from pm_traitbench.errors import EngineError, StageIOError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.schema import Rule
from pm_traitbench.tables.specs import ENGINE_TABLES, IDEAS, RULES
from pm_traitbench.tables.store import DataStore
from tests.engine.fixtures import MULTI_ASSET_PM_ID, NEUTRAL_PMS, stage_config, write_stage_inputs


def test_engine_stage_writes_four_tables_and_rules_gains_idea_rows(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    _, original_rules = write_stage_inputs(store, fixture_market, neutral_pm)

    run_stage(ENGINE_STAGE, config, store)

    for spec in ENGINE_TABLES:
        assert store.path(spec).exists()

    written_rules = store.read(RULES)
    original_keys = {(r.pm_id, r.rule_id) for r in original_rules}
    written_keys = {(r.pm_id, r.rule_id) for r in written_rules}
    assert original_keys <= written_keys
    assert len(written_keys) > len(original_keys)


def _idea_rule_counts(rules: list[Rule]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for rule in rules:
        if rule.scope == RuleScope.IDEA:
            key = (rule.pm_id, rule.trade_idea_id)
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_forced_rerun_replaces_idea_rules_instead_of_stacking_them(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    _, original_rules = write_stage_inputs(store, fixture_market, neutral_pm)

    run_stage(ENGINE_STAGE, config, store, force=True)
    first = store.read(RULES)
    run_stage(ENGINE_STAGE, config, store, force=True)
    second = store.read(RULES)

    assert len(second) == len(first)
    assert second == first
    assert [r for r in second if r.scope == RuleScope.PM] == original_rules
    idea_keys = {(idea.pm_id, idea.trade_idea_id) for idea in store.read(IDEAS)}
    counts = _idea_rule_counts(second)
    assert set(counts) == idea_keys
    assert counts == _idea_rule_counts(first)
    rule_keys = [(r.pm_id, r.rule_id) for r in second]
    assert len(rule_keys) == len(set(rule_keys))


def test_run_metadata_has_the_three_extras_and_skips_the_multi_asset_pm(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)

    run_stage(ENGINE_STAGE, config, store)

    metadata = json.loads((tmp_path / "run_metadata" / "engine.json").read_text(encoding="utf-8"))
    expected_pm_ids = {f"pm_{i:03d}" for i in range(1, len(NEUTRAL_PMS) + 1)}
    assert set(metadata["ideas_per_pm"]) == expected_pm_ids
    assert set(metadata["opportunities"]) == expected_pm_ids
    assert metadata["skipped"] == [MULTI_ASSET_PM_ID]


def test_second_run_without_force_raises_stage_io_error(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)

    with pytest.raises(StageIOError):
        run_stage(ENGINE_STAGE, config, store)


def test_build_views_returns_a_view_on_the_published_horizon(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)

    views = build_views(config, store, ["T"])

    axis = build_axis(config.timeline(), config.market.burn_in_days)
    assert views["T"].dates == axis.dates[axis.horizon]


def test_unknown_rule_field_fails_before_any_engine_table_is_written(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm, bad_field=True)

    with pytest.raises(EngineError):
        run_stage(ENGINE_STAGE, config, store)

    for spec in ENGINE_TABLES:
        assert not store.path(spec).exists()
