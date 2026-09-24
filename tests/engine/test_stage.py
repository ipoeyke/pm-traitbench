"""Tests for the engine stage: table I/O, run metadata, and failure-before-write."""

import json

import pytest

from pm_traitbench.config import Config
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.enums import Action, AssetClass, Op, RuleScope, RuleSource, Split, Typicality
from pm_traitbench.errors import EngineError, StageIOError
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.schema import Mandate, Persona, Rule, StatedProfile
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    ENGINE_TABLES,
    IDEAS,
    MARKET_CALENDAR,
    MARKET_CONSENSUS,
    MARKET_CURVES,
    MARKET_INSTRUMENTS,
    MARKET_PRICES,
    MARKET_REGIMES,
    PERSONAS,
    RULES,
    TRAITS,
)
from pm_traitbench.tables.store import DataStore

_NEUTRAL_PMS = [
    (AssetClass.EQUITIES, "equity_long_short"),
    (AssetClass.RATES_CREDIT, "sovereign_rates"),
    (AssetClass.RATES_CREDIT, "long_short_credit"),
    (AssetClass.COMMODITIES, "commodity_futures_directional"),
    (AssetClass.COMMODITIES, "curve_and_spread"),
]
_MULTI_ASSET_PM_ID = "pm_006"


def _stage_config() -> Config:
    """A config whose published horizon exactly covers the 60-day 'T' fixture market."""
    return Config.model_validate(
        {
            "calendar": {"start": "2026-01-05", "n_weeks": 12},
            "population": {
                "asset_classes": ["equities", "rates_credit", "commodities"],
                "market_seeds": ["T"],
                "pilot_market_seeds": ["T"],
                "pilot_per_cell": 1,
                "full_per_cell": 1,
            },
            "market": {
                "seeds": {"T": ["range", "risk_off", "risk_on"]},
                "real": {"seeds": {}},
                "boundary_weeks": [4, 8],
                "burn_in_days": 1,
            },
            "drift": {
                "bias_update_weeks": [2, 4],
                "preference_update_weeks": [2, 10],
                "dormant_weeks": [5, 7],
                "revive_weeks": [8, 9],
            },
            "engine": {"horizon_days": 10},
        }
    )


def _bad_field_rule(pm_id: str) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id="r_99",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="min_holding_period",
        field="not_a_real_field",
        op=Op.GE,
        level=5.0,
        unit=None,
        window=1,
        action=Action.HOLD,
        text="bogus rule",
    )


def _write_stage_inputs(store, fixture_market, neutral_pm, *, bad_field: bool = False):
    personas = []
    all_traits = []
    all_rules = []
    for i, (asset_class, sub_style) in enumerate(_NEUTRAL_PMS, start=1):
        persona, traits, rules = neutral_pm(asset_class, sub_style)
        pm_id = f"pm_{i:03d}"
        persona = persona.model_copy(update={"pm_id": pm_id})
        traits = [t.model_copy(update={"pm_id": pm_id}) for t in traits]
        rules = [r.model_copy(update={"pm_id": pm_id}) for r in rules]
        if bad_field and i == 1:
            rules.append(_bad_field_rule(pm_id))
        personas.append(persona)
        all_traits.extend(traits)
        all_rules.extend(rules)

    personas.append(
        Persona(
            pm_id=_MULTI_ASSET_PM_ID,
            market_seed="T",
            split=Split.PILOT,
            mandate=Mandate(
                asset_class=AssetClass.MULTI_ASSET,
                sub_style="global_macro",
                book_size=1e8,
                risk_unit="pct_nav_sleeve",
                benchmark="cash",
            ),
            stated_profile=StatedProfile(self_description="I mix sleeves across asset classes."),
            typicality=Typicality.TYPICAL,
        )
    )

    store.write(PERSONAS, personas)
    store.write(TRAITS, all_traits)
    store.write(RULES, all_rules)
    store.write(DRIFT_EVENTS, [])
    store.write(MARKET_INSTRUMENTS, fixture_market["instruments"])
    store.write(MARKET_PRICES, fixture_market["prices"])
    store.write(MARKET_CURVES, fixture_market["curves"])
    store.write(MARKET_CONSENSUS, fixture_market["consensus"])
    store.write(MARKET_CALENDAR, fixture_market["calendar"])
    store.write(MARKET_REGIMES, fixture_market["regimes"])
    return personas, all_rules


def test_engine_stage_writes_four_tables_and_rules_gains_idea_rows(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = _stage_config()
    store = DataStore(tmp_path, config.output)
    _, original_rules = _write_stage_inputs(store, fixture_market, neutral_pm)

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
    config = _stage_config()
    store = DataStore(tmp_path, config.output)
    _, original_rules = _write_stage_inputs(store, fixture_market, neutral_pm)

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
    config = _stage_config()
    store = DataStore(tmp_path, config.output)
    _write_stage_inputs(store, fixture_market, neutral_pm)

    run_stage(ENGINE_STAGE, config, store)

    metadata = json.loads((tmp_path / "run_metadata" / "engine.json").read_text(encoding="utf-8"))
    expected_pm_ids = {f"pm_{i:03d}" for i in range(1, len(_NEUTRAL_PMS) + 1)}
    assert set(metadata["ideas_per_pm"]) == expected_pm_ids
    assert set(metadata["opportunities"]) == expected_pm_ids
    assert metadata["skipped"] == [_MULTI_ASSET_PM_ID]


def test_second_run_without_force_raises_stage_io_error(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = _stage_config()
    store = DataStore(tmp_path, config.output)
    _write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)

    with pytest.raises(StageIOError):
        run_stage(ENGINE_STAGE, config, store)


def test_unknown_rule_field_fails_before_any_engine_table_is_written(
    tmp_path, fixture_market, neutral_pm
) -> None:
    config = _stage_config()
    store = DataStore(tmp_path, config.output)
    _write_stage_inputs(store, fixture_market, neutral_pm, bad_field=True)

    with pytest.raises(EngineError):
        run_stage(ENGINE_STAGE, config, store)

    for spec in ENGINE_TABLES:
        assert not store.path(spec).exists()
