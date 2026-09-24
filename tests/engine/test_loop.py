"""Tests for the per-PM daily loop: `run_pm` end to end on the fixture market."""

from dataclasses import replace

import pytest

from pm_traitbench.config import Config, SeedConfig
from pm_traitbench.engine import loop as loop_module
from pm_traitbench.engine.loop import run_pm
from pm_traitbench.engine.step import step as real_step
from pm_traitbench.enums import (
    Action,
    AssetClass,
    Op,
    RuleResponse,
    RuleScope,
    RuleSource,
    Side,
)
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Rule

_NEUTRAL_PMS = [
    (AssetClass.EQUITIES, "equity_long_short"),
    (AssetClass.RATES_CREDIT, "sovereign_rates"),
    (AssetClass.RATES_CREDIT, "long_short_credit"),
    (AssetClass.COMMODITIES, "commodity_futures_directional"),
    (AssetClass.COMMODITIES, "curve_and_spread"),
]


def _with_exit_deficiency(traits, value: float, *, active: bool):
    return [
        t.model_copy(update={"value": value, "active": active})
        if t.param == "exit_deficiency"
        else t
        for t in traits
    ]


def _run(
    neutral_pm, catalogue, fixture_view, asset_class, sub_style, config, *, exit_deficiency=0.0
):
    persona, traits, pm_rules = neutral_pm(asset_class, sub_style)
    traits = _with_exit_deficiency(traits, exit_deficiency, active=exit_deficiency > 0)
    return run_pm(persona, traits, pm_rules, [], fixture_view, config, catalogue)


def test_ledger_and_idea_rules_reference_ideas_that_exist(
    neutral_pm, catalogue, fixture_view, engine_config
) -> None:
    for asset_class, sub_style in _NEUTRAL_PMS:
        result = _run(neutral_pm, catalogue, fixture_view, asset_class, sub_style, engine_config)
        idea_ids = {idea.trade_idea_id for idea in result.ideas}
        for row in result.ledger:
            assert row.trade_idea_id in idea_ids
        for rule in result.idea_rules:
            assert rule.trade_idea_id in idea_ids


@pytest.mark.parametrize("exit_deficiency", [0.0, 0.5])
def test_signed_ledger_sizes_net_to_zero_per_idea_and_instrument(
    neutral_pm, catalogue, fixture_view, engine_config, exit_deficiency
) -> None:
    for asset_class, sub_style in _NEUTRAL_PMS:
        result = _run(
            neutral_pm,
            catalogue,
            fixture_view,
            asset_class,
            sub_style,
            engine_config,
            exit_deficiency=exit_deficiency,
        )
        assert result.ideas, f"{asset_class}/{sub_style} produced no ideas"
        net: dict[tuple[str, str], float] = {}
        for row in result.ledger:
            key = (row.trade_idea_id, row.instrument_id)
            sign = 1.0 if row.side == Side.BUY else -1.0
            net[key] = net.get(key, 0.0) + sign * row.size
        for key, total in net.items():
            assert total == pytest.approx(0.0, abs=1e-9), (asset_class, sub_style, key)


def test_every_idea_is_closed_with_exit_date_and_outcome(
    neutral_pm, catalogue, fixture_view, engine_config
) -> None:
    for asset_class, sub_style in _NEUTRAL_PMS:
        result = _run(neutral_pm, catalogue, fixture_view, asset_class, sub_style, engine_config)
        assert result.ideas, f"{asset_class}/{sub_style} produced no ideas"
        for idea in result.ideas:
            assert idea.exit_date is not None
            assert idea.outcome is not None


def test_run_pm_raises_if_an_idea_never_closes(
    neutral_pm, catalogue, fixture_view, engine_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona, traits, pm_rules = neutral_pm(AssetClass.EQUITIES, "equity_long_short")

    def fake_step(state, t, view, ctx, idea_rules_arg):
        new_state, day_out = real_step(state, t, view, ctx, idea_rules_arg)
        return new_state, replace(day_out, closed=())

    monkeypatch.setattr(loop_module, "step", fake_step)

    with pytest.raises(EngineError, match="never closed"):
        run_pm(persona, traits, pm_rules, [], fixture_view, engine_config, catalogue)


def test_no_position_day_before_its_idea_entry_date(
    neutral_pm, catalogue, fixture_view, engine_config
) -> None:
    for asset_class, sub_style in _NEUTRAL_PMS:
        result = _run(neutral_pm, catalogue, fixture_view, asset_class, sub_style, engine_config)
        entry_dates = {idea.trade_idea_id: idea.entry_date for idea in result.ideas}
        for pos_day in result.position_days:
            assert pos_day.date >= entry_dates[pos_day.trade_idea_id]


def test_rule_event_responses_are_only_acted_or_overridden_when_e_is_zero(
    neutral_pm, catalogue, fixture_view, engine_config
) -> None:
    total_events = 0
    for asset_class, sub_style in _NEUTRAL_PMS:
        result = _run(
            neutral_pm,
            catalogue,
            fixture_view,
            asset_class,
            sub_style,
            engine_config,
            exit_deficiency=0.0,
        )
        total_events += len(result.rule_events)
        for event in result.rule_events:
            assert event.response in (RuleResponse.ACTED, RuleResponse.OVERRIDDEN)
    assert total_events > 0


def test_opportunities_ideas_equals_len_ideas(
    neutral_pm, catalogue, fixture_view, engine_config
) -> None:
    for asset_class, sub_style in _NEUTRAL_PMS:
        result = _run(neutral_pm, catalogue, fixture_view, asset_class, sub_style, engine_config)
        assert result.opportunities["ideas"] == len(result.ideas)


def test_unknown_rule_field_raises_before_any_day_runs(
    neutral_pm, catalogue, fixture_view, engine_config
) -> None:
    persona, traits, pm_rules = neutral_pm(AssetClass.EQUITIES, "equity_long_short")
    bad_rule = Rule(
        pm_id=persona.pm_id,
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

    with pytest.raises(EngineError, match="r_99"):
        run_pm(persona, traits, [*pm_rules, bad_rule], [], fixture_view, engine_config, catalogue)


def test_multi_asset_persona_is_not_run(neutral_pm, catalogue, fixture_view, engine_config) -> None:
    persona, traits, pm_rules = neutral_pm(AssetClass.COMMODITIES, "commodity_futures_directional")
    persona = persona.model_copy(
        update={
            "mandate": persona.mandate.model_copy(
                update={"asset_class": AssetClass.MULTI_ASSET, "sub_style": "global_macro"}
            )
        }
    )

    with pytest.raises(EngineError):
        run_pm(persona, traits, pm_rules, [], fixture_view, engine_config, catalogue)


def test_exit_deficiency_breach_share_matches_the_set_value_pooled_over_seeds(
    neutral_pm, catalogue, fixture_view
) -> None:
    """Pools the five neutral PMs over 28 root seeds with exit_deficiency fixed at 0.06,
    and checks the pooled share of missed (non-'acted') responses lands near 0.06.

    Standard error of a 0.06 share at n=500 draws is about 0.011 (sqrt(0.06*0.94/500)),
    so a 0.03 tolerance is roughly 2.7 SE: wide enough to avoid flaking, tight enough to
    catch a real miscalibration of the exit_deficiency draw.
    """
    n_seeds = 28
    total = 0
    breached = 0

    for root_seed in range(n_seeds):
        config = Config(seed=SeedConfig(root=20260105 + root_seed))
        for asset_class, sub_style in _NEUTRAL_PMS:
            persona, traits, pm_rules = neutral_pm(asset_class, sub_style)
            traits = _with_exit_deficiency(traits, 0.06, active=True)
            result = run_pm(persona, traits, pm_rules, [], fixture_view, config, catalogue)
            for event in result.rule_events:
                if event.response == RuleResponse.OVERRIDDEN:
                    continue
                total += 1
                if event.response != RuleResponse.ACTED:
                    breached += 1

    assert total >= 500, f"only {total} non-overridden rule events over {n_seeds} seeds"
    share = breached / total
    assert abs(share - 0.06) <= 0.03
