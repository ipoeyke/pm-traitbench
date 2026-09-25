"""Tests for `carrier_pools`: dated engine evidence grouped by the trait it reveals."""

from datetime import date

import pytest

from pm_traitbench.catalogues.loader import REVEALED_PATTERNS
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import CarrierSource, DriftEventType, Expression, RuleResponse, Side, Tenor
from pm_traitbench.errors import PlanError
from pm_traitbench.signals.carriers import (
    BREACH_RESPONSES,
    FLAG_PREFIX_PARAM,
    HOLD_FLAGS,
    Carrier,
    carrier_pools,
)
from pm_traitbench.tables.schema import Leg
from tests.signals.conftest import (
    bias_trait,
    drift_event,
    idea_row,
    ledger_row,
    plan_inputs,
    position_day,
    pref_trait,
    rule_event,
)

_CURVE_LEGS = (
    Leg(instrument_id="RC-0001", tenor=Tenor.Y2, side=Side.BUY, weight=1.0),
    Leg(instrument_id="RC-0001", tenor=Tenor.Y10, side=Side.SELL, weight=1.0),
)

DEFAULT_DATE = date(2026, 1, 5)


def test_flag_prefix_param_covers_bias_params_except_extrapolation():
    assert set(FLAG_PREFIX_PARAM.values()) <= set(BIAS_PARAMS)
    assert set(BIAS_PARAMS) - set(FLAG_PREFIX_PARAM.values()) == {"extrapolation_theta"}


@pytest.mark.parametrize("prefix,param", sorted(FLAG_PREFIX_PARAM.items()))
def test_each_flag_prefix_maps_to_its_bias(prefix, param):
    pattern = REVEALED_PATTERNS[param][0]
    inputs = plan_inputs(
        traits=(bias_trait(param, trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag=f"{prefix}:{pattern}"),),
    )
    pools = carrier_pools(inputs)
    assert pools["t_01"] == (
        Carrier("t_01", "ti_001", DEFAULT_DATE, CarrierSource.LEDGER, pattern),
    )


def test_semicolon_joined_flag_yields_one_carrier_per_flag():
    inputs = plan_inputs(
        traits=(
            bias_trait("loss_aversion_lambda", trait_id="t_01"),
            bias_trait("disposition_ratio", trait_id="t_02"),
        ),
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag="loss_aversion:add;disposition:realise_gain_early"),),
    )
    pools = carrier_pools(inputs)
    assert len(pools["t_01"]) == 1
    assert len(pools["t_02"]) == 1


def test_flag_on_inactive_bias_yields_nothing():
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01", active=False),),
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag="loss_aversion:add"),),
    )
    pools = carrier_pools(inputs)
    assert "t_01" not in pools


def test_hold_flags_count_only_first_day_per_idea():
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        position_days=(
            position_day(date=date(2026, 1, 5), bias_flag="loss_aversion:hold"),
            position_day(date=date(2026, 1, 6), bias_flag="loss_aversion:hold"),
        ),
    )
    pools = carrier_pools(inputs)
    assert pools["t_01"] == (
        Carrier("t_01", "ti_001", date(2026, 1, 5), CarrierSource.POSITION_DAY, "hold"),
    )


def test_loss_aversion_add_on_position_days_yields_every_day():
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        position_days=(
            position_day(date=date(2026, 1, 5), bias_flag="loss_aversion:add"),
            position_day(date=date(2026, 1, 6), bias_flag="loss_aversion:add"),
        ),
    )
    pools = carrier_pools(inputs)
    assert len(pools["t_01"]) == 2


def test_acked_no_action_and_added_are_exit_deficiency_carriers():
    inputs = plan_inputs(
        traits=(bias_trait("exit_deficiency", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        rule_events=(
            rule_event(response=RuleResponse.ACKED_NO_ACTION, response_date=date(2026, 1, 10)),
        ),
    )
    pools = carrier_pools(inputs)
    assert pools["t_01"] == (
        Carrier("t_01", "ti_001", date(2026, 1, 10), CarrierSource.RULE_EVENT, "acked_no_action"),
    )
    assert set(BREACH_RESPONSES) == {RuleResponse.ACKED_NO_ACTION, RuleResponse.ADDED}


def test_acted_and_overridden_are_not_carriers():
    inputs = plan_inputs(
        traits=(bias_trait("exit_deficiency", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        rule_events=(
            rule_event(response=RuleResponse.ACTED),
            rule_event(response=RuleResponse.OVERRIDDEN),
        ),
    )
    pools = carrier_pools(inputs)
    assert pools["t_01"] == ()


def test_chased_trend_counts_only_when_extrapolation_active():
    idea = idea_row(chased_trend=True)
    inactive_inputs = plan_inputs(
        traits=(bias_trait("extrapolation_theta", trait_id="t_01", active=False),),
        ideas={"ti_001": idea},
    )
    assert "t_01" not in carrier_pools(inactive_inputs)

    active_inputs = plan_inputs(
        traits=(bias_trait("extrapolation_theta", trait_id="t_01"),),
        ideas={"ti_001": idea},
    )
    pools = carrier_pools(active_inputs)
    assert pools["t_01"] == (
        Carrier("t_01", "ti_001", idea.entry_date, CarrierSource.IDEA, "chased_trend"),
    )


def test_same_day_ledger_and_position_day_flag_dedupe_to_one_ledger_carrier():
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag="loss_aversion:add"),),
        position_days=(position_day(bias_flag="loss_aversion:add"),),
    )
    pools = carrier_pools(inputs)
    assert pools["t_01"] == (Carrier("t_01", "ti_001", DEFAULT_DATE, CarrierSource.LEDGER, "add"),)


def test_dormant_window_carriers_are_dropped():
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag="loss_aversion:add"),),
        drift_events=(
            drift_event("t_01", date(2026, 1, 1), DriftEventType.DORMANT),
            drift_event("t_01", date(2026, 2, 1), DriftEventType.REVIVE),
        ),
    )
    pools = carrier_pools(inputs)
    assert pools["t_01"] == ()


def test_unknown_prefix_raises_plan_error():
    inputs = plan_inputs(
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag="mystery:pattern"),),
    )
    with pytest.raises(PlanError):
        carrier_pools(inputs)


def test_unknown_pattern_for_known_prefix_raises_plan_error():
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01"),),
        ideas={"ti_001": idea_row()},
        ledger=(ledger_row(bias_flag="loss_aversion:bogus"),),
    )
    with pytest.raises(PlanError):
        carrier_pools(inputs)


def test_expression_preference_carriers_track_value_in_force():
    trait = pref_trait("duration_expression", "steepeners over outright duration", trait_id="t_50")
    curve_idea = idea_row(
        trade_idea_id="ti_001",
        expression=Expression.CURVE,
        entry_date=date(2026, 1, 5),
        legs=_CURVE_LEGS,
    )
    outright_before = idea_row(
        trade_idea_id="ti_002", expression=Expression.OUTRIGHT, entry_date=date(2026, 1, 6)
    )
    outright_after = idea_row(
        trade_idea_id="ti_003", expression=Expression.OUTRIGHT, entry_date=date(2026, 2, 1)
    )
    inputs = plan_inputs(
        traits=(trait,),
        ideas={idea.trade_idea_id: idea for idea in (curve_idea, outright_before, outright_after)},
        drift_events=(
            drift_event(
                "t_50",
                date(2026, 1, 20),
                DriftEventType.UPDATE,
                from_value="steepeners over outright duration",
                to_value="outright duration over curve trades",
            ),
        ),
    )
    pools = carrier_pools(inputs)
    assert pools["t_50"] == (
        Carrier("t_50", "ti_001", date(2026, 1, 5), CarrierSource.IDEA, None),
        Carrier("t_50", "ti_003", date(2026, 2, 1), CarrierSource.IDEA, None),
    )


def test_every_active_bias_key_exists_even_with_empty_pool():
    inputs = plan_inputs(traits=(bias_trait("anchoring_rho", trait_id="t_01"),))
    pools = carrier_pools(inputs)
    assert pools == {"t_01": ()}


def test_carrier_patterns_are_all_known_revealed_patterns():
    """Build a carrier from every flag and rule-event pattern and check it's in the bank."""
    traits = tuple(
        bias_trait(param, trait_id=f"t_{i:02d}") for i, param in enumerate(BIAS_PARAMS, start=1)
    )
    trait_by_param = {trait.param: trait for trait in traits}
    ideas: dict[str, object] = {}
    ledger_rows = []
    idea_n = 100
    for prefix, param in sorted(FLAG_PREFIX_PARAM.items()):
        for pattern in REVEALED_PATTERNS[param]:
            if f"{prefix}:{pattern}" in HOLD_FLAGS:
                continue
            idea_id = f"ti_{idea_n}"
            idea_n += 1
            ideas[idea_id] = idea_row(trade_idea_id=idea_id, entry_date=DEFAULT_DATE)
            ledger_rows.append(ledger_row(trade_idea_id=idea_id, bias_flag=f"{prefix}:{pattern}"))

    ideas["ti_201"] = idea_row(trade_idea_id="ti_201")
    ideas["ti_202"] = idea_row(trade_idea_id="ti_202")
    rule_events = (
        rule_event(trade_idea_id="ti_201", response=RuleResponse.ACKED_NO_ACTION),
        rule_event(trade_idea_id="ti_202", response=RuleResponse.ADDED),
    )
    chased_idea = idea_row(trade_idea_id="ti_300", chased_trend=True)
    ideas["ti_300"] = chased_idea

    traits = (*traits,)
    inputs = plan_inputs(
        traits=traits,
        ideas=ideas,
        ledger=tuple(ledger_rows),
        rule_events=rule_events,
    )
    pools = carrier_pools(inputs)
    for param, trait in trait_by_param.items():
        for carrier in pools[trait.trait_id]:
            if carrier.pattern is not None:
                assert carrier.pattern in REVEALED_PATTERNS[param]
