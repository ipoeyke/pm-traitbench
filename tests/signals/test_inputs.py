"""Tests for `PlanInputs` and `build_inputs`: partitioning by PM and reading drift state."""

from datetime import date

import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import DriftEventType
from pm_traitbench.errors import PlanError
from pm_traitbench.market.axis import build_axis
from pm_traitbench.signals.inputs import build_inputs, trading_days
from tests.signals.fixtures import (
    bias_trait,
    drift_event,
    idea_row,
    ledger_row,
    persona,
    plan_inputs,
    position_day,
    rule_event,
)


def test_trading_days_equals_engine_horizon() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    assert trading_days(config) == tuple(axis.dates[axis.horizon])


def test_dormant_windows_with_and_without_revive() -> None:
    inputs = plan_inputs(
        drift_events=(
            drift_event("t_01", date(2026, 1, 10), DriftEventType.DORMANT),
            drift_event("t_01", date(2026, 2, 10), DriftEventType.REVIVE),
            drift_event("t_01", date(2026, 3, 10), DriftEventType.DORMANT),
        )
    )
    assert inputs.dormant_windows("t_01") == (
        (date(2026, 1, 10), date(2026, 2, 10)),
        (date(2026, 3, 10), date.max),
    )
    assert inputs.dormant_windows("t_02") == ()


def test_is_dormant_boundaries() -> None:
    inputs = plan_inputs(
        drift_events=(
            drift_event("t_01", date(2026, 1, 10), DriftEventType.DORMANT),
            drift_event("t_01", date(2026, 2, 10), DriftEventType.REVIVE),
        )
    )
    assert inputs.is_dormant("t_01", date(2026, 1, 9)) is False
    assert inputs.is_dormant("t_01", date(2026, 1, 10)) is True
    assert inputs.is_dormant("t_01", date(2026, 2, 9)) is True
    assert inputs.is_dormant("t_01", date(2026, 2, 10)) is False


def test_value_at_before_on_and_after_update() -> None:
    inputs = plan_inputs(
        traits=(bias_trait("loss_aversion_lambda", trait_id="t_01", value=1.5, active=False),),
        drift_events=(
            drift_event(
                "t_01",
                date(2026, 2, 1),
                DriftEventType.UPDATE,
                from_value=1.5,
                to_value=2.5,
            ),
        ),
    )
    assert inputs.value_at("t_01", date(2026, 1, 31)) == 1.5
    assert inputs.value_at("t_01", date(2026, 2, 1)) == 2.5
    assert inputs.value_at("t_01", date(2026, 2, 2)) == 2.5


def test_trait_unknown_id_raises() -> None:
    inputs = plan_inputs()
    with pytest.raises(PlanError):
        inputs.trait("t_missing")


def test_build_inputs_skips_and_sorts_by_pm_id() -> None:
    personas = [persona_for("pm_002"), persona_for("pm_001")]
    result = build_inputs(personas, [], [], [], [], [], [], (date(2026, 1, 5),), skipped=set())
    assert [inputs.persona.pm_id for inputs in result] == ["pm_001", "pm_002"]


def test_build_inputs_respects_skipped() -> None:
    personas = [persona_for("pm_001"), persona_for("pm_002")]
    result = build_inputs(personas, [], [], [], [], [], [], (date(2026, 1, 5),), skipped={"pm_002"})
    assert [inputs.persona.pm_id for inputs in result] == ["pm_001"]


def test_build_inputs_raises_on_unknown_drift_trait() -> None:
    personas = [persona_for("pm_001")]
    traits = [bias_trait("loss_aversion_lambda", trait_id="t_01")]
    drift_events = [drift_event("t_99", date(2026, 1, 5), DriftEventType.DORMANT)]
    with pytest.raises(PlanError, match="pm_001.*t_99"):
        build_inputs(
            personas, traits, drift_events, [], [], [], [], (date(2026, 1, 5),), skipped=set()
        )


def test_build_inputs_raises_on_unknown_ledger_idea() -> None:
    personas = [persona_for("pm_001")]
    ideas = [idea_row(pm_id="pm_001", trade_idea_id="ti_001")]
    ledger = [ledger_row(pm_id="pm_001", trade_idea_id="ti_999")]
    with pytest.raises(PlanError, match="pm_001.*ti_999"):
        build_inputs(personas, [], [], ideas, ledger, [], [], (date(2026, 1, 5),), skipped=set())


def test_build_inputs_raises_on_unknown_rule_event_and_position_day_idea() -> None:
    personas = [persona_for("pm_001")]
    ideas = [idea_row(pm_id="pm_001", trade_idea_id="ti_001")]
    with pytest.raises(PlanError, match="pm_001.*ti_999"):
        build_inputs(
            personas,
            [],
            [],
            ideas,
            [],
            [rule_event(pm_id="pm_001", trade_idea_id="ti_999")],
            [],
            (date(2026, 1, 5),),
            skipped=set(),
        )
    with pytest.raises(PlanError, match="pm_001.*ti_999"):
        build_inputs(
            personas,
            [],
            [],
            ideas,
            [],
            [],
            [position_day(pm_id="pm_001", trade_idea_id="ti_999")],
            (date(2026, 1, 5),),
            skipped=set(),
        )


def persona_for(pm_id: str):
    return persona().model_copy(update={"pm_id": pm_id})
