"""Tests for gate 1's date splits: which splits apply, and which dates each one covers."""

from dataclasses import replace
from datetime import date

import pytest

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import Gate1Split, Regime
from pm_traitbench.gates.gate1.splits import days_for, splits_for
from tests.engine.conftest import fixture_market, fixture_view  # noqa: F401


def _with_multiplier(inputs, param: str, **mults):
    trait = inputs.traits[param].model_copy(update=mults)
    return replace(inputs, traits={**inputs.traits, param: trait})


def _with_drift(inputs, param: str, dates: tuple[date, ...]):
    return replace(inputs, drift_dates={**inputs.drift_dates, param: dates})


def test_splits_for_regime_multiplier_adds_only_that_regime(make_inputs) -> None:
    param = BIAS_PARAMS[0]
    inputs = _with_multiplier(make_inputs(), param, mult_risk_off=1.15)
    assert splits_for(inputs, param) == (Gate1Split.ALL, Gate1Split.REGIME_RISK_OFF)


def test_splits_for_multiplier_at_or_below_one_adds_nothing(make_inputs) -> None:
    param = BIAS_PARAMS[0]
    inputs = _with_multiplier(make_inputs(), param, mult_risk_off=1.0, mult_range=0.5)
    assert splits_for(inputs, param) == (Gate1Split.ALL,)


def test_splits_for_drift_event_adds_before_and_after(make_inputs) -> None:
    param = BIAS_PARAMS[0]
    inputs = _with_drift(make_inputs(), param, (date(2026, 2, 2),))
    assert splits_for(inputs, param) == (Gate1Split.ALL, Gate1Split.BEFORE, Gate1Split.AFTER)


def test_splits_for_order_is_all_then_regimes_then_before_after(make_inputs) -> None:
    param = BIAS_PARAMS[0]
    inputs = _with_multiplier(make_inputs(), param, mult_range=1.2, mult_risk_on=1.3)
    inputs = _with_drift(inputs, param, (date(2026, 2, 2),))
    assert splits_for(inputs, param) == (
        Gate1Split.ALL,
        Gate1Split.REGIME_RANGE,
        Gate1Split.REGIME_RISK_ON,
        Gate1Split.BEFORE,
        Gate1Split.AFTER,
    )


def test_days_for_all_is_every_horizon_date(make_inputs) -> None:
    inputs = make_inputs()
    assert days_for(Gate1Split.ALL, inputs, BIAS_PARAMS[0]) == frozenset(inputs.view.dates)


def test_days_for_regime_matches_view_regime(make_inputs) -> None:
    inputs = make_inputs()
    days = days_for(Gate1Split.REGIME_RANGE, inputs, BIAS_PARAMS[0])
    assert days
    for day in inputs.view.dates:
        expected = inputs.view.regime(inputs.day_index[day]) == Regime.RANGE
        assert (day in days) == expected


def test_days_for_update_event_splits_horizon_at_that_date(make_inputs) -> None:
    param = BIAS_PARAMS[0]
    inputs = make_inputs()
    split_date = inputs.view.dates[10]
    inputs = _with_drift(inputs, param, (split_date,))

    before = days_for(Gate1Split.BEFORE, inputs, param)
    after = days_for(Gate1Split.AFTER, inputs, param)

    assert before == frozenset(d for d in inputs.view.dates if d < split_date)
    assert after == frozenset(d for d in inputs.view.dates if d >= split_date)


def test_days_for_dormant_revive_pair_makes_after_the_dormant_window_only(make_inputs) -> None:
    param = BIAS_PARAMS[0]
    inputs = make_inputs()
    dormant_date = inputs.view.dates[10]
    revive_date = inputs.view.dates[20]
    inputs = _with_drift(inputs, param, (dormant_date, revive_date))

    after = days_for(Gate1Split.AFTER, inputs, param)

    assert after == frozenset(d for d in inputs.view.dates if dormant_date <= d < revive_date)
    assert revive_date not in after


def test_days_for_before_without_drift_raises(make_inputs) -> None:
    inputs = make_inputs()
    with pytest.raises(ValueError, match=BIAS_PARAMS[0]):
        days_for(Gate1Split.BEFORE, inputs, BIAS_PARAMS[0])


def test_days_for_after_without_drift_raises(make_inputs) -> None:
    inputs = make_inputs()
    with pytest.raises(ValueError, match=BIAS_PARAMS[0]):
        days_for(Gate1Split.AFTER, inputs, BIAS_PARAMS[0])
