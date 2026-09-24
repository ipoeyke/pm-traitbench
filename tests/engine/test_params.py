"""Tests for the effective bias parameter schedule: drift events and regime multipliers."""

import math
from datetime import date

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.params import ParamSchedule
from pm_traitbench.enums import DriftEventType, Kind, Regime
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import DriftEvent, Trait

_PM_ID = "pm_001"
_DAY = date(2026, 2, 1)


def _trait(
    param: str,
    value: float,
    *,
    trait_id: str = "t_01",
    active: bool = False,
    mult_range: float = 1.0,
    mult_risk_off: float = 1.0,
    mult_risk_on: float = 1.0,
) -> Trait:
    return Trait(
        pm_id=_PM_ID,
        trait_id=trait_id,
        kind=Kind.BIAS,
        param=param,
        value=value,
        active=active,
        mult_range=mult_range,
        mult_risk_off=mult_risk_off,
        mult_risk_on=mult_risk_on,
    )


def _all_bias_traits() -> list[Trait]:
    return [
        _trait(param, 0.3, trait_id=f"t_{i:02d}") for i, param in enumerate(BIAS_PARAMS, start=1)
    ]


def _drift(
    trait_id: str,
    day: date,
    event: DriftEventType,
    *,
    from_value: float | None = None,
    to_value: float | None = None,
) -> DriftEvent:
    return DriftEvent(
        pm_id=_PM_ID,
        date=day,
        event=event,
        trait_id=trait_id,
        **{"from": from_value, "to": to_value},
    )


def test_no_events_gives_base_values(engine_config: Config) -> None:
    traits = [
        _trait("loss_aversion_lambda", 1.4, trait_id="t_01", active=True),
        _trait("herding_weight", 0.2, trait_id="t_02", active=False),
    ]
    schedule = ParamSchedule.build(traits, [], engine_config)
    result = schedule.for_day(_DAY, Regime.RANGE)

    assert result.value("loss_aversion_lambda") == pytest.approx(1.4)
    assert result.value("herding_weight") == pytest.approx(0.2)
    assert result.is_active("loss_aversion_lambda") is True
    assert result.is_active("herding_weight") is False


def test_update_event_applies_from_its_date_not_before(engine_config: Config) -> None:
    trait = _trait("loss_aversion_lambda", 1.1, trait_id="t_01", active=True)
    event = _drift("t_01", date(2026, 2, 10), DriftEventType.UPDATE, from_value=1.1, to_value=2.6)
    schedule = ParamSchedule.build([trait], [event], engine_config)

    before = schedule.for_day(date(2026, 2, 9), Regime.RANGE)
    on_date = schedule.for_day(date(2026, 2, 10), Regime.RANGE)
    after = schedule.for_day(date(2026, 2, 20), Regime.RANGE)

    assert before.value("loss_aversion_lambda") == pytest.approx(1.1)
    assert on_date.value("loss_aversion_lambda") == pytest.approx(2.6)
    assert after.value("loss_aversion_lambda") == pytest.approx(2.6)


def test_dormant_sets_neutral_median_and_revive_restores(engine_config: Config) -> None:
    trait = _trait("herding_weight", 0.5, trait_id="t_01", active=True)
    neutral_median = engine_config.biases.params["herding_weight"].neutral.median_value()
    dormant_event = _drift("t_01", date(2026, 2, 5), DriftEventType.DORMANT)
    revive_event = _drift("t_01", date(2026, 2, 15), DriftEventType.REVIVE)
    schedule = ParamSchedule.build([trait], [dormant_event, revive_event], engine_config)

    before = schedule.for_day(date(2026, 2, 4), Regime.RANGE)
    dormant = schedule.for_day(date(2026, 2, 10), Regime.RANGE)
    revived = schedule.for_day(date(2026, 2, 20), Regime.RANGE)

    assert before.value("herding_weight") == pytest.approx(0.5)
    assert dormant.value("herding_weight") == pytest.approx(neutral_median)
    assert revived.value("herding_weight") == pytest.approx(0.5)


def test_multiplier_of_one_is_identity_on_every_param(engine_config: Config) -> None:
    traits = _all_bias_traits()
    schedule = ParamSchedule.build(traits, [], engine_config)

    for regime in Regime:
        result = schedule.for_day(_DAY, regime)
        for trait in traits:
            assert result.value(trait.param) == pytest.approx(trait.value, abs=1e-6)


def test_multiplier_on_herding_weight_uses_logit_scale(engine_config: Config) -> None:
    trait = _trait("herding_weight", 0.5, trait_id="t_01", active=True, mult_risk_on=1.3)
    schedule = ParamSchedule.build([trait], [], engine_config)
    result = schedule.for_day(_DAY, Regime.RISK_ON)

    expected = 1.0 / (1.0 + math.exp(-math.log(1.3)))
    assert result.value("herding_weight") == pytest.approx(expected, abs=1e-9)
    assert result.value("herding_weight") == pytest.approx(0.565, abs=1e-3)


def test_multiplier_on_herding_weight_stays_below_one_near_ceiling(engine_config: Config) -> None:
    trait = _trait("herding_weight", 0.99, trait_id="t_01", active=True, mult_risk_on=1.3)
    schedule = ParamSchedule.build([trait], [], engine_config)
    result = schedule.for_day(_DAY, Regime.RISK_ON)

    assert result.value("herding_weight") < 1.0


def test_multiplier_on_overconfidence_coverage_lowers_it(engine_config: Config) -> None:
    trait = _trait("overconfidence_coverage", 0.4, trait_id="t_01", active=True, mult_risk_on=1.3)
    schedule = ParamSchedule.build([trait], [], engine_config)
    result = schedule.for_day(_DAY, Regime.RISK_ON)

    assert result.value("overconfidence_coverage") < 0.4


def test_multiplier_on_loss_aversion_lambda_multiplies_plainly(engine_config: Config) -> None:
    trait = _trait("loss_aversion_lambda", 2.0, trait_id="t_01", active=True, mult_risk_on=1.3)
    schedule = ParamSchedule.build([trait], [], engine_config)
    result = schedule.for_day(_DAY, Regime.RISK_ON)

    assert result.value("loss_aversion_lambda") == pytest.approx(2.6)


def test_multiplier_on_disposition_ratio_multiplies_plainly(engine_config: Config) -> None:
    trait = _trait("disposition_ratio", 1.2, trait_id="t_01", active=True, mult_risk_on=1.3)
    schedule = ParamSchedule.build([trait], [], engine_config)
    result = schedule.for_day(_DAY, Regime.RISK_ON)

    assert result.value("disposition_ratio") == pytest.approx(1.56)


def test_build_rejects_drift_event_for_a_different_pm(engine_config: Config) -> None:
    trait = _trait("loss_aversion_lambda", 1.1, trait_id="t_01", active=True)
    other_pm_event = DriftEvent(
        pm_id="pm_002",
        date=date(2026, 2, 5),
        event=DriftEventType.UPDATE,
        trait_id="t_01",
        **{"from": 1.1, "to": 2.0},
    )

    with pytest.raises(EngineError):
        ParamSchedule.build([trait], [other_pm_event], engine_config)


def test_build_rejects_traits_for_more_than_one_pm(engine_config: Config) -> None:
    trait_a = _trait("loss_aversion_lambda", 1.1, trait_id="t_01", active=True)
    trait_b = Trait(
        pm_id="pm_002",
        trait_id="t_02",
        kind=Kind.BIAS,
        param="herding_weight",
        value=0.3,
        active=True,
        mult_range=1.0,
        mult_risk_off=1.0,
        mult_risk_on=1.0,
    )

    with pytest.raises(EngineError):
        ParamSchedule.build([trait_a, trait_b], [], engine_config)


def test_revive_without_matching_dormant_raises(engine_config: Config) -> None:
    trait = _trait("herding_weight", 0.5, trait_id="t_01", active=True)
    revive_event = _drift("t_01", date(2026, 2, 5), DriftEventType.REVIVE)
    schedule = ParamSchedule.build([trait], [revive_event], engine_config)

    with pytest.raises(EngineError, match="t_01"):
        schedule.for_day(date(2026, 2, 10), Regime.RANGE)


def test_second_dormant_does_not_overwrite_stashed_value(engine_config: Config) -> None:
    # If a second dormant re-stashed the (already dormant) current value, revive
    # would restore the neutral median instead of the true pre-dormant value.
    trait = _trait("herding_weight", 0.5, trait_id="t_01", active=True)
    first_dormant = _drift("t_01", date(2026, 2, 3), DriftEventType.DORMANT)
    second_dormant = _drift("t_01", date(2026, 2, 4), DriftEventType.DORMANT)
    revive_event = _drift("t_01", date(2026, 2, 20), DriftEventType.REVIVE)
    schedule = ParamSchedule.build(
        [trait], [first_dormant, second_dormant, revive_event], engine_config
    )

    revived = schedule.for_day(date(2026, 2, 25), Regime.RANGE)

    assert revived.value("herding_weight") == pytest.approx(0.5)


def test_event_on_inactive_param_still_applies_value_activity_unchanged(
    engine_config: Config,
) -> None:
    trait = _trait("exit_deficiency", 0.05, trait_id="t_01", active=False)
    event = _drift("t_01", date(2026, 1, 20), DriftEventType.UPDATE, from_value=0.05, to_value=0.4)
    schedule = ParamSchedule.build([trait], [event], engine_config)
    result = schedule.for_day(_DAY, Regime.RANGE)

    assert result.is_active("exit_deficiency") is False
    assert result.value("exit_deficiency") == pytest.approx(0.4)


def test_effective_params_unknown_param_raises_engine_error(engine_config: Config) -> None:
    trait = _trait("herding_weight", 0.3, trait_id="t_01", active=True)
    schedule = ParamSchedule.build([trait], [], engine_config)
    result = schedule.for_day(_DAY, Regime.RANGE)

    with pytest.raises(EngineError):
        result.value("not_a_real_param")
    with pytest.raises(EngineError):
        result.is_active("not_a_real_param")
