"""Shared builders for signal-plan tests: row builders re-exported from gate 1's fixtures,
plus a persona, trait, drift event and a full `PlanInputs` for a single PM.
"""

from datetime import date, timedelta

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import AssetClass, DriftEventType, Kind, Split, Typicality
from pm_traitbench.signals.inputs import PlanInputs
from pm_traitbench.tables.schema import DriftEvent, Mandate, Persona, StatedProfile, Trait
from tests.gates.fixtures import PM_ID, idea_row, ledger_row, position_day, rule_event  # noqa: F401

_BIAS_TRAIT_IDS: dict[str, str] = {
    param: f"t_{i:02d}" for i, param in enumerate(BIAS_PARAMS, start=1)
}
_PREF_TRAIT_ID = "t_90"


def _weekdays(start: date, n: int) -> tuple[date, ...]:
    days: list[date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


TRADING_DAYS: tuple[date, ...] = _weekdays(date(2026, 1, 5), 260)


def persona(asset_class: AssetClass = AssetClass.EQUITIES) -> Persona:
    """A single PM persona, overridable only by asset class since nothing else here reads it."""
    return Persona(
        pm_id=PM_ID,
        market_seed="T",
        split=Split.PILOT,
        mandate=Mandate(
            asset_class=asset_class,
            sub_style="value",
            book_size=1e8,
            risk_unit="pct_nav",
            benchmark="cash",
        ),
        stated_profile=StatedProfile(self_description="disciplined"),
        typicality=Typicality.TYPICAL,
    )


def bias_trait(
    param: str, *, active: bool = True, value: float | None = None, trait_id: str | None = None
) -> Trait:
    """A bias trait for `param`, active by default, at a fixed neutral value."""
    return Trait(
        pm_id=PM_ID,
        trait_id=trait_id or _BIAS_TRAIT_IDS[param],
        kind=Kind.BIAS,
        param=param,
        value=0.7 if value is None else value,
        active=active,
        mult_range=1.0,
        mult_risk_off=1.0,
        mult_risk_on=1.0,
    )


def pref_trait(param: str, value: str, *, trait_id: str = _PREF_TRAIT_ID) -> Trait:
    """A preference trait holding one of the shipped catalogue's values for `param`."""
    return Trait(
        pm_id=PM_ID,
        trait_id=trait_id,
        kind=Kind.PREFERENCE,
        param=param,
        value=value,
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )


def drift_event(
    trait_id: str,
    day: date,
    event: DriftEventType,
    from_value: float | str | None = None,
    to_value: float | str | None = None,
) -> DriftEvent:
    """A drift event for `trait_id` on `day`, overridable by keyword for `update` events."""
    return DriftEvent(
        pm_id=PM_ID,
        date=day,
        event=event,
        trait_id=trait_id,
        from_value=from_value,
        to_value=to_value,
    )


def plan_inputs(**overrides) -> PlanInputs:
    """A `PlanInputs` for one equities PM: eight neutral inactive biases, no preferences,
    no rows, and 260 weekdays starting 2026-01-05 as its trading days.
    """
    defaults = dict(
        persona=persona(),
        traits=tuple(bias_trait(param, active=False) for param in BIAS_PARAMS),
        drift_events=(),
        ideas={},
        ledger=(),
        rule_events=(),
        position_days=(),
        trading_days=TRADING_DAYS,
    )
    defaults.update(overrides)
    return PlanInputs(**defaults)


@pytest.fixture
def plan_config():
    return Config().plan
