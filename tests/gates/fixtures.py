"""Shared fixtures for gate 1 tests: a `PmInputs` factory and small row builders.

`make_inputs` builds a `PmInputs` on top of the engine fixture market view
(`fixture_view`), with empty rows and neutral traits by default, so a test
states only the fields it cares about. The row builders return one sensibly
defaulted row each, again so a test overrides only what matters.
"""

from datetime import date

import pytest

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.step import _OPPORTUNITY_KEYS
from pm_traitbench.enums import (
    AssetClass,
    Expression,
    InstrumentKind,
    Kind,
    PnlState,
    PositionAction,
    RuleResponse,
    Side,
    StreetView,
)
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.tables.schema import Idea, LedgerRow, Leg, PositionDay, RuleEvent, Trait

PM_ID = "pm_001"
DEFAULT_DATE = date(2026, 1, 5)
DEFAULT_IDEA_ID = "ti_001"


def idea_row(**overrides) -> Idea:
    """A single-leg outright idea entered on `DEFAULT_DATE`, overridable by keyword."""
    fields = dict(
        pm_id=PM_ID,
        trade_idea_id=DEFAULT_IDEA_ID,
        instrument_id="EQ-0001",
        expression=Expression.OUTRIGHT,
        side=Side.BUY,
        legs=(Leg(instrument_id="EQ-0001", tenor=None, side=Side.BUY, weight=1.0),),
        entry_date=DEFAULT_DATE,
        exit_date=None,
        entry_level=100.0,
        target_level=110.0,
        stop_level=90.0,
        thesis="thesis",
        outcome=None,
        own_signal=0.1,
        forecast=105.0,
        interval_lo=95.0,
        interval_hi=115.0,
        street_view_at_entry=StreetView.NEUTRAL,
        conflict=False,
        followed_street=None,
        conviction=3,
        size_rank=3,
        chased_trend=False,
    )
    fields.update(overrides)
    return Idea(**fields)


def ledger_row(**overrides) -> LedgerRow:
    """A single buy order for `DEFAULT_IDEA_ID` on `DEFAULT_DATE`, overridable by keyword."""
    fields = dict(
        pm_id=PM_ID,
        date=DEFAULT_DATE,
        trade_idea_id=DEFAULT_IDEA_ID,
        instrument_id="EQ-0001",
        tenor=None,
        instrument_type=InstrumentKind.EQUITY,
        side=Side.BUY,
        size=100.0,
        risk_amount=1.0,
        price_or_yield=100.0,
        stated_conviction=3,
        bias_flag=None,
        rule_id=None,
    )
    fields.update(overrides)
    return LedgerRow(**fields)


def rule_event(**overrides) -> RuleEvent:
    """A rule firing and being acked with no action, both dated `DEFAULT_DATE`."""
    fields = dict(
        pm_id=PM_ID,
        rule_id="r_01",
        trade_idea_id=DEFAULT_IDEA_ID,
        date_fired=DEFAULT_DATE,
        response=RuleResponse.ACKED_NO_ACTION,
        response_date=DEFAULT_DATE,
    )
    fields.update(overrides)
    return RuleEvent(**fields)


def position_day(**overrides) -> PositionDay:
    """A held, non-flat position-day snapshot on `DEFAULT_DATE`, overridable by keyword."""
    fields = dict(
        pm_id=PM_ID,
        date=DEFAULT_DATE,
        trade_idea_id=DEFAULT_IDEA_ID,
        pnl_unit=1.0,
        pnl_z=1.0,
        pnl_state=PnlState.GAIN,
        sessions_held=1,
        triggers_fired=0,
        trigger_pending=False,
        action=PositionAction.HOLD,
        bias_flag=None,
        anchor_level=None,
        effective_exit_level=None,
    )
    fields.update(overrides)
    return PositionDay(**fields)


def _neutral_traits() -> dict[str, Trait]:
    return {
        param: Trait(
            pm_id=PM_ID,
            trait_id=f"t_{i:02d}",
            kind=Kind.BIAS,
            param=param,
            value=0.3,
            active=False,
            mult_range=1.0,
            mult_risk_off=1.0,
            mult_risk_on=1.0,
        )
        for i, param in enumerate(BIAS_PARAMS, start=1)
    }


@pytest.fixture
def make_inputs(fixture_view):
    """Factory: a `PmInputs` on `fixture_view`, empty rows and neutral traits by default."""

    def _make(**overrides) -> PmInputs:
        day_index = {day: t for t, day in enumerate(fixture_view.dates)}
        defaults = dict(
            pm_id=PM_ID,
            seed=fixture_view.seed,
            asset_class=AssetClass.EQUITIES,
            is_real_seed=False,
            traits=_neutral_traits(),
            drift_dates={param: () for param in BIAS_PARAMS},
            ideas=(),
            series={},
            entry_risk={},
            entry_conviction={},
            position_days=(),
            rule_events=(),
            sell_dates=frozenset(),
            acted=frozenset(),
            view=fixture_view,
            day_index=day_index,
            horizon_days=20,
            engine_counts={key: 0 for key in _OPPORTUNITY_KEYS},
        )
        defaults.update(overrides)
        return PmInputs(**defaults)

    return _make
