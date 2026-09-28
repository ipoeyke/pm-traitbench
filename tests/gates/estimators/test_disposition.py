"""Tests for the disposition-effect estimator."""

from datetime import date

import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import PnlState, PositionAction
from pm_traitbench.gates.gate1.estimators import disposition
from tests.gates.fixtures import DEFAULT_DATE, position_day

KNOBS = Config().gate1
SELL_DATE_1 = DEFAULT_DATE
SELL_DATE_2 = date(2026, 1, 6)
NON_SELL_DATE = date(2026, 1, 7)
SELL_DATES = frozenset({SELL_DATE_1, SELL_DATE_2})


def _rows():
    return (
        position_day(
            trade_idea_id="ti_001",
            date=SELL_DATE_1,
            pnl_state=PnlState.GAIN,
            action=PositionAction.EXIT,
        ),
        position_day(
            trade_idea_id="ti_002",
            date=SELL_DATE_1,
            pnl_state=PnlState.GAIN,
            action=PositionAction.HOLD,
        ),
        position_day(
            trade_idea_id="ti_003",
            date=SELL_DATE_1,
            pnl_state=PnlState.GAIN,
            action=PositionAction.HOLD,
        ),
        position_day(
            trade_idea_id="ti_004",
            date=SELL_DATE_2,
            pnl_state=PnlState.LOSS,
            action=PositionAction.CUT,
        ),
        position_day(
            trade_idea_id="ti_005",
            date=SELL_DATE_2,
            pnl_state=PnlState.LOSS,
            action=PositionAction.HOLD,
        ),
        position_day(
            trade_idea_id="ti_006",
            date=SELL_DATE_2,
            pnl_state=PnlState.LOSS,
            action=PositionAction.HOLD,
        ),
        position_day(
            trade_idea_id="ti_007",
            date=SELL_DATE_2,
            pnl_state=PnlState.LOSS,
            action=PositionAction.HOLD,
        ),
        position_day(
            trade_idea_id="ti_008",
            date=SELL_DATE_2,
            pnl_z=0.0,
            pnl_state=PnlState.FLAT,
            action=PositionAction.HOLD,
        ),
    )


def test_pgr_over_plr_ratio(make_inputs):
    rows = _rows()
    inputs = make_inputs(position_days=rows, sell_dates=SELL_DATES)
    days = frozenset(inputs.view.dates)
    result = disposition.estimate(inputs, days, KNOBS)
    assert result.n == 8
    assert result.value == pytest.approx(4 / 3, abs=1e-12)


def test_no_loss_rows_gives_none(make_inputs):
    rows = tuple(row for row in _rows() if row.pnl_state != PnlState.LOSS)
    inputs = make_inputs(position_days=rows, sell_dates=SELL_DATES)
    result = disposition.estimate(inputs, frozenset(inputs.view.dates), KNOBS)
    assert result.value is None


def test_plr_zero_gives_none(make_inputs):
    rows = (
        position_day(
            trade_idea_id="ti_001",
            date=SELL_DATE_1,
            pnl_state=PnlState.GAIN,
            action=PositionAction.EXIT,
        ),
        position_day(
            trade_idea_id="ti_002",
            date=SELL_DATE_2,
            pnl_state=PnlState.LOSS,
            action=PositionAction.HOLD,
        ),
    )
    inputs = make_inputs(position_days=rows, sell_dates=SELL_DATES)
    result = disposition.estimate(inputs, frozenset(inputs.view.dates), KNOBS)
    assert result.value is None


def test_rows_on_non_sell_dates_are_ignored(make_inputs):
    rows = (
        position_day(
            trade_idea_id="ti_001",
            date=NON_SELL_DATE,
            pnl_state=PnlState.GAIN,
            action=PositionAction.EXIT,
        ),
    )
    inputs = make_inputs(position_days=rows, sell_dates=SELL_DATES)
    days = frozenset(inputs.view.dates)
    assert disposition.sell_day_rows(inputs, days) == []
    result = disposition.estimate(inputs, days, KNOBS)
    assert result.n == 0
    assert result.value is None


def test_restricting_days_drops_rows_outside_it(make_inputs):
    rows = _rows()
    inputs = make_inputs(position_days=rows, sell_dates=SELL_DATES)
    days = frozenset({SELL_DATE_1})
    result = disposition.estimate(inputs, days, KNOBS)
    assert result.n == 3
