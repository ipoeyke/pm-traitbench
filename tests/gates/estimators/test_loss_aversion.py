"""Tests for the loss-aversion estimator."""

from datetime import date

from pm_traitbench.config import Config
from pm_traitbench.enums import PnlState, PositionAction
from pm_traitbench.gates.gate1.estimators import loss_aversion
from tests.gates.fixtures import DEFAULT_DATE, position_day

KNOBS = Config().gate1
OTHER_DATE = date(2026, 1, 6)


def test_add_on_one_of_four_qualifying_rows_gives_quarter_share(make_inputs):
    rows = (
        position_day(trade_idea_id="ti_001", pnl_state=PnlState.LOSS, action=PositionAction.ADD),
        position_day(trade_idea_id="ti_002", pnl_state=PnlState.LOSS, action=PositionAction.HOLD),
        position_day(trade_idea_id="ti_003", pnl_state=PnlState.LOSS, action=PositionAction.HOLD),
        position_day(trade_idea_id="ti_004", pnl_state=PnlState.LOSS, action=PositionAction.CUT),
    )
    inputs = make_inputs(position_days=rows)
    days = frozenset(inputs.view.dates)
    result = loss_aversion.estimate(inputs, days, KNOBS)
    assert result.value == 0.25
    assert result.n == 4


def test_trigger_pending_roll_gain_and_last_date_rows_are_excluded(make_inputs):
    rows = (
        position_day(trade_idea_id="ti_001", pnl_state=PnlState.LOSS, action=PositionAction.HOLD),
        position_day(
            trade_idea_id="ti_002",
            pnl_state=PnlState.LOSS,
            trigger_pending=True,
        ),
        position_day(trade_idea_id="ti_003", pnl_state=PnlState.LOSS, action=PositionAction.ROLL),
        position_day(trade_idea_id="ti_004", pnl_state=PnlState.GAIN),
    )
    inputs = make_inputs(position_days=rows)
    last_date_row = position_day(
        trade_idea_id="ti_005", date=inputs.last_date, pnl_state=PnlState.LOSS
    )
    inputs = make_inputs(position_days=rows + (last_date_row,))
    result = loss_aversion.estimate(inputs, frozenset(inputs.view.dates), KNOBS)
    assert result.n == 1
    assert result.value == 0.0


def test_opportunities_restricted_by_days(make_inputs):
    rows = (
        position_day(trade_idea_id="ti_001", pnl_state=PnlState.LOSS),
        position_day(trade_idea_id="ti_002", date=OTHER_DATE, pnl_state=PnlState.LOSS),
    )
    inputs = make_inputs(position_days=rows)
    days = frozenset({DEFAULT_DATE})
    result = loss_aversion.estimate(inputs, days, KNOBS)
    assert result.n == 1
    assert loss_aversion.opportunities(inputs, days) == [rows[0]]


def test_no_rows_gives_none(make_inputs):
    inputs = make_inputs(position_days=())
    result = loss_aversion.estimate(inputs, frozenset(inputs.view.dates), KNOBS)
    assert result.value is None
    assert result.n == 0


def test_no_add_rule_excludes_untriggered_rows_from_the_estimate(make_inputs):
    rows = (
        position_day(trade_idea_id="ti_001", pnl_state=PnlState.LOSS, action=PositionAction.ADD),
        position_day(trade_idea_id="ti_002", pnl_state=PnlState.LOSS, action=PositionAction.HOLD),
        position_day(
            trade_idea_id="ti_003",
            pnl_state=PnlState.LOSS,
            action=PositionAction.ADD,
            triggers_fired=1,
        ),
        position_day(
            trade_idea_id="ti_004",
            pnl_state=PnlState.LOSS,
            action=PositionAction.HOLD,
            triggers_fired=1,
        ),
    )
    inputs = make_inputs(position_days=rows, no_add_before_trigger=True)
    days = frozenset(inputs.view.dates)
    result = loss_aversion.estimate(inputs, days, KNOBS)
    assert result.n == 2
    assert result.value == 0.5
    # The engine-matched opportunity set still counts every loss-side row.
    assert len(loss_aversion.opportunities(inputs, days)) == 4
