"""Tests for the anchoring estimator."""

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.enums import AssetClass, PositionAction, Side
from pm_traitbench.gates.gate1.estimators import anchoring
from tests.gates.conftest import DEFAULT_DATE, idea_row, position_day

KNOBS = Config().gate1
HORIZON = 20


def _equity_series(fixture_view, instrument_id="EQ-0001"):
    adapter = adapter_for(AssetClass.EQUITIES, "long_short", HORIZON)
    return adapter.outright_series(instrument_id)


def test_crossing_exited_same_day_is_a_hit_crossing_exited_later_is_not(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    dates = fixture_view.dates

    idea_hit = idea_row(
        trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0
    )
    row_hit = position_day(
        trade_idea_id="ti_001",
        date=dates[1],
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )

    idea_late = idea_row(
        trade_idea_id="ti_002", side=Side.BUY, entry_level=100.0, target_level=110.0
    )
    row_crossing = position_day(
        trade_idea_id="ti_002",
        date=dates[1],
        action=PositionAction.HOLD,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    row_later_exit = position_day(
        trade_idea_id="ti_002",
        date=dates[2],
        action=PositionAction.EXIT,
        pnl_unit=5.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )

    inputs = make_inputs(
        ideas=(idea_hit, idea_late),
        series={"ti_001": series, "ti_002": series},
        position_days=(row_hit, row_crossing, row_later_exit),
    )
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 2
    assert result.value == 0.5


def test_idea_that_never_crosses_is_not_counted(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    dates = fixture_view.dates
    idea = idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0)
    row = position_day(
        trade_idea_id="ti_001",
        date=dates[1],
        action=PositionAction.EXIT,
        pnl_unit=1.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series}, position_days=(row,))
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 0
    assert result.value is None


def test_crossing_on_the_last_horizon_date_is_not_counted(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    idea = idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0)
    inputs_probe = make_inputs()
    last_date = inputs_probe.last_date
    row = position_day(
        trade_idea_id="ti_001",
        date=last_date,
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series}, position_days=(row,))
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 0
    assert result.value is None


def test_acted_rule_event_on_the_crossing_day_is_not_a_hit(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    dates = fixture_view.dates
    idea = idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0)
    row = position_day(
        trade_idea_id="ti_001",
        date=dates[1],
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    inputs = make_inputs(
        ideas=(idea,),
        series={"ti_001": series},
        position_days=(row,),
        acted=frozenset({("ti_001", dates[1])}),
    )
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 1
    assert result.value == 0.0


def test_idea_with_no_anchor_is_skipped(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    dates = fixture_view.dates
    idea = idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0)
    row = position_day(
        trade_idea_id="ti_001",
        date=dates[1],
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=None,
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series}, position_days=(row,))
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 0
    assert result.value is None


def test_sell_side_bearish_series_hand_checked_crossing_direction(make_inputs, fixture_view):
    adapter = adapter_for(AssetClass.RATES_CREDIT, "sovereign_rates", HORIZON)
    series = adapter.outright_series("RT-USD")
    assert series.bullish_sign == -1
    dates = fixture_view.dates

    idea = idea_row(
        trade_idea_id="ti_001",
        instrument_id="RT-USD",
        side=Side.SELL,
        entry_level=100.0,
        target_level=80.0,
    )
    # bullish_sign=-1, side_sign=-1 (sell): level = 100 + (-1)*(-1)*pnl_unit = 100 + pnl_unit.
    # pnl_unit=-10.0 tracks the level down to 90, between entry (100) and target (80).
    row = position_day(
        trade_idea_id="ti_001",
        date=dates[1],
        action=PositionAction.EXIT,
        pnl_unit=-10.0,
        anchor_level=90.0,
        effective_exit_level=90.0,
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series}, position_days=(row,))
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 1
    assert result.value == 1.0
