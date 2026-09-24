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


def test_exit_at_anchor_is_inside_exit_at_target_is_outside(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    inputs_probe = make_inputs(series={"ti_001": series})
    t = inputs_probe.day_index[DEFAULT_DATE]
    band = KNOBS.anchor_band_k * fixture_view.sd_h(series, t, HORIZON)

    idea_inside = idea_row(
        trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0
    )
    row_inside = position_day(
        trade_idea_id="ti_001",
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    idea_outside = idea_row(
        trade_idea_id="ti_002", side=Side.BUY, entry_level=100.0, target_level=110.0
    )
    row_outside = position_day(
        trade_idea_id="ti_002",
        action=PositionAction.EXIT,
        pnl_unit=10.0,
        anchor_level=110.0 - band - 5.0,
        effective_exit_level=110.0 - band - 5.0,
    )
    inputs = make_inputs(
        ideas=(idea_inside, idea_outside),
        series={"ti_001": series, "ti_002": series},
        position_days=(row_inside, row_outside),
    )
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 2
    assert result.value == 0.5


def test_anchor_equal_to_target_is_not_counted(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    idea_skipped = idea_row(
        trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0
    )
    row_skipped = position_day(
        trade_idea_id="ti_001",
        action=PositionAction.EXIT,
        pnl_unit=10.0,
        anchor_level=110.0,
        effective_exit_level=110.0,
    )
    idea_counted = idea_row(
        trade_idea_id="ti_002", side=Side.BUY, entry_level=100.0, target_level=110.0
    )
    row_counted = position_day(
        trade_idea_id="ti_002",
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    inputs = make_inputs(
        ideas=(idea_skipped, idea_counted),
        series={"ti_001": series, "ti_002": series},
        position_days=(row_skipped, row_counted),
    )
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 1
    assert result.value == 1.0


def test_acted_rule_event_excludes_the_exit(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    idea = idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_level=100.0, target_level=110.0)
    row = position_day(
        trade_idea_id="ti_001",
        action=PositionAction.EXIT,
        pnl_unit=3.0,
        anchor_level=103.0,
        effective_exit_level=103.0,
    )
    inputs = make_inputs(
        ideas=(idea,),
        series={"ti_001": series},
        position_days=(row,),
        acted=frozenset({("ti_001", DEFAULT_DATE)}),
    )
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 0
    assert result.value is None


def test_last_date_row_is_not_counted(make_inputs, fixture_view):
    series = _equity_series(fixture_view)
    idea = idea_row(
        trade_idea_id="ti_001",
        side=Side.BUY,
        entry_level=100.0,
        target_level=110.0,
        entry_date=fixture_view.dates[0],
    )
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
    result = anchoring.estimate(inputs, frozenset({last_date}), KNOBS)
    assert result.n == 0
    assert result.value is None


def test_sell_side_bearish_series_hand_checked_exit_level(make_inputs, fixture_view):
    adapter = adapter_for(AssetClass.RATES_CREDIT, "sovereign_rates", HORIZON)
    series = adapter.outright_series("RT-USD")
    assert series.bullish_sign == -1

    idea = idea_row(
        trade_idea_id="ti_001",
        instrument_id="RT-USD",
        side=Side.SELL,
        entry_level=100.0,
        target_level=80.0,
    )
    # bullish_sign=-1, side_sign=-1 (sell): exit_level = 100 + (-1)*(-1)*5 = 105.
    row = position_day(
        trade_idea_id="ti_001",
        action=PositionAction.EXIT,
        pnl_unit=5.0,
        anchor_level=105.0,
        effective_exit_level=105.0,
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series}, position_days=(row,))
    result = anchoring.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 1
    assert result.value == 1.0
