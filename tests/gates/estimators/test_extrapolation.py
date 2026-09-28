"""Tests for the extrapolation estimator."""

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.enums import AssetClass, Side
from pm_traitbench.gates.gate1.estimators import extrapolation
from tests.gates.fixtures import idea_row

KNOBS = Config().gate1
HORIZON = 20


def _series(fixture_view, instrument_id="EQ-0004"):
    adapter = adapter_for(AssetClass.EQUITIES, "long_short", HORIZON)
    return adapter.outright_series(instrument_id)


def _run_up_date(fixture_view, series):
    """A date whose trailing 20-day move on EQ-0004 already exceeds one sd."""
    for day in fixture_view.dates:
        t = fixture_view.dates.index(day)
        if fixture_view.trailing_move(series, t, HORIZON) > fixture_view.sd_h(series, t, HORIZON):
            return day
    raise AssertionError("no run-up date found in fixture")


def test_buy_after_a_run_up_counts_sell_does_not(make_inputs, fixture_view):
    series = _series(fixture_view)
    entry_date = _run_up_date(fixture_view, series)
    t = {d: i for i, d in enumerate(fixture_view.dates)}[entry_date]
    trailing = fixture_view.trailing_move(series, t, HORIZON)
    sd = fixture_view.sd_h(series, t, HORIZON)
    assert trailing > sd  # sanity: confirm the fixture date is a run-up

    buy_idea = idea_row(
        trade_idea_id="ti_001", instrument_id="EQ-0004", side=Side.BUY, entry_date=entry_date
    )
    sell_idea = idea_row(
        trade_idea_id="ti_002", instrument_id="EQ-0004", side=Side.SELL, entry_date=entry_date
    )
    inputs = make_inputs(ideas=(buy_idea, sell_idea), series={"ti_001": series, "ti_002": series})
    days = frozenset({entry_date})
    result = extrapolation.estimate(inputs, days, KNOBS)
    assert result.n == 2
    assert result.value == 0.5
    assert extrapolation.after_run_count(inputs, days) == 1


def test_pairs_has_one_entry_per_idea(make_inputs, fixture_view):
    series = _series(fixture_view)
    entry_date = _run_up_date(fixture_view, series)
    idea = idea_row(
        trade_idea_id="ti_001", instrument_id="EQ-0004", side=Side.BUY, entry_date=entry_date
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series})
    t = {d: i for i, d in enumerate(fixture_view.dates)}[entry_date]
    expected_ratio = fixture_view.trailing_move(series, t, HORIZON) / fixture_view.sd_h(
        series, t, HORIZON
    )
    result = extrapolation.estimate(inputs, frozenset({entry_date}), KNOBS)
    assert len(result.pairs) == 1
    direction, ratio = result.pairs[0]
    assert direction == 1.0
    assert ratio == expected_ratio


def test_entries_outside_days_are_ignored(make_inputs, fixture_view):
    series = _series(fixture_view)
    entry_date = _run_up_date(fixture_view, series)
    other_date = fixture_view.dates[0]
    idea = idea_row(
        trade_idea_id="ti_001", instrument_id="EQ-0004", side=Side.BUY, entry_date=entry_date
    )
    inputs = make_inputs(ideas=(idea,), series={"ti_001": series})
    result = extrapolation.estimate(inputs, frozenset({other_date}), KNOBS)
    assert result.n == 0
    assert result.value is None
    assert extrapolation.after_run_count(inputs, frozenset({other_date})) == 0
