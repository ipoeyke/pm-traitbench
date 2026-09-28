"""Tests for the overconfidence estimator."""

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.enums import AssetClass
from pm_traitbench.gates.gate1.estimators import overconfidence
from tests.gates.fixtures import idea_row

KNOBS = Config().gate1
HORIZON = 20


def _series(fixture_view, instrument_id):
    adapter = adapter_for(AssetClass.EQUITIES, "long_short", HORIZON)
    return adapter.outright_series(instrument_id)


def test_two_of_five_realised_moves_inside_gives_two_fifths(make_inputs, fixture_view):
    dates = fixture_view.dates
    ideas = (
        idea_row(
            trade_idea_id="ti_001",
            instrument_id="EQ-0001",
            entry_date=dates[0],
            interval_lo=-15.0,
            interval_hi=-10.0,
        ),  # forward move ~ -12.3: inside
        idea_row(
            trade_idea_id="ti_002",
            instrument_id="EQ-0002",
            entry_date=dates[0],
            interval_lo=-1.0,
            interval_hi=1.0,
        ),  # forward move ~ -3.3: outside
        idea_row(
            trade_idea_id="ti_003",
            instrument_id="EQ-0003",
            entry_date=dates[0],
            interval_lo=-8.0,
            interval_hi=-6.0,
        ),  # forward move ~ -7.4: inside
        idea_row(
            trade_idea_id="ti_004",
            instrument_id="EQ-0004",
            entry_date=dates[0],
            interval_lo=10.0,
            interval_hi=20.0,
        ),  # forward move ~ 7.0: outside
        idea_row(
            trade_idea_id="ti_005",
            instrument_id="EQ-0001",
            entry_date=dates[5],
            interval_lo=0.0,
            interval_hi=5.0,
        ),  # forward move ~ -10.7: outside
    )
    series = {idea.trade_idea_id: _series(fixture_view, idea.instrument_id) for idea in ideas}
    inputs = make_inputs(ideas=ideas, series=series)
    result = overconfidence.estimate(inputs, frozenset(dates), KNOBS)
    assert result.n == 5
    assert result.value == 0.4


def test_entries_outside_days_are_ignored(make_inputs, fixture_view):
    dates = fixture_view.dates
    idea = idea_row(
        trade_idea_id="ti_001",
        instrument_id="EQ-0001",
        entry_date=dates[0],
        interval_lo=-15.0,
        interval_hi=-10.0,
    )
    series = {"ti_001": _series(fixture_view, "EQ-0001")}
    inputs = make_inputs(ideas=(idea,), series=series)
    result = overconfidence.estimate(inputs, frozenset({dates[5]}), KNOBS)
    assert result.n == 0
    assert result.value is None
