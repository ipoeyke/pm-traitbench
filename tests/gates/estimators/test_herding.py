"""Tests for the herding estimator."""

from pm_traitbench.config import Config
from pm_traitbench.enums import Side, StreetView
from pm_traitbench.gates.gate1.estimators import herding
from tests.gates.conftest import idea_row

KNOBS = Config().gate1


def test_three_of_four_agreeing_entries_gives_three_quarters_share(make_inputs, fixture_view):
    dates = fixture_view.dates
    # t=1 and t=2: street overweight (score > 0.1); t=3: overweight too; t=11: underweight.
    assert fixture_view.street_view("EQ-0001", 1) == StreetView.OVERWEIGHT
    assert fixture_view.street_view("EQ-0001", 2) == StreetView.OVERWEIGHT
    assert fixture_view.street_view("EQ-0001", 3) == StreetView.OVERWEIGHT
    assert fixture_view.street_view("EQ-0001", 11) == StreetView.UNDERWEIGHT

    ideas = (
        idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_date=dates[1]),  # agrees
        idea_row(trade_idea_id="ti_002", side=Side.BUY, entry_date=dates[2]),  # agrees
        idea_row(trade_idea_id="ti_003", side=Side.SELL, entry_date=dates[3]),  # disagrees
        idea_row(trade_idea_id="ti_004", side=Side.SELL, entry_date=dates[11]),  # agrees
    )
    inputs = make_inputs(ideas=ideas)
    days = frozenset(dates)
    result = herding.estimate(inputs, days, KNOBS)
    assert result.n == 4
    assert result.value == 0.75


def test_neutral_street_and_no_consensus_entries_are_skipped(make_inputs, fixture_view):
    dates = fixture_view.dates
    assert fixture_view.street_view("EQ-0001", 0) == StreetView.NEUTRAL
    assert fixture_view.street_view("RT-USD", 5) is None

    neutral_idea = idea_row(
        trade_idea_id="ti_001", instrument_id="EQ-0001", side=Side.BUY, entry_date=dates[0]
    )
    no_consensus_idea = idea_row(
        trade_idea_id="ti_002", instrument_id="RT-USD", side=Side.BUY, entry_date=dates[5]
    )
    inputs = make_inputs(ideas=(neutral_idea, no_consensus_idea))
    result = herding.estimate(inputs, frozenset(dates), KNOBS)
    assert result.n == 0
    assert result.value is None


def test_hidden_conflict_and_followed_street_columns_do_not_change_the_result(
    make_inputs, fixture_view
):
    dates = fixture_view.dates
    idea_a = idea_row(
        trade_idea_id="ti_001",
        side=Side.BUY,
        entry_date=dates[1],
        conflict=True,
        followed_street=False,
    )
    idea_b = idea_row(
        trade_idea_id="ti_002",
        side=Side.BUY,
        entry_date=dates[1],
        conflict=False,
        followed_street=None,
    )
    inputs = make_inputs(ideas=(idea_a,))
    result_a = herding.estimate(inputs, frozenset(dates), KNOBS)
    inputs_b = make_inputs(ideas=(idea_b,))
    result_b = herding.estimate(inputs_b, frozenset(dates), KNOBS)
    assert result_a.value == result_b.value == 1.0
    assert result_a.n == result_b.n == 1


def test_entries_outside_days_are_ignored(make_inputs, fixture_view):
    dates = fixture_view.dates
    idea = idea_row(trade_idea_id="ti_001", side=Side.BUY, entry_date=dates[1])
    inputs = make_inputs(ideas=(idea,))
    result = herding.estimate(inputs, frozenset({dates[0]}), KNOBS)
    assert result.n == 0
    assert result.value is None
