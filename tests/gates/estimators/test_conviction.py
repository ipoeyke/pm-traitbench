"""Tests for the conviction-inversion estimator."""

from datetime import date

from pm_traitbench.config import Config
from pm_traitbench.gates.gate1.estimators import conviction
from tests.gates.fixtures import DEFAULT_DATE, idea_row

KNOBS = Config().gate1
OTHER_DATE = date(2026, 1, 6)


def _ideas():
    return (
        idea_row(trade_idea_id="ti_001"),
        idea_row(trade_idea_id="ti_002"),
        idea_row(trade_idea_id="ti_003"),
    )


def test_risk_strictly_increasing_with_conviction_gives_zero(make_inputs):
    inputs = make_inputs(
        ideas=_ideas(),
        entry_risk={"ti_001": 1.0, "ti_002": 2.0, "ti_003": 3.0},
        entry_conviction={"ti_001": 1, "ti_002": 3, "ti_003": 5},
    )
    result = conviction.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value == 0.0
    assert result.n == 3


def test_risk_reversed_against_conviction_gives_two(make_inputs):
    inputs = make_inputs(
        ideas=_ideas(),
        entry_risk={"ti_001": 1.0, "ti_002": 2.0, "ti_003": 3.0},
        entry_conviction={"ti_001": 5, "ti_002": 3, "ti_003": 1},
    )
    result = conviction.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value == 2.0
    assert result.n == 3


def test_constant_conviction_gives_none(make_inputs):
    inputs = make_inputs(
        ideas=_ideas(),
        entry_risk={"ti_001": 1.0, "ti_002": 2.0, "ti_003": 3.0},
        entry_conviction={"ti_001": 3, "ti_002": 3, "ti_003": 3},
    )
    result = conviction.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value is None
    assert result.n == 3


def test_two_ideas_gives_none(make_inputs):
    ideas = (idea_row(trade_idea_id="ti_001"), idea_row(trade_idea_id="ti_002"))
    inputs = make_inputs(
        ideas=ideas,
        entry_risk={"ti_001": 1.0, "ti_002": 2.0},
        entry_conviction={"ti_001": 1, "ti_002": 3},
    )
    result = conviction.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value is None
    assert result.n == 2


def test_restricting_days_drops_ideas_outside_it(make_inputs):
    ideas = (
        idea_row(trade_idea_id="ti_001"),
        idea_row(trade_idea_id="ti_002"),
        idea_row(trade_idea_id="ti_003", entry_date=OTHER_DATE),
    )
    inputs = make_inputs(
        ideas=ideas,
        entry_risk={"ti_001": 1.0, "ti_002": 2.0, "ti_003": 3.0},
        entry_conviction={"ti_001": 1, "ti_002": 3, "ti_003": 5},
    )
    result = conviction.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 2
    assert result.value is None


def test_idea_with_no_entry_conviction_is_skipped(make_inputs):
    ideas = _ideas()
    inputs = make_inputs(
        ideas=ideas,
        entry_risk={"ti_001": 1.0, "ti_002": 2.0, "ti_003": 3.0},
        entry_conviction={"ti_001": 1, "ti_002": 3},
    )
    result = conviction.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 2
    assert result.value is None
