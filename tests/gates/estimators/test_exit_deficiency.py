"""Tests for the exit-deficiency estimator."""

from datetime import date

from pm_traitbench.config import Config
from pm_traitbench.enums import RuleResponse
from pm_traitbench.gates.gate1.estimators import exit_deficiency
from tests.gates.fixtures import DEFAULT_DATE, rule_event

KNOBS = Config().gate1
OTHER_DATE = date(2026, 1, 6)


def test_all_acked_no_action_gives_full_share(make_inputs):
    events = tuple(
        rule_event(rule_id=f"r_0{i}", response=RuleResponse.ACKED_NO_ACTION) for i in range(1, 5)
    )
    inputs = make_inputs(rule_events=events)
    result = exit_deficiency.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value == 1.0
    assert result.n == 4


def test_overridden_events_are_dropped_from_denominator(make_inputs):
    events = (
        rule_event(rule_id="r_01", response=RuleResponse.ACTED),
        rule_event(rule_id="r_02", response=RuleResponse.ADDED),
        rule_event(rule_id="r_03", response=RuleResponse.OVERRIDDEN),
    )
    inputs = make_inputs(rule_events=events)
    result = exit_deficiency.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value == 0.5
    assert result.n == 2


def test_no_events_gives_none(make_inputs):
    inputs = make_inputs(rule_events=())
    result = exit_deficiency.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value is None
    assert result.n == 0


def test_event_outside_days_is_ignored(make_inputs):
    events = (rule_event(rule_id="r_01", date_fired=OTHER_DATE, response_date=OTHER_DATE),)
    inputs = make_inputs(rule_events=events)
    result = exit_deficiency.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.value is None
    assert result.n == 0


def test_restricting_days_drops_events_outside_it(make_inputs):
    events = (
        rule_event(rule_id="r_01", response=RuleResponse.ACKED_NO_ACTION),
        rule_event(
            rule_id="r_02",
            date_fired=OTHER_DATE,
            response_date=OTHER_DATE,
            response=RuleResponse.ACTED,
        ),
    )
    inputs = make_inputs(rule_events=events)
    result = exit_deficiency.estimate(inputs, frozenset({DEFAULT_DATE}), KNOBS)
    assert result.n == 1
    assert result.value == 1.0
