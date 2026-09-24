"""Exit deficiency: the share of fired rules a PM leaves unacted on or adds to."""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import RuleResponse
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs

_INACTION = frozenset({RuleResponse.ACKED_NO_ACTION, RuleResponse.ADDED})


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of non-overridden rule firings in `days` acked with no action or added to.

    The denominator excludes `overridden` responses, since an override is the PM
    acting through the rule rather than on it.
    """
    events = [
        event
        for event in inputs.rule_events
        if event.date_fired in days and event.response != RuleResponse.OVERRIDDEN
    ]
    n = len(events)
    if n == 0:
        return Estimate(value=None, n=0)
    inaction = sum(1 for event in events if event.response in _INACTION)
    return Estimate(value=inaction / n, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
