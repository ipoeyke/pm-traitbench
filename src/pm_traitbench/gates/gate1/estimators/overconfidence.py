"""Overconfidence: the share of entries whose realised move lands inside the stated interval.

A narrower stated interval that still claims the same coverage traps fewer
realised outcomes, so a lower inside-share means more overconfidence: this is
the one estimator where a higher recovered value means a weaker bias.
"""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of entries in `days` whose forward move lands within `[interval_lo, interval_hi]`."""
    n = 0
    inside = 0
    for idea in inputs.ideas:
        if idea.entry_date not in days:
            continue
        s = inputs.series[idea.trade_idea_id]
        t = inputs.day_index[idea.entry_date]
        realised = inputs.view.forward_move(s, t, inputs.horizon_days)
        n += 1
        if idea.interval_lo <= realised <= idea.interval_hi:
            inside += 1
    if n == 0:
        return Estimate(value=None, n=0)
    return Estimate(value=inside / n, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=False)
