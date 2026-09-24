"""Conviction inversion: one minus the rank correlation of entry sizing and stated conviction."""

from datetime import date

from scipy.stats import spearmanr

from pm_traitbench.config import Gate1Config
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """One minus the Spearman rank correlation of entry risk against stated conviction.

    Stated conviction is read from `inputs.entry_conviction`, which comes from
    an idea's first entry-date ledger row; an idea with no such row is skipped.
    `None` when fewer than 3 ideas remain or either sequence is constant.
    """
    risk = []
    conviction_values = []
    for idea in inputs.ideas:
        if idea.entry_date not in days:
            continue
        stated = inputs.entry_conviction.get(idea.trade_idea_id)
        if stated is None:
            continue
        risk.append(inputs.entry_risk[idea.trade_idea_id])
        conviction_values.append(stated)

    n = len(risk)
    if n < 3 or len(set(risk)) == 1 or len(set(conviction_values)) == 1:
        return Estimate(value=None, n=n)
    rho = spearmanr(risk, conviction_values).statistic
    return Estimate(value=1 - rho, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
