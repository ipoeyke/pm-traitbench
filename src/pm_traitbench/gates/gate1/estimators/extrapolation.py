"""Extrapolation: the share of entries that chase a trailing move already past one sd."""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.gates.gate1.side_sign import SIDE_SIGN


def _entries(inputs: PmInputs, days: frozenset[date]):
    for idea in inputs.ideas:
        if idea.entry_date not in days:
            continue
        s = inputs.series[idea.trade_idea_id]
        t = inputs.day_index[idea.entry_date]
        direction = s.bullish_sign * SIDE_SIGN[idea.side]
        trailing = inputs.view.trailing_move(s, t, inputs.horizon_days)
        sd = inputs.view.sd_h(s, t, inputs.horizon_days)
        yield idea, direction, trailing, sd


def _after_run(direction: int, trailing: float, sd: float) -> bool:
    """Whether an entry chases a trailing move already past one sd, in bullish units."""
    return direction * trailing > sd


def after_run_count(inputs: PmInputs, days: frozenset[date]) -> int:
    """Number of entries in `days` that chase a trailing move already past one sd."""
    return sum(
        1
        for _, direction, trailing, sd in _entries(inputs, days)
        if _after_run(direction, trailing, sd)
    )


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of entries in `days` that chase a trailing move already past one sd.

    `pairs` carries, per idea, the entry's directional sign and its trailing
    move in units of that horizon's sd, for the calibration check.
    """
    n = 0
    after_run = 0
    pairs = []
    for _, direction, trailing, sd in _entries(inputs, days):
        n += 1
        if _after_run(direction, trailing, sd):
            after_run += 1
        pairs.append((float(direction), trailing / sd))
    if n == 0:
        return Estimate(value=None, n=0)
    return Estimate(value=after_run / n, n=n, pairs=tuple(pairs))


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
