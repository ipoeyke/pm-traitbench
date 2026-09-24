"""Extrapolation: the share of entries that chase a trailing move already past one sd."""

from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import Side
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs

_SIDE_SIGN: dict[Side, int] = {Side.BUY: 1, Side.SELL: -1}


def _entries(inputs: PmInputs, days: frozenset[date]):
    for idea in inputs.ideas:
        if idea.entry_date not in days:
            continue
        s = inputs.series[idea.trade_idea_id]
        t = inputs.day_index[idea.entry_date]
        side_sign = _SIDE_SIGN[idea.side]
        trailing = inputs.view.trailing_move(s, t, inputs.horizon_days)
        sd = inputs.view.sd_h(s, t, inputs.horizon_days)
        yield idea, s, side_sign, trailing, sd


def after_run_count(inputs: PmInputs, days: frozenset[date]) -> int:
    """Number of entries in `days` that chase a trailing move already past one sd."""
    return sum(
        1
        for _, s, side_sign, trailing, sd in _entries(inputs, days)
        if s.bullish_sign * side_sign * trailing > sd
    )


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of entries in `days` that chase a trailing move already past one sd.

    `pairs` carries, per idea, the entry's directional sign and its trailing
    move in units of that horizon's sd, for the calibration check.
    """
    n = 0
    after_run = 0
    pairs = []
    for _, s, side_sign, trailing, sd in _entries(inputs, days):
        n += 1
        direction = s.bullish_sign * side_sign
        if direction * trailing > sd:
            after_run += 1
        pairs.append((float(direction), trailing / sd))
    if n == 0:
        return Estimate(value=None, n=0)
    return Estimate(value=after_run / n, n=n, pairs=tuple(pairs))


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
