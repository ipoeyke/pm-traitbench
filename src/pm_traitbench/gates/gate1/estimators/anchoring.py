"""Anchoring: the share of discretionary exits that land at the anchor rather than the target."""

import math
from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.enums import PositionAction, Side
from pm_traitbench.gates.gate1.estimate import Estimate, EstimatorSpec
from pm_traitbench.gates.gate1.inputs import PmInputs

_SIDE_SIGN: dict[Side, int] = {Side.BUY: 1, Side.SELL: -1}


def _discretionary_exits(inputs: PmInputs, days: frozenset[date]):
    ideas = {idea.trade_idea_id: idea for idea in inputs.ideas}
    for row in inputs.position_days:
        if row.date not in days or row.action != PositionAction.EXIT:
            continue
        if row.date == inputs.last_date:
            continue
        if (row.trade_idea_id, row.date) in inputs.acted:
            continue
        if row.anchor_level is None:
            continue
        idea = ideas.get(row.trade_idea_id)
        if idea is None:
            continue
        if math.isclose(row.anchor_level, idea.target_level, rel_tol=1e-9, abs_tol=1e-9):
            continue
        yield idea, row


def estimate(inputs: PmInputs, days: frozenset[date], knobs: Gate1Config) -> Estimate:
    """Share of discretionary exits whose rebuilt level lands within a band of the anchor.

    A discretionary exit is one the PM chose (not a rule the PM acted on), not
    on the last horizon date, with an anchor recorded and distinct from the
    idea's target. The band is `anchor_band_k` horizon-vols either side of the
    anchor.
    """
    n = 0
    inside = 0
    for idea, row in _discretionary_exits(inputs, days):
        s = inputs.series[idea.trade_idea_id]
        t = inputs.day_index[row.date]
        side_sign = _SIDE_SIGN[idea.side]
        exit_level = idea.entry_level + s.bullish_sign * side_sign * row.pnl_unit
        band = knobs.anchor_band_k * inputs.view.sd_h(s, t, inputs.horizon_days)
        n += 1
        if abs(exit_level - row.anchor_level) <= band:
            inside += 1
    if n == 0:
        return Estimate(value=None, n=0)
    return Estimate(value=inside / n, n=n)


SPEC = EstimatorSpec(estimate, higher_is_stronger=True)
