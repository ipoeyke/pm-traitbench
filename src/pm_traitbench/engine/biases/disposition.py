"""Disposition effect: realising gains too readily, holding losers too long."""

import math

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import PnlState


def sell_hazard(
    progress: float, pnl_state: PnlState, params: EffectiveParams, config: Config
) -> float:
    """Daily discretionary sell hazard: scaled by the disposition ratio D at a gain or loss."""
    h = config.engine.base_hazard * (1 + max(progress, 0))
    d = params.value("disposition_ratio")
    if pnl_state == PnlState.GAIN:
        h *= math.sqrt(d)
    elif pnl_state == PnlState.LOSS:
        h /= math.sqrt(d)
    return min(max(h, 0.0), 1.0)


def draw_sell(h: float, rng: np.random.Generator) -> bool:
    """Draw whether the position is sold today, at hazard `h`."""
    return rng.uniform() < h


def flag(pnl_state: PnlState, progress: float, sold: bool, params: EffectiveParams) -> str | None:
    """The disposition flag for realising a gain early or holding onto a loser."""
    if not params.is_active("disposition_ratio"):
        return None
    if pnl_state == PnlState.GAIN and sold and progress < 1:
        return "disposition:realise_gain_early"
    if pnl_state == PnlState.LOSS and not sold:
        return "disposition:hold_loser"
    return None
