"""Herding bias: on entry, a PM may follow the street's consensus over its own view."""

from dataclasses import dataclass

import numpy as np

from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import Side, StreetView


@dataclass(frozen=True)
class HerdingDecision:
    """The outcome of resolving a possible conflict between own side and street view."""

    conflict: bool
    followed_street: bool | None
    side: Side
    flag: str | None


def _street_side(street: StreetView | None) -> Side | None:
    if street == StreetView.OVERWEIGHT:
        return Side.BUY
    if street == StreetView.UNDERWEIGHT:
        return Side.SELL
    return None


def decide(
    own_side: Side, street: StreetView | None, params: EffectiveParams, rng: np.random.Generator
) -> HerdingDecision:
    """Resolve a herding conflict, if any, by drawing against the herding weight."""
    street_side = _street_side(street)
    if street_side is None or street_side == own_side:
        return HerdingDecision(conflict=False, followed_street=None, side=own_side, flag=None)

    w = params.value("herding_weight")
    u = rng.uniform()
    if u < w:
        flag = "herding:followed_street" if params.is_active("herding_weight") else None
        return HerdingDecision(conflict=True, followed_street=True, side=street_side, flag=flag)
    return HerdingDecision(conflict=True, followed_street=False, side=own_side, flag=None)
