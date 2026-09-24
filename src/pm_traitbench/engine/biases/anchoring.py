"""Anchoring: an exit target drifts toward a nearby salient prior level."""

from collections.abc import Sequence
from dataclasses import dataclass

from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.state import Position


@dataclass(frozen=True)
class AnchoredExit:
    """The anchor used, the resulting effective exit level, and whether it is reached."""

    anchor_level: float
    effective_exit_level: float
    reached: bool


def evaluate(
    pos: Position, level_now: float, anchors: Sequence[float], params: EffectiveParams
) -> AnchoredExit:
    """Blend the position's target toward the nearest anchor on the target side."""
    favourable = -pos.adverse_dir
    candidates = [a for a in anchors if (a - level_now) * favourable > 0]
    if candidates:
        anchor = min(candidates, key=lambda a: abs(a - level_now))
    else:
        anchor = pos.target_level

    rho = params.value("anchoring_rho")
    effective = (1 - rho) * pos.target_level + rho * anchor
    reached = (level_now - effective) * favourable >= 0
    return AnchoredExit(anchor_level=anchor, effective_exit_level=effective, reached=reached)


def flag(params: EffectiveParams) -> str | None:
    """The anchoring flag when active, else None."""
    return "anchoring:exit_at_anchor" if params.is_active("anchoring_rho") else None
