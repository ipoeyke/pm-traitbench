"""Anchoring: an idea's exit is pulled toward a nearby round level instead of its target.

At entry, a candidate anchor sits at the round level `ANCHOR_FRACTION` of the
way from entry to target - a salient prior level, after Northcraft and Neale
(1987). With probability rho, drawn once at entry, the idea exits there
instead of at target.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pm_traitbench.engine.constants import ANCHOR_FRACTION
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import Series
from pm_traitbench.engine.state import Position

if TYPE_CHECKING:
    from pm_traitbench.engine.adapters.base import Adapter


@dataclass(frozen=True)
class AnchoredExit:
    """The idea's anchor (if any), its effective exit level, and whether that is reached."""

    anchor_level: float | None
    effective_exit_level: float
    reached: bool


def entry_anchor(
    adapter: "Adapter", series: Series, entry_level: float, target_level: float
) -> float | None:
    """The round level `ANCHOR_FRACTION` of the way from entry to target, or None if it does
    not qualify.

    Only a round level strictly between entry and target, in the direction from one to
    the other, shortens the trade; one that rounds onto or past the target, or back onto
    the entry side, would not, and is not an anchor.
    """
    fav = 1.0 if target_level >= entry_level else -1.0
    raw = entry_level + ANCHOR_FRACTION * (target_level - entry_level)
    anchor = adapter.round_step(series, raw)
    if (anchor - entry_level) * fav > 0 and (anchor - target_level) * fav < 0:
        return anchor
    return None


def evaluate(pos: Position, level_now: float) -> AnchoredExit:
    """Whether an anchored idea has reached its anchor; an unanchored one targets as usual."""
    if not pos.anchored:
        return AnchoredExit(
            anchor_level=pos.anchor_level, effective_exit_level=pos.target_level, reached=False
        )
    assert pos.anchor_level is not None, "an anchored idea always has an anchor, set at entry"
    fav = -pos.adverse_dir
    reached = (level_now - pos.anchor_level) * fav >= 0
    return AnchoredExit(
        anchor_level=pos.anchor_level, effective_exit_level=pos.anchor_level, reached=reached
    )


def flag(params: EffectiveParams) -> str | None:
    """The anchoring flag when active, else None."""
    return "anchoring:exit_at_anchor" if params.is_active("anchoring_rho") else None
