"""Loss aversion: hazards for cutting, holding and adding on a losing position.

Lambda raises the add hazard above 1 and lowers the cut hazard below its
neutral level, a calibrated construction: a linear prospect value over the
PM's own forecast leaves lambda no measurable effect at typical loss sizes.
"""

from dataclasses import dataclass

import numpy as np

from pm_traitbench.engine.constants import LOSS_ADD_CAP, LOSS_ADD_SLOPE, LOSS_CUT_HAZARD
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import PositionAction

_FLAG_BY_ACTION: dict[PositionAction, str | None] = {
    PositionAction.CUT: None,
    PositionAction.HOLD: "loss_aversion:hold",
    PositionAction.ADD: "loss_aversion:add",
}


@dataclass(frozen=True)
class LossSideChoice:
    """The action chosen for a losing position, and its bias flag."""

    action: PositionAction
    flag: str | None


def choose(params: EffectiveParams, rng: np.random.Generator, add_allowed: bool) -> LossSideChoice:
    """Draw cut, add or hold on a losing position from lambda-scaled hazards."""
    lam = params.value("loss_aversion_lambda")
    p_cut = LOSS_CUT_HAZARD / lam
    p_add = min(LOSS_ADD_CAP, LOSS_ADD_SLOPE * max(lam - 1.0, 0.0)) if add_allowed else 0.0

    u = rng.uniform()
    if u < p_cut:
        action = PositionAction.CUT
    elif u < p_cut + p_add:
        action = PositionAction.ADD
    else:
        action = PositionAction.HOLD

    active = params.is_active("loss_aversion_lambda")
    flag = _FLAG_BY_ACTION[action] if active else None
    return LossSideChoice(action, flag)


def hold_flag(params: EffectiveParams) -> str | None:
    """The hold flag when active, else None.

    Reused when a drawn add is blocked at the mandate cap, so that day still
    carries the same flag a drawn hold would.
    """
    return (
        _FLAG_BY_ACTION[PositionAction.HOLD] if params.is_active("loss_aversion_lambda") else None
    )
