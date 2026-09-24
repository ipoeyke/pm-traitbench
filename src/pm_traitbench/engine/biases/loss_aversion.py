"""Loss aversion: on a losing position, weighs cutting, holding and adding."""

from dataclasses import dataclass

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.engine.constants import ADD_FRACTION
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import PositionAction

# Fixed evaluation order for the softmax draw's cumulative selection.
_ACTION_ORDER = (PositionAction.CUT, PositionAction.HOLD, PositionAction.ADD)
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


def choose(
    pnl_z: float,
    forecast_remaining_z: float,
    params: EffectiveParams,
    config: Config,
    rng: np.random.Generator,
    add_allowed: bool,
) -> LossSideChoice:
    """Softmax-select cut, hold or add on a losing position at temperature `softmax_tau`."""
    lam = params.value("loss_aversion_lambda")
    loss = abs(pnl_z)
    value_by_action = {
        PositionAction.CUT: -lam * loss,
        PositionAction.HOLD: forecast_remaining_z,
        PositionAction.ADD: forecast_remaining_z * (1 + ADD_FRACTION) - lam * loss * ADD_FRACTION,
    }
    actions = tuple(a for a in _ACTION_ORDER if add_allowed or a != PositionAction.ADD)

    tau = config.engine.softmax_tau
    scores = np.array([value_by_action[a] for a in actions])
    weights = np.exp((scores - scores.max()) / tau)
    probs = weights / weights.sum()

    u = rng.uniform()
    cumulative = 0.0
    chosen = actions[-1]
    for action, p in zip(actions, probs, strict=True):
        cumulative += p
        if u < cumulative:
            chosen = action
            break

    active = params.is_active("loss_aversion_lambda")
    flag = _FLAG_BY_ACTION[chosen] if active else None
    return LossSideChoice(chosen, flag)
