"""Exit deficiency: a PM may fail to act appropriately on a rule breach."""

from dataclasses import dataclass

import numpy as np

from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import RuleResponse

LATE_ROLL_FLAG = "exit_deficiency:late_roll"


@dataclass(frozen=True)
class Response:
    """The outcome of testing a rule breach against exit deficiency."""

    response: RuleResponse
    flag: str | None


def respond(params: EffectiveParams, at_loss: bool, rng: np.random.Generator) -> Response:
    """Draw whether a breach is missed; a missed breach on a loss may instead add."""
    e = params.value("exit_deficiency")
    u = rng.uniform()
    if u < e:
        if params.is_active("loss_aversion_lambda") and at_loss:
            return Response(RuleResponse.ADDED, "exit_deficiency:added")
        return Response(RuleResponse.ACKED_NO_ACTION, None)
    return Response(RuleResponse.ACTED, None)
