"""Parameter-name grep: catches a PM turn that names a bias or preference parameter, or
a banned stance word, leaking a trait the transcript must never state directly.

Advisor turns are never grepped: the advisor never sees the PM's traits, and
regeneration to fix a failure only ever changes the narrator, not the advisor.
"""

from collections.abc import Sequence

from pm_traitbench.catalogues.loader import BANNED_STANCE_WORDS, matched_param
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import TurnRole
from pm_traitbench.tables.schema import DialogueLog


def grep_params(catalogue: Catalogue) -> tuple[str, ...]:
    """The bias params followed by every catalogue preference's param, in catalogue order."""
    return (*BIAS_PARAMS, *(entry.param for entry in catalogue.preferences))


def check_grep(log: DialogueLog, params: Sequence[str]) -> tuple[str, ...]:
    """Failure reasons for a session's PM turns naming a parameter or banned stance word."""
    reasons: set[str] = set()
    for turn in log.turns:
        if turn.role != TurnRole.PM:
            continue
        matched = matched_param(turn.text, params)
        if matched is not None:
            reasons.add(f"names a parameter: {matched}")
        lowered = turn.text.lower()
        for word in BANNED_STANCE_WORDS:
            if word in lowered:
                reasons.add(f"names a parameter: {word}")
    return tuple(sorted(reasons))
