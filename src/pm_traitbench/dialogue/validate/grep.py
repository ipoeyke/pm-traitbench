"""Parameter-name grep: catches a PM turn that names a bias or preference parameter, or
a banned stance word, leaking a trait the transcript must never state directly.

Advisor turns are never grepped: the advisor never sees the PM's traits, and
regeneration to fix a failure only ever changes the narrator, not the advisor.
"""

from collections.abc import Sequence

from pm_traitbench.catalogues.loader import banned_words_in, leak_param_names, matched_params
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.enums import TurnRole
from pm_traitbench.tables.schema import DialogueLog


def grep_params(catalogue: Catalogue) -> tuple[str, ...]:
    """The params a PM turn must never name: `leak_param_names` of the catalogue."""
    return leak_param_names(catalogue)


def check_grep(log: DialogueLog, params: Sequence[str]) -> tuple[str, ...]:
    """Failure reasons for a session's PM turns naming a parameter or banned stance word."""
    reasons: set[str] = set()
    for turn in log.turns:
        if turn.role != TurnRole.PM:
            continue
        lowered = turn.text.lower()
        for name in (*matched_params(lowered, params), *banned_words_in(lowered)):
            reasons.add(f"names a parameter: {name}")
    return tuple(sorted(reasons))
