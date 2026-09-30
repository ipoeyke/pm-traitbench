"""Parameter-name grep: catches a PM turn that names a bias or preference parameter, or
a banned bias phrase, leaking a trait the transcript must never state directly.

A bias param fails in either spelling. A preference param fails only as its raw
underscore identifier, since stating a preference in plain words is what a stated
stance asks for. Advisor turns are never grepped: the advisor never sees the PM's
traits, and regeneration to fix a failure only ever changes the narrator.
"""

import re
from collections.abc import Sequence

from pm_traitbench.catalogues.loader import banned_phrases_in
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import TurnRole
from pm_traitbench.tables.schema import DialogueLog


def _forms(param: str) -> tuple[str, ...]:
    raw = param.lower()
    if param in BIAS_PARAMS:
        return raw, raw.replace("_", " ")
    return (raw,) if "_" in raw else ()


def named_params(text: str, params: Sequence[str]) -> tuple[str, ...]:
    """Every param `text` names, whole-word and case-insensitive, in `params` order."""
    lowered = text.lower()
    return tuple(
        param
        for param in params
        if any(re.search(rf"\b{re.escape(form)}\b", lowered) for form in _forms(param))
    )


def check_grep(log: DialogueLog, params: Sequence[str]) -> tuple[str, ...]:
    """Failure reasons for a session's PM turns naming a parameter or banned bias phrase."""
    reasons: set[str] = set()
    for turn in log.turns:
        if turn.role != TurnRole.PM:
            continue
        for name in (*named_params(turn.text, params), *banned_phrases_in(turn.text)):
            reasons.add(f"names a parameter: {name}")
    return tuple(sorted(reasons))
