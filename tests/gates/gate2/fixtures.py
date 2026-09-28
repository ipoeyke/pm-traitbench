"""Shared gate 2 test fixtures: a `Session` builder used across the gate 2 test modules.

Extended by later gate 2 test modules, so a session's shape only has to
match `schema.py` in one place.
"""

from collections.abc import Sequence
from datetime import date

from pm_traitbench.enums import SessionKind, TurnRole
from pm_traitbench.signals.assemble import session_id
from pm_traitbench.tables.schema import Session, Turn

PM_A = "pm_001"
PM_B = "pm_002"


def session_of(
    pm_id: str,
    day: date,
    pm_texts: Sequence[str],
    advisor_texts: Sequence[str] | None = None,
    kind: SessionKind = SessionKind.CHECK_IN,
    trade_idea_ids: tuple[str, ...] = (),
    letter: int = 0,
) -> Session:
    """A `Session` for `pm_id` on `day`, alternating pm and advisor turns from the given texts.

    Advisor texts default to `"noted"` for each pm turn. `letter` selects the
    session's index among same-day sessions, per `signals.assemble.session_id`.
    """
    if advisor_texts is None:
        advisor_texts = ["noted"] * len(pm_texts)
    turns: list[Turn] = []
    for pm_text, advisor_text in zip(pm_texts, advisor_texts, strict=True):
        turns.append(Turn(role=TurnRole.PM, text=pm_text))
        turns.append(Turn(role=TurnRole.ADVISOR, text=advisor_text))
    return Session(
        session_id=session_id(pm_id, day, letter),
        pm_id=pm_id,
        date=day,
        kind=kind,
        trade_idea_ids=tuple(trade_idea_ids),
        turns=tuple(turns),
    )
