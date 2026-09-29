"""Gate 2 transcript rendering: a PM's sessions turned into text for a judge's prompt."""

from collections.abc import Sequence

from pm_traitbench.enums import TurnRole
from pm_traitbench.tables.schema import Session

_ROLE_LABELS = {TurnRole.PM: "PM", TurnRole.ADVISOR: "ADVISOR"}


def _date_then_id(session: Session) -> tuple:
    """The sort key that orders a PM's sessions the same way everywhere: date, then id."""
    return (session.date, session.session_id)


def render_session(session: Session) -> str:
    """`session` as an id/date header followed by one `ROLE: text` line per turn.

    The header omits the session kind: it is planning metadata a deployed
    copilot never sees, so a judge recovering biases from the transcript
    must not see it either.
    """
    lines = [f"Session {session.session_id}, {session.date.isoformat()}"]
    lines.extend(f"{_ROLE_LABELS[turn.role]}: {turn.text}" for turn in session.turns)
    return "\n".join(lines)


def render_pm(sessions: Sequence[Session]) -> str:
    """`sessions` rendered in date-then-id order, each session's block separated by a blank line."""
    ordered = sorted(sessions, key=_date_then_id)
    return "\n\n".join(render_session(session) for session in ordered)


def pm_turn_text(sessions: Sequence[Session]) -> str:
    """Every PM turn's text across `sessions` in date-then-id order, one per line.

    Sorted the same way as `render_pm` so a containment metric built from
    this text never depends on the order sessions happen to be passed in.
    """
    ordered = sorted(sessions, key=_date_then_id)
    return "\n".join(
        turn.text for session in ordered for turn in session.turns if turn.role == TurnRole.PM
    )
