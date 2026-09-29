"""Tests for gate 2's session transcript rendering."""

from datetime import date

from pm_traitbench.enums import SessionKind
from pm_traitbench.gates.gate2.transcript import pm_turn_text, render_pm, render_session
from tests.gates.gate2.fixtures import PM_A, session_of


def test_render_session_has_id_date_and_roles_without_kind() -> None:
    session = session_of(
        PM_A,
        date(2026, 1, 5),
        ["what's the flow", "size it up"],
        ["thin", "5y richer"],
        kind=SessionKind.DECISION,
    )
    rendered = render_session(session)
    lines = rendered.split("\n")
    assert lines[0] == f"Session {session.session_id}, 2026-01-05"
    assert lines[1] == "PM: what's the flow"
    assert lines[2] == "ADVISOR: thin"
    assert lines[3] == "PM: size it up"
    assert lines[4] == "ADVISOR: 5y richer"
    assert len(lines) == 5
    assert SessionKind.DECISION.value not in rendered


def test_render_pm_orders_by_date_then_id_and_separates_with_blank_line() -> None:
    later = session_of(PM_A, date(2026, 1, 6), ["later pm"], letter=0)
    earlier_b = session_of(PM_A, date(2026, 1, 5), ["earlier b"], letter=1)
    earlier_a = session_of(PM_A, date(2026, 1, 5), ["earlier a"], letter=0)

    rendered = render_pm([later, earlier_b, earlier_a])

    expected = "\n\n".join(render_session(s) for s in (earlier_a, earlier_b, later))
    assert rendered == expected


def test_pm_turn_text_keeps_only_pm_turns_and_sorts_by_date_then_id() -> None:
    earlier = session_of(PM_A, date(2026, 1, 5), ["pm one", "pm two"], ["adv one", "adv two"])
    later = session_of(PM_A, date(2026, 1, 6), ["pm three"])

    assert pm_turn_text([later, earlier]) == "pm one\npm two\npm three"
