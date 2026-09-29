from datetime import date

from pm_traitbench.enums import DriftEventType, Kind, Ownership, Valence
from pm_traitbench.gates.gate2.transcript import render_pm
from pm_traitbench.probes.context import (
    context_chars,
    in_context,
    own_confirm_ids,
    sessions_until,
    supporting_ids,
    third_party_signals,
)
from pm_traitbench.signals.assemble import session_id
from tests.probes.fixtures import PM_A, drift, session_of, signal, third_party, trait

D1, D2, D3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)


def sid(day):
    return session_id(PM_A, day, 0)


def test_sessions_until_includes_day_excludes_next_and_sorts():
    s3, s2, s1 = (session_of(PM_A, d, ["x"]) for d in (D3, D2, D1))
    assert sessions_until([s3, s2, s1], D2) == (s1, s2)


def test_sessions_until_orders_same_day_by_id():
    a, b = session_of(PM_A, D1, ["x"], letter=0), session_of(PM_A, D1, ["y"], letter=1)
    assert sessions_until([b, a], D1) == (a, b)


def test_context_chars_matches_render():
    sessions = [session_of(PM_A, D1, ["hello"]), session_of(PM_A, D2, ["world"])]
    assert context_chars(sessions) == len(render_pm(sessions))


def test_in_context_drops_signal_of_absent_session():
    s1, s2 = session_of(PM_A, D1, ["x"]), session_of(PM_A, D2, ["y"])
    kept = signal(PM_A, s1.session_id, D1, "t_01", signal_id="sg_001")
    dropped = signal(PM_A, s2.session_id, D2, "t_01", signal_id="sg_002")
    assert in_context([kept, dropped], {s1.session_id}, D2) == (kept,)


def test_in_context_drops_signal_after_day():
    s2 = session_of(PM_A, D2, ["y"])
    late = signal(PM_A, s2.session_id, D2, "t_01", signal_id="sg_002")
    assert in_context([late], {s2.session_id}, D1) == ()


def test_supporting_ids_exclude_retracted_and_third_party():
    s = session_of(PM_A, D1, ["x"])
    own = signal(PM_A, s.session_id, D1, "t_01", signal_id="sg_001")
    retracted = signal(
        PM_A, s.session_id, D1, "t_01", valence=Valence.RETRACTED, signal_id="sg_002"
    )
    colleague = third_party(PM_A, s.session_id, D1, "t_01", "v", "sg_003")
    other = signal(PM_A, s.session_id, D1, "t_02", signal_id="sg_004")
    bias = trait(PM_A, "t_01", "disposition_ratio", Kind.BIAS, 1.5)
    assert supporting_ids(bias, [own, retracted, colleague, other], [], D1) == ("sg_001",)


def test_supporting_ids_of_updated_preference_start_at_update():
    pref = trait(PM_A, "t_09", "response_format", Kind.PREFERENCE, "a")
    update_day = date(2026, 5, 20)
    events = [drift(PM_A, update_day, DriftEventType.UPDATE, "t_09", "a", "b")]
    old = signal(PM_A, sid(D1), D1, "t_09", signal_id="sg_001")
    on = signal(PM_A, sid(update_day), update_day, "t_09", signal_id="sg_002")
    assert supporting_ids(pref, [old, on], events, date(2026, 6, 1)) == ("sg_002",)
    assert supporting_ids(pref, [old], events, date(2026, 5, 19)) == ("sg_001",)


def test_own_confirm_ids_since():
    old = signal(PM_A, sid(D1), D1, "t_01", signal_id="sg_002")
    new = signal(PM_A, sid(D3), D3, "t_01", signal_id="sg_001")
    assert own_confirm_ids("t_01", [old, new]) == ("sg_001", "sg_002")
    assert own_confirm_ids("t_01", [old, new], since=D3) == ("sg_001",)


def test_third_party_signals_only_non_self_in_id_order():
    own = signal(PM_A, sid(D1), D1, "t_01", signal_id="sg_001")
    col = third_party(PM_A, sid(D1), D1, "t_01", "v", "sg_003")
    cli = third_party(PM_A, sid(D1), D1, "t_01", "v", "sg_002").model_copy(
        update={"ownership": Ownership.CLIENT}
    )
    assert third_party_signals("t_01", [own, col, cli]) == (cli, col)
