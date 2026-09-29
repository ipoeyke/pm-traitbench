"""Tests for gate 2's stated-signal classification request, turn location and scorer."""

import asyncio
from datetime import date

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import CachedClient
from pm_traitbench.enums import (
    AssetClass,
    Kind,
    Ownership,
    RuleScope,
    SignalMode,
    StanceEntry,
    Valence,
)
from pm_traitbench.errors import Gate2Error
from pm_traitbench.gates.gate2.classify import (
    ClassifyUnit,
    Statement,
    classify_request,
    classify_units,
    parse_classification,
    score_classification,
    send_classification,
    signal_turn_index,
)
from pm_traitbench.tables.schema import Stance
from tests.dialogue.fixtures import FakeClient, fake_message, rule
from tests.gates.fixtures import ledger_row
from tests.gates.gate2.fixtures import (
    PM_A,
    classify_reply,
    log_with_directives,
    session_of,
    signal,
    skeleton_with_stances,
)
from tests.signals.fixtures import persona as build_persona


def test_classify_units_keep_only_own_confirm_stated_signals_in_surviving_sessions():
    session_keep = session_of(
        PM_A, date(2026, 1, 5), ["I always cut losers fast", "I like short bullets"]
    )
    session_dropped = session_of(PM_A, date(2026, 1, 6), ["irrelevant"])

    sig_keep_a = signal(PM_A, session_keep.session_id, date(2026, 1, 5), "t_01", signal_id="sg_002")
    sig_keep_b = signal(PM_A, session_keep.session_id, date(2026, 1, 5), "t_02", signal_id="sg_001")
    sig_retracted = signal(
        PM_A,
        session_keep.session_id,
        date(2026, 1, 5),
        "t_03",
        valence=Valence.RETRACTED,
        signal_id="sg_003",
    )
    sig_colleague = signal(
        PM_A,
        session_keep.session_id,
        date(2026, 1, 5),
        "t_04",
        ownership=Ownership.COLLEAGUE,
        signal_id="sg_004",
    )
    sig_revealed = signal(
        PM_A,
        session_keep.session_id,
        date(2026, 1, 5),
        "t_05",
        mode=SignalMode.REVEALED,
        signal_id="sg_005",
    )
    sig_dropped = signal(
        PM_A, session_dropped.session_id, date(2026, 1, 6), "t_06", signal_id="sg_006"
    )
    signals = [sig_keep_a, sig_keep_b, sig_retracted, sig_colleague, sig_revealed, sig_dropped]

    stance_a = Stance(
        signal_id="sg_002",
        trait_id="t_01",
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance="I always cut losers fast",
    )
    stance_b = Stance(
        signal_id="sg_001",
        trait_id="t_02",
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance="I like short bullets",
    )
    skeleton_keep = skeleton_with_stances(session_keep, [stance_a, stance_b])
    log_keep = log_with_directives(
        session_keep, ["I always cut losers fast", "I like short bullets"]
    )

    sessions_by_id = {session_keep.session_id: session_keep}
    logs_by_id = {session_keep.session_id: log_keep}
    skeletons_by_id = {session_keep.session_id: skeleton_keep}

    units = classify_units(sessions_by_id, logs_by_id, skeletons_by_id, signals)

    assert len(units) == 1
    unit = units[0]
    assert unit.session.session_id == session_keep.session_id
    # Signals within a unit are sorted by signal id, not by trait id or insertion order.
    assert [s.signal_id for s in unit.signals] == ["sg_001", "sg_002"]
    assert unit.turn_index_by_signal == {"sg_001": 2, "sg_002": 0}
    n = len(unit.signals)
    assert n == 2


def test_classify_units_raises_when_a_surviving_session_has_no_log_or_skeleton():
    session = session_of(PM_A, date(2026, 1, 5), ["I always cut losers fast"])
    sig = signal(PM_A, session.session_id, date(2026, 1, 5), "t_01", signal_id="sg_001")

    with pytest.raises(Gate2Error, match=rf"session '{session.session_id}'"):
        classify_units({session.session_id: session}, {}, {}, [sig])


def test_signal_turn_index_finds_the_directive_turn_and_raises_on_zero_or_two_matches():
    session = session_of(PM_A, date(2026, 1, 5), ["yes I do that", "unrelated", "yes I do that"])
    good_stance = Stance(
        signal_id="sg_001",
        trait_id="t_01",
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance="yes I do that",
    )
    skeleton = skeleton_with_stances(session, [good_stance])

    log_one_match = log_with_directives(session, ["yes I do that", "something else", "nope"])
    assert signal_turn_index(skeleton, log_one_match, "sg_001") == 0

    log_zero_matches = log_with_directives(session, ["nope", "nope2", "nope3"])
    with pytest.raises(Gate2Error, match=r"stance matches 0 turns"):
        signal_turn_index(skeleton, log_zero_matches, "sg_001")

    log_two_matches = log_with_directives(session, ["yes I do that", "whatever", "yes I do that"])
    with pytest.raises(
        Gate2Error, match=rf"stance matches 2 turns in session '{session.session_id}'"
    ):
        signal_turn_index(skeleton, log_two_matches, "sg_001")

    skeleton_missing = skeleton_with_stances(session, [])
    with pytest.raises(Gate2Error, match=r"signal 'sg_001'"):
        signal_turn_index(skeleton_missing, log_one_match, "sg_001")


def test_classify_request_carries_rules_ledger_and_no_vocabulary():
    persona = build_persona(AssetClass.EQUITIES)
    pm_rule = rule(rule_id="r_01", text="Exit after a 5 percent drawdown.")
    stop_rule = rule(
        rule_id="r_02",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
        param="stop",
        text="Stop this idea at 90.",
    )
    target_rule = rule(
        rule_id="r_03",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
        param="target",
        text="Target this idea at 110.",
    )
    other_idea_rule = rule(
        rule_id="r_04",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_002",
        param="stop",
        text="Stop the other idea at 50.",
    )
    ledger = ledger_row(pm_id=persona.pm_id, trade_idea_id="ti_001", price_or_yield=101.5)
    ledger_after = ledger_row(
        pm_id=persona.pm_id, trade_idea_id="ti_001", date=date(2026, 1, 6), price_or_yield=999.0
    )
    session = session_of(
        persona.pm_id, date(2026, 1, 5), ["hello there"], trade_idea_ids=("ti_001",)
    )
    config = Config().gate2

    request = classify_request(
        persona,
        [pm_rule],
        [stop_rule, target_rule, other_idea_rule],
        [ledger, ledger_after],
        session,
        2,
        config,
    )

    assert set(request.keys()) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert request["max_tokens"] == 8000

    system = request["system"]
    user = request["messages"][0]["content"]
    assert pm_rule.text in system
    assert stop_rule.text in system
    assert target_rule.text in system
    assert other_idea_rule.text not in system
    assert "2026-01-05 ti_001 EQ-0001 buy size 100.0 at 101.5" in user
    assert "999.0" not in user
    assert "hello there" in user
    assert "Find 2 statements." in user

    dump = system + user
    assert not any(param in dump for param in BIAS_PARAMS)
    assert "Preferences to decide on" not in dump
    assert "Tendencies to decide on" not in dump


def test_classify_request_ledger_section_says_none_when_empty():
    persona = build_persona(AssetClass.EQUITIES)
    session = session_of(
        persona.pm_id, date(2026, 1, 5), ["hello there"], trade_idea_ids=("ti_001",)
    )
    config = Config().gate2

    request = classify_request(persona, [], [], [], session, 1, config)

    user = request["messages"][0]["content"]
    assert "Trades on the ideas discussed, on or before this session:\nnone" in user


def test_parse_classification_bounds_and_enum():
    zero = classify_reply([])
    assert parse_classification(zero, n=2) is None

    too_many = classify_reply([("a", "bias"), ("b", "preference"), ("c", "bias")])
    assert parse_classification(too_many, n=2) is None

    bad_kind = classify_reply([("a", "habit")])
    assert parse_classification(bad_kind, n=2) is None

    good = classify_reply([("a", "bias"), ("b", "preference")])
    statements = parse_classification(good, n=2)
    assert statements is not None
    assert [s.quote for s in statements] == ["a", "b"]
    assert [s.kind for s in statements] == [Kind.BIAS, Kind.PREFERENCE]


def test_score_classification_scores_each_signal_on_its_own_turn():
    session = session_of(
        PM_A, date(2026, 1, 5), ["I always cut losers fast", "I like short bullets"]
    )
    sig_a = signal(PM_A, session.session_id, date(2026, 1, 5), "t_01", signal_id="sg_001")
    sig_b = signal(PM_A, session.session_id, date(2026, 1, 5), "t_02", signal_id="sg_002")
    log = log_with_directives(session, [None, None])
    skeleton = skeleton_with_stances(session, [])
    unit = ClassifyUnit(
        session=session,
        log=log,
        skeleton=skeleton,
        signals=(sig_a, sig_b),
        turn_index_by_signal={sig_a.signal_id: 0, sig_b.signal_id: 2},
    )
    kind_by_trait = {"t_01": Kind.BIAS, "t_02": Kind.PREFERENCE}

    statements = (
        Statement(quote="I LIKE   SHORT bullets", kind=Kind.PREFERENCE),
        Statement(quote="i like short bullets", kind=Kind.BIAS),
        Statement(quote="this line never appears anywhere", kind=Kind.BIAS),
    )

    scores, warnings = score_classification(unit, statements, kind_by_trait)

    assert scores[sig_a.signal_id] == (None, False)
    assert scores[sig_b.signal_id] == (Kind.PREFERENCE, True)
    assert warnings == (
        f"session {session.session_id}: classification quote not in a PM turn: "
        '"this line never appears anywhere"',
    )


def test_score_classification_drops_empty_quote_and_credits_no_signal():
    session = session_of(
        PM_A, date(2026, 1, 5), ["I always cut losers fast", "I like short bullets"]
    )
    sig_a = signal(PM_A, session.session_id, date(2026, 1, 5), "t_01", signal_id="sg_001")
    sig_b = signal(PM_A, session.session_id, date(2026, 1, 5), "t_02", signal_id="sg_002")
    log = log_with_directives(session, [None, None])
    skeleton = skeleton_with_stances(session, [])
    unit = ClassifyUnit(
        session=session,
        log=log,
        skeleton=skeleton,
        signals=(sig_a, sig_b),
        turn_index_by_signal={sig_a.signal_id: 0, sig_b.signal_id: 2},
    )
    kind_by_trait = {"t_01": Kind.BIAS, "t_02": Kind.PREFERENCE}

    statements = (Statement(quote="  ", kind=Kind.BIAS),)

    scores, warnings = score_classification(unit, statements, kind_by_trait)

    assert scores[sig_a.signal_id] == (None, False)
    assert scores[sig_b.signal_id] == (None, False)
    assert warnings == (f"session {session.session_id}: classification quote is empty",)


def test_send_classification_labels_failure_by_session(tmp_path):
    unparsable = fake_message([{"type": "text", "text": "not json"}])
    fake = FakeClient(lambda _request: unparsable)
    client = CachedClient(lambda: fake, tmp_path, token_budget=None)
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(Gate2Error, match=r"^session sess_1: "):
        asyncio.run(send_classification(client, request, 2, "sess_1", max_retries=0))
