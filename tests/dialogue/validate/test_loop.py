"""Tests for the validate stage's per-session attempt loop."""

import asyncio
import re

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.session import SessionResult
from pm_traitbench.dialogue.validate.loop import (
    FEEDBACK_HEADER,
    feedback_text,
    run_session,
    validate_once,
)
from pm_traitbench.enums import SignalMode, StanceEntry, TurnRole, ValidationStatus
from pm_traitbench.errors import ValidateError
from pm_traitbench.tables.schema import Session, Turn
from tests.dialogue.fixtures import SESSION_ID, fake_message, turn_text, with_section
from tests.dialogue.validate.fixtures import (
    advisor_turn,
    forbidden_reply,
    leak_reply,
    level_mention,
    log_of,
    pm_turn,
    scripted_client,
    stance,
    stance_reply,
    trade_mention,
    validate_context,
)
from tests.gates.fixtures import ledger_row

_ADVISOR_PROMPT = "You are a market advisor for the PM's book."
_BAD_TEXT = "my loss_aversion_lambda is high"


def _raise(_request):
    raise AssertionError("this judge or narrator must not be called")


def _clean_forbidden(_request):
    return forbidden_reply([])


def _turn_reply(text: str, mentions: list[dict]) -> dict:
    return fake_message([turn_text(text, mentions)])


def _log_saying(pm_text: str):
    """A one-exchange log on the default session: the PM says `pm_text`, the advisor notes it."""
    return log_of(SESSION_ID, "pm_001", [pm_turn(pm_text), advisor_turn("noted")])


_CLEAN_LOG = _log_saying("all clear on the book")


def _session_from(ctx, texts: tuple[str, str]) -> Session:
    return Session(
        session_id=ctx.skeleton.session_id,
        pm_id=ctx.skeleton.pm_id,
        date=ctx.skeleton.date,
        kind=ctx.skeleton.kind,
        trade_idea_ids=ctx.skeleton.trade_idea_ids,
        turns=(Turn(role=TurnRole.PM, text=texts[0]), Turn(role=TurnRole.ADVISOR, text=texts[1])),
    )


def _once(ctx, client, *, log=_CLEAN_LOG, grep_params=(), trait_param_by_id=None):
    """`validate_once` with the default config and shipped catalogue and no ledger."""
    return asyncio.run(
        validate_once(
            ctx,
            log,
            client,
            Config(),
            load_catalogue(),
            ledger=(),
            grep_params=grep_params,
            trait_param_by_id=trait_param_by_id or {},
        )
    )


def _run(ctx, session, log, client, *, config=None, ledger=(), grep_params=()):
    """`run_session` with the shipped catalogue and no trait params."""
    return asyncio.run(
        run_session(
            ctx,
            session,
            log,
            client,
            config or Config(),
            load_catalogue(),
            _ADVISOR_PROMPT,
            ledger=ledger,
            grep_params=grep_params,
            trait_param_by_id={},
        )
    )


def test_clean_session_passes_on_attempt_one_with_no_narrator_call(market_lookup, tmp_path):
    ctx = validate_context(market_lookup)
    session = _session_from(ctx, ("all clear on the book", "noted"))
    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden)

    outcome = _run(
        ctx, session, _CLEAN_LOG, client, grep_params=("register", "loss_aversion_lambda")
    )

    assert len(outcome.rows) == 1
    assert outcome.rows[0].status == ValidationStatus.PASS
    assert outcome.rows[0].fallback_model is None
    assert outcome.rows[0].fallback_judges == ()
    assert outcome.final == SessionResult(
        session=session, log=_CLEAN_LOG, warnings=(), rejected_replies=0
    )


@pytest.mark.parametrize(
    ("mode", "entry", "expect_judged"),
    [
        (SignalMode.STATED, StanceEntry.STATED, False),
        (SignalMode.REVEALED, StanceEntry.REVEALED, True),
        (SignalMode.CONTRADICTION, StanceEntry.CLAIM, True),
    ],
)
def test_leak_judge_runs_only_with_a_revealed_or_contradiction_stance(
    market_lookup, tmp_path, mode, entry, expect_judged
):
    ctx = validate_context(market_lookup, stances=(stance("t_01", mode, entry, "line"),))
    calls = {"n": 0}

    def leak_called(_request):
        calls["n"] += 1
        return leak_reply(False, None)

    client = scripted_client(tmp_path, _raise, leak_called, _clean_forbidden)
    result = _once(ctx, client, trait_param_by_id={"t_01": "disposition_ratio"})

    assert result.leak_judged is expect_judged
    assert calls["n"] == int(expect_judged)


def test_explicit_label_of_the_revealed_param_fails_and_other_labels_pass(market_lookup, tmp_path):
    ctx = validate_context(
        market_lookup, stances=(stance("t_01", SignalMode.REVEALED, StanceEntry.REVEALED, "line"),)
    )

    log = _log_saying("I sold winners this week, the usual")

    def once(tmp_sub, leak_fn):
        client = scripted_client(tmp_path / tmp_sub, _raise, leak_fn, _clean_forbidden)
        return _once(ctx, client, log=log, trait_param_by_id={"t_01": "disposition_ratio"})

    revealed_label = once("a", lambda _r: leak_reply(True, "disposition_ratio", "I sold winners"))
    assert revealed_label.leak_reasons == ('leaks disposition_ratio: "I sold winners"',)
    assert revealed_label.passed is False

    unquoted = once("d", lambda _r: leak_reply(True, "disposition_ratio", "not in the turn"))
    assert unquoted.leak_reasons == ()
    assert unquoted.warnings == (
        f"session {SESSION_ID}: leak judge quote not in a PM turn: not in the turn",
    )
    assert unquoted.passed is True

    other_label = once(
        "b", lambda _r: leak_reply(True, "loss_aversion_lambda", "average down again")
    )
    assert other_label.leak_reasons == ()
    assert other_label.warnings == ()
    assert other_label.passed is True

    not_explicit = once("c", lambda _r: leak_reply(False, None, "quote"))
    assert not_explicit.leak_reasons == ()
    assert not_explicit.passed is True


def test_explicit_label_naming_a_stated_traits_param_passes(market_lookup, tmp_path):
    """A stated stance's param never needs judging; an explicit label naming it must not
    fail the session just because some other stance in the session happens to be revealed.
    """
    ctx = validate_context(
        market_lookup,
        stances=(
            stance("t_01", SignalMode.STATED, StanceEntry.STATED, "line a"),
            stance("t_02", SignalMode.REVEALED, StanceEntry.REVEALED, "line b"),
        ),
        pm_turns=2,
    )
    client = scripted_client(
        tmp_path,
        _raise,
        lambda _r: leak_reply(True, "loss_aversion_lambda", "average down again"),
        _clean_forbidden,
    )
    log = log_of(
        SESSION_ID,
        "pm_001",
        [pm_turn("all clear"), advisor_turn("noted"), pm_turn("still clear"), advisor_turn("ok")],
    )

    result = _once(
        ctx,
        client,
        log=log,
        trait_param_by_id={"t_01": "loss_aversion_lambda", "t_02": "disposition_ratio"},
    )

    assert result.leak_reasons == ()
    assert result.warnings == ()
    assert result.passed is True


def test_other_label_passes_without_a_warning(market_lookup, tmp_path):
    ctx = validate_context(
        market_lookup, stances=(stance("t_01", SignalMode.REVEALED, StanceEntry.REVEALED, "line"),)
    )
    client = scripted_client(
        tmp_path, _raise, lambda _r: leak_reply(True, "other", "quote"), _clean_forbidden
    )

    result = _once(ctx, client, trait_param_by_id={"t_01": "disposition_ratio"})

    assert result.leak_reasons == ()
    assert result.warnings == ()
    assert result.passed is True


def test_forbidden_violation_fails_with_the_avoid_line_and_out_of_range_index_is_dropped(
    market_lookup, tmp_path
):
    ctx = validate_context(
        market_lookup,
        avoid_lines=("never mention position size", "never predict a specific price"),
    )
    client = scripted_client(
        tmp_path,
        _raise,
        _raise,
        lambda _r: forbidden_reply([(1, "Quoted  the size"), (5, "out of range")]),
    )

    result = _once(ctx, client, log=_log_saying("I quoted the size to the desk"))

    assert result.forbidden_reasons == (
        'forbidden: never mention position size: "Quoted  the size"',
    )
    assert result.passed is False
    assert result.warnings == (
        f"session {ctx.skeleton.session_id}: forbidden judge index out of range: 5",
    )


def test_forbidden_violation_whose_quote_is_not_in_a_pm_turn_is_a_warning(market_lookup, tmp_path):
    ctx = validate_context(market_lookup, avoid_lines=("never mention position size",))
    client = scripted_client(
        tmp_path,
        _raise,
        _raise,
        lambda _r: forbidden_reply([(1, "ran it at twice the size")]),
    )

    result = _once(ctx, client, log=_log_saying("all clear, ran it at twice the size"))
    assert result.forbidden_reasons == (
        'forbidden: never mention position size: "ran it at twice the size"',
    )

    invented = _once(ctx, client)
    assert invented.forbidden_reasons == ()
    assert invented.passed is True
    assert invented.warnings == (
        f"session {ctx.skeleton.session_id}: forbidden judge quote not in a PM turn: "
        "ran it at twice the size",
    )


def test_forbidden_verdict_with_every_index_out_of_range_passes(market_lookup, tmp_path):
    ctx = validate_context(market_lookup, avoid_lines=("never mention position size",))
    client = scripted_client(
        tmp_path,
        _raise,
        _raise,
        lambda _r: forbidden_reply([(0, "zero index"), (5, "too high")]),
    )

    result = _once(ctx, client)

    assert result.forbidden_reasons == ()
    assert result.passed is True
    assert result.warnings == (
        f"session {ctx.skeleton.session_id}: forbidden judge index out of range: 0",
        f"session {ctx.skeleton.session_id}: forbidden judge index out of range: 5",
    )


def test_failed_session_is_regenerated_with_feedback_and_passes_on_attempt_two(
    market_lookup, tmp_path
):
    row = ledger_row()
    ctx = validate_context(market_lookup, day_trades=(row,))
    session = _session_from(ctx, (_BAD_TEXT, "noted"))
    log = _log_saying(_BAD_TEXT)

    calls = {"n": 0}

    def narrator_reply(request):
        assert "Attempt 2." in request["system"]
        assert "names a parameter: loss_aversion_lambda" in request["system"]
        mentions = [trade_mention(row).model_dump(mode="json")] if calls["n"] == 0 else []
        calls["n"] += 1
        return _turn_reply("feeling calm about the book today", mentions)

    client = scripted_client(tmp_path, narrator_reply, _raise, _clean_forbidden)

    outcome = _run(ctx, session, log, client, ledger=(row,), grep_params=("loss_aversion_lambda",))

    assert [r.status for r in outcome.rows] == [
        ValidationStatus.REGENERATE,
        ValidationStatus.PASS,
    ]
    assert outcome.final is not None
    assert any(
        turn.text == "feeling calm about the book today" for turn in outcome.final.session.turns
    )
    assert outcome.final.log != log
    assert any(turn.text == "feeling calm about the book today" for turn in outcome.final.log.turns)


def _run_always_bad_narrator(market_lookup, tmp_path, max_attempts):
    """Run a session whose narrator always names a banned param, up to `max_attempts`.

    Returns the outcome and every narrator request's `system`, grouped by the attempt
    number embedded in its feedback text.
    """
    ctx = validate_context(market_lookup)
    session = _session_from(ctx, (_BAD_TEXT, "noted"))
    systems_by_attempt: dict[int, set[str]] = {}

    def narrator_reply(request):
        match = re.search(rf"Attempt (\d+)\. {re.escape(FEEDBACK_HEADER)}", request["system"])
        assert match is not None
        systems_by_attempt.setdefault(int(match.group(1)), set()).add(request["system"])
        return _turn_reply("still naming my loss_aversion_lambda here", [])

    config = with_section(Config(), "validation", max_attempts=max_attempts)
    client = scripted_client(tmp_path, narrator_reply, _raise, _clean_forbidden)

    outcome = _run(
        ctx,
        session,
        _log_saying(_BAD_TEXT),
        client,
        config=config,
        grep_params=("loss_aversion_lambda",),
    )
    return outcome, systems_by_attempt


def test_drop_at_the_attempt_cap(market_lookup, tmp_path):
    outcome, systems_by_attempt = _run_always_bad_narrator(market_lookup, tmp_path, max_attempts=2)

    assert [r.status for r in outcome.rows] == [
        ValidationStatus.REGENERATE,
        ValidationStatus.DROPPED,
    ]
    assert outcome.final is None
    assert set(systems_by_attempt) == {2}


def test_drop_at_the_attempt_cap_after_two_regenerations(market_lookup, tmp_path):
    outcome, systems_by_attempt = _run_always_bad_narrator(market_lookup, tmp_path, max_attempts=3)

    assert [r.status for r in outcome.rows] == [
        ValidationStatus.REGENERATE,
        ValidationStatus.REGENERATE,
        ValidationStatus.DROPPED,
    ]
    assert outcome.final is None
    assert set(systems_by_attempt) == {2, 3}
    assert systems_by_attempt[2].isdisjoint(systems_by_attempt[3])


def test_max_attempts_one_drops_on_the_first_failure(market_lookup, tmp_path):
    outcome, systems_by_attempt = _run_always_bad_narrator(market_lookup, tmp_path, max_attempts=1)

    assert [r.status for r in outcome.rows] == [ValidationStatus.DROPPED]
    assert outcome.final is None
    assert systems_by_attempt == {}


def test_feedback_text_format():
    text = feedback_text(2, ["reason one", "reason two"])

    assert text == (
        "Attempt 2. A validator rejected the previous version of this session:\n"
        "- reason one\n"
        "- reason two\n"
        "Fix these and keep everything else as instructed."
    )


def test_stance_judge_sees_only_the_pm_turn_and_its_stance_line(market_lookup, tmp_path):
    line = "hold EQ-0001 through the drawdown"
    ctx = validate_context(
        market_lookup,
        stances=(stance("t_01", SignalMode.STATED, StanceEntry.STATED, line),),
        pm_turns=2,
    )
    log = log_of(
        SESSION_ID,
        "pm_001",
        [
            pm_turn("holding it"),
            advisor_turn("consider cutting"),
            pm_turn("fine"),
            advisor_turn("ok"),
        ],
    )
    seen: list[dict] = []

    def judge(request):
        seen.append(request)
        return stance_reply(False, "the PM cut the position")

    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden, stance=judge)
    result = _once(ctx, client, log=log)

    assert len(seen) == 1
    assert seen[0]["messages"][0]["content"] == (
        f"<pm_message>holding it</pm_message>\n\n<instruction>{line}</instruction>"
    )
    assert result.stance_reasons == (
        f'stance not carried out in PM turn 1: "{line}": the PM cut the position',
    )
    assert result.passed is False


def _refused(_request):
    return fake_message([{"type": "text", "text": "no"}], stop_reason="refusal")


def test_judge_refusals_fall_back_and_name_each_fallen_back_judge(market_lookup, tmp_path):
    config = Config()
    lines = ("hold EQ-0001 through the drawdown", "refuse to add to EQ-0002")
    ctx = validate_context(
        market_lookup,
        stances=tuple(
            stance(f"t_0{i}", SignalMode.STATED, StanceEntry.STATED, line)
            for i, line in enumerate(lines, 1)
        ),
        avoid_lines=("never mention position size",),
        pm_turns=2,
    )
    log = log_of(
        SESSION_ID,
        "pm_001",
        [pm_turn("holding it"), advisor_turn("cut?"), pm_turn("not adding"), advisor_turn("ok")],
    )
    primary = config.validation.judge_model

    def forbidden(request):
        return _refused(request) if request["model"] == primary else forbidden_reply([])

    def judge(request):
        refuse = request["model"] == primary and "not adding" in request["messages"][0]["content"]
        return _refused(request) if refuse else stance_reply(True, "carried out")

    client = scripted_client(tmp_path, _raise, _raise, forbidden, stance=judge)
    result = _once(ctx, client, log=log)

    assert result.passed is True
    assert result.fallback_judges == ("forbidden", "stance turn 2")
    assert result.rejected_replies == 2


def test_a_fallen_back_judge_is_recorded_on_the_validation_row(market_lookup, tmp_path):
    config = Config()
    ctx = validate_context(
        market_lookup,
        stances=(stance("t_01", SignalMode.STATED, StanceEntry.STATED, "hold EQ-0001"),),
    )
    session = _session_from(ctx, ("all clear on the book", "noted"))

    def judge(request):
        if request["model"] == config.validation.judge_model:
            return _refused(request)
        return stance_reply(True, "carried out")

    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden, stance=judge)
    outcome = _run(ctx, session, _CLEAN_LOG, client, config=config)

    (row,) = outcome.rows
    assert row.status == ValidationStatus.PASS
    assert row.judge_model == config.validation.judge_model
    assert row.fallback_model == config.validation.refusal_fallback_model
    assert row.fallback_judges == ("stance turn 1",)


def test_stance_judge_is_not_called_without_stances(market_lookup, tmp_path):
    ctx = validate_context(market_lookup)
    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden, stance=_raise)

    assert _once(ctx, client).stance_reasons == ()


def test_stance_check_rejects_a_log_that_does_not_match_the_turn_plan(market_lookup, tmp_path):
    ctx = validate_context(market_lookup, pm_turns=2)
    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden)

    with pytest.raises(ValidateError, match="log has 1 PM turns but the turn plan has 2"):
        _once(ctx, client)


def test_pm_level_mismatch_fails_the_session(market_lookup, tmp_path):
    ctx = validate_context(market_lookup)
    log = log_of(
        SESSION_ID,
        "pm_001",
        [
            pm_turn("marked at 999", mentions=(level_mention("EQ-0001", "price", 999.0),)),
            advisor_turn("noted"),
        ],
    )
    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden)

    result = _once(ctx, client, log=log)

    assert result.level_reasons == ("level not in market data: EQ-0001 - price 999.0",)
    assert result.passed is False
