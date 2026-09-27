"""Tests for the validate stage's per-session attempt loop."""

import asyncio
import re

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.session import SessionResult
from pm_traitbench.dialogue.validate.loop import feedback_text, run_session, validate_once
from pm_traitbench.enums import Side, SignalMode, StanceEntry, TurnRole, ValidationStatus
from pm_traitbench.tables.schema import Session, Turn
from tests.dialogue.conftest import fake_message, turn_text
from tests.dialogue.validate.conftest import (
    advisor_turn,
    forbidden_reply,
    leak_reply,
    ledger_row,
    log_of,
    pm_turn,
    scripted_client,
    stance,
    trade_mention,
    validate_context,
)
from tests.gates.conftest import DEFAULT_DATE, PM_ID

_ADVISOR_PROMPT = "You are a market advisor for the PM's book."


def _raise(_request):
    raise AssertionError("this judge or narrator must not be called")


def _clean_forbidden(_request):
    return forbidden_reply([])


def _turn_reply(text: str, mentions: list[dict]) -> dict:
    return fake_message([turn_text(text, mentions)])


def _session_from(ctx, texts: tuple[str, str]) -> Session:
    return Session(
        session_id=ctx.skeleton.session_id,
        pm_id=ctx.skeleton.pm_id,
        date=ctx.skeleton.date,
        kind=ctx.skeleton.kind,
        trade_idea_ids=ctx.skeleton.trade_idea_ids,
        turns=(Turn(role=TurnRole.PM, text=texts[0]), Turn(role=TurnRole.ADVISOR, text=texts[1])),
    )


def test_clean_session_passes_on_attempt_one_with_no_narrator_call(market_lookup, tmp_path):
    ctx = validate_context(market_lookup)
    session = _session_from(ctx, ("all clear on the book", "noted"))
    log = log_of(
        ctx.skeleton.session_id,
        ctx.skeleton.pm_id,
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden)

    outcome = asyncio.run(
        run_session(
            ctx,
            session,
            log,
            client,
            Config(),
            load_catalogue(),
            _ADVISOR_PROMPT,
            ledger=(),
            grep_params=("register", "loss_aversion_lambda"),
            trait_param_by_id={},
        )
    )

    assert len(outcome.rows) == 1
    assert outcome.rows[0].status == ValidationStatus.PASS
    assert outcome.final == SessionResult(session=session, log=log, warnings=(), rejected_replies=0)


def test_leak_judge_runs_only_with_a_revealed_or_contradiction_stance(market_lookup, tmp_path):
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    catalogue = load_catalogue()
    config = Config()

    stated_ctx = validate_context(
        market_lookup, stances=(stance("t_01", SignalMode.STATED, StanceEntry.STATED, "line"),)
    )
    stated_client = scripted_client(tmp_path / "stated", _raise, _raise, _clean_forbidden)
    stated_result = asyncio.run(
        validate_once(
            stated_ctx,
            log,
            stated_client,
            config,
            catalogue,
            ledger=(),
            grep_params=(),
            trait_param_by_id={"t_01": "disposition_ratio"},
        )
    )
    assert stated_result.leak_judged is False

    revealed_ctx = validate_context(
        market_lookup, stances=(stance("t_01", SignalMode.REVEALED, StanceEntry.REVEALED, "line"),)
    )
    calls = {"n": 0}

    def leak_called(_request):
        calls["n"] += 1
        return leak_reply(False, None)

    revealed_client = scripted_client(tmp_path / "revealed", _raise, leak_called, _clean_forbidden)
    revealed_result = asyncio.run(
        validate_once(
            revealed_ctx,
            log,
            revealed_client,
            config,
            catalogue,
            ledger=(),
            grep_params=(),
            trait_param_by_id={"t_01": "disposition_ratio"},
        )
    )
    assert revealed_result.leak_judged is True
    assert calls["n"] == 1

    contradiction_ctx = validate_context(
        market_lookup,
        stances=(stance("t_01", SignalMode.CONTRADICTION, StanceEntry.CLAIM, "line"),),
    )
    contradiction_calls = {"n": 0}

    def leak_called_contradiction(_request):
        contradiction_calls["n"] += 1
        return leak_reply(False, None)

    contradiction_client = scripted_client(
        tmp_path / "contradiction", _raise, leak_called_contradiction, _clean_forbidden
    )
    contradiction_result = asyncio.run(
        validate_once(
            contradiction_ctx,
            log,
            contradiction_client,
            config,
            catalogue,
            ledger=(),
            grep_params=(),
            trait_param_by_id={"t_01": "disposition_ratio"},
        )
    )
    assert contradiction_result.leak_judged is True
    assert contradiction_calls["n"] == 1


def test_explicit_label_of_the_revealed_param_fails_and_other_labels_pass(market_lookup, tmp_path):
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    catalogue = load_catalogue()
    config = Config()
    ctx = validate_context(
        market_lookup, stances=(stance("t_01", SignalMode.REVEALED, StanceEntry.REVEALED, "line"),)
    )
    trait_param_by_id = {"t_01": "disposition_ratio"}

    def _once(tmp_sub, leak_fn):
        client = scripted_client(tmp_path / tmp_sub, _raise, leak_fn, _clean_forbidden)
        return asyncio.run(
            validate_once(
                ctx,
                log,
                client,
                config,
                catalogue,
                ledger=(),
                grep_params=(),
                trait_param_by_id=trait_param_by_id,
            )
        )

    revealed_label = _once("a", lambda _r: leak_reply(True, "disposition effect", "I sold winners"))
    assert revealed_label.leak_reasons == ('leaks disposition_ratio: "I sold winners"',)
    assert revealed_label.passed is False

    other_label = _once("b", lambda _r: leak_reply(True, "loss aversion", "average down again"))
    assert other_label.leak_reasons == ()
    assert other_label.unmapped_labels == ()
    assert other_label.passed is True

    not_explicit = _once("c", lambda _r: leak_reply(False, "disposition effect", "quote"))
    assert not_explicit.leak_reasons == ()
    assert not_explicit.passed is True


def test_explicit_label_naming_a_stated_traits_param_passes(market_lookup, tmp_path):
    """A stated stance's param never needs judging; an explicit label naming it must not
    fail the session just because some other stance in the session happens to be revealed.
    """
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    catalogue = load_catalogue()
    ctx = validate_context(
        market_lookup,
        stances=(
            stance("t_01", SignalMode.STATED, StanceEntry.STATED, "line a"),
            stance("t_02", SignalMode.REVEALED, StanceEntry.REVEALED, "line b"),
        ),
    )
    client = scripted_client(
        tmp_path,
        _raise,
        lambda _r: leak_reply(True, "loss aversion", "average down again"),
        _clean_forbidden,
    )

    result = asyncio.run(
        validate_once(
            ctx,
            log,
            client,
            Config(),
            catalogue,
            ledger=(),
            grep_params=(),
            trait_param_by_id={"t_01": "loss_aversion_lambda", "t_02": "disposition_ratio"},
        )
    )

    assert result.leak_reasons == ()
    assert result.unmapped_labels == ()
    assert result.passed is True


def test_unmapped_label_is_a_warning_not_a_failure(market_lookup, tmp_path):
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    catalogue = load_catalogue()
    ctx = validate_context(
        market_lookup, stances=(stance("t_01", SignalMode.REVEALED, StanceEntry.REVEALED, "line"),)
    )
    client = scripted_client(
        tmp_path,
        _raise,
        lambda _r: leak_reply(True, "a label nobody catalogued", "quote"),
        _clean_forbidden,
    )

    result = asyncio.run(
        validate_once(
            ctx,
            log,
            client,
            Config(),
            catalogue,
            ledger=(),
            grep_params=(),
            trait_param_by_id={"t_01": "disposition_ratio"},
        )
    )

    assert result.leak_reasons == ()
    assert result.unmapped_labels == ("a label nobody catalogued",)
    assert result.passed is True

    blank_client = scripted_client(
        tmp_path / "blank", _raise, lambda _r: leak_reply(True, "   ", "quote"), _clean_forbidden
    )
    blank_result = asyncio.run(
        validate_once(
            ctx,
            log,
            blank_client,
            Config(),
            catalogue,
            ledger=(),
            grep_params=(),
            trait_param_by_id={"t_01": "disposition_ratio"},
        )
    )
    assert blank_result.unmapped_labels == ()
    assert blank_result.passed is True


def test_forbidden_violation_fails_with_the_avoid_line_and_out_of_range_index_is_dropped(
    market_lookup, tmp_path
):
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    ctx = validate_context(
        market_lookup,
        avoid_lines=("never mention position size", "never predict a specific price"),
    )
    client = scripted_client(
        tmp_path,
        _raise,
        _raise,
        lambda _r: forbidden_reply([(1, "quoted the size"), (5, "out of range")]),
    )

    result = asyncio.run(
        validate_once(
            ctx,
            log,
            client,
            Config(),
            load_catalogue(),
            ledger=(),
            grep_params=(),
            trait_param_by_id={},
        )
    )

    assert result.forbidden_reasons == (
        'forbidden: never mention position size: "quoted the size"',
    )
    assert result.passed is False


def test_forbidden_verdict_with_every_index_out_of_range_passes(market_lookup, tmp_path):
    log = log_of(
        "s_pm001_2026-01-05_a",
        "pm_001",
        [pm_turn("all clear on the book"), advisor_turn("noted")],
    )
    ctx = validate_context(market_lookup, avoid_lines=("never mention position size",))
    client = scripted_client(
        tmp_path,
        _raise,
        _raise,
        lambda _r: forbidden_reply([(0, "zero index"), (5, "too high")]),
    )

    result = asyncio.run(
        validate_once(
            ctx,
            log,
            client,
            Config(),
            load_catalogue(),
            ledger=(),
            grep_params=(),
            trait_param_by_id={},
        )
    )

    assert result.forbidden_reasons == ()
    assert result.passed is True


def test_failed_session_is_regenerated_with_feedback_and_passes_on_attempt_two(
    market_lookup, tmp_path
):
    row = ledger_row(PM_ID, DEFAULT_DATE, "ti_001", "EQ-0001", Side.BUY, 100.0)
    ctx = validate_context(market_lookup, day_trades=(row,))
    session = _session_from(ctx, ("my loss_aversion_lambda is high", "noted"))
    log = log_of(
        ctx.skeleton.session_id,
        ctx.skeleton.pm_id,
        [pm_turn("my loss_aversion_lambda is high"), advisor_turn("noted")],
    )

    calls = {"n": 0}

    def narrator_reply(request):
        assert "Attempt 2." in request["system"]
        assert "names a parameter: loss_aversion_lambda" in request["system"]
        mentions = [trade_mention(row).model_dump(mode="json")] if calls["n"] == 0 else []
        calls["n"] += 1
        return _turn_reply("feeling calm about the book today", mentions)

    client = scripted_client(tmp_path, narrator_reply, _raise, _clean_forbidden)

    outcome = asyncio.run(
        run_session(
            ctx,
            session,
            log,
            client,
            Config(),
            load_catalogue(),
            _ADVISOR_PROMPT,
            ledger=(row,),
            grep_params=("loss_aversion_lambda",),
            trait_param_by_id={},
        )
    )

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
    session = _session_from(ctx, ("my loss_aversion_lambda is high", "noted"))
    log = log_of(
        ctx.skeleton.session_id,
        ctx.skeleton.pm_id,
        [pm_turn("my loss_aversion_lambda is high"), advisor_turn("noted")],
    )
    systems_by_attempt: dict[int, set[str]] = {}

    def narrator_reply(request):
        match = re.search(r"Attempt (\d+)\.", request["system"])
        assert match is not None
        systems_by_attempt.setdefault(int(match.group(1)), set()).add(request["system"])
        return _turn_reply("still naming my loss_aversion_lambda here", [])

    config = Config(
        validation=Config().validation.model_copy(update={"max_attempts": max_attempts})
    )
    client = scripted_client(tmp_path, narrator_reply, _raise, _clean_forbidden)

    outcome = asyncio.run(
        run_session(
            ctx,
            session,
            log,
            client,
            config,
            load_catalogue(),
            _ADVISOR_PROMPT,
            ledger=(),
            grep_params=("loss_aversion_lambda",),
            trait_param_by_id={},
        )
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
    ctx = validate_context(market_lookup)
    session = _session_from(ctx, ("my loss_aversion_lambda is high", "noted"))
    log = log_of(
        ctx.skeleton.session_id,
        ctx.skeleton.pm_id,
        [pm_turn("my loss_aversion_lambda is high"), advisor_turn("noted")],
    )
    config = Config(validation=Config().validation.model_copy(update={"max_attempts": 1}))
    client = scripted_client(tmp_path, _raise, _raise, _clean_forbidden)

    outcome = asyncio.run(
        run_session(
            ctx,
            session,
            log,
            client,
            config,
            load_catalogue(),
            _ADVISOR_PROMPT,
            ledger=(),
            grep_params=("loss_aversion_lambda",),
            trait_param_by_id={},
        )
    )

    assert [r.status for r in outcome.rows] == [ValidationStatus.DROPPED]
    assert outcome.final is None


def test_feedback_text_format():
    text = feedback_text(2, ["reason one", "reason two"])

    assert text == (
        "Attempt 2. A validator rejected the previous version of this session:\n"
        "- reason one\n"
        "- reason two\n"
        "Fix these and keep everything else as instructed."
    )
