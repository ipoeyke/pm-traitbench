"""Tests for the async session driver that narrates one PM-copilot session."""

import asyncio

import pytest

from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import CachedClient, request_key
from pm_traitbench.dialogue.prompts import opening_line
from pm_traitbench.dialogue.session import TurnOutput, narrate_session, parse_turn
from pm_traitbench.dialogue.turns import Opening, PmDirective, TurnPlan
from pm_traitbench.enums import AdvisorTool, SignalMode, StanceEntry, TurnRole
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Stance
from tests.dialogue.conftest import (
    FakeClient,
    default_responder,
    fake_message,
    session_context,
    tool_use,
    turn_text,
)

_CONFIG = Config().dialogue
_ADVISOR_PROMPT = "You are a market advisor for the PM's book."


def _turn_plan(
    n_pm: int,
    *,
    violation_advisor_index: int | None = None,
    stances_by_index: dict[int, Stance] | None = None,
) -> TurnPlan:
    """A turn plan with `n_pm` PM turns, an opening on turn 0, and no day trades."""
    stances_by_index = stances_by_index or {}
    directives = tuple(
        PmDirective(
            stance=stances_by_index.get(i),
            trades=(),
            opening=Opening.SESSION_IDEAS if i == 0 else None,
        )
        for i in range(n_pm)
    )
    return TurnPlan(
        n_turns=2 * n_pm, pm_directives=directives, violation_advisor_index=violation_advisor_index
    )


def _cached_client(responder, tmp_path):
    fake = FakeClient(responder)
    return fake, CachedClient(lambda: fake, tmp_path, token_budget=None)


def test_session_alternates_pm_and_advisor_for_the_planned_turn_count(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(3))
    _, client = _cached_client(default_responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert len(result.session.turns) == 6
    assert len(result.log.turns) == 6
    for index, turn in enumerate(result.session.turns):
        expected = TurnRole.PM if index % 2 == 0 else TurnRole.ADVISOR
        assert turn.role == expected
    assert result.warnings == ()


def test_advisor_tool_round_runs_the_tool_and_logs_the_call(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    instrument_id = sorted(market_lookup.instruments)[0]
    calls = {"advisor": 0}

    def responder(request):
        if "tools" not in request:
            return fake_message([turn_text("Checking in on the book.")])
        calls["advisor"] += 1
        if calls["advisor"] == 1:
            return fake_message(
                [tool_use("get_quote", {"instrument": instrument_id}, "tu_1")],
                stop_reason="tool_use",
            )
        return fake_message([turn_text("Price looks fine, holding steady.")])

    fake, client = _cached_client(responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    advisor_log = result.log.turns[1]
    assert len(advisor_log.tool_calls) == 1
    assert advisor_log.tool_calls[0].name == AdvisorTool.GET_QUOTE
    assert advisor_log.tool_calls[0].is_error is False

    advisor_requests = [r for r in fake.requests if "tools" in r]
    assert len(advisor_requests) == 2
    last_message = advisor_requests[1]["messages"][-1]
    assert last_message["role"] == "user"
    assert last_message["content"][0]["type"] == "tool_result"

    assert result.session.turns[1].text == "Price looks fine, holding steady."
    assert {"role", "text"} == set(result.session.turns[1].model_dump())


def test_tool_round_cap_forces_a_final_reply_and_warns(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    instrument_id = sorted(market_lookup.instruments)[0]

    def responder(request):
        if "tools" not in request:
            return fake_message([turn_text("Quick check-in.")])
        if request.get("tool_choice") == {"type": "none"}:
            return fake_message([turn_text("No more digging, wrapping up.")])
        return fake_message(
            [tool_use("get_quote", {"instrument": instrument_id}, "tu")], stop_reason="tool_use"
        )

    _, client = _cached_client(responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert result.warnings == (
        f"{ctx.skeleton.session_id}: advisor reply 0 hit the tool-round cap",
    )
    advisor_log = result.log.turns[1]
    assert len(advisor_log.tool_calls) == _CONFIG.max_tool_rounds


def test_refusal_is_retried_and_not_cached(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    calls = {"pm": 0}

    def responder(request):
        if "tools" in request:
            return fake_message([turn_text("All good here.")])
        calls["pm"] += 1
        if calls["pm"] == 1:
            return fake_message([], stop_reason="refusal")
        return fake_message([turn_text("Feeling good about the book today.")])

    _, client = _cached_client(responder, tmp_path)

    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert calls["pm"] == 2
    cache_files = list(tmp_path.rglob("*.json"))
    # one cache file per accepted reply: the retried pm turn plus the advisor turn
    assert len(cache_files) == 2


def test_blank_reply_is_retried_and_never_committed(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    calls = {"pm": 0}

    def responder(request):
        if "tools" in request:
            return fake_message([turn_text("fine")])
        calls["pm"] += 1
        if calls["pm"] == 1:
            return fake_message([turn_text("   ")])
        return fake_message([turn_text("Feeling good about the book today.")])

    _, client = _cached_client(responder, tmp_path)

    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert calls["pm"] == 2
    cache_files = list(tmp_path.rglob("*.json"))
    # one cache file per accepted reply: the blank reply is never cached
    assert len(cache_files) == 2


def test_retry_resends_an_identical_request(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    calls = {"pm": 0}

    def responder(request):
        if "tools" in request:
            return fake_message([turn_text("fine")])
        calls["pm"] += 1
        if calls["pm"] == 1:
            return fake_message([], stop_reason="refusal")
        return fake_message([turn_text("Feeling good about the book today.")])

    fake, client = _cached_client(responder, tmp_path)

    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    narrator_requests = [r for r in fake.requests if "tools" not in r]
    assert len(narrator_requests) == 2
    assert narrator_requests[0] == narrator_requests[1]


def test_unknown_tool_name_is_rejected_and_retried(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    calls = {"advisor": 0}

    def responder(request):
        if "tools" not in request:
            return fake_message([turn_text("Quick check-in.")])
        calls["advisor"] += 1
        if calls["advisor"] == 1:
            return fake_message([tool_use("not_a_real_tool", {}, "tu_1")], stop_reason="tool_use")
        return fake_message([turn_text("All set.")])

    _, client = _cached_client(responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert calls["advisor"] == 2
    advisor_log = result.log.turns[1]
    assert advisor_log.tool_calls == ()
    assert len(advisor_log.request_hashes) == 1


def test_tool_use_block_with_a_missing_name_is_rejected_without_crashing(market_lookup, tmp_path):
    """A block with no `name` key mixed with a known-bad string name must not raise
    `TypeError` from sorting `None` against `str` while building the rejection reason.
    """
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    calls = {"advisor": 0}

    def responder(request):
        if "tools" not in request:
            return fake_message([turn_text("Quick check-in.")])
        calls["advisor"] += 1
        if calls["advisor"] == 1:
            nameless_block = {"type": "tool_use", "id": "tu_1", "input": {}}
            return fake_message(
                [nameless_block, tool_use("not_a_real_tool", {}, "tu_2")], stop_reason="tool_use"
            )
        return fake_message([turn_text("All set.")])

    _, client = _cached_client(responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert calls["advisor"] == 2
    advisor_log = result.log.turns[1]
    assert advisor_log.tool_calls == ()


def test_no_narrator_feed_after_the_last_advisor_reply(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(3))
    fake, client = _cached_client(default_responder, tmp_path)

    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    narrator_requests = [r for r in fake.requests if "tools" not in r]
    advisor_text = "Sounds reasonable, tell me more."
    # a mid-session narrator turn is fed the previous advisor reply
    assert any(
        m.get("role") == "user" and m.get("content") == advisor_text
        for m in narrator_requests[1]["messages"]
    )
    # exactly one narrator request per pm turn and one advisor call per
    # advisor turn: the session's last advisor reply never triggers a
    # further narrator request
    assert len(narrator_requests) == 3
    assert len(fake.requests) == 6


def test_usage_sums_and_request_hashes_are_ordered_across_a_tool_round(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    instrument_id = sorted(market_lookup.instruments)[0]
    calls = {"advisor": 0}

    def responder(request):
        if "tools" not in request:
            return fake_message([turn_text("Quick check-in.")])
        calls["advisor"] += 1
        if calls["advisor"] == 1:
            return fake_message(
                [tool_use("get_quote", {"instrument": instrument_id}, "tu_1")],
                stop_reason="tool_use",
                input_tokens=20,
                output_tokens=8,
            )
        return fake_message([turn_text("All set.")], input_tokens=15, output_tokens=6)

    fake, client = _cached_client(responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    advisor_log = result.log.turns[1]
    assert advisor_log.usage.input_tokens == 35
    assert advisor_log.usage.output_tokens == 14
    advisor_requests = [r for r in fake.requests if "tools" in r]
    scope = ctx.skeleton.session_id
    assert advisor_log.request_hashes == tuple(request_key(r, scope) for r in advisor_requests)


def test_pm_directive_is_stance_then_opening_then_none(market_lookup, tmp_path):
    stance = Stance(
        signal_id="sg_010",
        trait_id="t_10",
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance="I'm trimming into strength today.",
    )
    ctx = session_context(market_lookup, turn_plan=_turn_plan(3, stances_by_index={1: stance}))
    _, client = _cached_client(default_responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    pm_logs = [turn for turn in result.log.turns if turn.role == TurnRole.PM]
    assert pm_logs[0].directive == opening_line(ctx, Opening.SESSION_IDEAS)
    assert pm_logs[1].directive == "I'm trimming into strength today."
    assert pm_logs[2].directive is None


def test_advisor_history_preserves_thinking_and_tool_blocks_unchanged(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    instrument_id = sorted(market_lookup.instruments)[0]
    calls = {"advisor": 0}
    thinking_block = {"type": "thinking", "thinking": "let me check the quote first"}
    tool_block = tool_use("get_quote", {"instrument": instrument_id}, "tu_1")

    def responder(request):
        if "tools" not in request:
            return fake_message([turn_text("Quick check-in.")])
        calls["advisor"] += 1
        if calls["advisor"] == 1:
            return fake_message([thinking_block, tool_block], stop_reason="tool_use")
        return fake_message([turn_text("All set.")])

    fake, client = _cached_client(responder, tmp_path)

    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    advisor_requests = [r for r in fake.requests if "tools" in r]
    # the second advisor request's history carries the first reply's full
    # content, thinking and tool blocks included, unchanged
    assert advisor_requests[1]["messages"][-2] == {
        "role": "assistant",
        "content": [thinking_block, tool_block],
    }


def test_schema_invalid_output_is_retried_then_fails_after_max_retries(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    attempts = {"pm": 0}

    def responder(request):
        if "tools" in request:
            return fake_message([turn_text("fine")])
        attempts["pm"] += 1
        return fake_message([{"type": "text", "text": "not json"}])

    _, client = _cached_client(responder, tmp_path)

    with pytest.raises(DialogueError, match=ctx.skeleton.session_id):
        asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert attempts["pm"] == 1 + _CONFIG.max_retries


def test_violation_line_is_a_system_message_right_before_the_reaction_turn(market_lookup, tmp_path):
    reaction = Stance(
        signal_id="sg_001",
        trait_id="t_01",
        mode=SignalMode.REVEALED,
        entry=StanceEntry.REVEALED_REACTION,
        stance="That's not what I asked you to do.",
    )
    ctx = session_context(
        market_lookup,
        stances=(reaction,),
        turn_plan=_turn_plan(2, violation_advisor_index=0, stances_by_index={1: reaction}),
        advisor_violation="Sold half the position without asking.",
    )
    fake, client = _cached_client(default_responder, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    advisor_requests = [r for r in fake.requests if "tools" in r]
    assert advisor_requests[0]["messages"][-2] == {
        "role": "user",
        "content": "Feeling good about the book today.",
    }
    assert advisor_requests[0]["messages"][-1] == {
        "role": "system",
        "content": "Sold half the position without asking.",
    }
    advisor_log = result.log.turns[1]
    assert advisor_log.scripted_violation is True
    assert advisor_log.directive == "Sold half the position without asking."
    # only the scripted reply carries the violation
    assert result.log.turns[3].scripted_violation is False


def test_narrator_directives_are_system_messages_and_history_is_append_only(
    market_lookup, tmp_path
):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(3))
    fake, client = _cached_client(default_responder, tmp_path)

    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    narrator_requests = [r for r in fake.requests if "tools" not in r]
    assert len(narrator_requests) == 3
    for request in narrator_requests:
        assert request["messages"][-1]["role"] == "system"
    for earlier, later in zip(narrator_requests, narrator_requests[1:], strict=False):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


def test_rerun_from_cache_gives_identical_session_and_log_with_no_inner_calls(
    market_lookup, tmp_path
):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(2))

    warm = CachedClient(lambda: FakeClient(default_responder), tmp_path, token_budget=None)
    first = asyncio.run(narrate_session(ctx, warm, _CONFIG, _ADVISOR_PROMPT))

    def _fail() -> FakeClient:
        raise AssertionError("inner client must not be called on a fully cached rerun")

    cold = CachedClient(_fail, tmp_path, token_budget=None)
    second = asyncio.run(narrate_session(ctx, cold, _CONFIG, _ADVISOR_PROMPT))

    assert second.session == first.session
    assert second.log == first.log
    assert second.warnings == first.warnings


def test_feedback_reaches_the_narrator_system_prompt(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    fake, client = _cached_client(default_responder, tmp_path)

    asyncio.run(
        narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT, feedback="Stop inventing trades.")
    )

    narrator_requests = [r for r in fake.requests if "tools" not in r]
    assert "Stop inventing trades." in narrator_requests[0]["system"]


def test_parse_turn_requires_end_turn_stop_reason():
    response = fake_message([turn_text("hi")], stop_reason="max_tokens")
    assert parse_turn(response) is None


def test_parse_turn_requires_a_text_block():
    response = fake_message([{"type": "tool_use", "id": "t", "name": "get_quote", "input": {}}])
    assert parse_turn(response) is None


def test_parse_turn_rejects_an_invalid_mention():
    response = fake_message([turn_text("hi", mentions=[{"kind": "trade"}])])
    assert parse_turn(response) is None


def test_parse_turn_rejects_blank_text():
    assert parse_turn(fake_message([turn_text("")])) is None
    assert parse_turn(fake_message([turn_text("   ")])) is None


def test_parse_turn_uses_the_last_text_block():
    response = fake_message([turn_text("first"), turn_text("second")])
    output = parse_turn(response)
    assert isinstance(output, TurnOutput)
    assert output.text == "second"
