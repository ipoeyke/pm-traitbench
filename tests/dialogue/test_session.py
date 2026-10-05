"""Tests for the async session driver that narrates one PM-copilot session."""

import asyncio
import json
import re

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Voice
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import CachedClient, request_key
from pm_traitbench.dialogue.context import PmTables, build_contexts
from pm_traitbench.dialogue.prompts import opening_line
from pm_traitbench.dialogue.session import TurnOutput, _send_accepted, narrate_session, parse_turn
from pm_traitbench.dialogue.turns import Opening, PmDirective, TurnPlan
from pm_traitbench.enums import (
    AdvisorTool,
    Kind,
    RuleScope,
    SessionKind,
    SignalMode,
    StanceEntry,
    TurnRole,
)
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Skeleton, Stance
from tests.dialogue.fixtures import (
    FakeClient,
    default_responder,
    fake_message,
    rule,
    session_context,
    tool_use,
    turn_text,
)
from tests.gates.fixtures import DEFAULT_DATE, idea_row
from tests.signals.fixtures import bias_trait, persona, pref_trait

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
    assert result.rejected_replies == 0


def test_send_accepted_refreshes_past_a_cached_reply_that_fails_validation(tmp_path):
    """A cache entry that validated under an older, looser check can fail today's `_classify`;
    the retry must bypass that entry rather than replaying it forever.
    """
    request = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        return fake_message([turn_text("Feeling good about the book today.")])

    inner = FakeClient(responder)
    client = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(request, "s_test")
    shard = tmp_path / key[:2]
    shard.mkdir(parents=True)
    stale = fake_message([turn_text("hi", mentions=[{"kind": "trade"}])])
    (shard / f"{key}.json").write_text(
        json.dumps({"key": key, "response": stale}), encoding="utf-8"
    )

    reply, accepted, rejected, _ = asyncio.run(
        _send_accepted(client, request, _CONFIG, "s_test", allow_tool_use=False)
    )

    assert calls["n"] == 1  # exactly one fresh inner call, no wasted retries
    assert rejected == 1
    assert accepted.output.text == "Feeling good about the book today."
    stored = json.loads((shard / f"{key}.json").read_text(encoding="utf-8"))
    assert stored["response"] == reply.response


def test_send_accepted_refresh_stays_sticky_after_the_first_stale_cache_hit(tmp_path):
    """Once a stale cached reply is rejected, every later attempt must go to the API, not
    alternate back onto the same stale entry when a fresh reply is also rejected.
    """
    request = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        return fake_message([turn_text("   ")])  # always blank: always rejected

    inner = FakeClient(responder)
    client = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(request, "s_test")
    shard = tmp_path / key[:2]
    shard.mkdir(parents=True)
    stale = fake_message([turn_text("hi", mentions=[{"kind": "trade"}])])
    (shard / f"{key}.json").write_text(
        json.dumps({"key": key, "response": stale}), encoding="utf-8"
    )

    with pytest.raises(DialogueError):
        asyncio.run(_send_accepted(client, request, _CONFIG, "s_test", allow_tool_use=False))

    assert calls["n"] == _CONFIG.max_retries  # every attempt after the stale hit is fresh
    assert client.totals.cache_hits == 1


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
        f"session {ctx.skeleton.session_id}: advisor reply 0 hit the tool-round cap",
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

    # Without a fallback model, a refusal is retried on the same model and never cached.
    result = asyncio.run(narrate_session(ctx, client, _NO_FALLBACK, _ADVISOR_PROMPT))

    assert calls["pm"] == 2
    cache_files = list(tmp_path.rglob("*.json"))
    # one cache file per accepted reply: the retried pm turn plus the advisor turn
    assert len(cache_files) == 2
    assert result.rejected_replies == 1


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

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert calls["pm"] == 2
    cache_files = list(tmp_path.rglob("*.json"))
    # one cache file per accepted reply: the blank reply is never cached
    assert len(cache_files) == 2
    assert result.rejected_replies == 1


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

    asyncio.run(narrate_session(ctx, client, _NO_FALLBACK, _ADVISOR_PROMPT))

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


def test_session_level_requests_never_leak_across_the_narrator_or_advisor_history(
    market_lookup, tmp_path
):
    """Run `narrate_session` on a context built from real catalogue content, with a
    violation in play, then scan every recorded narrator and advisor request the driver
    produced, not just one hand-built request.
    """
    catalogue = load_catalogue()
    idea = idea_row(trade_idea_id="ti_001")
    pm_persona = persona()
    asset_class = pm_persona.mandate.asset_class

    loss_aversion = bias_trait("loss_aversion_lambda", trait_id="t_01", value=1.7345)
    herding = bias_trait("herding_weight", trait_id="t_02", value=0.5137)
    register_values = next(p.values for p in catalogue.preferences if p.param == "register")
    register_pref = pref_trait("register", register_values[0], trait_id="t_90")

    pm_rule = rule(rule_id="r_01", text="Cap total book risk at the mandate ceiling.")
    idea_rule = rule(
        rule_id="r_02",
        text="Trim ti_001 by half once it reaches target.",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
    )

    reaction_line = catalogue.stances.lines(
        "communication", StanceEntry.REVEALED_REACTION, asset_class
    )[0].format(value=register_values[0])
    reaction = Stance(
        signal_id="sg_001",
        trait_id=register_pref.trait_id,
        mode=SignalMode.REVEALED,
        entry=StanceEntry.REVEALED_REACTION,
        stance=reaction_line,
    )

    skeleton = Skeleton(
        session_id="s_pm001_2026-01-05_a",
        pm_id=pm_persona.pm_id,
        date=DEFAULT_DATE,
        kind=SessionKind.DECISION,
        trade_idea_ids=("ti_001",),
        stances=(reaction,),
        advisor_violation="Sold half the position without asking.",
        forbidden_trait_ids=(loss_aversion.trait_id, herding.trait_id),
        forbidden_pref_params=("register", "pushback_style"),
    )
    pm = PmTables(
        persona=pm_persona,
        traits=(loss_aversion, herding, register_pref),
        drift_events=(),
        rules=(pm_rule, idea_rule),
        ideas={"ti_001": idea},
        ledger=(),
        position_days=(),
        skeletons=(skeleton,),
    )
    voice = Voice(voice_id="v_01", line="terse trader shorthand, drops articles")
    ctx = build_contexts(pm, voice, market_lookup, catalogue, Config())[0]

    fake, client = _cached_client(default_responder, tmp_path)
    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    narrator_requests = [r for r in fake.requests if "tools" not in r]
    advisor_requests = [r for r in fake.requests if "tools" in r]
    assert narrator_requests and advisor_requests

    for request in narrator_requests:
        serialized = json.dumps(request).lower()
        for trait in pm.traits:
            assert trait.trait_id.lower() not in serialized
        for param in BIAS_PARAMS:
            assert param.lower() not in serialized
        for trait in pm.traits:
            if trait.kind == Kind.BIAS:
                assert str(trait.value).lower() not in serialized
        for mode in SignalMode:
            assert mode.value.lower() not in serialized
        assert re.search(r"\bbias\b", serialized, re.IGNORECASE) is None

    for request in advisor_requests:
        serialized = json.dumps(request)
        assert ctx.persona.stated_profile.self_description not in serialized
        assert pm_rule.text not in serialized
        assert idea_rule.text not in serialized
        assert idea.thesis not in serialized
        assert reaction.stance not in serialized


# --- refusal fallback -------------------------------------------------------------------------

_REFUSAL = fake_message(
    [{"type": "text", "text": "I can't help with that."}], stop_reason="refusal"
)
_REFUSAL["stop_details"] = {
    "type": "refusal",
    "category": "reasoning_extraction",
    "explanation": "asked to narrate its reasoning",
}
_NO_FALLBACK = _CONFIG.model_copy(update={"refusal_fallback_model": None})


def _refuse_primary(request):
    if request["model"] == _CONFIG.narrator_model and "tools" not in request:
        return _REFUSAL
    return default_responder(request)


def test_refused_narrator_turn_is_answered_by_the_fallback_model(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    fake, client = _cached_client(_refuse_primary, tmp_path)

    result = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    narrator_requests = [r for r in fake.requests if "tools" not in r]
    assert [r["model"] for r in narrator_requests] == [
        _CONFIG.narrator_model,
        _CONFIG.refusal_fallback_model,
    ]
    assert narrator_requests[1] == {**narrator_requests[0], "model": _CONFIG.refusal_fallback_model}
    pm_turn = result.log.turns[0]
    assert pm_turn.model == _CONFIG.refusal_fallback_model
    assert result.log.turns[1].model == _CONFIG.advisor_model
    assert result.rejected_replies == 1


def test_cached_refusal_replays_straight_to_the_fallback(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    fake, client = _cached_client(_refuse_primary, tmp_path)
    asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))
    calls_before = len(fake.requests)

    again = asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    assert len(fake.requests) == calls_before
    assert again.log.turns[0].model == _CONFIG.refusal_fallback_model


def test_refusal_without_a_fallback_fails_with_the_category(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    fake, client = _cached_client(_refuse_primary, tmp_path)

    with pytest.raises(
        DialogueError,
        match=r"the reply was refused \(reasoning_extraction: asked to narrate its reasoning\)",
    ):
        asyncio.run(narrate_session(ctx, client, _NO_FALLBACK, _ADVISOR_PROMPT))

    assert {r["model"] for r in fake.requests} == {_CONFIG.narrator_model}
    assert len(fake.requests) == 1 + _CONFIG.max_retries


def test_refusal_by_the_fallback_too_fails_without_a_second_fallback(market_lookup, tmp_path):
    ctx = session_context(market_lookup, turn_plan=_turn_plan(1))
    fake, client = _cached_client(lambda _request: _REFUSAL, tmp_path)

    with pytest.raises(DialogueError, match="the reply was refused"):
        asyncio.run(narrate_session(ctx, client, _CONFIG, _ADVISOR_PROMPT))

    models = [r["model"] for r in fake.requests]
    assert models[0] == _CONFIG.narrator_model
    assert set(models[1:]) == {_CONFIG.refusal_fallback_model}
    assert len(models) == 1 + 1 + _CONFIG.max_retries
