"""Tests for the narrator and advisor system prompts and request builders."""

import dataclasses
import json

import pytest

from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import request_key
from pm_traitbench.dialogue.prompts import (
    NARRATOR_OPENING_MESSAGE,
    TURN_SCHEMA,
    advisor_request,
    advisor_system,
    narrator_directive,
    narrator_request,
    narrator_system,
    read_advisor_prompt,
)
from pm_traitbench.enums import Action, Op, RuleScope, RuleSource, SignalMode, StanceEntry
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Idea, Leg, Rule, Side, Stance
from tests.dialogue.conftest import session_context
from tests.gates.conftest import DEFAULT_DATE, ledger_row

_CONFIG = Config()


def _rule(
    rule_id: str, text: str, *, scope=RuleScope.PM, trade_idea_id=None, pm_id="pm_001"
) -> Rule:
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=RuleSource.SELF,
        scope=scope,
        trade_idea_id=trade_idea_id,
        param="stop_loss",
        field="pnl_pct",
        op=Op.LE,
        level=-5.0,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text=text,
    )


def test_narrator_request_has_exactly_the_binding_keys(market_lookup):
    ctx = session_context(market_lookup)
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]

    request = narrator_request(ctx, messages, _CONFIG.dialogue, None)

    assert set(request) == {
        "model",
        "max_tokens",
        "system",
        "messages",
        "output_config",
        "cache_control",
    }
    assert request["model"] == _CONFIG.dialogue.narrator_model
    assert request["max_tokens"] == _CONFIG.dialogue.max_output_tokens
    assert request["output_config"] == {
        "effort": _CONFIG.dialogue.effort.value,
        "format": {"type": "json_schema", "schema": TURN_SCHEMA},
    }
    assert request["cache_control"] == {"type": "ephemeral"}


def test_narrator_system_contains_voice_rules_and_avoid_lines(market_lookup):
    ctx = session_context(market_lookup)
    rule = _rule("r_01", "Cut a position after two consecutive stop triggers.")
    ctx = dataclasses.replace(
        ctx, pm_rules=(rule,), avoid_lines=("do not describe averaging into a losing position",)
    )

    system = narrator_system(ctx, None)

    assert ctx.voice.line in system
    assert rule.text in system
    for line in ctx.avoid_lines:
        assert line in system


def test_narrator_request_never_contains_trait_ids_params_or_values(market_lookup):
    stance = Stance(
        signal_id="sg_001",
        trait_id="t_07",
        mode=SignalMode.REVEALED,
        entry=StanceEntry.STATED,
        stance="I like adding to a position once the move keeps confirming through the session.",
    )
    ctx = session_context(market_lookup, stances=(stance,))
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]
    for i in range(len(ctx.turn_plan.pm_directives)):
        messages.append({"role": "system", "content": narrator_directive(ctx, i)})

    request = narrator_request(ctx, messages, _CONFIG.dialogue, None)

    serialized = json.dumps(request).lower()
    assert stance.trait_id.lower() not in serialized
    assert stance.mode.value.lower() not in serialized
    for param in BIAS_PARAMS:
        assert param.lower() not in serialized
    assert "bias" not in serialized


def test_directive_carries_the_stance_line_and_day_trades(market_lookup):
    stance = Stance(
        signal_id="sg_001",
        trait_id="t_02",
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance="I want to talk through today's book before anything else.",
    )
    trade = ledger_row(trade_idea_id="ti_001", date=DEFAULT_DATE)
    ctx = session_context(market_lookup, stances=(stance,), day_trades=(trade,))

    stance_index = next(
        i
        for i, directive in enumerate(ctx.turn_plan.pm_directives)
        if directive.stance is not None and directive.stance.trait_id == stance.trait_id
    )
    directive_text = narrator_directive(ctx, stance_index)
    assert f"In this message: {stance.stance}" in directive_text
    assert stance.trait_id not in directive_text
    assert stance.mode.value not in directive_text

    opening_text = narrator_directive(ctx, 0)
    assert "Mention each of these trades" in opening_text
    assert trade.trade_idea_id in opening_text


def test_feedback_changes_the_narrator_request_key(market_lookup):
    ctx = session_context(market_lookup)
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]

    base = narrator_request(ctx, messages, _CONFIG.dialogue, None)
    changed = narrator_request(
        ctx, messages, _CONFIG.dialogue, "Keep this PM's trades tighter to the skeleton."
    )

    assert base["system"] != changed["system"]
    assert request_key(base) != request_key(changed)


def test_advisor_request_contains_no_persona_rules_ideas_or_skeleton_text(market_lookup):
    stance = Stance(
        signal_id="sg_001",
        trait_id="t_03",
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance="A distinctive planted stance line about how I size positions.",
    )
    ctx = session_context(market_lookup, stances=(stance,))
    rule = _rule("r_01", "A distinctive rule text about trimming winners at target.")
    idea = Idea(
        pm_id=ctx.persona.pm_id,
        trade_idea_id="ti_001",
        instrument_id="EQ-0001",
        expression=ctx.ideas[0].expression,
        side=Side.BUY,
        legs=(Leg(instrument_id="EQ-0001", tenor=None, side=Side.BUY, weight=1.0),),
        entry_date=DEFAULT_DATE,
        exit_date=None,
        entry_level=100.0,
        target_level=110.0,
        stop_level=90.0,
        thesis="A distinctive thesis about durable moat compounding over the horizon.",
        outcome=None,
        own_signal=0.1,
        forecast=105.0,
        interval_lo=95.0,
        interval_hi=115.0,
        street_view_at_entry=ctx.ideas[0].street_view_at_entry,
        conflict=False,
        followed_street=None,
        conviction=3,
        size_rank=3,
        chased_trend=False,
    )
    ctx = dataclasses.replace(ctx, pm_rules=(rule,), ideas=(idea,))

    advisor_prompt = read_advisor_prompt(None)
    system = advisor_system(advisor_prompt, ctx.skeleton.date)
    messages = [{"role": "user", "content": "Any levels I should know about on my names?"}]
    request = advisor_request(system, messages, _CONFIG.dialogue)

    serialized = json.dumps(request)
    assert ctx.persona.stated_profile.self_description not in serialized
    assert rule.text not in serialized
    assert idea.thesis not in serialized
    assert stance.stance not in serialized


def test_advisor_request_with_tools_disabled_sets_tool_choice_none():
    messages = [{"role": "user", "content": "hi"}]

    default_request = advisor_request("system prompt", messages, _CONFIG.dialogue)
    disabled_request = advisor_request(
        "system prompt", messages, _CONFIG.dialogue, tools_disabled=True
    )

    assert "tool_choice" not in default_request
    assert disabled_request["tool_choice"] == {"type": "none"}
    assert set(disabled_request) - set(default_request) == {"tool_choice"}


def test_read_advisor_prompt_defaults_to_the_packaged_file():
    text = read_advisor_prompt(None)

    assert "investment copilot" in text


def test_read_advisor_prompt_missing_file_raises(tmp_path):
    with pytest.raises(DialogueError):
        read_advisor_prompt(tmp_path / "missing.md")
