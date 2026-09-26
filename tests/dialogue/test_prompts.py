"""Tests for the narrator and advisor system prompts and request builders."""

import dataclasses
import json
import re

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Voice
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import request_key
from pm_traitbench.dialogue.context import PmTables, build_contexts
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
from pm_traitbench.dialogue.tools import TOOL_DEFINITIONS
from pm_traitbench.enums import (
    InstrumentKind,
    Kind,
    RuleScope,
    SessionKind,
    SignalMode,
    StanceEntry,
    Tenor,
)
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import Idea, Leg, Side, Skeleton, Stance
from tests.dialogue.conftest import rule, session_context
from tests.gates.conftest import DEFAULT_DATE, idea_row, ledger_row
from tests.signals.conftest import bias_trait, persona, pref_trait

_CONFIG = Config()


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


def test_advisor_request_has_exactly_the_binding_keys():
    messages = [{"role": "user", "content": "hi"}]

    request = advisor_request("system prompt", messages, _CONFIG.dialogue)

    assert set(request) == {
        "model",
        "max_tokens",
        "system",
        "messages",
        "output_config",
        "cache_control",
        "tools",
    }
    assert request["tools"] == list(TOOL_DEFINITIONS)


def test_two_builds_from_equal_inputs_give_the_same_request_key(market_lookup):
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]

    ctx_1 = session_context(market_lookup)
    ctx_2 = session_context(market_lookup)
    request_1 = narrator_request(ctx_1, messages, _CONFIG.dialogue, None)
    request_2 = narrator_request(ctx_2, list(messages), _CONFIG.dialogue, None)

    assert request_1 == request_2
    assert request_key(request_1) == request_key(request_2)


def test_narrator_system_contains_voice_rules_and_avoid_lines(market_lookup):
    ctx = session_context(market_lookup)
    pm_rule = rule(rule_id="r_01", text="Cut a position after two consecutive stop triggers.")
    ctx = dataclasses.replace(
        ctx,
        pm_rules=(pm_rule,),
        avoid_lines=("do not describe averaging into a losing position",),
    )

    system = narrator_system(ctx, None)

    assert ctx.voice.line in system
    assert pm_rule.text in system
    for line in ctx.avoid_lines:
        assert line in system


def test_narrator_system_omits_the_avoid_header_when_there_are_no_avoid_lines(market_lookup):
    ctx = session_context(market_lookup)
    assert ctx.avoid_lines == ()

    system = narrator_system(ctx, None)

    assert "Never do any of the following" not in system


def test_narrator_request_never_leaks_trait_or_bias_information(market_lookup):
    """Build a context with real traits, catalogue avoid lines, rules and a real stance
    line, then scan the whole narrator request for every way a trait could leak.
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

    stance_line = catalogue.stances.lines("loss_aversion_lambda", StanceEntry.STATED, asset_class)[
        0
    ]
    stance = Stance(
        signal_id="sg_001",
        trait_id=loss_aversion.trait_id,
        mode=SignalMode.STATED,
        entry=StanceEntry.STATED,
        stance=stance_line,
    )

    skeleton = Skeleton(
        session_id="s_pm001_2026-01-05_a",
        pm_id=pm_persona.pm_id,
        date=DEFAULT_DATE,
        kind=SessionKind.DECISION,
        trade_idea_ids=("ti_001",),
        stances=(stance,),
        advisor_violation=None,
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

    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]
    for i in range(len(ctx.turn_plan.pm_directives)):
        messages.append({"role": "system", "content": narrator_directive(ctx, i)})
    request = narrator_request(ctx, messages, _CONFIG.dialogue, None)
    serialized_raw = json.dumps(request)

    # The scan is only meaningful if real content flowed through.
    assert ctx.avoid_lines and ctx.pm_rules and ctx.idea_rules
    assert stance_line in serialized_raw

    serialized = serialized_raw.lower()
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


def test_directive_trade_line_includes_the_tenor_when_set(market_lookup):
    trade = ledger_row(
        trade_idea_id="ti_001",
        date=DEFAULT_DATE,
        instrument_id="RT-USD",
        tenor=Tenor.Y10,
        instrument_type=InstrumentKind.SOVEREIGN_CURVE,
    )
    ctx = session_context(market_lookup, day_trades=(trade,))

    directive_text = narrator_directive(ctx, 0)

    assert f"tenor {Tenor.Y10.value}" in directive_text


def test_check_in_with_no_open_positions_directive(market_lookup):
    ctx = session_context(market_lookup, kind=SessionKind.CHECK_IN)
    assert ctx.open_positions == ()

    directive_text = narrator_directive(ctx, 0)

    assert directive_text == (
        "Open with a routine check-in. You hold no open positions today, so ask the "
        "advisor about the markets you trade without naming a position."
    )


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
    pm_rule = rule(rule_id="r_01", text="A distinctive rule text about trimming winners at target.")
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
    ctx = dataclasses.replace(ctx, pm_rules=(pm_rule,), ideas=(idea,))

    advisor_prompt = read_advisor_prompt(None)
    system = advisor_system(advisor_prompt, ctx.skeleton.date)
    messages = [{"role": "user", "content": "Any levels I should know about on my names?"}]
    request = advisor_request(system, messages, _CONFIG.dialogue)

    serialized = json.dumps(request)
    assert ctx.persona.stated_profile.self_description not in serialized
    assert pm_rule.text not in serialized
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
