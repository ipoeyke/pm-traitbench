"""Tests for the narrator and advisor system prompts and request builders."""

import dataclasses
import json
import re

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import Voice
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import request_key
from pm_traitbench.dialogue.context import PmTables, TradeNote, build_contexts, trade_key
from pm_traitbench.dialogue.prompts import (
    ADVISOR_TURN_SCHEMA,
    NARRATOR_OPENING_MESSAGE,
    advisor_request,
    advisor_system,
    market_levels_section,
    narrator_directive,
    narrator_request,
    narrator_system,
    narrator_turn_schema,
    read_advisor_prompt,
)
from pm_traitbench.dialogue.session import parse_turn
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
from tests.dialogue.fixtures import fake_message, rule, session_context, turn_text
from tests.gates.fixtures import DEFAULT_DATE, PM_ID, idea_row, ledger_row
from tests.signals.fixtures import bias_trait, persona, pref_trait

_CONFIG = Config()


def test_narrator_request_has_only_the_allowed_keys(market_lookup):
    ctx = session_context(market_lookup)
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]

    request = narrator_request(ctx, 0, messages, _CONFIG.dialogue, None)

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
        "format": {"type": "json_schema", "schema": narrator_turn_schema([])},
    }
    assert request["cache_control"] == {"type": "ephemeral"}


def test_advisor_request_has_only_the_allowed_keys():
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
    request_1 = narrator_request(ctx_1, 0, messages, _CONFIG.dialogue, None)
    request_2 = narrator_request(ctx_2, 0, list(messages), _CONFIG.dialogue, None)

    assert request_1 == request_2
    scope = ctx_1.skeleton.session_id
    assert request_key(request_1, scope) == request_key(request_2, scope)


def test_turn_schema_mention_enums_come_from_the_enum_classes():
    schema = narrator_turn_schema(["ti_001"])
    trade_schema, level_schema = schema["properties"]["mentions"]["items"]["anyOf"]

    assert trade_schema["properties"]["kind"] == {"const": "trade"}
    assert level_schema["properties"]["kind"] == {"const": "level"}
    tenor_enum = [t.value for t in Tenor] + [None]
    assert trade_schema["properties"]["tenor"]["enum"] == tenor_enum
    assert level_schema["properties"]["tenor"]["enum"] == tenor_enum
    assert trade_schema["properties"]["side"]["enum"] == [s.value for s in Side]
    assert level_schema["properties"]["side"] == {"type": "null"}
    assert trade_schema["properties"]["field"] == {"type": "null"}
    assert trade_schema["properties"]["value"] == {"type": "null"}
    assert level_schema["properties"]["field"] == {"type": "string"}
    assert level_schema["properties"]["value"] == {"type": "number"}
    assert trade_schema["properties"]["trade_idea_id"] == {"enum": ["ti_001"]}
    for mention_schema in (trade_schema, level_schema):
        assert mention_schema["additionalProperties"] is False
        assert set(mention_schema["required"]) == set(mention_schema["properties"])


def test_parse_turn_accepts_each_mention_shape_the_schema_allows():
    trade_mention = {
        "kind": "trade",
        "instrument_id": "EQ-0001",
        "trade_idea_id": "ti_001",
        "tenor": None,
        "side": "buy",
        "size": 100.0,
        "field": None,
        "value": None,
    }
    level_mention = {
        "kind": "level",
        "instrument_id": "EQ-0001",
        "trade_idea_id": None,
        "tenor": Tenor.Y10.value,
        "side": None,
        "size": None,
        "field": "price",
        "value": 101.25,
    }
    response = fake_message([turn_text("hi", mentions=[trade_mention, level_mention])])

    output = parse_turn(response)

    assert output is not None
    assert len(output.mentions) == 2


def test_turn_schema_rejects_an_off_enum_tenor():
    mention = {
        "kind": "level",
        "instrument_id": "EQ-0001",
        "trade_idea_id": None,
        "tenor": "10y",
        "side": None,
        "size": None,
        "field": "price",
        "value": 101.25,
    }
    response = fake_message([turn_text("hi", mentions=[mention])])

    assert parse_turn(response) is None


def test_advisor_turn_schema_offers_only_level_mentions():
    """The advisor never sees idea ids, so its reply schema must not offer the trade branch."""
    mentions_items = ADVISOR_TURN_SCHEMA["properties"]["mentions"]["items"]

    assert "anyOf" not in mentions_items
    assert mentions_items["properties"]["kind"] == {"const": "level"}
    narrator_items = narrator_turn_schema(["ti_001"])["properties"]["mentions"]["items"]
    assert mentions_items == narrator_items["anyOf"][1]


def test_narrator_turn_schema_offers_both_mention_kinds_when_trades_are_listed():
    mentions_items = narrator_turn_schema(["ti_001"])["properties"]["mentions"]["items"]

    assert {branch["properties"]["kind"]["const"] for branch in mentions_items["anyOf"]} == {
        "trade",
        "level",
    }


def test_narrator_turn_schema_drops_the_trade_branch_without_listed_trades():
    assert narrator_turn_schema([]) == ADVISOR_TURN_SCHEMA


def test_narrator_request_limits_trade_ids_to_turn_zero_listed_trades(market_lookup):
    trades = tuple(
        ledger_row(trade_idea_id=idea_id, date=DEFAULT_DATE) for idea_id in ("ti_002", "ti_001")
    )
    ctx = session_context(market_lookup, day_trades=trades)
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]

    first = narrator_request(ctx, 0, messages, _CONFIG.dialogue, None)
    later = narrator_request(ctx, 1, messages, _CONFIG.dialogue, None)

    schema = first["output_config"]["format"]["schema"]
    assert schema == narrator_turn_schema(["ti_001", "ti_002"])
    assert later["output_config"]["format"]["schema"] == narrator_turn_schema([])


def test_advisor_request_uses_the_advisor_only_turn_schema():
    request = advisor_request(
        "system prompt", [{"role": "user", "content": "hi"}], _CONFIG.dialogue
    )

    assert request["output_config"]["format"]["schema"] == ADVISOR_TURN_SCHEMA


def test_advisor_mentions_instruction_names_level_for_a_curve_point():
    system = advisor_system("AUTHORED PROMPT TEXT", DEFAULT_DATE)

    assert "level" in system
    assert "curve" in system.lower()


def test_advisor_system_appends_the_mentions_instruction_before_the_date():
    system = advisor_system("AUTHORED PROMPT TEXT", DEFAULT_DATE)

    assert "AUTHORED PROMPT TEXT" in system
    assert "mentions" in system
    assert system.index("AUTHORED PROMPT TEXT") < system.rindex("mentions")
    assert system.rindex("mentions") < system.index(f"Today is {DEFAULT_DATE.isoformat()}")


def test_advisor_system_asks_for_a_summarised_price_history():
    system = advisor_system("AUTHORED PROMPT TEXT", DEFAULT_DATE)

    assert "Summarise a price history" in system


def test_narrator_system_says_the_advisor_never_changes_a_stance(market_lookup):
    system = narrator_system(session_context(market_lookup), None)

    assert "the advisor's replies never change it" in system
    assert "carries the decision out anyway" in system
    assert "the PM states only that preference and adds no other request" in system


def test_market_levels_section_lists_each_idea_instrument_close(market_lookup):
    ctx = session_context(market_lookup)
    close = market_lookup.latest_price("EQ-0001", ctx.skeleton.date)
    assert close is not None

    section = market_levels_section(ctx)

    assert section.startswith("Latest market levels, the only market numbers you may state")
    assert f"(EQ-0001): price {close.price:.6g} (close {close.date.isoformat()})" in section
    assert section in narrator_system(ctx, None)


def test_market_levels_section_lists_curve_levels_by_tenor(market_lookup):
    curve_idea = idea_row(
        instrument_id="RT-USD",
        legs=(Leg(instrument_id="RT-USD", tenor=None, side=Side.BUY, weight=1.0),),
    )
    ctx = dataclasses.replace(session_context(market_lookup), ideas=(curve_idea,))
    found = market_lookup.curve_on_or_before("RT-USD", ctx.skeleton.date)
    assert found is not None
    curve_date, levels = found

    section = market_levels_section(ctx)

    assert "(RT-USD) curve, field level: " in section
    assert f"{Tenor.Y10.value} {levels[Tenor.Y10]:.6g}" in section
    assert f"(close {curve_date.isoformat()})" in section


def test_market_levels_section_forbids_levels_without_positions_or_ideas(market_lookup):
    ctx = dataclasses.replace(session_context(market_lookup), ideas=(), open_positions=())

    assert market_levels_section(ctx) == (
        "Latest market levels: none are available to you, so state no market level."
    )


def test_narrator_system_limits_trade_mentions_to_the_directive(market_lookup):
    system = narrator_system(session_context(market_lookup), None)

    assert 'trade a turn\'s "Mention each of these trades" directive lists' in system
    assert "never with a trade mention" in system


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
    request = narrator_request(ctx, 0, messages, _CONFIG.dialogue, None)
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
    assert f"In this message: {stance.stance}. This is your decision" in directive_text
    assert "carry this out anyway" in directive_text
    assert "state only that one and ask nothing else" in directive_text
    assert stance.trait_id not in directive_text
    assert stance.mode.value not in directive_text

    opening_text = narrator_directive(ctx, 0)
    assert "Mention each of these trades" in opening_text
    assert trade.trade_idea_id in opening_text


def test_directive_trade_line_says_what_the_trade_does_and_why(market_lookup):
    trade = ledger_row(trade_idea_id="ti_001", date=DEFAULT_DATE)
    ctx = session_context(market_lookup, day_trades=(trade,))

    assert "(opens the position)" in narrator_directive(ctx, 0)

    closing = dataclasses.replace(
        ctx,
        trade_notes={
            trade_key(trade): TradeNote(
                "closes the whole position", "on your rule: Exit after a 5 percent drawdown."
            )
        },
    )
    directive_text = narrator_directive(closing, 0)

    assert (
        "(closes the whole position, on your rule: Exit after a 5 percent drawdown.)"
        in directive_text
    )
    assert "not a trim" in narrator_system(ctx, None)


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

    base = narrator_request(ctx, 0, messages, _CONFIG.dialogue, None)
    changed = narrator_request(
        ctx, 0, messages, _CONFIG.dialogue, "Keep this PM's trades tighter to the skeleton."
    )

    assert base["system"] != changed["system"]
    scope = ctx.skeleton.session_id
    assert request_key(base, scope) != request_key(changed, scope)


def test_two_same_date_check_ins_of_one_pm_get_distinct_cache_keys(market_lookup):
    """Two check-in sessions of one PM, same date, no stances or day trades, render the
    identical narrator turn-0 request body; real plan output does put several sessions
    on one date, so only the session-scoped cache key, never the body, may tell them apart.
    """
    catalogue = load_catalogue()
    pm_persona = persona()
    voice = Voice(voice_id="v_01", line="terse trader shorthand, drops articles")
    pm_prefix = f"s_{PM_ID.replace('_', '')}_{DEFAULT_DATE.isoformat()}"
    skeleton_a = Skeleton(
        session_id=f"{pm_prefix}_a",
        pm_id=PM_ID,
        date=DEFAULT_DATE,
        kind=SessionKind.CHECK_IN,
        trade_idea_ids=(),
        stances=(),
        advisor_violation=None,
        forbidden_trait_ids=(),
        forbidden_pref_params=(),
    )
    skeleton_b = skeleton_a.model_copy(update={"session_id": f"{pm_prefix}_b"})
    pm = PmTables(
        persona=pm_persona,
        traits=(),
        drift_events=(),
        rules=(),
        ideas={},
        ledger=(),
        position_days=(),
        skeletons=(skeleton_a, skeleton_b),
    )

    ctx_a, ctx_b = build_contexts(pm, voice, market_lookup, catalogue, Config())
    messages = [{"role": "user", "content": NARRATOR_OPENING_MESSAGE}]
    body_a = narrator_request(ctx_a, 0, messages, _CONFIG.dialogue, None)
    body_b = narrator_request(ctx_b, 0, messages, _CONFIG.dialogue, None)

    assert body_a == body_b
    assert request_key(body_a, ctx_a.skeleton.session_id) != request_key(
        body_b, ctx_b.skeleton.session_id
    )


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
