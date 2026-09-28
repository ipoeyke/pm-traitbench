"""Shared dialogue-test fixtures: fake message builders, an in-memory `LlmClient`, a
`MarketLookup` built on the shared fixture market, and a small `SessionContext` builder.

Consumed by client, tools, prompt and session tests, so a canned response's
shape only has to match `Message.to_dict()` in one place.
"""

import json
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

import pytest

from pm_traitbench.catalogues.models import Voice
from pm_traitbench.config import Config, TurnRanges
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.dialogue.turns import TurnPlan, plan_turns
from pm_traitbench.enums import Action, Op, RuleScope, RuleSource, SessionKind
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import LedgerRow, Rule, Skeleton, Stance
from tests.gates.fixtures import PM_ID, idea_row
from tests.signals.fixtures import persona


def fake_message(
    content: list[dict],
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict:
    """A dict shaped like `Message.to_dict()`, for a fake client's canned responses."""
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def turn_text(text: str, mentions: list[dict] | None = None) -> dict:
    """A text content block whose text is a turn's `{text, mentions}` JSON payload."""
    return {"type": "text", "text": json.dumps({"text": text, "mentions": mentions})}


def tool_use(name: str, tool_input: dict, tool_use_id: str) -> dict:
    """A tool-use content block."""
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}


class FakeClient:
    """An in-memory `LlmClient` scripted by a responder function; records every request."""

    def __init__(self, responder: Callable[[Mapping[str, Any]], dict]) -> None:
        self._responder = responder
        self.requests: list[Mapping[str, Any]] = []
        self.closed = False

    async def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return self._responder(request)

    async def aclose(self) -> None:
        self.closed = True


def default_responder(request: Mapping[str, Any]) -> dict:
    """Canned replies: advisor requests carry `tools`, narrator requests do not."""
    if "tools" in request:
        return fake_message([turn_text("Sounds reasonable, tell me more.")])
    return fake_message([turn_text("Feeling good about the book today.")])


@pytest.fixture(scope="module")
def market_lookup(fixture_market: dict) -> MarketLookup:
    """A `MarketLookup` built from the shared fixture market, seed 'T'."""
    return MarketLookup.build(
        seed="T",
        instruments=fixture_market["instruments"],
        prices=fixture_market["prices"],
        curves=fixture_market["curves"],
        consensus=fixture_market["consensus"],
        calendar=fixture_market["calendar"],
    )


def rule(**overrides) -> Rule:
    """A self-imposed, PM-scope stop-loss rule, overridable by keyword."""
    fields = dict(
        pm_id=PM_ID,
        rule_id="r_01",
        source=RuleSource.SELF,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param="stop_loss",
        field="pnl_pct",
        op=Op.LE,
        level=-5.0,
        unit="pct",
        window=1,
        action=Action.EXIT,
        text="Exit a position after a 5 percent drawdown from entry.",
    )
    fields.update(overrides)
    return Rule(**fields)


def session_context(
    lookup: MarketLookup,
    *,
    kind: SessionKind = SessionKind.DECISION,
    stances: tuple[Stance, ...] = (),
    advisor_violation: str | None = None,
    day_trades: tuple[LedgerRow, ...] = (),
    turn_plan: TurnPlan | None = None,
) -> SessionContext:
    """A small, internally consistent `SessionContext` for one PM's session.

    Every field but `kind`, `stances`, `advisor_violation`, `day_trades` and
    `turn_plan` is a fixed default built from `tests.signals.fixtures.persona`
    and `tests.gates.fixtures.idea_row`. A silence session drops its stances,
    trade and violation overrides, since a silence skeleton forbids them.
    """
    pm = persona()
    voice = Voice(voice_id="v_01", line="terse trader shorthand, drops articles")
    idea = idea_row()
    is_silence = kind == SessionKind.SILENCE
    resolved_day_trades = () if is_silence else day_trades

    skeleton = Skeleton(
        session_id=f"s_{pm.pm_id.replace('_', '')}_2026-01-05_a",
        pm_id=pm.pm_id,
        date=date(2026, 1, 5),
        kind=kind,
        trade_idea_ids=() if is_silence else (idea.trade_idea_id,),
        stances=() if is_silence else stances,
        advisor_violation=None if is_silence else advisor_violation,
        forbidden_trait_ids=(),
        forbidden_pref_params=(),
    )

    resolved_turn_plan = turn_plan
    if resolved_turn_plan is None:
        resolved_turn_plan = plan_turns(
            skeleton,
            resolved_day_trades,
            TurnRanges(),
            stream(Config().seed.root, "dialogue", pm.pm_id, skeleton.session_id, "turns"),
        )

    question_instrument = None
    instrument_names: dict[str, str] = {}
    if is_silence:
        question_instrument = sorted(lookup.instruments.values(), key=lambda i: i.instrument_id)[0]
        instrument_names[question_instrument.instrument_id] = question_instrument.name
    else:
        instrument_names[idea.instrument_id] = lookup.instruments[idea.instrument_id].name
    for trade in resolved_day_trades:
        instrument_names.setdefault(
            trade.instrument_id, lookup.instruments[trade.instrument_id].name
        )

    return SessionContext(
        skeleton=skeleton,
        persona=pm,
        voice=voice,
        pm_rules=(),
        idea_rules=(),
        ideas=() if is_silence else (idea,),
        day_trades=resolved_day_trades,
        open_positions=(),
        question_instrument=question_instrument,
        instrument_names=instrument_names,
        avoid_lines=(),
        lookup=lookup,
        turn_plan=resolved_turn_plan,
    )
