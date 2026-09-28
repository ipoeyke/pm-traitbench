"""Shared validate-stage test fixtures and builders: turns, dialogue logs, skeletons,
mentions and judge replies built from plausible constants.

Consumed by every validate-stage test module, so a fixture's shape only has
to match `schema.py` in one place.
"""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pm_traitbench.dialogue.client import CachedClient
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.enums import (
    MentionKind,
    SessionKind,
    SignalMode,
    StanceEntry,
    Tenor,
    TurnRole,
)
from pm_traitbench.signals.assemble import session_id
from pm_traitbench.tables.schema import (
    CallUsage,
    DialogueLog,
    LedgerRow,
    Mention,
    Skeleton,
    Stance,
    ToolCall,
    TurnLog,
)
from tests.dialogue.fixtures import FakeClient, default_responder, fake_message, session_context

_MODEL = "claude-opus-5-5"
_REQUEST_HASH = "0" * 64
_ZERO_USAGE = CallUsage(
    input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
)


def _turn(
    role: TurnRole,
    text: str,
    *,
    mentions: tuple[Mention, ...] = (),
    directive: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
) -> TurnLog:
    """A turn log entry with one request hash and zero usage."""
    return TurnLog(
        role=role,
        text=text,
        mentions=mentions,
        directive=directive,
        scripted_violation=False,
        tool_calls=tool_calls,
        model=_MODEL,
        request_hashes=(_REQUEST_HASH,),
        usage=_ZERO_USAGE,
    )


def pm_turn(text: str, mentions: tuple[Mention, ...] = (), directive: str | None = None) -> TurnLog:
    """A PM turn log entry with no tool calls."""
    return _turn(TurnRole.PM, text, mentions=mentions, directive=directive)


def advisor_turn(
    text: str, mentions: tuple[Mention, ...] = (), tool_calls: tuple[ToolCall, ...] = ()
) -> TurnLog:
    """An advisor turn log entry with no directive."""
    return _turn(TurnRole.ADVISOR, text, mentions=mentions, tool_calls=tool_calls)


def log_of(session_id: str, pm_id: str, turns: Sequence[TurnLog]) -> DialogueLog:
    """A `DialogueLog` on voice `v_01` from an alternating turn sequence."""
    return DialogueLog(session_id=session_id, pm_id=pm_id, voice_id="v_01", turns=tuple(turns))


def trade_mention(row: LedgerRow, size: float | None = None) -> Mention:
    """A trade mention built from a ledger row, optionally with a different stated size."""
    return Mention(
        kind=MentionKind.TRADE,
        instrument_id=row.instrument_id,
        trade_idea_id=row.trade_idea_id,
        tenor=row.tenor,
        side=row.side,
        size=row.size if size is None else size,
        field=None,
        value=None,
    )


def level_mention(
    instrument_id: str, field: str, value: float, tenor: Tenor | None = None
) -> Mention:
    """A market-level mention for `field` on `instrument_id`."""
    return Mention(
        kind=MentionKind.LEVEL,
        instrument_id=instrument_id,
        trade_idea_id=None,
        tenor=tenor,
        side=None,
        size=None,
        field=field,
        value=value,
    )


def leak_reply(explicit: bool, label: str | None, quote: str = "") -> dict:
    """A `fake_message` body whose single text block is a leak verdict JSON object."""
    payload = json.dumps({"explicit": explicit, "label": label, "quote": quote})
    return fake_message([{"type": "text", "text": payload}])


def forbidden_reply(violations: list[tuple[int, str]]) -> dict:
    """A `fake_message` body whose single text block is a forbidden verdict JSON object."""
    payload = json.dumps(
        {"violations": [{"index": index, "quote": quote} for index, quote in violations]}
    )
    return fake_message([{"type": "text", "text": payload}])


def _schema_title(request: Mapping[str, Any]) -> str | None:
    """The request's output schema title, or `None` for a request whose schema carries none."""
    return request.get("output_config", {}).get("format", {}).get("schema", {}).get("title")


def is_leak_request(request: Mapping[str, Any]) -> bool:
    """True when `request` is a leakage judge request, by its schema's title."""
    return _schema_title(request) == "leak_verdict"


def is_forbidden_request(request: Mapping[str, Any]) -> bool:
    """True when `request` is a forbidden-trait judge request, by its schema's title."""
    return _schema_title(request) == "forbidden_verdict"


def validate_context(
    lookup: MarketLookup,
    *,
    stances: tuple[Stance, ...] = (),
    avoid_lines: tuple[str, ...] = (),
    day_trades: tuple[LedgerRow, ...] = (),
    kind: SessionKind = SessionKind.DECISION,
) -> SessionContext:
    """A `SessionContext` for the validate stage, with its own avoid lines."""
    ctx = session_context(lookup, kind=kind, stances=stances, day_trades=day_trades)
    return dataclasses.replace(ctx, avoid_lines=avoid_lines)


def stance(trait_id: str, mode: SignalMode, entry: StanceEntry, text: str) -> Stance:
    """A stance on `trait_id` with a fixed signal id, for tests that don't care which."""
    return Stance(signal_id="sg_001", trait_id=trait_id, mode=mode, entry=entry, stance=text)


def routing_responder(
    narrator: Callable[[dict], dict],
    leak: Callable[[dict], dict],
    forbidden: Callable[[dict], dict],
) -> Callable[[Mapping[str, Any]], dict]:
    """A `FakeClient` responder routing judge, advisor and narrator requests to their scripts."""

    def responder(request: Mapping[str, Any]) -> dict:
        if is_leak_request(request):
            return leak(request)
        if is_forbidden_request(request):
            return forbidden(request)
        if "tools" in request:
            return default_responder(request)
        return narrator(request)

    return responder


def scripted_client(
    tmp_path: Path,
    narrator_reply: Callable[[dict], dict],
    leak: Callable[[dict], dict],
    forbidden: Callable[[dict], dict],
) -> CachedClient:
    """A `CachedClient` over a `FakeClient` that routes each request to its own script."""
    fake = FakeClient(routing_responder(narrator_reply, leak, forbidden))
    return CachedClient(lambda: fake, tmp_path, token_budget=None)


def skeleton_of(
    pm_id: str,
    date: date,
    trade_idea_ids: tuple[str, ...],
    stances: tuple[Stance, ...] = (),
    forbidden_trait_ids: tuple[str, ...] = (),
    forbidden_pref_params: tuple[str, ...] = (),
    kind: SessionKind = SessionKind.DECISION,
) -> Skeleton:
    """A skeleton with no advisor violation, sitting on the given pm and date."""
    return Skeleton(
        session_id=session_id(pm_id, date, 0),
        pm_id=pm_id,
        date=date,
        kind=kind,
        trade_idea_ids=trade_idea_ids,
        stances=stances,
        advisor_violation=None,
        forbidden_trait_ids=forbidden_trait_ids,
        forbidden_pref_params=forbidden_pref_params,
    )
