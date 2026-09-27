"""Shared validate-stage test fixtures and builders: turns, dialogue logs, ledger rows,
skeletons and mentions built from plausible constants.

Consumed by every validate-stage test module, so a fixture's shape only has
to match `schema.py` in one place.
"""

from collections.abc import Sequence
from datetime import date

from pm_traitbench.enums import InstrumentKind, MentionKind, SessionKind, Side, Tenor, TurnRole
from pm_traitbench.tables.schema import (
    CallUsage,
    DialogueLog,
    LedgerRow,
    Mention,
    Skeleton,
    ToolCall,
    TurnLog,
)
from tests.dialogue.conftest import fixture_market, market_lookup  # noqa: F401

_MODEL = "claude-opus-5-5"
_REQUEST_HASH = "0" * 64
_ZERO_USAGE = CallUsage(
    input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
)


def pm_turn(text: str, mentions: tuple[Mention, ...] = (), directive: str | None = None) -> TurnLog:
    """A PM turn log entry with no tool calls, one request hash and zero usage."""
    return TurnLog(
        role=TurnRole.PM,
        text=text,
        mentions=mentions,
        directive=directive,
        scripted_violation=False,
        tool_calls=(),
        model=_MODEL,
        request_hashes=(_REQUEST_HASH,),
        usage=_ZERO_USAGE,
    )


def advisor_turn(
    text: str, mentions: tuple[Mention, ...] = (), tool_calls: tuple[ToolCall, ...] = ()
) -> TurnLog:
    """An advisor turn log entry with no directive, one request hash and zero usage."""
    return TurnLog(
        role=TurnRole.ADVISOR,
        text=text,
        mentions=mentions,
        directive=None,
        scripted_violation=False,
        tool_calls=tool_calls,
        model=_MODEL,
        request_hashes=(_REQUEST_HASH,),
        usage=_ZERO_USAGE,
    )


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


def ledger_row(
    pm_id: str,
    date: date,
    trade_idea_id: str,
    instrument_id: str,
    side: Side,
    size: float,
    tenor: Tenor | None = None,
) -> LedgerRow:
    """A ledger row with plausible constants for the fields not under test."""
    return LedgerRow(
        pm_id=pm_id,
        date=date,
        trade_idea_id=trade_idea_id,
        instrument_id=instrument_id,
        tenor=tenor,
        instrument_type=InstrumentKind.EQUITY,
        side=side,
        size=size,
        risk_amount=1.0,
        price_or_yield=100.0,
        stated_conviction=3,
        bias_flag=None,
        rule_id=None,
    )


def skeleton_of(
    pm_id: str,
    date: date,
    trade_idea_ids: tuple[str, ...],
    stances: tuple = (),
    forbidden_trait_ids: tuple[str, ...] = (),
    forbidden_pref_params: tuple[str, ...] = (),
    kind: SessionKind = SessionKind.DECISION,
) -> Skeleton:
    """A skeleton with no advisor violation, sitting on the given pm and date."""
    return Skeleton(
        session_id=f"s_{pm_id.replace('_', '')}_{date.isoformat()}_a",
        pm_id=pm_id,
        date=date,
        kind=kind,
        trade_idea_ids=trade_idea_ids,
        stances=stances,
        advisor_violation=None,
        forbidden_trait_ids=forbidden_trait_ids,
        forbidden_pref_params=forbidden_pref_params,
    )
