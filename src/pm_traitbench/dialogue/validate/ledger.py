"""Ledger consistency: checks a narrated session's trade and level mentions against the
ledger and the market.

Trade mismatches are failures, since stage 6 must not narrate a trade that
was never placed or leave a placed one unmentioned. Level mismatches are
warnings only, never failures, since the advisor may derive a figure no
tool call returned.
"""

import json
from collections.abc import Iterator, Sequence
from datetime import date
from typing import Any

from pm_traitbench.dialogue.tools import MarketLookup
from pm_traitbench.enums import MentionKind, TurnRole
from pm_traitbench.tables.schema import DialogueLog, LedgerRow, Mention, Skeleton, TurnLog


def _trade_reason(prefix: str, mention_or_row: Mention | LedgerRow) -> str:
    tenor = mention_or_row.tenor.value if mention_or_row.tenor is not None else "-"
    return (
        f"trade {prefix}: {mention_or_row.trade_idea_id} {mention_or_row.instrument_id} "
        f"{tenor} {mention_or_row.side.value} {mention_or_row.size}"
    )


def _matches(mention: Mention, row: LedgerRow, skeleton: Skeleton, size_tolerance: float) -> bool:
    return (
        row.pm_id == skeleton.pm_id
        and row.trade_idea_id == mention.trade_idea_id
        and row.instrument_id == mention.instrument_id
        and row.tenor == mention.tenor
        and row.side == mention.side
        and row.date <= skeleton.date
        and abs(mention.size - row.size) <= size_tolerance * row.size
    )


def check_trades(
    log: DialogueLog, skeleton: Skeleton, ledger: Sequence[LedgerRow], size_tolerance: float
) -> tuple[str, ...]:
    """Failure reasons for a session's trade mentions; empty when the layer passes."""
    trade_mentions = [
        mention
        for turn in log.turns
        if turn.role == TurnRole.PM
        for mention in turn.mentions
        if mention.kind == MentionKind.TRADE
    ]

    reasons: set[str] = set()
    for mention in trade_mentions:
        if not any(_matches(mention, row, skeleton, size_tolerance) for row in ledger):
            reasons.add(_trade_reason("not in ledger", mention))

    for row in ledger:
        if row.date != skeleton.date or row.trade_idea_id not in skeleton.trade_idea_ids:
            continue
        if not any(_matches(mention, row, skeleton, size_tolerance) for mention in trade_mentions):
            reasons.add(_trade_reason("not mentioned", row))

    return tuple(sorted(reasons))


def _agree(value: float, reference: float, tolerance: float) -> bool:
    return abs(value - reference) <= tolerance * max(abs(reference), 1e-9)


def _numbers_in(value: Any) -> Iterator[float]:
    """Every number nested in `value`, recursing into dicts and lists; bools are not numbers."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers_in(item)


def _confirmed_by_tool_calls(turn: TurnLog, mention: Mention, level_tolerance: float) -> bool:
    for call in turn.tool_calls:
        if call.is_error:
            continue
        parsed = json.loads(call.result_json)
        if any(_agree(mention.value, number, level_tolerance) for number in _numbers_in(parsed)):
            return True
    return False


def _confirmed_by_market(
    mention: Mention, today: date, lookup: MarketLookup, level_tolerance: float
) -> bool:
    if mention.field in ("price", "spread_bp"):
        row = lookup.latest_price(mention.instrument_id, today)
        reference = None if row is None else getattr(row, mention.field)
    elif mention.field == "level":
        if mention.tenor is None:
            return False
        found = lookup.curve_on_or_before(mention.instrument_id, today)
        reference = None if found is None else found[1].get(mention.tenor)
    elif mention.field in ("street_score", "positioning_pct"):
        row = lookup.latest_consensus(mention.instrument_id, today)
        reference = None if row is None else getattr(row, mention.field)
    else:
        return False
    if reference is None:
        return False
    return _agree(mention.value, reference, level_tolerance)


def count_level_warnings(
    log: DialogueLog, skeleton: Skeleton, lookup: MarketLookup, level_tolerance: float
) -> int:
    """Number of level mentions that are not confirmed by a tool result or the market."""
    warnings = 0
    for turn in log.turns:
        for mention in turn.mentions:
            if mention.kind != MentionKind.LEVEL:
                continue
            if turn.role == TurnRole.ADVISOR:
                confirmed = _confirmed_by_tool_calls(turn, mention, level_tolerance)
            else:
                confirmed = _confirmed_by_market(mention, skeleton.date, lookup, level_tolerance)
            if not confirmed:
                warnings += 1
    return warnings
