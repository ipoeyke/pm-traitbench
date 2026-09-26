"""Read-only market lookup tools the simulated advisor calls instead of stating a market fact
from memory.

`MarketLookup` indexes one seed's stage 2 market tables (prices, curves,
consensus, calendar) for the five `AdvisorTool` lookups. `run_tool` dispatches
one tool call and never raises: a bad tool name, unknown instrument, empty
result or out-of-range window all come back as an error `ToolOutcome`. No
lookup returns a row dated after the session date, except a calendar's own
future-scheduled rows, whose surprise is masked rather than omitted.
"""

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from difflib import get_close_matches
from typing import Any

from pm_traitbench.enums import AdvisorTool, InstrumentKind, Tenor
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import CalendarEvent, ConsensusRow, CurvePoint, Instrument, Price

_MAX_CLOSE_MATCHES = 5
_WINDOW_MIN, _WINDOW_MAX = 0, 20
_HISTORY_MIN, _HISTORY_MAX = 1, 60


@dataclass(frozen=True)
class ToolOutcome:
    """One tool call's result and whether it is an error."""

    result: dict[str, Any]
    is_error: bool


def _error(message: str) -> ToolOutcome:
    return ToolOutcome(result={"error": message}, is_error=True)


@dataclass(frozen=True)
class MarketLookup:
    """One market seed's tables, indexed for the advisor's read-only tool calls."""

    seed: str
    dates: tuple[date, ...]
    instruments: Mapping[str, Instrument]
    _prices: Mapping[str, tuple[Price, ...]] = field(repr=False)
    _curve_dates: Mapping[str, tuple[date, ...]] = field(repr=False)
    _curve_levels: Mapping[tuple[str, date], Mapping[Tenor, float]] = field(repr=False)
    _consensus: Mapping[str, tuple[ConsensusRow, ...]] = field(repr=False)
    _calendar_by_instrument: Mapping[str, tuple[CalendarEvent, ...]] = field(repr=False)
    _calendar_market_wide: tuple[CalendarEvent, ...] = field(repr=False)

    @classmethod
    def build(
        cls,
        seed: str,
        instruments: Sequence[Instrument],
        prices: Sequence[Price],
        curves: Sequence[CurvePoint],
        consensus: Sequence[ConsensusRow],
        calendar: Sequence[CalendarEvent],
    ) -> "MarketLookup":
        """Build one seed's lookup; rows for other seeds are ignored.

        Raises `DialogueError` if a curve row's `curve_id` matches no
        instrument, since the advisor must never be handed a curve it cannot
        name back to the PM.
        """
        seed_prices = [row for row in prices if row.seed == seed]
        seed_curves = [row for row in curves if row.seed == seed]
        seed_consensus = [row for row in consensus if row.seed == seed]
        seed_calendar = [row for row in calendar if row.seed == seed]

        all_ids = {instrument.instrument_id for instrument in instruments}
        for curve in seed_curves:
            if curve.curve_id not in all_ids:
                raise DialogueError(f"curve id '{curve.curve_id}' matches no instrument")

        dates = tuple(sorted({row.date for row in seed_prices}))
        covered_ids = {row.instrument_id for row in seed_prices} | {
            row.curve_id for row in seed_curves
        }
        instruments_by_id = {
            instrument.instrument_id: instrument
            for instrument in instruments
            if instrument.instrument_id in covered_ids
        }

        prices_by_instrument: dict[str, list[Price]] = {}
        for row in seed_prices:
            prices_by_instrument.setdefault(row.instrument_id, []).append(row)

        curve_levels: dict[tuple[str, date], dict[Tenor, float]] = {}
        for row in seed_curves:
            curve_levels.setdefault((row.curve_id, row.date), {})[row.tenor] = row.level
        curve_dates: dict[str, set[date]] = {}
        for curve_id, curve_date in curve_levels:
            curve_dates.setdefault(curve_id, set()).add(curve_date)

        consensus_by_instrument: dict[str, list[ConsensusRow]] = {}
        for row in seed_consensus:
            consensus_by_instrument.setdefault(row.instrument_id, []).append(row)

        calendar_by_instrument: dict[str, list[CalendarEvent]] = {}
        calendar_market_wide: list[CalendarEvent] = []
        for row in seed_calendar:
            if row.instrument_id is None:
                calendar_market_wide.append(row)
            else:
                calendar_by_instrument.setdefault(row.instrument_id, []).append(row)

        return cls(
            seed=seed,
            dates=dates,
            instruments=instruments_by_id,
            _prices={
                instrument_id: tuple(sorted(rows, key=lambda r: r.date))
                for instrument_id, rows in prices_by_instrument.items()
            },
            _curve_dates={curve_id: tuple(sorted(days)) for curve_id, days in curve_dates.items()},
            _curve_levels=curve_levels,
            _consensus={
                instrument_id: tuple(sorted(rows, key=lambda r: r.date))
                for instrument_id, rows in consensus_by_instrument.items()
            },
            _calendar_by_instrument={
                instrument_id: tuple(sorted(rows, key=lambda r: r.date))
                for instrument_id, rows in calendar_by_instrument.items()
            },
            _calendar_market_wide=tuple(sorted(calendar_market_wide, key=lambda r: r.date)),
        )

    def resolve(self, instrument: str) -> Instrument | None:
        """Resolve an instrument by exact id, else by case-insensitive exact name."""
        exact = self.instruments.get(instrument)
        if exact is not None:
            return exact
        lowered = instrument.lower()
        for candidate in self.instruments.values():
            if candidate.name.lower() == lowered:
                return candidate
        return None

    def _latest_price(self, instrument_id: str, today: date) -> Price | None:
        rows = self._prices.get(instrument_id, ())
        idx = bisect.bisect_right(rows, today, key=lambda r: r.date) - 1
        return rows[idx] if idx >= 0 else None

    def _history(self, instrument_id: str, today: date, n_days: int) -> tuple[Price, ...]:
        rows = self._prices.get(instrument_id, ())
        idx = bisect.bisect_right(rows, today, key=lambda r: r.date) - 1
        if idx < 0:
            return ()
        return rows[max(0, idx - n_days + 1) : idx + 1]

    def _latest_consensus(self, instrument_id: str, today: date) -> ConsensusRow | None:
        rows = self._consensus.get(instrument_id, ())
        idx = bisect.bisect_right(rows, today, key=lambda r: r.date) - 1
        return rows[idx] if idx >= 0 else None

    def _latest_curve_date(self, curve_id: str, today: date) -> date | None:
        dates = self._curve_dates.get(curve_id, ())
        idx = bisect.bisect_right(dates, today) - 1
        return dates[idx] if idx >= 0 else None

    def _calendar_window(self, today: date, days_back: int, days_forward: int) -> tuple[date, date]:
        """Trading-day window spanning `days_back` before through `days_forward` after `today`."""
        if not self.dates:
            return today, today
        idx = bisect.bisect_left(self.dates, today)
        if idx == len(self.dates) or self.dates[idx] != today:
            idx = max(0, idx - 1)
        lo_idx = max(0, idx - days_back)
        hi_idx = min(len(self.dates) - 1, idx + days_forward)
        return self.dates[lo_idx], self.dates[hi_idx]

    def _calendar_rows(self, instrument_id: str, lo: date, hi: date) -> tuple[CalendarEvent, ...]:
        """Own and market-wide calendar rows within `[lo, hi]`, in a deterministic order."""
        own = self._calendar_by_instrument.get(instrument_id, ())
        combined = [row for row in own if lo <= row.date <= hi]
        combined += [row for row in self._calendar_market_wide if lo <= row.date <= hi]
        return tuple(sorted(combined, key=lambda r: (r.date, r.event.value, r.instrument_id or "")))


def _instrument_schema(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": AdvisorTool.GET_QUOTE.value,
        "description": (
            "Look up the latest known price, and spread for a credit issuer, of an instrument."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"instrument": _instrument_schema("Instrument id or name.")},
            "required": ["instrument"],
            "additionalProperties": False,
        },
    },
    {
        "name": AdvisorTool.GET_CURVE.value,
        "description": "Look up the latest known tenor levels of a sovereign or commodity curve.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": _instrument_schema(
                    "Sovereign curve or commodity instrument id or name."
                )
            },
            "required": ["instrument"],
            "additionalProperties": False,
        },
    },
    {
        "name": AdvisorTool.GET_CONSENSUS.value,
        "description": (
            "Look up the street's latest consensus view and positioning on an instrument."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"instrument": _instrument_schema("Instrument id or name.")},
            "required": ["instrument"],
            "additionalProperties": False,
        },
    },
    {
        "name": AdvisorTool.GET_CALENDAR.value,
        "description": (
            "Look up scheduled and past market events near today for an instrument, "
            "including market-wide events."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": _instrument_schema("Instrument id or name."),
                "days_back": {
                    "type": "integer",
                    "description": "Trading days before today to include, 0 to 20.",
                },
                "days_forward": {
                    "type": "integer",
                    "description": "Trading days after today to include, 0 to 20.",
                },
            },
            "required": ["instrument", "days_back", "days_forward"],
            "additionalProperties": False,
        },
    },
    {
        "name": AdvisorTool.GET_HISTORY.value,
        "description": (
            "Look up the last n trading days of an instrument's price or spread history."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": _instrument_schema("Instrument id or name."),
                "n_days": {
                    "type": "integer",
                    "description": "Number of trailing rows to return, 1 to 60.",
                },
            },
            "required": ["instrument", "n_days"],
            "additionalProperties": False,
        },
    },
)


def _resolve_or_error(lookup: MarketLookup, instrument: str) -> Instrument | ToolOutcome:
    resolved = lookup.resolve(instrument)
    if resolved is not None:
        return resolved
    names = [candidate.name for candidate in lookup.instruments.values()]
    matches = get_close_matches(instrument, names, n=_MAX_CLOSE_MATCHES)
    suffix = f"; closest names: {', '.join(matches)}" if matches else ""
    return _error(f"unknown instrument '{instrument}'{suffix}")


def _no_row_error(name: str, today: date) -> ToolOutcome:
    return _error(f"no row for '{name}' on or before {today.isoformat()}")


def _get_quote(lookup: MarketLookup, tool_input: Mapping[str, Any], today: date) -> ToolOutcome:
    resolved = _resolve_or_error(lookup, tool_input["instrument"])
    if isinstance(resolved, ToolOutcome):
        return resolved
    row = lookup._latest_price(resolved.instrument_id, today)
    if row is None:
        return _no_row_error(resolved.name, today)
    return ToolOutcome(
        result={
            "instrument_id": resolved.instrument_id,
            "name": resolved.name,
            "date": row.date.isoformat(),
            "price": row.price,
            "spread_bp": row.spread_bp,
        },
        is_error=False,
    )


def _get_curve(lookup: MarketLookup, tool_input: Mapping[str, Any], today: date) -> ToolOutcome:
    resolved = _resolve_or_error(lookup, tool_input["instrument"])
    if isinstance(resolved, ToolOutcome):
        return resolved
    if resolved.instrument_id not in lookup._curve_dates:
        return _error(f"'{resolved.name}' has no curve")
    curve_date = lookup._latest_curve_date(resolved.instrument_id, today)
    if curve_date is None:
        return _no_row_error(resolved.name, today)
    raw_levels = lookup._curve_levels[(resolved.instrument_id, curve_date)]
    levels = {tenor.value: raw_levels[tenor] for tenor in Tenor if tenor in raw_levels}
    return ToolOutcome(
        result={
            "instrument_id": resolved.instrument_id,
            "name": resolved.name,
            "date": curve_date.isoformat(),
            "levels": levels,
        },
        is_error=False,
    )


def _get_consensus(lookup: MarketLookup, tool_input: Mapping[str, Any], today: date) -> ToolOutcome:
    resolved = _resolve_or_error(lookup, tool_input["instrument"])
    if isinstance(resolved, ToolOutcome):
        return resolved
    if resolved.instrument_id not in lookup._consensus:
        return _error(f"'{resolved.name}' has no consensus rows")
    row = lookup._latest_consensus(resolved.instrument_id, today)
    if row is None:
        return _no_row_error(resolved.name, today)
    return ToolOutcome(
        result={
            "instrument_id": resolved.instrument_id,
            "name": resolved.name,
            "date": row.date.isoformat(),
            "street_score": row.street_score,
            "street_view": row.street_view.value,
            "positioning_pct": row.positioning_pct,
            "positioning": row.positioning.value,
        },
        is_error=False,
    )


def _get_calendar(lookup: MarketLookup, tool_input: Mapping[str, Any], today: date) -> ToolOutcome:
    days_back = tool_input["days_back"]
    days_forward = tool_input["days_forward"]
    if not (_WINDOW_MIN <= days_back <= _WINDOW_MAX):
        return _error(f"days_back must be between {_WINDOW_MIN} and {_WINDOW_MAX}, got {days_back}")
    if not (_WINDOW_MIN <= days_forward <= _WINDOW_MAX):
        return _error(
            f"days_forward must be between {_WINDOW_MIN} and {_WINDOW_MAX}, got {days_forward}"
        )
    resolved = _resolve_or_error(lookup, tool_input["instrument"])
    if isinstance(resolved, ToolOutcome):
        return resolved
    lo, hi = lookup._calendar_window(today, days_back, days_forward)
    rows = lookup._calendar_rows(resolved.instrument_id, lo, hi)
    events = [
        {
            "date": row.date.isoformat(),
            "event": row.event.value,
            "surprise": row.surprise if row.date <= today else None,
        }
        for row in rows
    ]
    return ToolOutcome(
        result={"instrument_id": resolved.instrument_id, "name": resolved.name, "events": events},
        is_error=False,
    )


def _get_history(lookup: MarketLookup, tool_input: Mapping[str, Any], today: date) -> ToolOutcome:
    n_days = tool_input["n_days"]
    if not (_HISTORY_MIN <= n_days <= _HISTORY_MAX):
        return _error(f"n_days must be between {_HISTORY_MIN} and {_HISTORY_MAX}, got {n_days}")
    resolved = _resolve_or_error(lookup, tool_input["instrument"])
    if isinstance(resolved, ToolOutcome):
        return resolved
    rows = lookup._history(resolved.instrument_id, today, n_days)
    if not rows:
        return _no_row_error(resolved.name, today)
    is_credit = resolved.kind == InstrumentKind.CREDIT_ISSUER
    points = [
        {"date": row.date.isoformat(), "value": row.spread_bp if is_credit else row.price}
        for row in rows
    ]
    return ToolOutcome(
        result={
            "instrument_id": resolved.instrument_id,
            "name": resolved.name,
            "field": "spread_bp" if is_credit else "price",
            "points": points,
        },
        is_error=False,
    )


_HANDLERS: dict[str, Callable[[MarketLookup, Mapping[str, Any], date], ToolOutcome]] = {
    AdvisorTool.GET_QUOTE.value: _get_quote,
    AdvisorTool.GET_CURVE.value: _get_curve,
    AdvisorTool.GET_CONSENSUS.value: _get_consensus,
    AdvisorTool.GET_CALENDAR.value: _get_calendar,
    AdvisorTool.GET_HISTORY.value: _get_history,
}


def run_tool(
    lookup: MarketLookup, name: str, tool_input: Mapping[str, Any], today: date
) -> ToolOutcome:
    """Dispatch one advisor tool call; never raises, always returns a `ToolOutcome`."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return _error(f"unknown tool '{name}'")
    try:
        return handler(lookup, tool_input, today)
    except (KeyError, TypeError, ValueError) as exc:
        return _error(f"invalid tool input: {exc}")
