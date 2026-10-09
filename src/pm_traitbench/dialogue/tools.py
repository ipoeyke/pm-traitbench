"""Read-only market lookup tools the simulated advisor calls instead of stating a market fact
from memory.

`MarketLookup` indexes one seed's stage 2 market tables (prices, curves,
consensus, calendar) for the five `AdvisorTool` lookups. `run_tool` dispatches
one tool call and never raises: every handler validates its own input first
(missing keys, wrong types, out-of-range windows), so a bad tool name,
unknown instrument or malformed argument all come back as an error
`ToolOutcome` rather than an exception. No lookup returns a row dated after
the session date, except a calendar's own future-scheduled rows, whose
surprise is masked rather than omitted.
"""

import bisect
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher, get_close_matches
from types import MappingProxyType
from typing import Any

from pm_traitbench.enums import AdvisorTool, Family, InstrumentKind, Tenor
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import CalendarEvent, ConsensusRow, CurvePoint, Instrument, Price

_MAX_CLOSE_MATCHES = 5
# difflib's own cutoff, applied to the similarity ranking so ids sort by id on a tie.
_MIN_SIMILARITY = 0.6
# Desk shorthand for a sovereign curve, by the currency that names it in the universe.
_SOVEREIGN_SHORTHAND: Mapping[str, str] = MappingProxyType(
    {
        "ust": "USD",
        "usts": "USD",
        "treasury": "USD",
        "treasuries": "USD",
        "bund": "EUR",
        "bunds": "EUR",
        "btp": "EUR",
        "btps": "EUR",
        "oat": "EUR",
        "oats": "EUR",
        "gilt": "GBP",
        "gilts": "GBP",
        "jgb": "JPY",
        "jgbs": "JPY",
    }
)
_NO_NAMES: Mapping[str, str] = MappingProxyType({})
_WINDOW_MIN, _WINDOW_MAX = 0, 20
_HISTORY_MIN, _HISTORY_MAX = 1, 60
_CURVE_KINDS = frozenset({InstrumentKind.SOVEREIGN_CURVE, InstrumentKind.COMMODITY})


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
        instrument, or matches one that is not a sovereign curve or
        commodity, since the advisor must never be handed a curve it cannot
        correctly name back to the PM.
        """
        seed_prices = [row for row in prices if row.seed == seed]
        seed_curves = [row for row in curves if row.seed == seed]
        seed_consensus = [row for row in consensus if row.seed == seed]
        seed_calendar = [row for row in calendar if row.seed == seed]

        instrument_by_id = {instrument.instrument_id: instrument for instrument in instruments}
        for curve in seed_curves:
            instrument = instrument_by_id.get(curve.curve_id)
            if instrument is None:
                raise DialogueError(f"curve id '{curve.curve_id}' matches no instrument")
            if instrument.kind not in _CURVE_KINDS:
                raise DialogueError(
                    f"curve id '{curve.curve_id}' names a '{instrument.kind.value}' instrument, "
                    "expected sovereign_curve or commodity"
                )

        dates = tuple(sorted({row.date for row in seed_prices}))
        covered_ids = {row.instrument_id for row in seed_prices} | {
            row.curve_id for row in seed_curves
        }
        instruments_by_id = {
            instrument_id: instrument_by_id[instrument_id]
            for instrument_id in sorted(covered_ids)
            if instrument_id in instrument_by_id
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

    def latest_price(self, instrument_id: str, today: date) -> Price | None:
        """The most recent price row for `instrument_id` on or before `today`, if any."""
        rows = self._prices.get(instrument_id, ())
        idx = bisect.bisect_right(rows, today, key=lambda r: r.date) - 1
        return rows[idx] if idx >= 0 else None

    def history(self, instrument_id: str, today: date, n_days: int) -> tuple[Price, ...]:
        """Up to the last `n_days` price rows for `instrument_id` on or before `today`."""
        rows = self._prices.get(instrument_id, ())
        idx = bisect.bisect_right(rows, today, key=lambda r: r.date) - 1
        if idx < 0:
            return ()
        return rows[max(0, idx - n_days + 1) : idx + 1]

    def has_curve(self, instrument_id: str) -> bool:
        """Whether `instrument_id` has any curve row on this seed."""
        return instrument_id in self._curve_dates

    def curve_on_or_before(
        self, instrument_id: str, today: date
    ) -> tuple[date, Mapping[Tenor, float]] | None:
        """The instrument's curve date and tenor levels on the latest date on or before `today`."""
        dates = self._curve_dates.get(instrument_id, ())
        idx = bisect.bisect_right(dates, today) - 1
        if idx < 0:
            return None
        curve_date = dates[idx]
        return curve_date, self._curve_levels[(instrument_id, curve_date)]

    def has_consensus(self, instrument_id: str) -> bool:
        """Whether `instrument_id` has any consensus row on this seed."""
        return instrument_id in self._consensus

    def latest_consensus(self, instrument_id: str, today: date) -> ConsensusRow | None:
        """The most recent consensus row for `instrument_id` on or before `today`, if any."""
        rows = self._consensus.get(instrument_id, ())
        idx = bisect.bisect_right(rows, today, key=lambda r: r.date) - 1
        return rows[idx] if idx >= 0 else None

    def calendar_events(
        self, instrument_id: str, today: date, days_back: int, days_forward: int
    ) -> tuple[CalendarEvent, ...]:
        """Own and market-wide calendar rows in a trading-day window around `today`.

        The window spans `days_back` trading days before through
        `days_forward` trading days after `today`, always including `today`
        itself even when it falls on a holiday.
        """
        lo, hi = self._calendar_window(today, days_back, days_forward)
        own = self._calendar_by_instrument.get(instrument_id, ())
        combined = [row for row in own if lo <= row.date <= hi]
        combined += [row for row in self._calendar_market_wide if lo <= row.date <= hi]
        return tuple(sorted(combined, key=lambda r: (r.date, r.event.value, r.instrument_id or "")))

    def _calendar_window(self, today: date, days_back: int, days_forward: int) -> tuple[date, date]:
        """The trading-day window `[lo, hi]` around `today`, widened to include `today` itself.

        `lo` is `days_back` trading days before `today`, `hi` is
        `days_forward` trading days after, each clamped to the seed's date
        range and pulled in to cover `today` when it falls outside every
        trading day (e.g. a holiday).
        """
        if not self.dates:
            return today, today
        n = len(self.dates)
        lo_idx = max(bisect.bisect_left(self.dates, today) - days_back, 0)
        hi_idx = min(bisect.bisect_right(self.dates, today) - 1 + days_forward, n - 1)
        if lo_idx > hi_idx:
            return today, today
        return min(self.dates[lo_idx], today), max(self.dates[hi_idx], today)


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


def _require_str(tool_input: Mapping[str, Any], key: str) -> str | ToolOutcome:
    """`tool_input[key]` as a `str`, or an error outcome if missing or the wrong type."""
    if key not in tool_input:
        return _error(f"missing required field '{key}'")
    value = tool_input[key]
    if not isinstance(value, str):
        return _error(f"field '{key}' must be a string, got {type(value).__name__}")
    return value


def _require_int(tool_input: Mapping[str, Any], key: str) -> int | ToolOutcome:
    """`tool_input[key]` as an `int`, or an error outcome if missing or the wrong type.

    `bool` is a subclass of `int` in Python but is never a valid window or count here.
    """
    if key not in tool_input:
        return _error(f"missing required field '{key}'")
    value = tool_input[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return _error(f"field '{key}' must be an integer, got {type(value).__name__}")
    return value


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _suggestions(lookup: MarketLookup, instrument: str) -> list[str]:
    """Names the query could mean, best first: those sharing a word with it ("usd curve"),
    a sovereign curve its words name by shorthand or currency ("bund", "ust"), then the
    closest names and ids by edit similarity (a near-miss id such as "EQ-001")."""
    instruments = list(lookup.instruments.values())
    by_instrument = {
        c.instrument_id: _tokens(c.name) | _tokens(c.instrument_id) for c in instruments
    }
    # A word on more than a quarter of the universe ("eq", "issuer") picks nothing out.
    counts = Counter(token for tokens in by_instrument.values() for token in tokens)
    common = {token for token, n in counts.items() if n > len(by_instrument) / 4}
    query = _tokens(instrument) - common
    shared = sorted(
        ((len(query & by_instrument[c.instrument_id]), c.instrument_id, c) for c in instruments),
        key=lambda item: (-item[0], item[1]),
    )
    ranked = [c.name for n, _, c in shared if n]
    curves = [c for c in instruments if c.family == Family.RATES]
    named = {_SOVEREIGN_SHORTHAND.get(t, t.upper()) for t in _tokens(instrument)}
    ranked += [c.name for c in curves if c.currency in named]
    ranked += [
        c.name for c in curves if get_close_matches(instrument.lower(), [c.currency.lower()])
    ]
    lowered = instrument.lower()
    similar = sorted(
        ((SequenceMatcher(None, lowered, text.lower()).ratio(), text) for text in _names(lookup)),
        key=lambda item: (-item[0], item[1]),
    )
    ranked += [text for ratio, text in similar[:_MAX_CLOSE_MATCHES] if ratio >= _MIN_SIMILARITY]
    return list(dict.fromkeys(ranked))[:_MAX_CLOSE_MATCHES]


def _names(lookup: MarketLookup) -> list[str]:
    return [c.name for c in lookup.instruments.values()] + list(lookup.instruments)


def _resolve_or_error(
    lookup: MarketLookup, instrument: str, session_names: Mapping[str, str]
) -> Instrument | ToolOutcome:
    """The instrument, or an error naming the closest lookup names and the session's own
    instruments: the advisor only ever hears those in the PM's words, so a guessed ticker
    must be answered with the names and ids the lookup knows."""
    resolved = lookup.resolve(instrument)
    if resolved is not None:
        return resolved
    message = f"unknown instrument '{instrument}'"
    matches = _suggestions(lookup, instrument)
    if matches:
        message += f"; closest names: {', '.join(matches)}"
    listed = ", ".join(f"{name} ({instrument_id})" for instrument_id, name in session_names.items())
    if listed:
        message += f"; the PM's instruments: {listed}"
    return _error(message)


def _no_row_error(name: str, today: date) -> ToolOutcome:
    return _error(f"no row for '{name}' on or before {today.isoformat()}")


def _get_quote(
    lookup: MarketLookup,
    tool_input: Mapping[str, Any],
    today: date,
    session_names: Mapping[str, str],
) -> ToolOutcome:
    instrument = _require_str(tool_input, "instrument")
    if isinstance(instrument, ToolOutcome):
        return instrument
    resolved = _resolve_or_error(lookup, instrument, session_names)
    if isinstance(resolved, ToolOutcome):
        return resolved
    row = lookup.latest_price(resolved.instrument_id, today)
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


def _get_curve(
    lookup: MarketLookup,
    tool_input: Mapping[str, Any],
    today: date,
    session_names: Mapping[str, str],
) -> ToolOutcome:
    instrument = _require_str(tool_input, "instrument")
    if isinstance(instrument, ToolOutcome):
        return instrument
    resolved = _resolve_or_error(lookup, instrument, session_names)
    if isinstance(resolved, ToolOutcome):
        return resolved
    if not lookup.has_curve(resolved.instrument_id):
        return _error(f"'{resolved.name}' has no curve")
    found = lookup.curve_on_or_before(resolved.instrument_id, today)
    if found is None:
        return _no_row_error(resolved.name, today)
    curve_date, raw_levels = found
    levels = {tenor.value: raw_levels[tenor] for tenor in Tenor if tenor in raw_levels}
    return ToolOutcome(
        result={
            "instrument_id": resolved.instrument_id,
            "name": resolved.name,
            "date": curve_date.isoformat(),
            "field": "level",
            "levels": levels,
        },
        is_error=False,
    )


def _get_consensus(
    lookup: MarketLookup,
    tool_input: Mapping[str, Any],
    today: date,
    session_names: Mapping[str, str],
) -> ToolOutcome:
    instrument = _require_str(tool_input, "instrument")
    if isinstance(instrument, ToolOutcome):
        return instrument
    resolved = _resolve_or_error(lookup, instrument, session_names)
    if isinstance(resolved, ToolOutcome):
        return resolved
    if not lookup.has_consensus(resolved.instrument_id):
        return _error(f"'{resolved.name}' has no consensus rows")
    row = lookup.latest_consensus(resolved.instrument_id, today)
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


def _get_calendar(
    lookup: MarketLookup,
    tool_input: Mapping[str, Any],
    today: date,
    session_names: Mapping[str, str],
) -> ToolOutcome:
    instrument = _require_str(tool_input, "instrument")
    if isinstance(instrument, ToolOutcome):
        return instrument
    days_back = _require_int(tool_input, "days_back")
    if isinstance(days_back, ToolOutcome):
        return days_back
    days_forward = _require_int(tool_input, "days_forward")
    if isinstance(days_forward, ToolOutcome):
        return days_forward
    if not (_WINDOW_MIN <= days_back <= _WINDOW_MAX):
        return _error(f"days_back must be between {_WINDOW_MIN} and {_WINDOW_MAX}, got {days_back}")
    if not (_WINDOW_MIN <= days_forward <= _WINDOW_MAX):
        return _error(
            f"days_forward must be between {_WINDOW_MIN} and {_WINDOW_MAX}, got {days_forward}"
        )
    resolved = _resolve_or_error(lookup, instrument, session_names)
    if isinstance(resolved, ToolOutcome):
        return resolved
    rows = lookup.calendar_events(resolved.instrument_id, today, days_back, days_forward)
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


def _get_history(
    lookup: MarketLookup,
    tool_input: Mapping[str, Any],
    today: date,
    session_names: Mapping[str, str],
) -> ToolOutcome:
    instrument = _require_str(tool_input, "instrument")
    if isinstance(instrument, ToolOutcome):
        return instrument
    n_days = _require_int(tool_input, "n_days")
    if isinstance(n_days, ToolOutcome):
        return n_days
    if not (_HISTORY_MIN <= n_days <= _HISTORY_MAX):
        return _error(f"n_days must be between {_HISTORY_MIN} and {_HISTORY_MAX}, got {n_days}")
    resolved = _resolve_or_error(lookup, instrument, session_names)
    if isinstance(resolved, ToolOutcome):
        return resolved
    rows = lookup.history(resolved.instrument_id, today, n_days)
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
    lookup: MarketLookup,
    name: str,
    tool_input: Mapping[str, Any],
    today: date,
    session_names: Mapping[str, str] = _NO_NAMES,
) -> ToolOutcome:
    """Dispatch one advisor tool call; malformed input returns an error outcome.

    `session_names` maps the session's instrument ids to names, listed back when an
    instrument matches nothing close.

    Each handler validates its own `tool_input` first, so a bad tool name,
    unknown instrument or malformed argument never raises. A bug elsewhere
    in the lookup is not caught here and propagates as an exception.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return _error(f"unknown tool '{name}'")
    return handler(lookup, tool_input, today, session_names)
