"""Event calendar: scheduled and random market events, and their price jumps.

Placement runs on horizon indices `0..H-1` so burn-in days never carry an
event; indices convert to axis dates afterward. Event dates are drawn from a
seed-independent stream (every market seed shares one calendar), while
surprise magnitudes and signs are drawn from a seed-dependent stream (each
market seed sees its own surprises on the same calendar).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import (
    NULL_SURPRISE_EVENTS,
    CommodityGroup,
    EventType,
    Family,
    InstrumentKind,
)
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.constants import HORIZON_DAYS_PER_YEAR
from pm_traitbench.tables.schema import CalendarEvent, Instrument

RngFor = Callable[..., np.random.Generator]

EVENT_TARGETS: dict[EventType, Callable[[Instrument], bool]] = {
    EventType.EARNINGS: lambda i: i.family == Family.EQUITIES,
    EventType.RATING_DOWNGRADE: lambda i: i.family == Family.CREDIT,
    EventType.RATING_UPGRADE: lambda i: i.family == Family.CREDIT,
    EventType.CB_MEETING: lambda i: i.kind == InstrumentKind.SOVEREIGN_CURVE,
    EventType.INVENTORY_REPORT: lambda i: i.commodity_group == CommodityGroup.ENERGY,
    EventType.CROP_REPORT: lambda i: i.commodity_group == CommodityGroup.AGRICULTURE,
}
# macro_print has no per-instrument target: it is drawn once, market-wide.


def _grid_days(
    rng: np.random.Generator, per_year: float, jitter_days: int, horizon_days: int
) -> np.ndarray:
    """Evenly spaced day indices with a random phase offset and per-point jitter."""
    years = horizon_days / HORIZON_DAYS_PER_YEAR
    count = round(per_year * years)
    if count <= 0:
        return np.array([], dtype=int)
    spacing = horizon_days / count
    offset = rng.uniform(0, spacing)
    jitter = rng.integers(-jitter_days, jitter_days + 1, size=count)
    k = np.arange(count)
    days = np.floor(offset + k * spacing).astype(int) + jitter
    days = np.clip(days, 0, horizon_days - 1)
    return np.unique(days)


def _poisson_days(rng: np.random.Generator, per_year: float, horizon_days: int) -> np.ndarray:
    """Poisson-counted day indices drawn without replacement from the horizon."""
    years = horizon_days / HORIZON_DAYS_PER_YEAR
    count = min(int(rng.poisson(per_year * years)), horizon_days)
    if count <= 0:
        return np.array([], dtype=int)
    return np.sort(rng.choice(horizon_days, count, replace=False))


def row_sort_key(row: CalendarEvent) -> tuple[date, str, EventType]:
    """Sort rows by date, then instrument (market-wide rows first), then event type."""
    return (row.date, row.instrument_id or "", row.event)


def _surprises(rng: np.random.Generator, event: EventType, k: int) -> np.ndarray:
    """Draw k surprises: magnitude first, then sign for two-sided event types."""
    magnitude = rng.beta(2, 2, size=k)
    if event == EventType.RATING_DOWNGRADE:
        return -magnitude
    if event == EventType.RATING_UPGRADE:
        return magnitude
    sign = rng.choice([-1.0, 1.0], size=k)
    return magnitude * sign


@dataclass(frozen=True)
class SampledEvents:
    """Randomly sampled calendar events and the row count drawn per type."""

    rows: list[CalendarEvent]
    drawn: dict[EventType, int]


def sample_events(
    instruments: Sequence[Instrument],
    axis: SimAxis,
    config: Config,
    rng_for: RngFor,
    seed: str,
) -> SampledEvents:
    """Sample every configured event type's dates and surprises for one market seed."""
    horizon_days = axis.n_days - axis.n_burn
    rows: list[CalendarEvent] = []
    drawn: dict[EventType, int] = {}

    for event, spec in config.market.events.items():
        drawn[event] = 0
        if event == EventType.MACRO_PRINT:
            targets = [("macro", "all")]
        else:
            predicate = EVENT_TARGETS[event]
            targets = [
                (instrument.instrument_id, instrument.family.value)
                for instrument in instruments
                if predicate(instrument)
            ]

        for target_key, affected in targets:
            date_rng = rng_for("events", target_key, event.value)
            if spec.placement == "grid":
                day_indices = _grid_days(date_rng, spec.per_year, spec.jitter_days, horizon_days)
            else:
                day_indices = _poisson_days(date_rng, spec.per_year, horizon_days)
            if len(day_indices) == 0:
                continue

            surprise_rng = rng_for(seed, "surprise", target_key, event.value)
            surprises = _surprises(surprise_rng, event, len(day_indices))
            instrument_id = None if event == EventType.MACRO_PRINT else target_key
            for day_index, surprise in zip(day_indices, surprises, strict=True):
                rows.append(
                    CalendarEvent(
                        seed=seed,
                        date=axis.dates[axis.n_burn + int(day_index)],
                        instrument_id=instrument_id,
                        event=event,
                        surprise=float(surprise),
                        affected=affected,
                    )
                )
            drawn[event] += len(day_indices)

    rows.sort(key=row_sort_key)
    return SampledEvents(rows=rows, drawn=drawn)


def third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given calendar month."""
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(days=14)


def generated_rows(
    instruments: Sequence[Instrument], axis: SimAxis, seed: str
) -> list[CalendarEvent]:
    """Deterministic, un-sampled rows: commodity contract expiries and weekly positioning."""
    horizon_dates = axis.dates[axis.horizon]
    horizon_set = set(horizon_dates)
    commodities = [i for i in instruments if i.family == Family.COMMODITIES]

    months = {(day.year, day.month) for day in horizon_dates}
    candidate_expiries = (third_friday(year, month) for year, month in months)
    expiries = sorted(expiry for expiry in candidate_expiries if expiry in horizon_set)

    rows: list[CalendarEvent] = []
    for expiry in expiries:
        for commodity in commodities:
            rows.append(
                CalendarEvent(
                    seed=seed,
                    date=expiry,
                    instrument_id=commodity.instrument_id,
                    event=EventType.CONTRACT_EXPIRY,
                    surprise=None,
                    affected="commodities",
                )
            )

    for day in horizon_dates:
        if day.weekday() == 4:
            rows.append(
                CalendarEvent(
                    seed=seed,
                    date=day,
                    instrument_id=None,
                    event=EventType.POSITIONING_REPORT,
                    surprise=None,
                    affected="all",
                )
            )

    rows.sort(key=row_sort_key)
    return rows


@dataclass(frozen=True)
class EventJumps:
    """Per-axis-day sum of `surprise x jump_size`, per instrument and market-wide."""

    by_instrument: Mapping[str, np.ndarray]
    macro: np.ndarray

    def for_instrument(self, instrument_id: str) -> np.ndarray:
        if instrument_id in self.by_instrument:
            return self.by_instrument[instrument_id]
        return np.zeros_like(self.macro)


def build_jumps(rows: Sequence[CalendarEvent], axis: SimAxis, config: Config) -> EventJumps:
    """Accumulate sampled events' surprise x jump_size onto their instrument's axis day."""
    by_instrument: dict[str, np.ndarray] = {}
    macro = np.zeros(axis.n_days)

    for row in rows:
        if row.event in NULL_SURPRISE_EVENTS:
            continue
        value = row.surprise * config.market.events[row.event].jump_size
        day_index = axis.index(row.date)
        if row.instrument_id is None:
            macro[day_index] += value
        else:
            arr = by_instrument.setdefault(row.instrument_id, np.zeros(axis.n_days))
            arr[day_index] += value

    macro.flags.writeable = False
    for arr in by_instrument.values():
        arr.flags.writeable = False
    return EventJumps(by_instrument=by_instrument, macro=macro)


def event_day_indices(rows: Sequence[CalendarEvent], axis: SimAxis) -> dict[str, set[int]]:
    """Map each instrument to the axis indices of its sampled events."""
    result: dict[str, set[int]] = {}
    for row in rows:
        if row.instrument_id is None or row.event in NULL_SURPRISE_EVENTS:
            continue
        result.setdefault(row.instrument_id, set()).add(axis.index(row.date))
    return result
