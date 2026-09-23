"""Shared calendar helpers: sort order, deterministic rows and event-day lookup.

`row_sort_key` orders every calendar row the same way regardless of its
source; contract expiries and weekly positioning reports are computed
directly from the axis, with no random draw.
"""

from collections.abc import Callable, Sequence
from datetime import date, timedelta

import numpy as np

from pm_traitbench.enums import NULL_SURPRISE_EVENTS, EventType, Family
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.tables.schema import CalendarEvent, Instrument

RngFor = Callable[..., np.random.Generator]


def row_sort_key(row: CalendarEvent) -> tuple[date, str, EventType]:
    """Sort rows by date, then instrument (market-wide rows first), then event type."""
    return (row.date, row.instrument_id or "", row.event)


def third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given calendar month."""
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(days=14)


def generated_rows(
    instruments: Sequence[Instrument], axis: SimAxis, seed: str, report_weekday: int
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
        if day.weekday() == report_weekday:
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


def event_day_indices(rows: Sequence[CalendarEvent], axis: SimAxis) -> dict[str, set[int]]:
    """Map each instrument to the axis indices of its sampled events."""
    result: dict[str, set[int]] = {}
    for row in rows:
        if row.instrument_id is None or row.event in NULL_SURPRISE_EVENTS:
            continue
        result.setdefault(row.instrument_id, set()).add(axis.index(row.date))
    return result
