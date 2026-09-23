"""Real-market calendar: fixed macro dates plus the Nasdaq earnings feed,
priced into surprises from the seed's own simulated series.

Event dates and targets carry no randomness: FOMC, WASDE and NFP dates are a
public schedule, earnings dates come from the Nasdaq calendar, and inventory
and crop reports follow a fixed weekday or that same public schedule.
Surprise magnitude and sign come from the realised market move, so no
sampling stream is used here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from pm_traitbench.config import Config, RealSeedSpec
from pm_traitbench.enums import CommodityGroup, EventType, Family, InstrumentKind
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import row_sort_key
from pm_traitbench.market.check import EVENT_FAMILY
from pm_traitbench.market.constants import ANNUALISATION_DAYS
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.sources import REAL_INSTRUMENTS
from pm_traitbench.market.real.universe import real_axis_dates
from pm_traitbench.tables.schema import CalendarEvent, Instrument

FOMC_DATES: tuple[date, ...] = (
    date(2018, 6, 13),
    date(2018, 8, 1),
    date(2018, 9, 26),
    date(2018, 11, 8),
    date(2018, 12, 19),
    date(2019, 1, 30),
    date(2019, 3, 20),
    date(2019, 5, 1),
)
# USDA WASDE archive; no January 2019 report because of the government shutdown.
WASDE_DATES: tuple[date, ...] = (
    date(2018, 6, 12),
    date(2018, 7, 12),
    date(2018, 8, 10),
    date(2018, 9, 12),
    date(2018, 10, 11),
    date(2018, 11, 8),
    date(2018, 12, 11),
    date(2019, 2, 8),
    date(2019, 3, 8),
    date(2019, 4, 9),
    date(2019, 5, 10),
)
# BLS Employment Situation release schedule.
NFP_DATES: tuple[date, ...] = (
    date(2018, 7, 6),
    date(2018, 8, 3),
    date(2018, 9, 7),
    date(2018, 10, 5),
    date(2018, 11, 2),
    date(2018, 12, 7),
    date(2019, 1, 4),
    date(2019, 2, 1),
    date(2019, 3, 8),
    date(2019, 4, 5),
    date(2019, 5, 3),
)

_TICKER_TO_INSTRUMENT_ID: dict[str, str] = {
    inst.series: inst.instrument_id for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES
}

# The event types a real seed can draw; rating actions are never generated
# for real seeds, so they always count as zero.
_COUNTED_EVENT_TYPES: tuple[EventType, ...] = tuple(EVENT_FAMILY)


@dataclass(frozen=True)
class RealEvent:
    """One dated, targeted calendar event, before its surprise is priced."""

    instrument_id: str | None
    event: EventType
    day: int


def real_event_days(
    instruments: Sequence[Instrument],
    cache: RawCache,
    spec: RealSeedSpec,
    axis: SimAxis,
    calendar_start: date,
) -> list[RealEvent]:
    """Build the real event calendar's dates and targets, horizon days only.

    Every axis day's real date is a weekday: `real_axis_dates` maps Monday to
    Monday, so the offset between `calendar_start` and `spec.window_start` is
    a whole number of weeks. The Nasdaq feed is fetched for weekdays only, so
    `cache.nasdaq_symbols` is only ever asked about a weekday too.
    """
    real_dates = real_axis_dates(spec, axis, calendar_start)
    horizon = range(axis.n_burn, axis.n_days)
    real_date_to_day = {real_dates[t]: t for t in horizon}

    events: list[RealEvent] = []

    for t in horizon:
        symbols = cache.nasdaq_symbols(real_dates[t]) & set(_TICKER_TO_INSTRUMENT_ID)
        for symbol in sorted(symbols):
            instrument_id = _TICKER_TO_INSTRUMENT_ID[symbol]
            events.append(RealEvent(instrument_id=instrument_id, event=EventType.EARNINGS, day=t))

    curve_ids = [i.instrument_id for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
    for fomc_date in FOMC_DATES:
        day = real_date_to_day.get(fomc_date)
        if day is None:
            continue
        for instrument_id in curve_ids:
            events.append(
                RealEvent(instrument_id=instrument_id, event=EventType.CB_MEETING, day=day)
            )

    energy_ids = [
        i.instrument_id for i in instruments if i.commodity_group == CommodityGroup.ENERGY
    ]
    for t in horizon:
        if axis.dates[t].weekday() == 2:
            for instrument_id in energy_ids:
                events.append(
                    RealEvent(instrument_id=instrument_id, event=EventType.INVENTORY_REPORT, day=t)
                )

    agriculture_ids = [
        i.instrument_id for i in instruments if i.commodity_group == CommodityGroup.AGRICULTURE
    ]
    for wasde_date in WASDE_DATES:
        day = real_date_to_day.get(wasde_date)
        if day is None:
            continue
        for instrument_id in agriculture_ids:
            events.append(
                RealEvent(instrument_id=instrument_id, event=EventType.CROP_REPORT, day=day)
            )

    for nfp_date in NFP_DATES:
        day = real_date_to_day.get(nfp_date)
        if day is None:
            continue
        events.append(RealEvent(instrument_id=None, event=EventType.MACRO_PRINT, day=day))

    return events


def _clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _log_return(prices: np.ndarray, t: int) -> float:
    return float(np.log(prices[t]) - np.log(prices[t - 1]))


def surprise_rows(
    events: Sequence[RealEvent],
    output: ProcessOutput,
    spy_log_return: np.ndarray,
    betas: Mapping[str, float],
    y10_bp: np.ndarray,
    axis: SimAxis,
    seed: str,
    config: Config,
) -> list[CalendarEvent]:
    """Price each RealEvent's surprise from the seed's own simulated series and
    turn it into a CalendarEvent row, sorted for the calendar table.
    """
    daily_vol = config.market.families.equity.market_vol / np.sqrt(ANNUALISATION_DAYS)

    rows: list[CalendarEvent] = []
    for event in events:
        t = event.day
        jump_size = config.market.events[event.event].jump_size

        if event.event == EventType.EARNINGS:
            r_i = _log_return(output.prices[event.instrument_id], t)
            beta = betas[event.instrument_id]
            surprise = _clip((r_i - beta * spy_log_return[t]) / jump_size)
        elif event.event == EventType.CB_MEETING:
            surprise = _clip(-(y10_bp[t] - y10_bp[t - 1]) / jump_size)
        elif event.event in (EventType.INVENTORY_REPORT, EventType.CROP_REPORT):
            r_c = _log_return(output.prices[event.instrument_id], t)
            surprise = _clip(r_c / jump_size)
        elif event.event == EventType.MACRO_PRINT:
            surprise = _clip((spy_log_return[t] / daily_vol) / jump_size)
        else:
            raise ValueError(f"real events do not support {event.event.value}")

        family = EVENT_FAMILY.get(event.event)
        affected = family.value if family is not None else "all"

        rows.append(
            CalendarEvent(
                seed=seed,
                date=axis.dates[t],
                instrument_id=event.instrument_id,
                event=event.event,
                surprise=surprise,
                affected=affected,
            )
        )

    rows.sort(key=row_sort_key)
    return rows


def drawn_counts(rows: Sequence[CalendarEvent]) -> dict[EventType, int]:
    """Row count per event type; rating types map to 0 since real seeds never
    draw them.
    """
    counts = dict.fromkeys(_COUNTED_EVENT_TYPES, 0)
    for row in rows:
        counts[row.event] = counts.get(row.event, 0) + 1
    return counts
