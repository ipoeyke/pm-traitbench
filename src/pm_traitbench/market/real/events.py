"""Real-market calendar: fixed macro dates priced into surprises from the
seed's own simulated series.

Event dates and targets carry no randomness: FOMC, WASDE and NFP dates are a
public schedule, and inventory and crop reports follow a fixed weekday or
that same public schedule. A real seed has no earnings feed, so it never
draws an EARNINGS event. Surprise magnitude and sign come from the realised
market move, so no sampling stream is used here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from pm_traitbench.config import RealSeedSpec
from pm_traitbench.enums import CommodityGroup, EventType, InstrumentKind
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import row_sort_key
from pm_traitbench.market.check import EVENT_FAMILY
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.real.event_dates import EVENT_DATES_COVER
from pm_traitbench.market.real.universe import real_axis_dates
from pm_traitbench.tables.schema import CalendarEvent, Instrument

__all__ = [
    "EVENT_DATES_COVER",
    "FOMC_DATES",
    "NFP_DATES",
    "SURPRISE_SD_SCALE",
    "WASDE_DATES",
    "RealEvent",
    "drawn_counts",
    "real_event_days",
    "surprise_rows",
]

# A 1-sd day scores 1/3, and only a move beyond 3 sd reaches +-1, so a real
# event's surprise keeps its relative size instead of saturating every day.
SURPRISE_SD_SCALE = 3.0

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

# The event types a real seed can draw; rating actions and earnings are
# never generated for real seeds, so they always count as zero.
_COUNTED_EVENT_TYPES: tuple[EventType, ...] = tuple(EVENT_FAMILY)


@dataclass(frozen=True)
class RealEvent:
    """One dated, targeted calendar event, before its surprise is priced."""

    instrument_id: str | None
    event: EventType
    day: int


def real_event_days(
    instruments: Sequence[Instrument],
    spec: RealSeedSpec,
    axis: SimAxis,
    calendar_start: date,
) -> list[RealEvent]:
    """Build the real event calendar's dates and targets, horizon days only.

    Every axis day's real date is a weekday: `real_axis_dates` maps Monday to
    Monday, so the offset between `calendar_start` and `spec.window_start` is
    a whole number of weeks. A real seed draws no EARNINGS event, since it
    has no earnings feed.
    """
    real_dates = real_axis_dates(spec, axis, calendar_start)
    horizon = range(axis.n_burn, axis.n_days)
    real_date_to_day = {real_dates[t]: t for t in horizon}

    events: list[RealEvent] = []

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


def _daily_sd(diffs: np.ndarray, seed: str, name: str) -> float:
    """Sample sd (ddof=1) of one series' daily moves over the seed's full axis.
    A flat series has zero sd, which would divide by zero below.
    """
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        raise StageIOError(f"real seed '{seed}': '{name}' has zero daily sd over the axis")
    return sd


def surprise_rows(
    events: Sequence[RealEvent],
    output: ProcessOutput,
    spy_log_return: np.ndarray,
    y10_bp: np.ndarray,
    axis: SimAxis,
    seed: str,
) -> list[CalendarEvent]:
    """Price each RealEvent's surprise from the seed's own simulated series and
    turn it into a CalendarEvent row, sorted for the calendar table.

    Each surprise is the event's own daily move scaled by that series' own
    sd over the whole axis, so magnitude reflects how unusual the move was
    rather than a single fixed jump size.
    """
    y10_diff = np.diff(y10_bp)
    y10_sd_cache: float | None = None

    commodity_returns: dict[str, np.ndarray] = {}
    commodity_sd: dict[str, float] = {}
    spy_returns = spy_log_return[1:]
    spy_sd_cache: float | None = None

    def _commodity_returns(instrument_id: str) -> np.ndarray:
        if instrument_id not in commodity_returns:
            commodity_returns[instrument_id] = np.diff(np.log(output.prices[instrument_id]))
        return commodity_returns[instrument_id]

    rows: list[CalendarEvent] = []
    for event in events:
        t = event.day

        if event.event == EventType.CB_MEETING:
            if y10_sd_cache is None:
                y10_sd_cache = _daily_sd(y10_diff, seed, event.instrument_id or "curve")
            move = -(y10_bp[t] - y10_bp[t - 1])
            surprise = _clip(move / (SURPRISE_SD_SCALE * y10_sd_cache))
        elif event.event in (EventType.INVENTORY_REPORT, EventType.CROP_REPORT):
            returns = _commodity_returns(event.instrument_id)
            if event.instrument_id not in commodity_sd:
                commodity_sd[event.instrument_id] = _daily_sd(returns, seed, event.instrument_id)
            move = returns[t - 1]
            surprise = _clip(move / (SURPRISE_SD_SCALE * commodity_sd[event.instrument_id]))
        elif event.event == EventType.MACRO_PRINT:
            if spy_sd_cache is None:
                spy_sd_cache = _daily_sd(spy_returns, seed, "SPY")
            move = spy_log_return[t]
            surprise = _clip(move / (SURPRISE_SD_SCALE * spy_sd_cache))
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
