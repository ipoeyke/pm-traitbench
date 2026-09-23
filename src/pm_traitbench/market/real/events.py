"""Real-market calendar: fixed macro dates and SEC EDGAR earnings filings,
priced into surprises from the seed's own simulated series.

FOMC, WASDE and NFP dates are a public schedule; inventory and crop reports
follow a fixed weekday or that same public schedule. Earnings dates come
from each equity's 8-K filings with item 2.02 (Results of Operations),
converted from their EDGAR acceptance time to a real date. Surprise
magnitude and sign come from the realised market move, so no sampling
stream is used here.
"""

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from pm_traitbench.config import RealSeedSpec
from pm_traitbench.enums import CommodityGroup, EventType, Family, InstrumentKind
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import row_sort_key
from pm_traitbench.market.check import EVENT_FAMILY
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.real.event_dates import EVENT_DATES_COVER
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.sources import REAL_INSTRUMENTS
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

# A 1-sd day scores about 0.32 (tanh(1/3)); the value approaches +-1 as the
# move grows but never reaches it, so even a large outlier keeps its
# ranking relative to a bigger one instead of saturating to the same score.
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

# The event types a real seed can draw; rating actions are never generated
# for real seeds, so they always count as zero.
_COUNTED_EVENT_TYPES: tuple[EventType, ...] = tuple(EVENT_FAMILY)

_SERIES_BY_ID: dict[str, str] = {inst.instrument_id: inst.series for inst in REAL_INSTRUMENTS}
_CIKS_BY_ID: dict[str, tuple[str, ...]] = {
    inst.instrument_id: inst.ciks for inst in REAL_INSTRUMENTS if inst.ciks
}

# A results 8-K filed from 16:00 local time on counts as reported the next
# trading day, matching when the market could first react to it.
_MARKET_CLOSE = time(16, 0)
_EASTERN = ZoneInfo("America/New_York")


def _next_open_day(t: int, last: int, real_dates: Sequence[date], values: dict[date, float]) -> int:
    """First axis day at or after `t`, up to `last`, whose real date has a source
    value: a closed day (the series was filled, with no source value) moves the
    report to the next trading day that has one.
    """
    day = t
    while day < last and real_dates[day] not in values:
        day += 1
    return day


def _earnings_event_day(
    accepted: datetime, real_dates: Sequence[date], last_day: int, values: dict[date, float]
) -> int | None:
    """The axis day one 8-K filing's earnings event lands on, or None when its
    date falls outside the axis (before its start, e.g. an older filing from
    well before the fetch window, or a filing too close to the horizon's end).

    Before 16:00 America/New_York the filing's own real date is used, from
    16:00 on the next one; either way, the result then moves forward to the
    first axis day with a source value, same as a closed inventory report.
    """
    local = accepted.astimezone(_EASTERN)
    real_date = local.date() if local.time() < _MARKET_CLOSE else local.date() + timedelta(days=1)
    if real_date < real_dates[0]:
        return None
    start_t = bisect.bisect_left(real_dates, real_date)
    if start_t > last_day:
        return None
    day = _next_open_day(start_t, last_day, real_dates, values)
    if real_dates[day] not in values:
        return None
    return day


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
    cache: RawCache,
) -> list[RealEvent]:
    """Build the real event calendar's dates and targets, horizon days only.

    Every axis day's real date is a weekday: `real_axis_dates` maps Monday to
    Monday, so the offset between `calendar_start` and `spec.window_start` is
    a whole number of weeks. An earnings row comes from an equity's 8-K
    filings with item 2.02; several such filings landing on the same axis
    day for one equity still count as a single row. An inventory report due
    on a Wednesday whose commodity series was closed (filled, with no
    source value) moves to the next trading day with one.
    """
    real_dates = real_axis_dates(spec, axis, calendar_start)
    horizon = range(axis.n_burn, axis.n_days)
    real_date_to_day = {real_dates[t]: t for t in horizon}
    last_day = axis.n_days - 1

    events: list[RealEvent] = []

    equity_ids = [i.instrument_id for i in instruments if i.family == Family.EQUITIES]
    for instrument_id in equity_ids:
        ciks = _CIKS_BY_ID.get(instrument_id)
        if not ciks:
            continue
        values = cache.yahoo(_SERIES_BY_ID[instrument_id])
        days: set[int] = set()
        for filing in cache.edgar_filings(ciks):
            if filing.form != "8-K" or "2.02" not in filing.items:
                continue
            day = _earnings_event_day(filing.accepted, real_dates, last_day, values)
            if day is None or day < axis.n_burn:
                continue
            days.add(day)
        for day in sorted(days):
            events.append(RealEvent(instrument_id=instrument_id, event=EventType.EARNINGS, day=day))

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
    energy_values = {
        instrument_id: cache.yahoo(_SERIES_BY_ID[instrument_id]) for instrument_id in energy_ids
    }
    for t in horizon:
        if axis.dates[t].weekday() == 2:
            for instrument_id in energy_ids:
                day = _next_open_day(t, last_day, real_dates, energy_values[instrument_id])
                events.append(
                    RealEvent(
                        instrument_id=instrument_id, event=EventType.INVENTORY_REPORT, day=day
                    )
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


def _squash(value: float) -> float:
    """Map a raw sd-scaled move onto (-1, 1), preserving sign and relative
    size instead of saturating a large outlier to the same score as any
    other move past the bound.
    """
    return float(np.tanh(value))


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
    betas: Mapping[str, float],
) -> list[CalendarEvent]:
    """Price each RealEvent's surprise from the seed's own simulated series and
    turn it into a CalendarEvent row, sorted for the calendar table.

    Each surprise is the event's own daily move scaled by that series' own
    sd over the whole axis, so magnitude reflects how unusual the move was
    rather than a single fixed jump size. An earnings surprise instead
    scales the equity's abnormal return (its own move net of `beta` times
    SPY's) by that abnormal series' own sd, since a beta-driven move is not
    itself the surprise; an equity with no entry in `betas` raises.
    """
    y10_diff = np.diff(y10_bp)
    y10_sd_cache: float | None = None

    commodity_returns: dict[str, np.ndarray] = {}
    commodity_sd: dict[str, float] = {}
    spy_returns = spy_log_return[1:]
    spy_sd_cache: float | None = None

    equity_abnormal: dict[str, np.ndarray] = {}
    equity_sd: dict[str, float] = {}

    def _commodity_returns(instrument_id: str) -> np.ndarray:
        if instrument_id not in commodity_returns:
            commodity_returns[instrument_id] = np.diff(np.log(output.prices[instrument_id]))
        return commodity_returns[instrument_id]

    def _abnormal_return(instrument_id: str) -> np.ndarray:
        if instrument_id not in equity_abnormal:
            if instrument_id not in betas:
                raise StageIOError(f"real seed '{seed}': no beta for equity '{instrument_id}'")
            r_i = np.diff(np.log(output.prices[instrument_id]))
            a = np.zeros(len(spy_log_return))
            a[1:] = r_i - betas[instrument_id] * spy_returns
            equity_abnormal[instrument_id] = a
        return equity_abnormal[instrument_id]

    rows: list[CalendarEvent] = []
    for event in events:
        t = event.day

        if event.event == EventType.CB_MEETING:
            if y10_sd_cache is None:
                y10_sd_cache = _daily_sd(y10_diff, seed, event.instrument_id or "curve")
            move = -(y10_bp[t] - y10_bp[t - 1])
            surprise = _squash(move / (SURPRISE_SD_SCALE * y10_sd_cache))
        elif event.event in (EventType.INVENTORY_REPORT, EventType.CROP_REPORT):
            returns = _commodity_returns(event.instrument_id)
            if event.instrument_id not in commodity_sd:
                commodity_sd[event.instrument_id] = _daily_sd(returns, seed, event.instrument_id)
            move = returns[t - 1]
            surprise = _squash(move / (SURPRISE_SD_SCALE * commodity_sd[event.instrument_id]))
        elif event.event == EventType.MACRO_PRINT:
            if spy_sd_cache is None:
                spy_sd_cache = _daily_sd(spy_returns, seed, "SPY")
            move = spy_log_return[t]
            surprise = _squash(move / (SURPRISE_SD_SCALE * spy_sd_cache))
        elif event.event == EventType.EARNINGS:
            a = _abnormal_return(event.instrument_id)
            if event.instrument_id not in equity_sd:
                equity_sd[event.instrument_id] = _daily_sd(a[1:], seed, event.instrument_id)
            move = a[t]
            surprise = _squash(move / (SURPRISE_SD_SCALE * equity_sd[event.instrument_id]))
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
