"""Tests for the real-market event calendar: dates, targets and surprises."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import CommodityGroup, EventType, Family, InstrumentKind
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import SimAxis, build_axis
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.real.events import (
    FOMC_DATES,
    NFP_DATES,
    SURPRISE_SD_SCALE,
    WASDE_DATES,
    RealEvent,
    drawn_counts,
    real_event_days,
    surprise_rows,
)
from pm_traitbench.market.real.fetch import EdgarFiling, RawCache
from pm_traitbench.market.real.sources import REAL_INSTRUMENTS
from pm_traitbench.market.real.universe import build_real_universe, real_axis_dates
from pm_traitbench.tables.schema import CalendarEvent, Instrument

_WINDOW_START = date(2018, 6, 4)
_WINDOW_END = date(2019, 5, 31)

_AAPL_CIK = next(i.ciks[0] for i in REAL_INSTRUMENTS if i.series == "AAPL")


def _equity_instrument(instrument_id: str = "EQ-R001", beta: float = 1.0) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name=instrument_id,
        currency="USD",
        sector="sector_01",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=beta,
        expiry_rule=None,
    )


class _FakeEarningsCache:
    """A minimal hand-built cache exposing only what earnings events need:
    EDGAR filings by CIK and each equity's raw Yahoo close-date set.
    """

    def __init__(
        self, filings: dict[str, list[EdgarFiling]], yahoo: dict[str, dict[date, float]]
    ) -> None:
        self._filings = filings
        self._yahoo = yahoo

    def edgar_filings(self, ciks) -> list[EdgarFiling]:
        return [filing for cik in ciks for filing in self._filings.get(cik, [])]

    def yahoo(self, ticker: str) -> dict[date, float]:
        return self._yahoo.get(ticker, {})


def _weekday_axis(start: date, n: int) -> SimAxis:
    """`n` consecutive weekdays from `start` (a Monday), no burn-in."""
    dates = []
    day = start
    while len(dates) < n:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return SimAxis(dates=tuple(dates), n_burn=0)


def _all_trading_days(axis: SimAxis) -> dict[date, float]:
    return dict.fromkeys(axis.dates, 100.0)


def _filing(accepted: datetime, items: str = "2.02,9.01", form: str = "8-K") -> EdgarFiling:
    return EdgarFiling(form=form, accepted=accepted, items=tuple(items.split(",")) if items else ())


@pytest.mark.parametrize("dates", [FOMC_DATES, WASDE_DATES, NFP_DATES])
def test_fixed_date_lists_are_ascending_and_within_the_seed_window(dates: tuple[date, ...]) -> None:
    assert list(dates) == sorted(dates)
    assert len(set(dates)) == len(dates)
    for day in dates:
        assert _WINDOW_START <= day <= _WINDOW_END


def _real_event_days(config: Config, result):
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    spec = config.market.real.seeds["R1"]
    instruments = build_real_universe(config, cache, axis)
    events = real_event_days(instruments, spec, axis, config.calendar.start, cache)
    return axis, spec, instruments, events


def test_no_events_fall_in_burn_in(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    axis, _, _, events = _real_event_days(config, result)

    assert events
    assert all(e.day >= axis.n_burn for e in events)


def test_earnings_events_are_drawn_from_the_fixture(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, instruments, events = _real_event_days(config, result)

    equity_ids = {i.instrument_id for i in instruments if i.family == Family.EQUITIES}
    earnings = [e for e in events if e.event == EventType.EARNINGS]
    assert earnings
    assert {e.instrument_id for e in earnings} <= equity_ids


_EARNINGS_WINDOW_START = date(2018, 6, 4)


def _earnings_days(filings: list[EdgarFiling], *, n_days: int = 10) -> list[RealEvent]:
    axis = _weekday_axis(_EARNINGS_WINDOW_START, n_days)
    spec = Config().market.real.seeds["R1"].model_copy(update={"window_start": axis.dates[0]})
    cache = _FakeEarningsCache(
        filings={_AAPL_CIK: filings}, yahoo={"AAPL": _all_trading_days(axis)}
    )
    events = real_event_days([_equity_instrument()], spec, axis, axis.dates[0], cache)
    return [e for e in events if e.event == EventType.EARNINGS]


def test_earnings_filing_after_the_close_lands_on_the_next_trading_day() -> None:
    accepted = datetime(2018, 6, 6, 20, 30, tzinfo=UTC)  # 16:30 EDT, after the 16:00 close
    events = _earnings_days([_filing(accepted)])
    assert len(events) == 1
    assert events[0].day == 3  # 2018-06-07, the next weekday after 2018-06-06


def test_earnings_filing_before_the_close_lands_on_the_same_day() -> None:
    accepted = datetime(2018, 6, 8, 12, 0, tzinfo=UTC)  # 08:00 EDT, before the close
    events = _earnings_days([_filing(accepted)])
    assert len(events) == 1
    assert events[0].day == 4  # 2018-06-08 itself


def test_earnings_filing_without_item_202_is_ignored() -> None:
    accepted = datetime(2018, 6, 6, 12, 0, tzinfo=UTC)
    events = _earnings_days([_filing(accepted, items="5.02")])
    assert events == []


def test_two_earnings_filings_on_one_day_give_one_row() -> None:
    first = datetime(2018, 6, 6, 12, 0, tzinfo=UTC)
    second = datetime(2018, 6, 6, 13, 0, tzinfo=UTC)
    events = _earnings_days([_filing(first), _filing(second)])
    assert len(events) == 1


def test_earnings_filing_just_before_the_horizon_gives_no_row() -> None:
    # The axis's last weekday, after the close: the shifted date falls on the
    # weekend right after the horizon ends, with no axis day to land on.
    axis = _weekday_axis(_EARNINGS_WINDOW_START, 10)
    last_date = axis.dates[-1]
    accepted = datetime(last_date.year, last_date.month, last_date.day, 20, 30, tzinfo=UTC)
    events = _earnings_days([_filing(accepted)])
    assert events == []


def test_earnings_events_instrument_id_is_the_registry_id() -> None:
    accepted = datetime(2018, 6, 6, 12, 0, tzinfo=UTC)
    events = _earnings_days([_filing(accepted)])
    assert events[0].instrument_id == "EQ-R001"


def test_cb_meeting_fires_once_per_fomc_date_for_the_usd_curve(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, _, events = _real_event_days(config, result)

    cb = [e for e in events if e.event == EventType.CB_MEETING]
    assert len(cb) == len(FOMC_DATES)
    assert {e.instrument_id for e in cb} == {"RT-USD"}


def test_inventory_report_fires_every_horizon_wednesday_for_energy_commodities(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    axis, _, instruments, events = _real_event_days(config, result)

    energy_ids = {
        i.instrument_id for i in instruments if i.commodity_group == CommodityGroup.ENERGY
    }
    assert len(energy_ids) == 4

    inventory = [e for e in events if e.event == EventType.INVENTORY_REPORT]
    expected_wednesdays = sum(
        1 for t in range(axis.n_burn, axis.n_days) if axis.dates[t].weekday() == 2
    )
    assert len(inventory) == expected_wednesdays * len(energy_ids)
    assert {e.instrument_id for e in inventory} == energy_ids
    for e in inventory:
        assert axis.dates[e.day].weekday() == 2


def test_closed_wednesday_inventory_report_moves_to_the_next_trading_day(fake_cache) -> None:
    config = Config()
    result = fake_cache(config, holiday_weekday=2)
    axis, spec, instruments, events = _real_event_days(config, result)

    real_dates = real_axis_dates(spec, axis, config.calendar.start)
    holiday_day = real_dates.index(result.holiday)
    assert axis.n_burn <= holiday_day < axis.n_days
    assert axis.dates[holiday_day].weekday() == 2
    assert axis.dates[holiday_day + 1].weekday() == 3

    energy_ids = {
        i.instrument_id for i in instruments if i.commodity_group == CommodityGroup.ENERGY
    }
    inventory = [e for e in events if e.event == EventType.INVENTORY_REPORT]
    on_holiday = {e.instrument_id for e in inventory if e.day == holiday_day}
    moved = {e.instrument_id for e in inventory if e.day == holiday_day + 1}

    assert on_holiday == set()
    assert moved == energy_ids


def test_open_wednesday_inventory_report_stays_on_its_own_day(fake_cache) -> None:
    config = Config()
    result = fake_cache(config, holiday_weekday=2)
    axis, spec, instruments, events = _real_event_days(config, result)

    real_dates = real_axis_dates(spec, axis, config.calendar.start)
    holiday_day = real_dates.index(result.holiday)

    energy_ids = {
        i.instrument_id for i in instruments if i.commodity_group == CommodityGroup.ENERGY
    }
    inventory = [e for e in events if e.event == EventType.INVENTORY_REPORT]
    other_wednesdays = [
        t
        for t in range(axis.n_burn, axis.n_days)
        if axis.dates[t].weekday() == 2 and t != holiday_day
    ]
    assert other_wednesdays

    for t in other_wednesdays:
        on_day = {e.instrument_id for e in inventory if e.day == t}
        assert on_day == energy_ids


def test_crop_report_fires_on_wasde_dates_for_agriculture_commodities_only(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, instruments, events = _real_event_days(config, result)

    agriculture_ids = {
        i.instrument_id for i in instruments if i.commodity_group == CommodityGroup.AGRICULTURE
    }
    assert len(agriculture_ids) == 7

    crop = [e for e in events if e.event == EventType.CROP_REPORT]
    assert len(crop) == len(WASDE_DATES) * len(agriculture_ids)
    assert {e.instrument_id for e in crop} == agriculture_ids


def test_macro_print_fires_on_nfp_dates_with_no_instrument(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    _, _, _, events = _real_event_days(config, result)

    macro = [e for e in events if e.event == EventType.MACRO_PRINT]
    assert len(macro) == len(NFP_DATES)
    assert all(e.instrument_id is None for e in macro)


_AXIS_3D = SimAxis(dates=(date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)), n_burn=0)


def _sd_ratio_pair(move: float, k: float) -> tuple[float, float]:
    """Two daily changes whose sample sd (ddof=1) is `abs(move) / k`, with the
    first change equal to `move`.
    """
    return move, move * (1 - np.sqrt(2) / k)


def test_unsupported_event_type_raises() -> None:
    with pytest.raises(ValueError, match="rating_downgrade"):
        surprise_rows(
            [RealEvent(instrument_id="CR-X", event=EventType.RATING_DOWNGRADE, day=1)],
            ProcessOutput(),
            spy_log_return=np.zeros(3),
            y10_bp=np.zeros(3),
            axis=_AXIS_3D,
            seed="S",
        )


def test_earnings_move_of_exactly_one_sd_scores_one_third() -> None:
    d0, d1 = _sd_ratio_pair(1.0, 1.0)
    prices = 100.0 * np.exp(np.array([0.0, d0, d0 + d1]))
    rows = surprise_rows(
        [RealEvent(instrument_id="EQ-TEST", event=EventType.EARNINGS, day=1)],
        ProcessOutput(prices={"EQ-TEST": prices}),
        spy_log_return=np.zeros(3),
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
        betas={"EQ-TEST": 1.0},
    )
    assert rows[0].surprise == pytest.approx(1 / SURPRISE_SD_SCALE)
    assert rows[0].affected == "equities"


def test_earnings_abnormal_return_subtracts_beta_times_spy_return() -> None:
    """A beta-scaled co-movement with SPY baked into the equity's own return
    is subtracted out, leaving the same surprise as the SPY-flat case.
    """
    beta = 1.5
    spy_log_return = np.array([0.0, 0.03, -0.02])
    spy_returns = spy_log_return[1:]
    d0, d1 = _sd_ratio_pair(1.0, 1.0)
    equity_returns = beta * spy_returns + np.array([d0, d1])
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(equity_returns)]))
    rows = surprise_rows(
        [RealEvent(instrument_id="EQ-TEST", event=EventType.EARNINGS, day=1)],
        ProcessOutput(prices={"EQ-TEST": prices}),
        spy_log_return=spy_log_return,
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
        betas={"EQ-TEST": beta},
    )
    assert rows[0].surprise == pytest.approx(1 / SURPRISE_SD_SCALE)


def test_cb_meeting_move_of_exactly_one_sd_scores_one_third() -> None:
    d0, d1 = _sd_ratio_pair(-1.0, 1.0)  # move = -d0 = 1.0, a positive rate cut is good
    y10_bp = np.array([100.0, 100.0 + d0, 100.0 + d0 + d1])
    rows = surprise_rows(
        [RealEvent(instrument_id="RT-USD", event=EventType.CB_MEETING, day=1)],
        ProcessOutput(),
        spy_log_return=np.zeros(3),
        y10_bp=y10_bp,
        axis=_AXIS_3D,
        seed="S",
    )
    assert rows[0].surprise == pytest.approx(1 / SURPRISE_SD_SCALE)
    assert rows[0].affected == "rates"


def test_inventory_report_move_of_exactly_one_sd_scores_one_third() -> None:
    d0, d1 = _sd_ratio_pair(1.0, 1.0)
    prices = 100.0 * np.exp(np.array([0.0, d0, d0 + d1]))
    rows = surprise_rows(
        [RealEvent(instrument_id="CM-CRD", event=EventType.INVENTORY_REPORT, day=1)],
        ProcessOutput(prices={"CM-CRD": prices}),
        spy_log_return=np.zeros(3),
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
    )
    assert rows[0].surprise == pytest.approx(1 / SURPRISE_SD_SCALE)
    assert rows[0].affected == "commodities"


def test_crop_report_move_of_exactly_minus_one_sd_scores_minus_one_third() -> None:
    d0, d1 = _sd_ratio_pair(-1.0, 1.0)
    prices = 100.0 * np.exp(np.array([0.0, d0, d0 + d1]))
    rows = surprise_rows(
        [RealEvent(instrument_id="CM-WHT", event=EventType.CROP_REPORT, day=1)],
        ProcessOutput(prices={"CM-WHT": prices}),
        spy_log_return=np.zeros(3),
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
    )
    assert rows[0].surprise == pytest.approx(-1 / SURPRISE_SD_SCALE)
    assert rows[0].affected == "commodities"


def test_macro_print_move_of_exactly_one_sd_scores_one_third() -> None:
    d0, d1 = _sd_ratio_pair(1.0, 1.0)
    spy_log_return = np.array([0.0, d0, d1])
    rows = surprise_rows(
        [RealEvent(instrument_id=None, event=EventType.MACRO_PRINT, day=1)],
        ProcessOutput(),
        spy_log_return=spy_log_return,
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
    )
    assert rows[0].instrument_id is None
    assert rows[0].affected == "all"
    assert rows[0].surprise == pytest.approx(1 / SURPRISE_SD_SCALE)


def test_a_four_sd_move_clips_to_one_and_a_minus_four_sd_move_clips_to_minus_one() -> None:
    d0, d1 = _sd_ratio_pair(4.0, 4.0)
    spy_log_return = np.array([0.0, d0, d1])
    positive = surprise_rows(
        [RealEvent(instrument_id=None, event=EventType.MACRO_PRINT, day=1)],
        ProcessOutput(),
        spy_log_return=spy_log_return,
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
    )
    assert positive[0].surprise == 1.0

    d0, d1 = _sd_ratio_pair(-4.0, 4.0)
    spy_log_return = np.array([0.0, d0, d1])
    negative = surprise_rows(
        [RealEvent(instrument_id=None, event=EventType.MACRO_PRINT, day=1)],
        ProcessOutput(),
        spy_log_return=spy_log_return,
        y10_bp=np.zeros(3),
        axis=_AXIS_3D,
        seed="S",
    )
    assert negative[0].surprise == -1.0


def test_flat_series_raises_naming_the_seed() -> None:
    with pytest.raises(StageIOError, match=r"real seed 'S': 'SPY' has zero daily sd"):
        surprise_rows(
            [RealEvent(instrument_id=None, event=EventType.MACRO_PRINT, day=1)],
            ProcessOutput(),
            spy_log_return=np.zeros(3),
            y10_bp=np.zeros(3),
            axis=_AXIS_3D,
            seed="S",
        )


def test_drawn_counts_counts_rows_per_type_with_zeros_for_rating_types() -> None:
    rows = [
        CalendarEvent(
            seed="S",
            date=date(2020, 1, 1),
            instrument_id="X",
            event=EventType.EARNINGS,
            surprise=0.1,
            affected="equities",
        ),
        CalendarEvent(
            seed="S",
            date=date(2020, 1, 1),
            instrument_id="X",
            event=EventType.EARNINGS,
            surprise=0.2,
            affected="equities",
        ),
        CalendarEvent(
            seed="S",
            date=date(2020, 1, 1),
            instrument_id=None,
            event=EventType.MACRO_PRINT,
            surprise=0.1,
            affected="all",
        ),
    ]

    counts = drawn_counts(rows)
    assert counts[EventType.EARNINGS] == 2
    assert counts[EventType.MACRO_PRINT] == 1
    assert counts[EventType.CB_MEETING] == 0
    assert counts[EventType.INVENTORY_REPORT] == 0
    assert counts[EventType.CROP_REPORT] == 0
    assert counts[EventType.RATING_DOWNGRADE] == 0
    assert counts[EventType.RATING_UPGRADE] == 0
