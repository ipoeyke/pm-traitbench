"""Tests for the real-market event calendar: dates, targets and surprises."""

from datetime import date

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import CommodityGroup, EventType, Family
from pm_traitbench.market.axis import SimAxis, build_axis
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.real.events import (
    FOMC_DATES,
    NFP_DATES,
    WASDE_DATES,
    RealEvent,
    _roll_to_weekday,
    drawn_counts,
    real_event_days,
    surprise_rows,
)
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.universe import build_real_universe, real_axis_dates
from pm_traitbench.tables.schema import CalendarEvent

_WINDOW_START = date(2018, 6, 4)
_WINDOW_END = date(2019, 5, 31)


@pytest.mark.parametrize("dates", [FOMC_DATES, WASDE_DATES, NFP_DATES])
def test_fixed_date_lists_are_ascending_and_within_the_seed_window(dates: tuple[date, ...]) -> None:
    assert list(dates) == sorted(dates)
    assert len(set(dates)) == len(dates)
    for day in dates:
        assert _WINDOW_START <= day <= _WINDOW_END


def test_roll_to_weekday_moves_a_weekend_date_to_the_next_monday() -> None:
    assert _roll_to_weekday(date(2018, 6, 9)) == date(2018, 6, 11)  # Saturday
    assert _roll_to_weekday(date(2018, 6, 10)) == date(2018, 6, 11)  # Sunday


def test_roll_to_weekday_leaves_a_weekday_in_place() -> None:
    assert _roll_to_weekday(date(2018, 6, 6)) == date(2018, 6, 6)


def _real_event_days(config: Config, result):
    cache = RawCache.open(result.data_dir)
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    spec = config.market.real.seeds["R1"]
    instruments = build_real_universe(config, cache, axis)
    events = real_event_days(instruments, cache, spec, axis, config.calendar.start)
    return axis, spec, instruments, events


def test_no_events_fall_in_burn_in(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    axis, _, _, events = _real_event_days(config, result)

    assert events
    assert all(e.day >= axis.n_burn for e in events)


def test_earnings_rows_are_registry_tickers_and_the_holiday_report_stays(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    axis, spec, instruments, events = _real_event_days(config, result)

    earnings = [e for e in events if e.event == EventType.EARNINGS]
    assert earnings

    registry_ids = {i.instrument_id for i in instruments if i.family == Family.EQUITIES}
    for e in earnings:
        assert e.instrument_id in registry_ids

    real_dates = real_axis_dates(spec, axis, config.calendar.start)
    holiday_day = real_dates.index(result.holiday)
    assert any(e.day == holiday_day and e.instrument_id == "EQ-R001" for e in earnings)


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


def test_surprise_formulas_match_the_binding_examples() -> None:
    config = Config()
    axis = SimAxis(dates=(date(2020, 1, 1), date(2020, 1, 2)), n_burn=0)

    earnings_rows = surprise_rows(
        [RealEvent(instrument_id="EQ-TEST", event=EventType.EARNINGS, day=1)],
        ProcessOutput(prices={"EQ-TEST": np.array([100.0, 100.0 * np.exp(0.10)])}),
        spy_log_return=np.array([0.0, 0.01]),
        betas={"EQ-TEST": 1.0},
        y10_bp=np.zeros(2),
        axis=axis,
        seed="S",
        config=config,
    )
    assert earnings_rows[0].surprise == pytest.approx(1.0)
    assert earnings_rows[0].affected == "equities"

    cb_rows = surprise_rows(
        [RealEvent(instrument_id="RT-USD", event=EventType.CB_MEETING, day=1)],
        ProcessOutput(),
        spy_log_return=np.zeros(2),
        betas={},
        y10_bp=np.array([100.0, 104.0]),
        axis=axis,
        seed="S",
        config=config,
    )
    assert cb_rows[0].surprise == pytest.approx(-0.5)
    assert cb_rows[0].affected == "rates"


def test_surprise_is_clipped_to_the_unit_range() -> None:
    config = Config()
    axis = SimAxis(dates=(date(2020, 1, 1), date(2020, 1, 2)), n_burn=0)
    rows = surprise_rows(
        [RealEvent(instrument_id="CM-CRD", event=EventType.INVENTORY_REPORT, day=1)],
        ProcessOutput(prices={"CM-CRD": np.array([100.0, 100.0 * np.exp(1.0)])}),
        spy_log_return=np.zeros(2),
        betas={},
        y10_bp=np.zeros(2),
        axis=axis,
        seed="S",
        config=config,
    )
    assert rows[0].surprise == 1.0
    assert rows[0].affected == "commodities"


def test_macro_print_surprise_row_is_market_wide() -> None:
    config = Config()
    axis = SimAxis(dates=(date(2020, 1, 1), date(2020, 1, 2)), n_burn=0)
    r_spy = 0.005
    rows = surprise_rows(
        [RealEvent(instrument_id=None, event=EventType.MACRO_PRINT, day=1)],
        ProcessOutput(),
        spy_log_return=np.array([0.0, r_spy]),
        betas={},
        y10_bp=np.zeros(2),
        axis=axis,
        seed="S",
        config=config,
    )
    assert rows[0].instrument_id is None
    assert rows[0].affected == "all"
    daily_vol = config.market.families.equity.market_vol / np.sqrt(252)
    jump_size = config.market.events[EventType.MACRO_PRINT].jump_size
    expected = max(-1.0, min(1.0, (r_spy / daily_vol) / jump_size))
    assert rows[0].surprise == pytest.approx(expected)


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
