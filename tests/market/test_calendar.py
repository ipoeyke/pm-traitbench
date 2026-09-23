"""Tests for shared calendar helpers: contract expiries, positioning reports
and event-day lookup.
"""

from pm_traitbench.config import Config
from pm_traitbench.enums import EventType, Family
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import event_day_indices, generated_rows, third_friday
from pm_traitbench.tables.schema import CalendarEvent


def test_third_friday_is_a_friday_in_the_third_week() -> None:
    friday = third_friday(2026, 3)
    assert friday.weekday() == 4
    assert 15 <= friday.day <= 21


def test_generated_rows_contract_expiry_and_positioning_report(build_axis_and_universe) -> None:
    config = Config()
    axis, instruments = build_axis_and_universe(config, 1)
    rows = generated_rows(instruments, axis, "A")

    commodities = [i for i in instruments if i.family == Family.COMMODITIES]
    horizon_dates = axis.dates[axis.horizon]
    horizon_fridays = [day for day in horizon_dates if day.weekday() == 4]
    horizon_set = set(horizon_dates)
    months = {(day.year, day.month) for day in horizon_dates}
    expected_expiries = {third_friday(year, month) for year, month in months} & horizon_set

    expiry_rows = [row for row in rows if row.event == EventType.CONTRACT_EXPIRY]
    positioning_rows = [row for row in rows if row.event == EventType.POSITIONING_REPORT]

    for commodity in commodities:
        commodity_rows = [
            row for row in expiry_rows if row.instrument_id == commodity.instrument_id
        ]
        assert {row.date for row in commodity_rows} == expected_expiries
        assert all(row.affected == "commodities" for row in commodity_rows)

    assert len(expiry_rows) == len(expected_expiries) * len(commodities)
    assert {row.date for row in positioning_rows} == set(horizon_fridays)
    assert len(positioning_rows) == len(horizon_fridays)
    assert all(row.instrument_id is None and row.affected == "all" for row in positioning_rows)
    assert all(row.surprise is None for row in rows)
    assert all(row.date >= config.calendar.start for row in rows)


def test_event_day_indices_excludes_macro_and_generated_rows() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    d1 = axis.dates[axis.n_burn]
    d2 = axis.dates[axis.n_burn + 1]

    rows = [
        CalendarEvent(
            seed="A",
            date=d1,
            instrument_id="EQ-0001",
            event=EventType.EARNINGS,
            surprise=0.5,
            affected="equities",
        ),
        CalendarEvent(
            seed="A",
            date=d2,
            instrument_id=None,
            event=EventType.MACRO_PRINT,
            surprise=-0.3,
            affected="all",
        ),
        CalendarEvent(
            seed="A",
            date=d2,
            instrument_id="CM-CRD",
            event=EventType.CONTRACT_EXPIRY,
            surprise=None,
            affected="commodities",
        ),
    ]

    indices = event_day_indices(rows, axis)
    assert indices == {"EQ-0001": {axis.n_burn}}
