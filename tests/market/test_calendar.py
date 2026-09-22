"""Tests for the event calendar: sampled events, generated rows and jump sizing."""

import functools

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import EventType, Family, InstrumentKind
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.calendar import (
    EVENT_TARGETS,
    build_jumps,
    event_day_indices,
    generated_rows,
    sample_events,
    third_friday,
)
from pm_traitbench.market.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import CalendarEvent, Instrument

_TWO_SIDED_EVENTS = (
    EventType.EARNINGS,
    EventType.CB_MEETING,
    EventType.INVENTORY_REPORT,
    EventType.CROP_REPORT,
    EventType.MACRO_PRINT,
)

# Root seed 7 keeps every sampled type's pooled count within 10% of per_year
# over the long horizon below; date draws are seed-independent, so the root
# alone determines the counts.
_LONG_ROOT = 7


def _long_config() -> Config:
    return Config.model_validate(
        {"calendar": {"n_weeks": 520}, "market": {"boundary_weeks": (100, 300)}}
    )


def _rng_for(root: int):
    return functools.partial(stream, root, "market")


def _build(config: Config, root: int):
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instruments = build_universe(config, stream(root, "market", "universe"))
    return axis, instruments


def _long_sample():
    config = _long_config()
    axis, instruments = _build(config, _LONG_ROOT)
    result = sample_events(instruments, axis, config, _rng_for(_LONG_ROOT), "A")
    return config, axis, instruments, result


def test_per_type_mean_counts_within_10_percent_of_per_year() -> None:
    config, axis, instruments, result = _long_sample()
    years = (axis.n_days - axis.n_burn) / 260

    for event, spec in config.market.events.items():
        if event == EventType.MACRO_PRINT:
            n_targets = 1
        else:
            n_targets = sum(1 for i in instruments if EVENT_TARGETS[event](i))
        expected = spec.per_year * n_targets * years
        actual = result.drawn[event]
        assert abs(actual - expected) / expected <= 0.10, (event, expected, actual)


def test_earnings_exact_count_per_equity_with_unique_dates() -> None:
    config, axis, instruments, result = _long_sample()
    years = (axis.n_days - axis.n_burn) / 260
    expected_count = round(4 * years)

    equities = [i for i in instruments if i.family == Family.EQUITIES]
    assert equities
    for equity in equities:
        dates = [
            row.date
            for row in result.rows
            if row.event == EventType.EARNINGS and row.instrument_id == equity.instrument_id
        ]
        assert len(dates) == expected_count
        assert len(set(dates)) == len(dates)


def test_rating_sign_matches_label_and_surprises_are_bounded() -> None:
    _, _, _, result = _long_sample()
    for row in result.rows:
        if row.surprise is not None:
            assert -1 <= row.surprise <= 1
        if row.event == EventType.RATING_DOWNGRADE:
            assert row.surprise < 0
        elif row.event == EventType.RATING_UPGRADE:
            assert row.surprise > 0


def test_two_sided_event_types_produce_both_signs() -> None:
    _, _, _, result = _long_sample()
    for event in _TWO_SIDED_EVENTS:
        surprises = [row.surprise for row in result.rows if row.event == event]
        assert any(s > 0 for s in surprises)
        assert any(s < 0 for s in surprises)


def test_no_sampled_row_dated_before_calendar_start() -> None:
    config, _, _, result = _long_sample()
    assert all(row.date >= config.calendar.start for row in result.rows)


def test_drawn_counts_equal_row_counts_per_type() -> None:
    config, _, _, result = _long_sample()
    for event in config.market.events:
        assert result.drawn[event] == sum(1 for row in result.rows if row.event == event)


def test_same_root_different_market_seeds_share_dates_but_not_surprises() -> None:
    config = Config()
    axis, instruments = _build(config, 1)
    rng_for = _rng_for(1)
    result_a = sample_events(instruments, axis, config, rng_for, "A")
    result_b = sample_events(instruments, axis, config, rng_for, "B")

    keys_a = {(row.date, row.instrument_id, row.event) for row in result_a.rows}
    keys_b = {(row.date, row.instrument_id, row.event) for row in result_b.rows}
    assert keys_a == keys_b

    surprises_a = [row.surprise for row in result_a.rows]
    surprises_b = [row.surprise for row in result_b.rows]
    assert surprises_a != surprises_b


def test_jitter_clipped_duplicate_grid_day_is_dropped_and_drawn_reduced() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    instrument = Instrument(
        instrument_id="EQ-0001",
        family=Family.EQUITIES,
        kind=InstrumentKind.EQUITY,
        name="Equity 0001",
        currency="USD",
        sector="sector_01",
        rating_band=None,
        commodity_group=None,
        duration_years=None,
        beta=1.0,
        expiry_rule=None,
    )

    class _ScriptedRng:
        """Stub returning fixed values, to force a jitter-clipped collision."""

        def __init__(self, uniform: float, integers: list[int]) -> None:
            self._uniform = uniform
            self._integers = integers

        def uniform(self, low: float, high: float) -> float:
            return self._uniform

        def integers(self, low: int, high: int, size: int) -> np.ndarray:
            return np.array(self._integers)

    def rng_for(*keys: str):
        # Force the second of four earnings grid points far negative, so it
        # clips to day 0 and collides with the first point.
        if keys == ("events", "EQ-0001", "earnings"):
            return _ScriptedRng(uniform=0.0, integers=[0, -100, 0, 0])
        return stream(1, "market", *keys)

    result = sample_events([instrument], axis, config, rng_for, "A")
    earnings_rows = [row for row in result.rows if row.event == EventType.EARNINGS]

    assert len(earnings_rows) == 3
    assert len({row.date for row in earnings_rows}) == 3
    assert result.drawn[EventType.EARNINGS] == 3


def test_third_friday_is_a_friday_in_the_third_week() -> None:
    friday = third_friday(2026, 3)
    assert friday.weekday() == 4
    assert 15 <= friday.day <= 21


def test_generated_rows_contract_expiry_and_positioning_report() -> None:
    config = Config()
    axis, instruments = _build(config, 1)
    rows = generated_rows(instruments, axis, "A")

    commodities = [i for i in instruments if i.family == Family.COMMODITIES]
    horizon_dates = axis.dates[axis.horizon]
    horizon_fridays = [day for day in horizon_dates if day.weekday() == 4]

    expiry_rows = [row for row in rows if row.event == EventType.CONTRACT_EXPIRY]
    positioning_rows = [row for row in rows if row.event == EventType.POSITIONING_REPORT]

    expiry_dates = sorted({row.date for row in expiry_rows})
    for expiry_date in expiry_dates:
        assert expiry_date.weekday() == 4
        assert 15 <= expiry_date.day <= 21
        instrument_ids = {row.instrument_id for row in expiry_rows if row.date == expiry_date}
        assert instrument_ids == {c.instrument_id for c in commodities}

    assert len(expiry_rows) == len(expiry_dates) * len(commodities)
    assert {row.date for row in positioning_rows} == set(horizon_fridays)
    assert len(positioning_rows) == len(horizon_fridays)
    assert all(row.instrument_id is None and row.affected == "all" for row in positioning_rows)
    assert all(row.surprise is None for row in rows)
    assert all(row.date >= config.calendar.start for row in rows)


def test_build_jumps_hand_built_two_rows() -> None:
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
    ]

    jumps = build_jumps(rows, axis, config)
    earnings_jump_size = config.market.events[EventType.EARNINGS].jump_size
    macro_jump_size = config.market.events[EventType.MACRO_PRINT].jump_size

    expected_eq = np.zeros(axis.n_days)
    expected_eq[axis.n_burn] = 0.5 * earnings_jump_size
    assert np.allclose(jumps.for_instrument("EQ-0001"), expected_eq)
    assert np.allclose(jumps.for_instrument("EQ-9999"), np.zeros(axis.n_days))

    expected_macro = np.zeros(axis.n_days)
    expected_macro[axis.n_burn + 1] = -0.3 * macro_jump_size
    assert np.allclose(jumps.macro, expected_macro)


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
