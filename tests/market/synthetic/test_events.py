"""Tests for sampled calendar events and the price jumps they drive."""

import functools

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import EventType, Family, InstrumentKind
from pm_traitbench.market.axis import build_axis
from pm_traitbench.market.synthetic.events import EVENT_TARGETS, build_jumps, sample_events
from pm_traitbench.market.synthetic.universe import build_universe
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import CalendarEvent, Instrument

_TWO_SIDED_EVENTS = (
    EventType.EARNINGS,
    EventType.CB_MEETING,
    EventType.INVENTORY_REPORT,
    EventType.CROP_REPORT,
    EventType.MACRO_PRINT,
)

# An ordinary root, not tuned to pass; verified locally for roots 0-19.
_LONG_ROOT = 0


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


def _row_count(rows: list[CalendarEvent], event: EventType, target_key: str) -> int:
    if target_key == "macro":
        return sum(1 for row in rows if row.event == event and row.instrument_id is None)
    return sum(1 for row in rows if row.event == event and row.instrument_id == target_key)


def test_grid_counts_are_exact_and_poisson_counts_are_within_tolerance() -> None:
    config, axis, instruments, result = _long_sample()
    years = (axis.n_days - axis.n_burn) / 260

    for event, spec in config.market.events.items():
        if event == EventType.MACRO_PRINT:
            targets = ["macro"]
        else:
            targets = [i.instrument_id for i in instruments if EVENT_TARGETS[event](i)]

        if spec.placement == "grid":
            expected = round(spec.per_year * years)
            for target in targets:
                actual = _row_count(result.rows, event, target)
                assert actual == expected
        else:
            # A pooled Poisson count's spread shrinks slowly with its mean, so
            # a flat percentage band fails most roots at this per-issuer rate.
            mu = spec.per_year * len(targets) * years
            actual = result.drawn[event]
            assert abs(actual - mu) <= 4 * mu**0.5


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


def test_build_jumps_arrays_are_frozen() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    d1 = axis.dates[axis.n_burn]

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
            date=d1,
            instrument_id=None,
            event=EventType.MACRO_PRINT,
            surprise=-0.3,
            affected="all",
        ),
    ]

    jumps = build_jumps(rows, axis, config)

    with pytest.raises(ValueError):
        jumps.for_instrument("EQ-0001")[0] = 1.0
    with pytest.raises(ValueError):
        jumps.macro[0] = 1.0
