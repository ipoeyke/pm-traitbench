"""Tests for the simulation axis: burn-in and horizon date sequencing."""

from datetime import timedelta

import pytest

from pm_traitbench.config import Config
from pm_traitbench.market.axis import build_axis


def test_default_axis_has_burn_in_plus_horizon_dates() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    assert axis.n_burn == 60
    assert axis.n_days == 60 + 260
    assert len(axis.dates) == 320


def test_axis_dates_are_all_weekdays_and_strictly_increasing() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    assert all(day.weekday() < 5 for day in axis.dates)
    for prev, curr in zip(axis.dates, axis.dates[1:], strict=False):
        assert curr > prev


def test_axis_has_no_gap_except_weekends() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    for prev, curr in zip(axis.dates, axis.dates[1:], strict=False):
        gap = (curr - prev).days
        assert gap == (3 if prev.weekday() == 4 else 1)


def test_horizon_start_matches_calendar_start() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    assert axis.dates[axis.n_burn] == config.calendar.start


def test_last_burn_in_date_is_friday_before_start() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    friday_before_start = config.calendar.start - timedelta(days=3)
    assert friday_before_start.weekday() == 4
    assert axis.dates[axis.n_burn - 1] == friday_before_start


def test_zero_burn_in_gives_horizon_only() -> None:
    config = Config()
    axis = build_axis(config.timeline(), 0)
    assert axis.n_burn == 0
    assert axis.n_days == 260
    assert axis.dates[0] == config.calendar.start


def test_horizon_slice_selects_only_horizon_dates() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    horizon_dates = axis.dates[axis.horizon]
    assert len(horizon_dates) == 260
    assert horizon_dates[0] == config.calendar.start


def test_index_returns_position_for_axis_date() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    assert axis.index(config.calendar.start) == axis.n_burn
    assert axis.index(axis.dates[0]) == 0
    assert axis.index(axis.dates[-1]) == axis.n_days - 1


def test_index_of_weekend_date_raises_value_error() -> None:
    config = Config()
    axis = build_axis(config.timeline(), config.market.burn_in_days)
    weekend_date = config.calendar.start - timedelta(days=2)
    assert weekend_date.weekday() == 5
    with pytest.raises(ValueError):
        axis.index(weekend_date)
