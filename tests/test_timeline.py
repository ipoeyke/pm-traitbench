from datetime import date

import pytest

from pm_traitbench.timeline import Timeline


@pytest.fixture
def timeline() -> Timeline:
    return Timeline(date(2026, 1, 5), 52)


def test_week_start_first_week(timeline: Timeline) -> None:
    assert timeline.week_start(1) == date(2026, 1, 5)


def test_week_start_mid_year(timeline: Timeline) -> None:
    assert timeline.week_start(22) == date(2026, 6, 1)


def test_week_start_zero_raises(timeline: Timeline) -> None:
    with pytest.raises(ValueError):
        timeline.week_start(0)


def test_week_start_past_last_week_raises(timeline: Timeline) -> None:
    with pytest.raises(ValueError):
        timeline.week_start(53)


def test_weekdays_in_single_week(timeline: Timeline) -> None:
    assert timeline.weekdays_in_weeks(1, 1) == [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
    ]


def test_weekdays_in_week_range(timeline: Timeline) -> None:
    days = timeline.weekdays_in_weeks(18, 30)
    assert len(days) == 65
    assert all(day.weekday() < 5 for day in days)
    assert days[0] == timeline.week_start(18)


def test_weekdays_in_weeks_first_after_last_raises(timeline: Timeline) -> None:
    with pytest.raises(ValueError):
        timeline.weekdays_in_weeks(5, 3)


def test_construct_with_non_monday_start_raises() -> None:
    with pytest.raises(ValueError):
        Timeline(date(2026, 1, 6), 52)
