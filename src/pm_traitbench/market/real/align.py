"""Forward-fill a raw real-market series onto a set of axis-mapped dates.

Shared by the real universe (equity beta inputs) and the real market build
(price and curve series), so both fill gaps and enforce the same limit the
same way.
"""

import bisect
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from pm_traitbench.config import REAL_FILL_LIMIT
from pm_traitbench.errors import StageIOError


def _weekdays_between(start: date, end: date) -> int:
    """Count of weekdays strictly after `start` through `end`, inclusive."""
    count = 0
    day = start
    while day < end:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count += 1
    return count


def aligned_series(
    values: dict[date, float], dates: Sequence[date], *, name: str
) -> tuple[np.ndarray, int]:
    """Forward-fill `values` onto `dates`, in order.

    Day 0 may fill from any earlier date in `values` (the fetch buffer covers
    a gap right at the window's start); every later gap fills from the prior
    aligned day. The weekdays between that earlier date and day 0 count as
    part of day 0's own fill run. Returns the aligned array and the longest
    run of filled (non-exact) days. Raises `StageIOError` naming `name` when
    day 0 has no value on or before it within the buffer, or the longest run
    exceeds `REAL_FILL_LIMIT`.
    """
    dates = list(dates)
    if not dates:
        return np.array([], dtype=float), 0

    keys = sorted(values)
    start_idx = bisect.bisect_right(keys, dates[0]) - 1
    if start_idx < 0:
        raise StageIOError(
            f"{name}: no value on or before day 0 ({dates[0]}) within the fetch buffer"
        )
    last_known = keys[start_idx]
    last_value = values[last_known]

    result = np.empty(len(dates), dtype=float)
    longest_run = 0
    current_run = _weekdays_between(last_known, dates[0]) - 1
    for i, day in enumerate(dates):
        if day in values:
            last_value = values[day]
            current_run = 0
        else:
            current_run += 1
            longest_run = max(longest_run, current_run)
        result[i] = last_value

    if longest_run > REAL_FILL_LIMIT:
        raise StageIOError(
            f"{name}: longest fill run of {longest_run} days exceeds the limit of {REAL_FILL_LIMIT}"
        )
    return result, longest_run
