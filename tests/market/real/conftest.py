"""Fixtures for real-market tests: a deterministic fake raw cache and opener,
used instead of the network by every test in this package.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pm_traitbench.config import Config, real_window_end, referenced_seeds
from pm_traitbench.enums import Family
from pm_traitbench.market.real.fetch import fetch_all, fetch_range, fred_url, nasdaq_url, yahoo_url
from pm_traitbench.market.real.sources import REAL_INSTRUMENTS, fred_series, yahoo_tickers


def _fake_opener(files: dict[str, bytes]) -> Callable[[str], bytes]:
    """An opener returning canned bytes for known URLs, failing loudly on any other."""

    def _opener(url: str) -> bytes:
        try:
            return files[url]
        except KeyError:
            raise AssertionError(f"unexpected fetch: {url}") from None

    return _opener


@pytest.fixture
def fake_opener() -> Callable[[dict[str, bytes]], Callable[[str], bytes]]:
    """A factory building an opener that serves canned bytes for known URLs."""
    return _fake_opener


def _weekdays(start: date, end: date) -> list[date]:
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _quarter_bounds(day: date) -> tuple[date, date]:
    q_start_month = 3 * ((day.month - 1) // 3) + 1
    start = date(day.year, q_start_month, 1)
    if q_start_month == 10:
        end = date(day.year, 12, 31)
    else:
        end = date(day.year, q_start_month + 3, 1) - timedelta(days=1)
    return start, end


def _quarters_overlapping(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar quarters overlapping [start, end], each clipped to that range."""
    quarters = []
    q_start, q_end = _quarter_bounds(start)
    while q_start <= end:
        quarters.append((max(q_start, start), min(q_end, end)))
        q_start, q_end = _quarter_bounds(q_end + timedelta(days=1))
    return quarters


def _fred_csv(series: str, days: Sequence[date], values: dict[date, float]) -> bytes:
    """A FRED-shaped CSV: every day in `days` gets a row, missing days get an
    empty value, matching FRED's real holiday convention (e.g. '2018-07-04,').
    """
    lines = [f"observation_date,{series}"]
    for day in days:
        value = values.get(day)
        value_str = f"{value:.4f}" if value is not None else ""
        lines.append(f"{day.isoformat()},{value_str}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _yahoo_json(values: dict[date, float]) -> bytes:
    days = sorted(values)
    timestamps = [
        int(datetime(d.year, d.month, d.day, 13, 30, tzinfo=UTC).timestamp()) for d in days
    ]
    adjclose = [values[d] for d in days]
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {},
                    "timestamp": timestamps,
                    "indicators": {"adjclose": [{"adjclose": adjclose}], "quote": [{}]},
                }
            ],
            "error": None,
        }
    }
    return json.dumps(payload).encode("utf-8")


def _nasdaq_json(symbols: list[str]) -> bytes:
    payload = {
        "data": {"asOf": None, "headers": {}, "rows": [{"symbol": s} for s in symbols] or None},
        "message": None,
        "status": {"rCode": 200},
    }
    return json.dumps(payload).encode("utf-8")


@dataclass(frozen=True)
class FakeCache:
    """A built fake raw cache: its data dir and the shared holiday-gap weekday."""

    data_dir: Path
    holiday: date


@pytest.fixture
def fake_cache(tmp_path: Path) -> Callable[[Config], FakeCache]:
    """Build a deterministic fake raw cache for `config`'s real seeds.

    Covers SPY and every registry series over the fetch range. FRED and Yahoo
    share one holiday-gap weekday: FRED keeps the row with an empty value
    (its real convention), Yahoo omits the row entirely. Yahoo timestamps sit
    at 13:30 UTC. The Nasdaq calendar has every equity ticker reporting once
    per calendar quarter, with the first equity ticker's report in the
    holiday's quarter moved onto the holiday weekday itself (replacing, not
    duplicating, that ticker's report for that quarter) rather than any
    weekend date, since a real cache never holds a weekend file.
    """

    def _build(config: Config) -> FakeCache:
        data_dir = tmp_path / "data"
        start, end = fetch_range(config)
        weekdays = _weekdays(start, end)
        holiday = weekdays[len(weekdays) // 2]
        series_days = [day for day in weekdays if day != holiday]

        rng = np.random.default_rng(0)
        files: dict[str, bytes] = {}

        for series in fred_series():
            walk = 100.0 + np.cumsum(rng.normal(0, 1, size=len(series_days)))
            values = dict(zip(series_days, walk, strict=True))
            files[fred_url(series, start, end)] = _fred_csv(series, weekdays, values)

        for ticker in yahoo_tickers():
            walk = 100.0 + np.cumsum(rng.normal(0, 1, size=len(series_days)))
            values = dict(zip(series_days, walk, strict=True))
            files[yahoo_url(ticker, start, end)] = _yahoo_json(values)

        used = set(referenced_seeds(config))
        real_seeds = {name: spec for name, spec in config.market.real.seeds.items() if name in used}
        first_seed = next(iter(real_seeds.values()))
        window_end = real_window_end(first_seed, config.calendar.n_weeks)
        nasdaq_days = _weekdays(first_seed.window_start, window_end)

        equities = [inst.series for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES]
        quarters = _quarters_overlapping(first_seed.window_start, window_end)
        holiday_quarter_index = next(
            (i for i, (q_start, q_end) in enumerate(quarters) if q_start <= holiday <= q_end),
            None,
        )
        assert holiday_quarter_index is not None, "holiday falls outside the Nasdaq window"

        symbols_by_date: dict[date, list[str]] = {}
        target_ticker = equities[0]
        for ticker in equities:
            for qi, (q_start, q_end) in enumerate(quarters):
                if ticker == target_ticker and qi == holiday_quarter_index:
                    report_day = holiday
                else:
                    candidates = _weekdays(q_start, q_end)
                    report_day = candidates[int(rng.integers(0, len(candidates)))]
                symbols_by_date.setdefault(report_day, []).append(ticker)

        for day in nasdaq_days:
            files[nasdaq_url(day)] = _nasdaq_json(symbols_by_date.get(day, []))

        fetch_all(config, data_dir, opener=_fake_opener(files), sleeper=lambda _: None)
        return FakeCache(data_dir=data_dir, holiday=holiday)

    return _build
