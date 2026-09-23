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
from pm_traitbench.market.real.sources import (
    REAL_INSTRUMENTS,
    REFERENCE_EQUITY,
    fred_series,
    yahoo_tickers,
)


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
    """A built fake raw cache: its data dir, the shared holiday-gap weekday, and
    the known beta each equity ticker was generated with.
    """

    data_dir: Path
    holiday: date
    betas: dict[str, float]


@pytest.fixture
def fake_cache(tmp_path: Path) -> Callable[[Config], FakeCache]:
    """Build a deterministic fake raw cache for `config`'s real seeds.

    Covers SPY and every registry series over the fetch range. FRED and Yahoo
    share one holiday-gap weekday: FRED keeps the row with an empty value
    (its real convention), Yahoo omits the row entirely. Yahoo timestamps sit
    at 13:30 UTC. Every equity is generated as `beta * spy_return + noise`
    with a known beta, so a beta regression against SPY can be checked
    against a known slope. The Nasdaq calendar covers every referenced real
    seed's own window and has every equity ticker reporting once per
    calendar quarter, with the first equity ticker's report in the
    holiday's quarter moved onto the holiday weekday itself (replacing, not
    duplicating, that ticker's report for that quarter) rather than any
    weekend date, since a real cache never holds a weekend file.
    """

    def _build(config: Config) -> FakeCache:
        data_dir = tmp_path / "data"
        start, end = fetch_range(config)
        weekdays = _weekdays(start, end)

        used = set(referenced_seeds(config))
        real_seeds = {name: spec for name, spec in config.market.real.seeds.items() if name in used}
        nasdaq_day_set: set[date] = set()
        quarters: list[tuple[date, date]] = []
        for spec in real_seeds.values():
            window_end = real_window_end(spec, config.calendar.n_weeks)
            nasdaq_day_set.update(_weekdays(spec.window_start, window_end))
            quarters.extend(_quarters_overlapping(spec.window_start, window_end))
        nasdaq_days = sorted(nasdaq_day_set)

        # Picked from the union of the seeds' own windows (not the wider
        # fetch range, which can include a gap between non-adjacent windows
        # that no seed's Nasdaq calendar covers).
        holiday = nasdaq_days[len(nasdaq_days) // 2]
        series_days = [day for day in weekdays if day != holiday]

        rng = np.random.default_rng(0)
        files: dict[str, bytes] = {}

        # Corporate yields (DAAA, DBAA) always sit above the 20Y Treasury they're
        # spread against; a large fixed offset keeps that true for a real seed's
        # whole window without changing the draw count for any other series.
        credit_spread_offset = {"DAAA": 1000.0, "DBAA": 1000.0}

        for series in fred_series():
            walk = 100.0 + np.cumsum(rng.normal(0, 1, size=len(series_days)))
            walk = walk + credit_spread_offset.get(series, 0.0)
            values = dict(zip(series_days, walk, strict=True))
            files[fred_url(series, start, end)] = _fred_csv(series, weekdays, values)

        n_returns = len(series_days) - 1
        spy_returns = rng.normal(0, 0.01, size=n_returns)
        spy_log_prices = np.log(100.0) + np.concatenate([[0.0], np.cumsum(spy_returns)])
        spy_values = dict(zip(series_days, np.exp(spy_log_prices), strict=True))
        files[yahoo_url(REFERENCE_EQUITY, start, end)] = _yahoo_json(spy_values)

        equities_by_ticker = {
            inst.series: inst for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES
        }
        betas: dict[str, float] = {}
        beta_draws = rng.uniform(0.5, 1.5, size=len(equities_by_ticker))
        for ticker, beta in zip(equities_by_ticker, beta_draws, strict=True):
            inst = equities_by_ticker[ticker]
            # Small relative to the SPY return so a beta regression over the
            # window recovers the known slope tightly enough to assert on.
            noise = rng.normal(0, 0.001, size=n_returns)
            log_returns = beta * spy_returns + noise
            log_prices = np.log(100.0) + np.concatenate([[0.0], np.cumsum(log_returns)])
            values = dict(zip(series_days, np.exp(log_prices), strict=True))
            files[yahoo_url(ticker, start, end)] = _yahoo_json(values)
            betas[inst.instrument_id] = float(beta)

        other_tickers = [
            ticker
            for ticker in yahoo_tickers()
            if ticker != REFERENCE_EQUITY and ticker not in equities_by_ticker
        ]
        for ticker in other_tickers:
            walk = 100.0 + np.cumsum(rng.normal(0, 1, size=len(series_days)))
            values = dict(zip(series_days, walk, strict=True))
            files[yahoo_url(ticker, start, end)] = _yahoo_json(values)

        equities = [inst.series for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES]
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
        return FakeCache(data_dir=data_dir, holiday=holiday, betas=betas)

    return _build
