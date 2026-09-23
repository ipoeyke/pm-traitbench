"""Fixtures for real-market tests: a deterministic fake raw cache and opener,
used instead of the network by every test in this package.
"""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pm_traitbench.config import Config
from pm_traitbench.enums import Family
from pm_traitbench.market.real.fetch import (
    Manifest,
    ManifestEntry,
    cache_dir,
    fetch_all,
    fetch_range,
    fred_url,
    nasdaq_url,
    yahoo_url,
)
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


def _fred_csv(series: str, values: dict[date, float]) -> bytes:
    lines = [f"observation_date,{series}"]
    lines.extend(f"{day.isoformat()},{values[day]:.4f}" for day in sorted(values))
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


@pytest.fixture
def fake_cache(tmp_path: Path) -> Callable[[Config], Path]:
    """Build a deterministic fake raw cache for `config`'s real seeds and return its data dir.

    Covers SPY and every registry series over the fetch range, with one shared
    holiday gap, Yahoo timestamps at 13:30 UTC, and a Nasdaq calendar where
    every equity ticker reports once per calendar quarter (plus one forced
    Saturday report, to exercise the weekend case).
    """

    def _build(config: Config) -> Path:
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
            files[fred_url(series, start, end)] = _fred_csv(series, values)

        for ticker in yahoo_tickers():
            walk = 100.0 + np.cumsum(rng.normal(0, 1, size=len(series_days)))
            values = dict(zip(series_days, walk, strict=True))
            files[yahoo_url(ticker, start, end)] = _yahoo_json(values)

        real_seeds = config.market.real.seeds
        first_seed = next(iter(real_seeds.values()))
        window_end = first_seed.window_start + timedelta(weeks=config.calendar.n_weeks)
        nasdaq_days = _weekdays(first_seed.window_start, window_end)

        equities = [inst.series for inst in REAL_INSTRUMENTS if inst.family == Family.EQUITIES]
        quarters = _quarters_overlapping(first_seed.window_start, window_end)
        symbols_by_date: dict[date, list[str]] = {}
        for ticker in equities:
            for q_start, q_end in quarters:
                candidates = _weekdays(q_start, q_end)
                report_day = candidates[int(rng.integers(0, len(candidates)))]
                symbols_by_date.setdefault(report_day, []).append(ticker)

        # Force exactly one report onto a Saturday, to exercise the weekend case.
        saturday = first_seed.window_start
        while saturday.weekday() != 5:
            saturday += timedelta(days=1)
        symbols_by_date.setdefault(saturday, []).append(equities[0])

        for day in nasdaq_days:
            files[nasdaq_url(day)] = _nasdaq_json(symbols_by_date.get(day, []))

        manifest = fetch_all(config, data_dir, opener=_fake_opener(files), sleeper=lambda _: None)

        # The Saturday file falls outside fetch_all's own weekday-only day list,
        # so add it directly through fetch.py's own manifest entry and writer.
        root = cache_dir(data_dir)
        body = _nasdaq_json(symbols_by_date[saturday])
        rel_path = f"nasdaq/{saturday.isoformat()}.json"
        (root / rel_path).write_bytes(body)
        entry = ManifestEntry(
            path=rel_path,
            url=nasdaq_url(saturday),
            retrieved_at=datetime.now(UTC).isoformat(),
            sha256=hashlib.sha256(body).hexdigest(),
            size=len(body),
        )
        manifest = Manifest(window=manifest.window, entries=manifest.entries + (entry,))
        (root / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

        return data_dir

    return _build
