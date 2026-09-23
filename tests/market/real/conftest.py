"""Fixtures for real-market tests: a deterministic fake raw cache and opener,
used instead of the network by every test in this package.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pm_traitbench.config import Config, referenced_seeds
from pm_traitbench.enums import Family
from pm_traitbench.market.real.fetch import (
    edgar_followup_url,
    edgar_url,
    fetch_all,
    fetch_range,
    fred_url,
    yahoo_url,
)
from pm_traitbench.market.real.sources import (
    REAL_INSTRUMENTS,
    REFERENCE_EQUITY,
    fred_series,
    yahoo_tickers,
)

_CREDIT_SERIES: frozenset[str] = frozenset({"DAAA", "DBAA"})

# Realistic starting yields (percent) so a curve tenor never nears the 0.0
# floor; steps are scaled to about 0.05 percentage points a day.
_TREASURY_START_PCT: dict[str, float] = {
    "DGS2": 2.5,
    "DGS5": 2.7,
    "DGS10": 2.9,
    "DGS20": 3.0,
    "DGS30": 3.1,
}
_TREASURY_STEP_PCT = 0.05


def _fake_opener(files: dict[str, bytes]) -> Callable[[str, Mapping[str, str]], bytes]:
    """An opener returning canned bytes for known URLs, failing loudly on any other."""

    def _opener(url: str, headers: Mapping[str, str]) -> bytes:
        try:
            return files[url]
        except KeyError:
            raise AssertionError(f"unexpected fetch: {url}") from None

    return _opener


@pytest.fixture
def fake_opener() -> Callable[[dict[str, bytes]], Callable[[str, Mapping[str, str]], bytes]]:
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


def _edgar_accepted(day: date) -> str:
    """A 20:30 UTC acceptance timestamp, EDGAR's own format."""
    return f"{day.isoformat()}T20:30:00.000Z"


def _edgar_lists(filings: Sequence[tuple[date, str, str]]) -> dict[str, list[str]]:
    return {
        "form": [form for _, form, _ in filings],
        "filingDate": [day.isoformat() for day, _, _ in filings],
        "acceptanceDateTime": [_edgar_accepted(day) for day, _, _ in filings],
        "items": [items for _, _, items in filings],
    }


def _edgar_main_json(recent: Sequence[tuple[date, str, str]], files: list[dict[str, str]]) -> bytes:
    payload = {"filings": {"recent": _edgar_lists(recent), "files": files}}
    return json.dumps(payload).encode("utf-8")


def _edgar_older_json(filings: Sequence[tuple[date, str, str]]) -> bytes:
    return json.dumps(_edgar_lists(filings)).encode("utf-8")


def _referenced_window_starts(config: Config) -> list[date]:
    used = set(referenced_seeds(config))
    return [spec.window_start for name, spec in config.market.real.seeds.items() if name in used]


@dataclass(frozen=True)
class FakeCache:
    """A built fake raw cache: its data dir, the shared holiday-gap weekday, and
    the known beta each equity ticker was generated with.
    """

    data_dir: Path
    holiday: date
    betas: dict[str, float]


@pytest.fixture
def fake_cache(tmp_path: Path) -> Callable[..., FakeCache]:
    """Build a deterministic fake raw cache for `config`'s real seeds.

    Covers SPY and every registry series over the fetch range. FRED and Yahoo
    share one holiday-gap weekday: FRED keeps the row with an empty value
    (its real convention), Yahoo omits the row entirely. Yahoo timestamps sit
    at 13:30 UTC. Every equity is generated as `beta * spy_return + noise`
    with a known beta, so a beta regression against SPY can be checked
    against a known slope. `holiday_weekday` picks which weekday (0=Monday)
    the shared gap falls on, defaulting to the range's middle weekday.
    """

    def _build(config: Config, *, holiday_weekday: int | None = None) -> FakeCache:
        data_dir = tmp_path / "data"
        start, end = fetch_range(config)
        weekdays = _weekdays(start, end)

        if holiday_weekday is None:
            holiday = weekdays[len(weekdays) // 2]
        else:
            candidates = [day for day in weekdays if day.weekday() == holiday_weekday]
            holiday = candidates[len(candidates) // 2]
        series_days = [day for day in weekdays if day != holiday]

        rng = np.random.default_rng(0)
        files: dict[str, bytes] = {}

        # One draw per series, in fred_series() order, so the draw count and
        # sequence match every other series exactly; only how DAAA/DBAA turn
        # their own draw into a walk differs.
        draws = {series: rng.normal(0, 1, size=len(series_days)) for series in fred_series()}
        walks = {
            series: (
                _TREASURY_START_PCT[series] + _TREASURY_STEP_PCT * np.cumsum(draw)
                if series in _TREASURY_START_PCT
                else 100.0 + np.cumsum(draw)
            )
            for series, draw in draws.items()
            if series not in _CREDIT_SERIES
        }
        # Corporate yields track the 20Y Treasury plus a positive, slowly
        # drifting spread built from their own draw: about 60bp and 180bp,
        # moving a few bp a day, never going negative like two independent
        # walks eventually would.
        walks["DAAA"] = walks["DGS20"] + 0.6 * np.exp(np.cumsum(0.03 * draws["DAAA"]))
        walks["DBAA"] = walks["DGS20"] + 1.8 * np.exp(np.cumsum(0.03 * draws["DBAA"]))

        for series in fred_series():
            values = dict(zip(series_days, walks[series], strict=True))
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

        # EDGAR: four quarterly 2.02 filings per seed window (comfortably
        # above the 3-report coverage floor) in the main "recent" file, plus
        # a 2.02 and a non-2.02 8-K before window_start in a separate older
        # file (exercising the filings.files merge path; excluded from
        # coverage since they fall in the burn-in, not the horizon).
        window_starts = _referenced_window_starts(config)
        for inst in REAL_INSTRUMENTS:
            if inst.family != Family.EQUITIES:
                continue
            older_filings: list[tuple[date, str, str]] = []
            recent_filings: list[tuple[date, str, str]] = []
            for window_start in window_starts:
                older_filings.append((window_start - timedelta(days=40), "8-K", "5.02"))
                older_filings.append((window_start - timedelta(days=20), "8-K", "2.02,9.01"))
                for offset in (45, 136, 227, 318):
                    recent_filings.append(
                        (window_start + timedelta(days=offset), "8-K", "2.02,9.01")
                    )
                recent_filings.append((window_start + timedelta(days=200), "8-K", "5.02"))
            older_filings.sort()
            recent_filings.sort()

            older_name = f"CIK{inst.cik}-submissions-001.json"
            files[edgar_followup_url(older_name)] = _edgar_older_json(older_filings)
            files_list = [
                {
                    "name": older_name,
                    "filingFrom": start.isoformat(),
                    "filingTo": (min(window_starts) - timedelta(days=1)).isoformat(),
                }
            ]
            files[edgar_url(inst.cik)] = _edgar_main_json(recent_filings, files_list)

        fetch_all(config, data_dir, opener=_fake_opener(files), sleeper=lambda _: None)
        return FakeCache(data_dir=data_dir, holiday=holiday, betas=betas)

    return _build
