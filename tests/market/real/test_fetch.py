"""Tests for the real-market raw-data fetch and its manifest cache."""

import http.client
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import pm_traitbench.market.real.fetch as fetch_module
from pm_traitbench.config import REAL_REQUEST_INTERVAL_S, REAL_RETRIES, Config
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.real.fetch import (
    RawCache,
    fetch_all,
    fetch_range,
    fred_url,
    nasdaq_url,
    urlopen_bytes,
    yahoo_url,
)


def _isolate_to_one_fred_series(monkeypatch: pytest.MonkeyPatch, series: str) -> None:
    """Restrict fetch_all's task list to a single FRED series, for retry/error tests."""
    monkeypatch.setattr(fetch_module, "fred_series", lambda: [series])
    monkeypatch.setattr(fetch_module, "yahoo_tickers", lambda: [])
    monkeypatch.setattr(fetch_module, "_nasdaq_days", lambda config: [])


def _isolate_to_one_yahoo_ticker(monkeypatch: pytest.MonkeyPatch, ticker: str) -> None:
    monkeypatch.setattr(fetch_module, "fred_series", lambda: [])
    monkeypatch.setattr(fetch_module, "yahoo_tickers", lambda: [ticker])
    monkeypatch.setattr(fetch_module, "_nasdaq_days", lambda config: [])


def _weekdays_of(start: date, end: date) -> list[date]:
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _holiday_of(config: Config) -> date:
    """The fixture's one shared holiday-gap weekday (see tests/market/real/conftest.py)."""
    start, end = fetch_range(config)
    weekdays = _weekdays_of(start, end)
    return weekdays[len(weekdays) // 2]


def test_fred_url_matches_the_binding_form() -> None:
    url = fred_url("DGS10", date(2018, 6, 4), date(2019, 6, 3))
    assert url == (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2018-06-04&coed=2019-06-03"
    )


def test_yahoo_url_matches_the_binding_form() -> None:
    start = date(2018, 6, 4)
    end = date(2018, 6, 5)
    url = yahoo_url("AAPL", start, end)
    expected_period1 = int(datetime(2018, 6, 4, tzinfo=UTC).timestamp())
    expected_period2 = int(datetime(2018, 6, 6, tzinfo=UTC).timestamp())
    assert url == (
        "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
        f"?period1={expected_period1}&period2={expected_period2}&interval=1d"
    )


def test_nasdaq_url_matches_the_binding_form() -> None:
    url = nasdaq_url(date(2018, 7, 6))
    assert url == "https://api.nasdaq.com/api/calendar/earnings?date=2018-07-06"


def test_fetch_range_starts_seventy_weekdays_before_window_start_and_ends_at_window_end() -> None:
    config = Config()
    spec = next(iter(config.market.real.seeds.values()))
    start, end = fetch_range(config)

    day = spec.window_start
    remaining = config.market.burn_in_days + 10
    while remaining > 0:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            remaining -= 1
    assert start == day
    assert end == spec.window_start + timedelta(weeks=config.calendar.n_weeks)


def test_fetch_all_writes_every_file_and_a_matching_manifest(fake_cache) -> None:
    config = Config()
    data_dir = fake_cache(config)
    cache = RawCache.open(data_dir)
    assert len(cache.manifest.entries) > 0
    for entry in cache.manifest.entries:
        assert (data_dir / "raw" / "market" / entry.path).exists()


def test_second_fetch_without_force_makes_no_requests(fake_cache) -> None:
    config = Config()
    data_dir = fake_cache(config)

    def _opener(url: str) -> bytes:
        raise AssertionError(f"should not be called: {url}")

    fetch_all(config, data_dir, opener=_opener, sleeper=lambda _: None)


def test_second_fetch_with_force_refetches_every_file(fake_cache) -> None:
    config = Config()
    data_dir = fake_cache(config)
    old = RawCache.open(data_dir).manifest
    bodies_by_url = {
        entry.url: (data_dir / "raw" / "market" / entry.path).read_bytes() for entry in old.entries
    }

    calls: list[str] = []

    def _opener(url: str) -> bytes:
        calls.append(url)
        return bodies_by_url[url]

    fetch_all(config, data_dir, force=True, opener=_opener, sleeper=lambda _: None)
    assert len(calls) == len(old.entries)


def test_kept_file_must_match_both_sha256_and_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed fetch window changes a series' URL; a stale-window file refetches
    without --force even if its bytes still happen to match the old manifest sha256.
    """
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    body = b"observation_date,DGS2\n2018-01-02,2.0\n"
    fetch_all(config, tmp_path, opener=lambda url: body, sleeper=lambda _: None)

    spec = next(iter(config.market.real.seeds.values()))
    shifted_spec = spec.model_copy(
        update={
            "window_start": spec.window_start - timedelta(weeks=1),
            "regime_starts": tuple(
                (regime, start - timedelta(weeks=1)) for regime, start in spec.regime_starts
            ),
        }
    )
    shifted_config = config.model_copy(
        update={
            "market": config.market.model_copy(
                update={
                    "real": config.market.real.model_copy(update={"seeds": {"R1": shifted_spec}})
                }
            )
        }
    )

    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        return body

    fetch_all(shifted_config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == 1


def test_transient_429s_retry_then_succeed_with_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    body = b"observation_date,DGS2\n2018-01-02,2.0\n"
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise HTTPError(url, 429, "Too Many Requests", hdrs=None, fp=None)
        return body

    sleeps: list[float] = []
    manifest = fetch_all(config, tmp_path, opener=_opener, sleeper=lambda s: sleeps.append(s))

    assert calls["n"] == 3
    assert len(manifest.entries) == 1
    assert sleeps[0] == REAL_REQUEST_INTERVAL_S * 2**0
    assert sleeps[1] == REAL_REQUEST_INTERVAL_S * 2**1


def test_five_5xx_responses_raise_stage_io_error_naming_the_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        raise HTTPError(url, 503, "Service Unavailable", hdrs=None, fp=None)

    with pytest.raises(StageIOError, match="DGS2") as exc_info:
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == REAL_RETRIES
    assert "503" in str(exc_info.value)


def test_non_retryable_http_error_raises_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == 1


def test_url_error_retries_like_5xx_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        raise URLError("connection refused")

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == REAL_RETRIES


def test_timeout_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    body = b"observation_date,DGS2\n2018-01-02,2.0\n"
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("timed out")
        return body

    manifest = fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == 3
    assert len(manifest.entries) == 1


def test_connection_reset_retries_like_5xx_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()

    def _opener(url: str) -> bytes:
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)


def test_incomplete_read_retries_like_5xx_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()

    def _opener(url: str) -> bytes:
        raise http.client.IncompleteRead(b"partial")

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)


def test_malformed_fred_body_raises_naming_the_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()

    def _opener(url: str) -> bytes:
        return b"not,a,fred,csv\n"

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)


def test_malformed_yahoo_body_raises_naming_the_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_yahoo_ticker(monkeypatch, "AAPL")
    config = Config()

    def _opener(url: str) -> bytes:
        return b'{"chart": {"result": [{}], "error": null}}'

    with pytest.raises(StageIOError, match="AAPL"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)


def test_malformed_nasdaq_body_raises_naming_the_date() -> None:
    from pm_traitbench.market.real.fetch import _validate_nasdaq

    with pytest.raises(StageIOError, match="2018-07-06"):
        _validate_nasdaq(b'{"data": {}}', "2018-07-06")
    with pytest.raises(StageIOError, match="2018-07-06"):
        _validate_nasdaq(b'{"data": {"rows": "oops"}}', "2018-07-06")
    # null or empty rows are valid (no reports that day), not malformed
    _validate_nasdaq(b'{"data": {"rows": null}}', "2018-07-06")
    _validate_nasdaq(b'{"data": {"rows": []}}', "2018-07-06")


def test_raw_cache_open_fails_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(StageIOError, match="run fetch-market first"):
        RawCache.open(tmp_path)


def test_raw_cache_open_fails_on_tampered_file_naming_its_path(fake_cache) -> None:
    config = Config()
    data_dir = fake_cache(config)
    cache = RawCache.open(data_dir)
    tampered_path = data_dir / "raw" / "market" / cache.manifest.entries[0].path
    tampered_path.write_bytes(b"tampered")

    with pytest.raises(StageIOError) as exc_info:
        RawCache.open(data_dir)
    assert str(tampered_path) in str(exc_info.value)


def test_fetch_all_raises_on_corrupt_manifest_naming_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    manifest_path = fetch_module.cache_dir(tmp_path) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("not json", encoding="utf-8")

    with pytest.raises(StageIOError) as exc_info:
        fetch_all(config, tmp_path, opener=lambda url: b"unused", sleeper=lambda _: None)
    assert str(manifest_path) in str(exc_info.value)


def test_raw_cache_open_raises_on_corrupt_manifest_naming_its_path(tmp_path: Path) -> None:
    manifest_path = fetch_module.cache_dir(tmp_path) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"window": ["a", "b"]}', encoding="utf-8")  # missing "entries"

    with pytest.raises(StageIOError) as exc_info:
        RawCache.open(tmp_path)
    assert str(manifest_path) in str(exc_info.value)


def test_fred_skips_missing_marker(fake_cache) -> None:
    config = Config()
    data_dir = fake_cache(config)
    holiday = _holiday_of(config)
    start, end = fetch_range(config)
    weekdays = _weekdays_of(start, end)

    cache = RawCache.open(data_dir)
    values = cache.fred("DGS2")
    assert holiday not in values
    assert len(values) == len(weekdays) - 1
    for day in weekdays:
        if day != holiday:
            assert isinstance(values[day], float)


def test_yahoo_maps_a_known_timestamp_and_skips_a_null_adjclose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_yahoo_ticker(monkeypatch, "AAPL")
    config = Config()
    known_ts = int(datetime(2018, 6, 4, 13, 30, tzinfo=UTC).timestamp())
    null_ts = int(datetime(2018, 6, 5, 13, 30, tzinfo=UTC).timestamp())
    body = json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "timestamp": [known_ts, null_ts],
                        "indicators": {
                            "adjclose": [{"adjclose": [123.45, None]}],
                            "quote": [{}],
                        },
                    }
                ],
                "error": None,
            }
        }
    ).encode("utf-8")

    fetch_all(config, tmp_path, opener=lambda url: body, sleeper=lambda _: None)
    cache = RawCache.open(tmp_path)
    values = cache.yahoo("AAPL")
    assert values == {date(2018, 6, 4): 123.45}


def test_nasdaq_symbols_returns_the_set(fake_cache) -> None:
    config = Config()
    data_dir = fake_cache(config)
    holiday = _holiday_of(config)

    cache = RawCache.open(data_dir)
    symbols = cache.nasdaq_symbols(holiday)
    assert isinstance(symbols, set)
    assert len(symbols) > 0


def test_fake_opener_raises_on_unknown_url(fake_opener) -> None:
    opener = fake_opener({"https://known": b"body"})
    assert opener("https://known") == b"body"
    with pytest.raises(AssertionError):
        opener("https://unknown")


@pytest.mark.network
def test_urlopen_bytes_fetches_a_real_fred_series() -> None:
    body = urlopen_bytes(fred_url("DGS10", date(2018, 6, 4), date(2018, 6, 8)))
    assert body.startswith(b"observation_date,DGS10")
