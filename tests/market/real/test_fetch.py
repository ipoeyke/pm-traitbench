"""Tests for the real-market raw-data fetch and its manifest cache."""

import http.client
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import pm_traitbench.market.real.fetch as fetch_module
from pm_traitbench.config import REAL_REQUEST_INTERVAL_S, REAL_RETRIES, Config, real_window_end
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


def _only_synthetic(config: Config) -> Config:
    return config.model_copy(
        update={"population": config.population.model_copy(update={"pilot_market_seeds": ("A",)})}
    )


def _two_real_seed_config() -> Config:
    """A config referencing two real seeds, R1's default window plus a second,
    non-adjacent window 104 weeks later.
    """
    config = Config()
    r1 = config.market.real.seeds["R1"]
    r2 = r1.model_copy(
        update={
            "window_start": r1.window_start + timedelta(weeks=104),
            "regime_starts": tuple(
                (regime, start + timedelta(weeks=104)) for regime, start in r1.regime_starts
            ),
            "note": "second window for multi-seed fetch testing",
        }
    )
    return config.model_copy(
        update={
            "market": config.market.model_copy(
                update={
                    "real": config.market.real.model_copy(update={"seeds": {"R1": r1, "R2": r2}})
                }
            ),
            "population": config.population.model_copy(update={"pilot_market_seeds": ("R1", "R2")}),
        }
    )


def test_nasdaq_days_covers_every_referenced_real_seeds_window() -> None:
    config = _two_real_seed_config()
    r1 = config.market.real.seeds["R1"]
    r2 = config.market.real.seeds["R2"]

    days = fetch_module._nasdaq_days(config)

    r1_days = _weekdays_of(r1.window_start, real_window_end(r1, config.calendar.n_weeks))
    r2_days = _weekdays_of(r2.window_start, real_window_end(r2, config.calendar.n_weeks))
    assert set(r1_days) <= set(days)
    assert set(r2_days) <= set(days)
    assert days == sorted(set(r1_days) | set(r2_days))


def test_fetch_all_fetches_nasdaq_days_for_every_referenced_real_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch_module, "fred_series", lambda: [])
    monkeypatch.setattr(fetch_module, "yahoo_tickers", lambda: [])

    config = _two_real_seed_config()
    r1 = config.market.real.seeds["R1"]
    r2 = config.market.real.seeds["R2"]

    body = b'{"data": {"rows": null}}'
    manifest = fetch_all(config, tmp_path, opener=lambda url: body, sleeper=lambda _: None)

    paths = {entry.path for entry in manifest.entries}
    assert f"nasdaq/{r1.window_start.isoformat()}.json" in paths
    r2_end = real_window_end(r2, config.calendar.n_weeks)
    assert f"nasdaq/{r2_end.isoformat()}.json" in paths


def test_fetch_all_with_no_referenced_real_seeds_fetches_nothing(tmp_path: Path) -> None:
    only_synthetic = _only_synthetic(Config())

    def _opener(url: str) -> bytes:
        raise AssertionError(f"should not be called: {url}")

    manifest = fetch_all(only_synthetic, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert manifest.entries == ()
    assert manifest.complete is True
    assert not (fetch_module.cache_dir(tmp_path) / "manifest.json").exists()


def test_fetch_all_with_no_referenced_real_seeds_returns_existing_manifest_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    body = b"observation_date,DGS2\n2018-01-02,2.0\n"
    old = fetch_all(config, tmp_path, opener=lambda url: body, sleeper=lambda _: None)

    manifest_path = fetch_module.cache_dir(tmp_path) / "manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    only_synthetic = _only_synthetic(config)

    def _opener(url: str) -> bytes:
        raise AssertionError(f"should not be called: {url}")

    result = fetch_all(only_synthetic, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert result == old
    assert manifest_path.read_text(encoding="utf-8") == before


def test_fetch_all_writes_every_file_and_a_matching_manifest(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    assert len(cache.manifest.entries) > 0
    for entry in cache.manifest.entries:
        assert (result.data_dir / "raw" / "market" / entry.path).exists()


def test_second_fetch_without_force_makes_no_requests(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)

    def _opener(url: str) -> bytes:
        raise AssertionError(f"should not be called: {url}")

    fetch_all(config, result.data_dir, opener=_opener, sleeper=lambda _: None)


def test_second_fetch_with_force_refetches_every_file(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    old = RawCache.open(result.data_dir).manifest
    bodies_by_url = {
        entry.url: (result.data_dir / "raw" / "market" / entry.path).read_bytes()
        for entry in old.entries
    }

    calls: list[str] = []

    def _opener(url: str) -> bytes:
        calls.append(url)
        return bodies_by_url[url]

    fetch_all(config, result.data_dir, force=True, opener=_opener, sleeper=lambda _: None)
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


def _read_manifest(tmp_path: Path) -> fetch_module.Manifest:
    manifest_path = fetch_module.cache_dir(tmp_path) / "manifest.json"
    return fetch_module.Manifest.from_json(manifest_path.read_text(encoding="utf-8"))


def test_manifest_checkpoints_after_each_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure partway through a fetch leaves the manifest holding the files
    already fetched, not just the ones from before this run started, marked
    incomplete since RawCache.open would otherwise refuse to read it.
    """
    monkeypatch.setattr(fetch_module, "fred_series", lambda: ["DGS2", "DGS5", "DGS10"])
    monkeypatch.setattr(fetch_module, "yahoo_tickers", lambda: [])
    monkeypatch.setattr(fetch_module, "_nasdaq_days", lambda config: [])
    config = Config()
    body = b"observation_date,X\n2018-01-02,1.0\n"
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 3:
            raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        return body

    with pytest.raises(StageIOError):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)

    manifest = _read_manifest(tmp_path)
    assert manifest.complete is False
    assert [entry.path for entry in manifest.entries] == ["fred/DGS2.csv", "fred/DGS5.csv"]


def test_checkpoint_seeds_still_valid_old_entries_for_tasks_not_yet_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file this run hasn't reached yet keeps its still-valid old entry in
    the interim checkpoint, so a failure elsewhere doesn't discard it.
    """
    monkeypatch.setattr(fetch_module, "fred_series", lambda: ["DGS2", "DGS5", "DGS10"])
    monkeypatch.setattr(fetch_module, "yahoo_tickers", lambda: [])
    monkeypatch.setattr(fetch_module, "_nasdaq_days", lambda config: [])
    config = Config()
    body = b"observation_date,X\n2018-01-02,1.0\n"

    old = fetch_all(config, tmp_path, opener=lambda url: body, sleeper=lambda _: None)
    assert old.complete is True
    assert len(old.entries) == 3

    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 2:
            raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        return body

    with pytest.raises(StageIOError):
        fetch_all(config, tmp_path, force=True, opener=_opener, sleeper=lambda _: None)

    manifest = _read_manifest(tmp_path)
    assert manifest.complete is False
    kept = {entry.path: entry for entry in manifest.entries}
    assert kept["fred/DGS10.csv"] == old.entries[2]


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


def test_url_error_with_non_transient_reason_raises_immediately(
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
    assert calls["n"] == 1


def test_url_error_wrapping_a_timeout_retries_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        raise URLError(TimeoutError("timed out"))

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
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == REAL_RETRIES


def test_incomplete_read_retries_like_5xx_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()

    def _opener(url: str) -> bytes:
        raise http.client.IncompleteRead(b"partial")

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)


def test_other_os_error_raises_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SSL, permission and DNS failures are OSErrors but never transient timeouts
    or connection failures, so they raise without retrying.
    """
    _isolate_to_one_fred_series(monkeypatch, "DGS2")
    config = Config()
    calls = {"n": 0}

    def _opener(url: str) -> bytes:
        calls["n"] += 1
        raise PermissionError("permission denied")

    with pytest.raises(StageIOError, match="DGS2"):
        fetch_all(config, tmp_path, opener=_opener, sleeper=lambda _: None)
    assert calls["n"] == 1


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


def test_raw_cache_open_fails_on_incomplete_manifest(tmp_path: Path) -> None:
    root = fetch_module.cache_dir(tmp_path)
    root.mkdir(parents=True)
    manifest = fetch_module.Manifest(
        window=("2018-01-01", "2018-01-02"), entries=(), complete=False
    )
    (root / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

    with pytest.raises(StageIOError, match="is incomplete; run fetch-market again"):
        RawCache.open(tmp_path)


def test_raw_cache_fred_raises_for_an_unlisted_series(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    with pytest.raises(StageIOError, match="UNKNOWN"):
        cache.fred("UNKNOWN")


def test_raw_cache_yahoo_raises_for_an_unlisted_ticker(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    with pytest.raises(StageIOError, match="UNKNOWN"):
        cache.yahoo("UNKNOWN")


def test_raw_cache_nasdaq_symbols_raises_for_an_unlisted_date(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    with pytest.raises(StageIOError, match="2099-01-01"):
        cache.nasdaq_symbols(date(2099, 1, 1))


def test_raw_cache_open_fails_on_tampered_file_naming_its_path(fake_cache) -> None:
    config = Config()
    result = fake_cache(config)
    cache = RawCache.open(result.data_dir)
    tampered_path = result.data_dir / "raw" / "market" / cache.manifest.entries[0].path
    tampered_path.write_bytes(b"tampered")

    with pytest.raises(StageIOError) as exc_info:
        RawCache.open(result.data_dir)
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
    result = fake_cache(config)
    start, end = fetch_range(config)
    weekdays = _weekdays_of(start, end)

    cache = RawCache.open(result.data_dir)
    values = cache.fred("DGS2")
    assert result.holiday not in values
    assert len(values) == len(weekdays) - 1
    for day in weekdays:
        if day != result.holiday:
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
    result = fake_cache(config)

    cache = RawCache.open(result.data_dir)
    symbols = cache.nasdaq_symbols(result.holiday)
    assert isinstance(symbols, set)
    assert "AAPL" in symbols


def test_fake_opener_raises_on_unknown_url(fake_opener) -> None:
    opener = fake_opener({"https://known": b"body"})
    assert opener("https://known") == b"body"
    with pytest.raises(AssertionError):
        opener("https://unknown")


@pytest.mark.network
def test_urlopen_bytes_fetches_a_real_fred_series() -> None:
    body = urlopen_bytes(fred_url("DGS10", date(2018, 6, 4), date(2018, 6, 8)))
    assert body.startswith(b"observation_date,DGS10")
