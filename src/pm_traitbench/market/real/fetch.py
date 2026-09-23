"""Raw-data fetch for the real market: FRED, Yahoo Finance and Nasdaq into a
local cache with a sha256 manifest, so no other pipeline code ever touches
the network.
"""

import hashlib
import http.client
import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError

from pm_traitbench.config import (
    REAL_FETCH_BUFFER_DAYS,
    REAL_REQUEST_INTERVAL_S,
    REAL_RETRIES,
    Config,
    RealSeedSpec,
    real_window_end,
)
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.real.sources import fred_series, yahoo_tickers
from pm_traitbench.market.stage import referenced_seeds

Opener = Callable[[str], bytes]

_USER_AGENT = "Mozilla/5.0 (compatible; pm-traitbench/1.0; +data fetcher)"
_TIMEOUT_S = 30.0


def urlopen_bytes(url: str) -> bytes:
    """Fetch `url` with a browser-like User-Agent and a 30 second timeout."""
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return response.read()


def cache_dir(data_dir: Path) -> Path:
    return data_dir / "raw" / "market"


def fred_url(series: str, start: date, end: date) -> str:
    return (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        f"&cosd={start.isoformat()}&coed={end.isoformat()}"
    )


def _unix_seconds(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def yahoo_url(ticker: str, start: date, end: date) -> str:
    period1 = _unix_seconds(start)
    period2 = _unix_seconds(end + timedelta(days=1))
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )


def nasdaq_url(day: date) -> str:
    return f"https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}"


def _weekdays_before(day: date, n: int) -> date:
    """Return the date `n` weekdays before `day` (`day` itself excluded)."""
    current = day
    remaining = n
    while remaining > 0:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _referenced_real_seeds(config: Config) -> dict[str, RealSeedSpec]:
    """Real seeds the population's pilot/full market-seed lists use."""
    used = set(referenced_seeds(config))
    return {name: spec for name, spec in config.market.real.seeds.items() if name in used}


def fetch_range(config: Config) -> tuple[date, date]:
    """Union over referenced real seeds of (burn-in start minus the fetch buffer,
    window end). Falls back to the calendar start when no real seed is referenced.
    """
    seeds = _referenced_real_seeds(config)
    if not seeds:
        return (config.calendar.start, config.calendar.start)
    back = config.market.burn_in_days + REAL_FETCH_BUFFER_DAYS
    starts = [_weekdays_before(spec.window_start, back) for spec in seeds.values()]
    ends = [real_window_end(spec, config.calendar.n_weeks) for spec in seeds.values()]
    return (min(starts), max(ends))


def _nasdaq_days(config: Config) -> list[date]:
    """Every weekday from the first referenced real seed's window start to its end."""
    seeds = _referenced_real_seeds(config)
    if not seeds:
        return []
    first = next(iter(seeds.values()))
    end = real_window_end(first, config.calendar.n_weeks)
    days = []
    day = first.window_start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    url: str
    retrieved_at: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Manifest:
    window: tuple[str, str]
    entries: tuple[ManifestEntry, ...]

    def to_json(self) -> str:
        payload = {
            "window": list(self.window),
            "entries": [
                {
                    "path": entry.path,
                    "url": entry.url,
                    "retrieved_at": entry.retrieved_at,
                    "sha256": entry.sha256,
                    "size": entry.size,
                }
                for entry in self.entries
            ],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        payload = json.loads(text)
        window = payload["window"]
        return cls(
            window=(window[0], window[1]),
            entries=tuple(ManifestEntry(**entry) for entry in payload["entries"]),
        )


def _fetch_with_retry(
    opener: Opener, sleeper: Callable[[float], None], interval: float, url: str, label: str
) -> bytes:
    # REAL_RETRIES counts attempts (including the first), not additional retries:
    # the last attempt raises instead of sleeping and trying again.
    attempt = 0
    while True:
        try:
            body = opener(url)
        except HTTPError as e:
            retryable = e.code == 429 or 500 <= e.code < 600
            if not retryable or attempt >= REAL_RETRIES - 1:
                raise StageIOError(f"{label}: fetch failed (HTTP {e.code})") from e
            sleeper(interval * 2**attempt)
            attempt += 1
            continue
        except URLError as e:
            # Only a timeout or connection failure wrapped in a URLError is
            # transient; anything else (SSL, DNS...) raises at once below.
            transient = isinstance(e.reason, TimeoutError | ConnectionError)
            if not transient or attempt >= REAL_RETRIES - 1:
                raise StageIOError(f"{label}: fetch failed ({e.reason})") from e
            sleeper(interval * 2**attempt)
            attempt += 1
            continue
        except (TimeoutError, ConnectionError, http.client.IncompleteRead) as e:
            if attempt >= REAL_RETRIES - 1:
                raise StageIOError(f"{label}: fetch failed ({e})") from e
            sleeper(interval * 2**attempt)
            attempt += 1
            continue
        except OSError as e:
            # A non-transient OSError (permission, DNS, SSL...) never retries.
            raise StageIOError(f"{label}: fetch failed ({e})") from e
        sleeper(interval)
        return body


def _validate_fred(body: bytes, label: str) -> None:
    if not body.startswith(b"observation_date,"):
        raise StageIOError(f"malformed FRED response for series '{label}'")


def _validate_yahoo(body: bytes, label: str) -> None:
    try:
        payload = json.loads(body)
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise StageIOError(f"malformed Yahoo response for ticker '{label}'") from e
    if len(timestamps) != len(adjclose):
        raise StageIOError(f"malformed Yahoo response for ticker '{label}'")


def _validate_nasdaq(body: bytes, label: str) -> None:
    try:
        payload = json.loads(body)
        rows = payload["data"]["rows"]
    except (KeyError, TypeError, ValueError) as e:
        raise StageIOError(f"malformed Nasdaq response for date '{label}'") from e
    if rows is not None and not isinstance(rows, list):
        raise StageIOError(f"malformed Nasdaq response for date '{label}'")


def _validate(rel_path: str, body: bytes, label: str) -> None:
    if rel_path.startswith("fred/"):
        _validate_fred(body, label)
    elif rel_path.startswith("yahoo/"):
        _validate_yahoo(body, label)
    elif rel_path.startswith("nasdaq/"):
        _validate_nasdaq(body, label)


def _read_manifest(path: Path) -> Manifest:
    try:
        return Manifest.from_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise StageIOError(f"corrupt manifest at {path}: {e}") from e


def _load_old_manifest(data_dir: Path) -> Manifest | None:
    path = cache_dir(data_dir) / "manifest.json"
    if not path.exists():
        return None
    return _read_manifest(path)


def _write_manifest(root: Path, manifest: Manifest) -> None:
    manifest_path = root / "manifest.json"
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_path.write_text(manifest.to_json(), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def fetch_all(
    config: Config,
    data_dir: Path,
    *,
    force: bool = False,
    opener: Opener = urlopen_bytes,
    sleeper: Callable[[float], None] | None = None,
) -> Manifest:
    """Fetch every real-market raw series into the cache and write its manifest.

    Without `force`, a file already on disk whose sha256 and URL still match
    the previous manifest is kept and not refetched; with `force` every file
    is refetched. Requests are spaced `REAL_REQUEST_INTERVAL_S` apart, with
    exponential backoff on a retryable failure. The manifest is checkpointed
    after every file, so a failure partway through keeps prior progress.
    When no real seed is referenced by the population's market seeds, no
    file is fetched and an empty manifest is written.
    """
    sleep = sleeper if sleeper is not None else time.sleep
    root = cache_dir(data_dir)
    for sub in ("fred", "yahoo", "nasdaq"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    start, end = fetch_range(config)
    old = _load_old_manifest(data_dir)
    old_by_path = {entry.path: entry for entry in old.entries} if old else {}

    tasks: list[tuple[str, str, str]] = []
    if _referenced_real_seeds(config):
        for series in fred_series():
            tasks.append((f"fred/{series}.csv", fred_url(series, start, end), series))
        for ticker in yahoo_tickers():
            tasks.append((f"yahoo/{ticker}.json", yahoo_url(ticker, start, end), ticker))
        for day in _nasdaq_days(config):
            label = day.isoformat()
            tasks.append((f"nasdaq/{label}.json", nasdaq_url(day), label))

    entries: list[ManifestEntry] = []
    for rel_path, url, label in tasks:
        full_path = root / rel_path
        old_entry = old_by_path.get(rel_path)
        entry: ManifestEntry | None = None
        # A kept file must match both the manifest sha256 and the URL it was
        # fetched from; a changed fetch window changes the URL even when a
        # coincidentally identical file is still on disk.
        if not force and old_entry is not None and old_entry.url == url and full_path.exists():
            actual = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if actual == old_entry.sha256:
                entry = old_entry

        if entry is None:
            body = _fetch_with_retry(opener, sleep, REAL_REQUEST_INTERVAL_S, url, label)
            _validate(rel_path, body, label)
            full_path.write_bytes(body)
            entry = ManifestEntry(
                path=rel_path,
                url=url,
                retrieved_at=datetime.now(UTC).isoformat(),
                sha256=hashlib.sha256(body).hexdigest(),
                size=len(body),
            )

        entries.append(entry)
        # Checkpoint immediately: a failure on the next file keeps this progress.
        _write_manifest(
            root, Manifest(window=(start.isoformat(), end.isoformat()), entries=tuple(entries))
        )

    manifest = Manifest(window=(start.isoformat(), end.isoformat()), entries=tuple(entries))
    _write_manifest(root, manifest)
    return manifest


class RawCache:
    """Reads validated raw series from a fetched cache, verifying manifest hashes."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self._root = root
        self._manifest = manifest

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @classmethod
    def open(cls, data_dir: Path) -> "RawCache":
        root = cache_dir(data_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise StageIOError(f"no raw market cache at {root}; run fetch-market first")
        manifest = _read_manifest(manifest_path)
        for entry in manifest.entries:
            path = root / entry.path
            if not path.exists():
                raise StageIOError(f"raw market cache file missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != entry.sha256:
                raise StageIOError(f"raw market cache file does not match its manifest: {path}")
        return cls(root, manifest)

    def fred(self, series: str) -> dict[date, float]:
        """FRED values by date, skipping a missing observation: an empty value
        (FRED's actual holiday convention, e.g. '2018-07-04,') or a '.' marker.
        """
        text = (self._root / "fred" / f"{series}.csv").read_text(encoding="utf-8")
        values: dict[date, float] = {}
        for line in text.splitlines()[1:]:
            if not line:
                continue
            day_str, value_str = line.split(",", 1)
            value_str = value_str.strip()
            if value_str in ("", "."):
                continue
            values[date.fromisoformat(day_str)] = float(value_str)
        return values

    def yahoo(self, ticker: str) -> dict[date, float]:
        """Adjusted close by the UTC date of each timestamp, skipping null values."""
        payload = json.loads((self._root / "yahoo" / f"{ticker}.json").read_text(encoding="utf-8"))
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
        values: dict[date, float] = {}
        for ts, value in zip(timestamps, adjclose, strict=True):
            if value is None:
                continue
            values[datetime.fromtimestamp(ts, tz=UTC).date()] = float(value)
        return values

    def nasdaq_symbols(self, day: date) -> set[str]:
        """The set of symbols reporting earnings on `day`."""
        payload = json.loads(
            (self._root / "nasdaq" / f"{day.isoformat()}.json").read_text(encoding="utf-8")
        )
        rows = payload["data"]["rows"] or []
        return {row["symbol"] for row in rows}
