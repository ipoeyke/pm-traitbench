"""Raw-data fetch for the real market: FRED and Yahoo Finance into a local
cache with a sha256 manifest, so no other pipeline code ever touches the
network.
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
    referenced_seeds,
)
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.real.sources import fred_series, yahoo_tickers

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
    complete: bool

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
            "complete": self.complete,
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        payload = json.loads(text)
        window = payload["window"]
        return cls(
            window=(window[0], window[1]),
            entries=tuple(ManifestEntry(**entry) for entry in payload["entries"]),
            complete=payload["complete"],
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


def _validate(rel_path: str, body: bytes, label: str) -> None:
    if rel_path.startswith("fred/"):
        _validate_fred(body, label)
    elif rel_path.startswith("yahoo/"):
        _validate_yahoo(body, label)


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


def _mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise StageIOError(f"cannot create directory {path}: {e}") from e


def _write_bytes(path: Path, body: bytes) -> None:
    try:
        path.write_bytes(body)
    except OSError as e:
        raise StageIOError(f"cannot write {path}: {e}") from e


def _write_manifest(root: Path, manifest: Manifest) -> None:
    manifest_path = root / "manifest.json"
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        tmp_path.write_text(manifest.to_json(), encoding="utf-8")
        os.replace(tmp_path, manifest_path)
    except OSError as e:
        raise StageIOError(f"cannot write manifest at {manifest_path}: {e}") from e


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
    (`complete=False`) after every file, seeded with any still-valid old
    entry for a file not yet reached, so a failure partway through keeps
    prior progress instead of discarding it; the final write sets
    `complete=True`. When no real seed is referenced by the population's
    market seeds, nothing is fetched: the existing manifest is returned
    unchanged if one exists, otherwise an empty complete manifest, and
    nothing is written.
    """
    sleep = sleeper if sleeper is not None else time.sleep
    root = cache_dir(data_dir)
    old = _load_old_manifest(data_dir)

    if not _referenced_real_seeds(config):
        if old is not None:
            return old
        start, end = fetch_range(config)
        return Manifest(window=(start.isoformat(), end.isoformat()), entries=(), complete=True)

    for sub in ("fred", "yahoo"):
        _mkdir(root / sub)

    start, end = fetch_range(config)
    old_by_path = {entry.path: entry for entry in old.entries} if old else {}

    tasks: list[tuple[str, str, str]] = []
    for series in fred_series():
        tasks.append((f"fred/{series}.csv", fred_url(series, start, end), series))
    for ticker in yahoo_tickers():
        tasks.append((f"yahoo/{ticker}.json", yahoo_url(ticker, start, end), ticker))

    # Which old entries are still trustworthy (same URL, same on-disk sha256):
    # computed once, since a file this run hasn't reached yet doesn't change.
    valid_old_by_path: dict[str, ManifestEntry] = {}
    for rel_path, url, _label in tasks:
        old_entry = old_by_path.get(rel_path)
        if old_entry is None or old_entry.url != url:
            continue
        full_path = root / rel_path
        if not full_path.exists():
            continue
        if hashlib.sha256(full_path.read_bytes()).hexdigest() == old_entry.sha256:
            valid_old_by_path[rel_path] = old_entry

    entries: list[ManifestEntry] = []
    for i, (rel_path, url, label) in enumerate(tasks):
        full_path = root / rel_path
        # A kept file must match both the manifest sha256 and the URL it was
        # fetched from; a changed fetch window changes the URL even when a
        # coincidentally identical file is still on disk. force skips reuse.
        entry = valid_old_by_path.get(rel_path) if not force else None

        if entry is None:
            body = _fetch_with_retry(opener, sleep, REAL_REQUEST_INTERVAL_S, url, label)
            _validate(rel_path, body, label)
            _write_bytes(full_path, body)
            entry = ManifestEntry(
                path=rel_path,
                url=url,
                retrieved_at=datetime.now(UTC).isoformat(),
                sha256=hashlib.sha256(body).hexdigest(),
                size=len(body),
            )
            # A fresh fetch replaces any placeholder used below for checkpoints.
            valid_old_by_path.pop(rel_path, None)

        entries.append(entry)
        # Checkpoint immediately: a failure on the next file keeps this
        # progress, plus a safety net of still-valid old entries for every
        # file not yet reached.
        remaining = [valid_old_by_path[p] for p, _, _ in tasks[i + 1 :] if p in valid_old_by_path]
        checkpoint = Manifest(
            window=(start.isoformat(), end.isoformat()),
            entries=tuple(entries) + tuple(remaining),
            complete=False,
        )
        _write_manifest(root, checkpoint)

    manifest = Manifest(
        window=(start.isoformat(), end.isoformat()), entries=tuple(entries), complete=True
    )
    _write_manifest(root, manifest)
    return manifest


class RawCache:
    """Reads validated raw series from a fetched cache, verifying manifest hashes."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self._root = root
        self._manifest = manifest
        self._entries_by_path = {entry.path: entry for entry in manifest.entries}

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
        if not manifest.complete:
            raise StageIOError(f"raw market cache at {root} is incomplete; run fetch-market again")
        for entry in manifest.entries:
            path = root / entry.path
            if not path.exists():
                raise StageIOError(f"raw market cache file missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != entry.sha256:
                raise StageIOError(f"raw market cache file does not match its manifest: {path}")
        return cls(root, manifest)

    def _entry(self, rel_path: str, description: str) -> ManifestEntry:
        entry = self._entries_by_path.get(rel_path)
        if entry is None:
            raise StageIOError(f"{description} is not in the raw cache manifest")
        return entry

    def fred(self, series: str) -> dict[date, float]:
        """FRED values by date, skipping a missing observation: an empty value
        (FRED's actual holiday convention, e.g. '2018-07-04,') or a '.' marker.
        """
        entry = self._entry(f"fred/{series}.csv", f"FRED series '{series}'")
        text = (self._root / entry.path).read_text(encoding="utf-8")
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
        entry = self._entry(f"yahoo/{ticker}.json", f"Yahoo ticker '{ticker}'")
        payload = json.loads((self._root / entry.path).read_text(encoding="utf-8"))
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
        values: dict[date, float] = {}
        for ts, value in zip(timestamps, adjclose, strict=True):
            if value is None:
                continue
            values[datetime.fromtimestamp(ts, tz=UTC).date()] = float(value)
        return values
