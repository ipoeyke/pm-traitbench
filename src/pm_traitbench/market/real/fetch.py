"""Raw-data fetch for the real market: FRED, Yahoo Finance and SEC EDGAR into
a local cache with a sha256 manifest, so no other pipeline code ever touches
the network.
"""

import hashlib
import http.client
import json
import os
import re
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
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
from pm_traitbench.market.real.sources import REAL_INSTRUMENTS, fred_series, yahoo_tickers

Opener = Callable[[str, Mapping[str, str]], bytes]

_USER_AGENT = "Mozilla/5.0 (compatible; pm-traitbench/1.0; +data fetcher)"
_DEFAULT_HEADERS: dict[str, str] = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
_TIMEOUT_S = 30.0


def urlopen_bytes(url: str, headers: Mapping[str, str]) -> bytes:
    """Fetch `url` with the given headers and a 30 second timeout."""
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return response.read()


def _edgar_headers(config: Config) -> dict[str, str]:
    return {"User-Agent": config.market.real.sec_user_agent, "Accept": "application/json"}


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


def edgar_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def edgar_followup_url(name: str) -> str:
    return f"https://data.sec.gov/submissions/{name}"


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
    opener: Opener,
    sleeper: Callable[[float], None],
    interval: float,
    url: str,
    headers: Mapping[str, str],
    label: str,
) -> bytes:
    # REAL_RETRIES counts attempts (including the first), not additional retries:
    # the last attempt raises instead of sleeping and trying again.
    attempt = 0
    while True:
        try:
            body = opener(url, headers)
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


# A main submissions file is named exactly "CIK<10 digits>.json"; an older
# file discovered through filings.files carries a "-submissions-NNN" suffix.
_EDGAR_MAIN_PATH_RE = re.compile(r"^edgar/CIK\d{10}\.json$")
_EDGAR_LIST_FIELDS: tuple[str, ...] = ("form", "filingDate", "acceptanceDateTime", "items")


def _validate_edgar_files_list(files: object, label: str) -> None:
    """`filings.files` must be a list, and every entry a string `name` and
    ISO `filingFrom`/`filingTo` dates.
    """
    if not isinstance(files, list):
        raise StageIOError(f"malformed EDGAR response for CIK '{label}'")
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise StageIOError(f"malformed EDGAR response for CIK '{label}'")
        try:
            date.fromisoformat(entry.get("filingFrom"))
            date.fromisoformat(entry.get("filingTo"))
        except (TypeError, ValueError) as e:
            raise StageIOError(f"malformed EDGAR response for CIK '{label}'") from e


def _validate_edgar(rel_path: str, body: bytes, label: str) -> None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError) as e:
        raise StageIOError(f"malformed EDGAR response for CIK '{label}'") from e
    if _EDGAR_MAIN_PATH_RE.match(rel_path):
        try:
            node = payload["filings"]["recent"]
            files = payload["filings"]["files"]
        except (KeyError, TypeError) as e:
            raise StageIOError(f"malformed EDGAR response for CIK '{label}'") from e
        _validate_edgar_files_list(files, label)
    else:
        node = payload
    try:
        lengths = {len(node[field]) for field in _EDGAR_LIST_FIELDS}
    except (KeyError, TypeError) as e:
        raise StageIOError(f"malformed EDGAR response for CIK '{label}'") from e
    if len(lengths) != 1:
        raise StageIOError(f"malformed EDGAR response for CIK '{label}'")


def _validate(rel_path: str, body: bytes, label: str) -> None:
    if rel_path.startswith("fred/"):
        _validate_fred(body, label)
    elif rel_path.startswith("yahoo/"):
        _validate_yahoo(body, label)
    elif rel_path.startswith("edgar/"):
        _validate_edgar(rel_path, body, label)


def _edgar_followup_names(body: bytes, start: date, end: date) -> list[str]:
    """Names, from `filings.files`, of older EDGAR files whose date range
    overlaps [start, end]. The main file's `filings.files` shape was
    already checked by `_validate`, so every entry is trusted here.
    """
    files = json.loads(body)["filings"]["files"]
    names = []
    for entry in files:
        filing_from = date.fromisoformat(entry["filingFrom"])
        filing_to = date.fromisoformat(entry["filingTo"])
        if filing_from <= end and filing_to >= start:
            names.append(entry["name"])
    return names


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


_Task = tuple[str, str, str, Mapping[str, str]]


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

    Each equity's EDGAR submissions file is fetched once per CIK; its own
    `filings.files` list is then read to discover any older filing history
    file whose date range overlaps the fetch window, and those are fetched
    too, going into the same manifest as every other file.
    """
    sleep = sleeper if sleeper is not None else time.sleep
    root = cache_dir(data_dir)
    old = _load_old_manifest(data_dir)

    if not _referenced_real_seeds(config):
        if old is not None:
            return old
        start, end = fetch_range(config)
        return Manifest(window=(start.isoformat(), end.isoformat()), entries=(), complete=True)

    for sub in ("fred", "yahoo", "edgar"):
        _mkdir(root / sub)

    start, end = fetch_range(config)
    old_by_path = {entry.path: entry for entry in old.entries} if old else {}
    edgar_headers = _edgar_headers(config)

    def _valid_old(rel_path: str, url: str) -> ManifestEntry | None:
        old_entry = old_by_path.get(rel_path)
        if old_entry is None or old_entry.url != url:
            return None
        full_path = root / rel_path
        if not full_path.exists():
            return None
        if hashlib.sha256(full_path.read_bytes()).hexdigest() == old_entry.sha256:
            return old_entry
        return None

    tasks: list[_Task] = []
    for series in fred_series():
        url = fred_url(series, start, end)
        tasks.append((f"fred/{series}.csv", url, series, _DEFAULT_HEADERS))
    for ticker in yahoo_tickers():
        url = yahoo_url(ticker, start, end)
        tasks.append((f"yahoo/{ticker}.json", url, ticker, _DEFAULT_HEADERS))
    for inst in REAL_INSTRUMENTS:
        for cik in inst.ciks:
            tasks.append((f"edgar/CIK{cik}.json", edgar_url(cik), cik, edgar_headers))

    # Which old entries are still trustworthy (same URL, same on-disk sha256):
    # computed once, since a file this run hasn't reached yet doesn't change.
    valid_old_by_path: dict[str, ManifestEntry] = {}
    for rel_path, url, _label, _headers in tasks:
        entry = _valid_old(rel_path, url)
        if entry is not None:
            valid_old_by_path[rel_path] = entry

    def _fetch_one(rel_path: str, url: str, label: str, headers: Mapping[str, str]) -> bytes | None:
        """Fetch or reuse one file, write it and return its bytes if freshly
        fetched (None if reused, so the caller can lazily read it back).
        """
        entry = valid_old_by_path.get(rel_path) if not force else None
        if entry is not None:
            entries.append(entry)
            return None
        body = _fetch_with_retry(opener, sleep, REAL_REQUEST_INTERVAL_S, url, headers, label)
        _validate(rel_path, body, label)
        _write_bytes(root / rel_path, body)
        entries.append(
            ManifestEntry(
                path=rel_path,
                url=url,
                retrieved_at=datetime.now(UTC).isoformat(),
                sha256=hashlib.sha256(body).hexdigest(),
                size=len(body),
            )
        )
        valid_old_by_path.pop(rel_path, None)
        return body

    entries: list[ManifestEntry] = []
    followups: list[_Task] = []

    for i, (rel_path, url, label, headers) in enumerate(tasks):
        body = _fetch_one(rel_path, url, label, headers)
        if rel_path.startswith("edgar/"):
            if body is None:
                body = (root / rel_path).read_bytes()
            for name in _edgar_followup_names(body, start, end):
                followups.append((f"edgar/{name}", edgar_followup_url(name), label, edgar_headers))

        remaining = [
            valid_old_by_path[p] for p, _, _, _ in tasks[i + 1 :] if p in valid_old_by_path
        ]
        _write_manifest(
            root,
            Manifest(
                window=(start.isoformat(), end.isoformat()),
                entries=tuple(entries) + tuple(remaining),
                complete=False,
            ),
        )

    # Discovered dynamically from each main file, so checked for reuse only
    # now, the same way the static tasks were checked above.
    for rel_path, url, _label, _headers in followups:
        entry = _valid_old(rel_path, url)
        if entry is not None:
            valid_old_by_path[rel_path] = entry

    for j, (rel_path, url, label, headers) in enumerate(followups):
        _fetch_one(rel_path, url, label, headers)
        remaining = [
            valid_old_by_path[p] for p, _, _, _ in followups[j + 1 :] if p in valid_old_by_path
        ]
        _write_manifest(
            root,
            Manifest(
                window=(start.isoformat(), end.isoformat()),
                entries=tuple(entries) + tuple(remaining),
                complete=False,
            ),
        )

    manifest = Manifest(
        window=(start.isoformat(), end.isoformat()), entries=tuple(entries), complete=True
    )
    _write_manifest(root, manifest)
    return manifest


@dataclass(frozen=True)
class EdgarFiling:
    """One SEC EDGAR filing entry from a company's submissions history."""

    form: str
    accepted: datetime
    items: tuple[str, ...]


def _parse_edgar_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _parse_edgar_filings(node: dict) -> list[EdgarFiling]:
    filings = []
    for form, accepted, item_str in zip(
        node["form"], node["acceptanceDateTime"], node["items"], strict=True
    ):
        items = tuple(item.strip() for item in item_str.split(",")) if item_str else ()
        accepted_at = _parse_edgar_datetime(accepted)
        filings.append(EdgarFiling(form=form, accepted=accepted_at, items=items))
    return filings


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

    def edgar_filings(self, ciks: Sequence[str]) -> list[EdgarFiling]:
        """Every filing across every CIK in `ciks`: each one's main submissions
        file merged with its own cached older files, found in the manifest by
        path prefix. A company that reorganised under a new holding company
        lists more than one CIK.
        """
        filings: list[EdgarFiling] = []
        for cik in ciks:
            entry = self._entry(f"edgar/CIK{cik}.json", f"EDGAR CIK '{cik}'")
            main = json.loads((self._root / entry.path).read_text(encoding="utf-8"))
            filings.extend(_parse_edgar_filings(main["filings"]["recent"]))

            prefix = f"edgar/CIK{cik}-"
            for path in sorted(self._entries_by_path):
                if not path.startswith(prefix):
                    continue
                older_entry = self._entries_by_path[path]
                older = json.loads((self._root / older_entry.path).read_text(encoding="utf-8"))
                filings.extend(_parse_edgar_filings(older))
        return filings
