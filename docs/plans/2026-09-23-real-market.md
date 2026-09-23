# Real Market Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the pilot split a real, disguised historical market (seed `R1`, Jun 2018 to May 2019) built from a manifest-verified local cache of free public data, side by side with the synthetic market that the full split keeps using.

**Architecture:** The `market/` package is restructured into shared modules plus `synthetic/` and `real/` subpackages with symmetric `build_seed(...) -> SeedMarket` entry points. A `fetch-market` CLI command fills `<data-dir>/raw/market/` with raw series and a sha256 manifest; the `market` stage dispatches each referenced seed to the real or synthetic builder and writes the same six tables. Real series keep their daily returns but are remapped onto the configured calendar and rebased to the synthetic starting levels.

**Tech Stack:** Python 3.13, numpy, pydantic v2, `urllib` (stdlib) for HTTP, pytest, ruff, uv.

## Global Constraints

- Worktree: `/Users/ianpoey/code/github/pm-traitbench-real-market`, branch `feat/real-market-pilot`. Run every command from there with `uv run`. Never touch another checkout.
- No file in the repo (code, docstrings, comments, tests, README, commit messages) may mention `docs/`, a spec or a plan. Docstrings state the rule itself. Citing a public data source (FRED series id, Yahoo Finance, Nasdaq, Federal Reserve, USDA, BLS) or a paper as the basis of a value is allowed.
- Raw market data is never committed. `data/` is already gitignored; the cache lives at `<data-dir>/raw/market/` and `--data-dir` defaults to `data` for every command.
- Never commit anything under `docs/`.
- Commit messages: one Conventional Commits subject line, no body, then a blank line and the single harness-provided `Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer for the committing model. No other trailer.
- Plain dash `-`, never an em dash. No filler words: genuinely, actually, truly, really.
- Docstrings concise; inline comments 1-2 lines. Match surrounding style: frozen pydantic models with `extra="forbid"`, `Field(description=...)` on every row-model field, `json_schema_extra={"basis": ..., "note": ...}` on every config leaf. `Config.dump_with_basis()` must keep working.
- Randomness only through `pm_traitbench.rng.stream(root_seed, *keys)`.
- No test may need the network unless marked `network`; such tests are skipped without `--run-network`.
- Lint and tests green after every task: `uv run ruff check && uv run ruff format --check && uv run pytest -q -W error::RuntimeWarning`.
- Every seed is built and checked before any market table is written.
- No test may pass only because of a hand-picked seed; a statistical assertion must hold for roots 0-9.

## Decisions made while planning (binding)

1. The restructure (Task 1) is a pure move: no behaviour change, same test bodies apart from imports and paths. `git mv` where possible so history follows.
2. After Task 2 the default config already names pilot seed `R1`, so until Task 7 lands the stage raises `StageIOError("real market seed 'R1' needs a raw cache; run fetch-market first")` for real seeds. Tests that run the market stage before Task 7 override `population.pilot_market_seeds` to a synthetic seed.
3. `DataStore` exposes `data_dir` as a read-only property; the raw cache path is `data_dir / "raw" / "market"`.
4. `SeedMarket` gains `fills: Mapping[str, int]` (longest run of filled days per series, default empty).
5. `family_indices` moves to the shared `market/check.py` so the real check can report realised moments with the same index definitions.
6. Real event rows use the same `EventType` values and `affected` rule as synthetic rows. Rating actions are not generated for real seeds.
7. Yahoo continuous front-month series are used as `M1` and as `price`; tenors M2-M12 are `M1 * (1 + slope_g * (k-1)/12)`, no noise.
8. The fetch step's HTTP opener is injectable: `fetch_all(config, data_dir, *, force, opener=urlopen_bytes)` where `opener(url: str) -> bytes`.

---

### Task 1: Restructure `market/` into shared, `synthetic/` and `real/`

**Files:**
- Create: `src/pm_traitbench/market/synthetic/__init__.py`, `src/pm_traitbench/market/real/__init__.py` (docstring only, real says it is filled by later work: "Real historical market data: registry, fetch, build and checks.")
- Move (git mv): `market/universe.py -> market/synthetic/universe.py`; `market/drivers.py -> market/synthetic/drivers.py`; `market/processes/ -> market/synthetic/processes/` (whole directory); `market/generate.py -> market/synthetic/build.py`
- Create by splitting: `market/output.py` (from `processes/common.py`: `ProcessOutput` only), `market/seed.py` (from `generate.py`: `SeedMarket`, `MarketRows`, `market_rng`, `to_rows`), `market/synthetic/schedule.py` (from `regimes.py`: `build_schedule`), `market/synthetic/events.py` (from `calendar.py`: `_grid_days`, `_poisson_days`, `_surprises`, `SampledEvents`, `EVENT_TARGETS`, `sample_events`, `EventJumps`, `build_jumps`), `market/synthetic/check.py` (from `check.py`: `implied_moments`, `round_level_test_count`, `_round_level_metrics`, `check_market`, and the vol/corr metric code)
- Modify: `market/calendar.py` (keeps `RngFor`, `row_sort_key`, `third_friday`, `generated_rows`, `event_day_indices`), `market/regimes.py` (keeps `RegimeLookup`, `RegimePath`, `regime_path`, `constant_path`), `market/check.py` (keeps `CheckMetric`, `CheckReport`, `family_indices`, `_changes`, `_fx_currencies`, `_fx_currency_log_value`, `_count_metric`, `_date_mismatch_metric`, `_count_metrics`, `_check_error`; make `count_metrics` and `check_error` public), `market/stage.py` (imports), `market/consensus.py` (import `ProcessOutput` from `market/output.py`)
- Tests: move `tests/market/test_universe.py`, `test_drivers.py`, `test_equities.py`, `test_rates.py`, `test_credit.py`, `test_commodities.py`, `test_fx.py`, `test_process_common.py`, `test_generate.py` (renamed `test_build.py`), `test_check.py` into `tests/market/synthetic/`; split `tests/market/test_calendar.py` so the sampling and jump tests go to `tests/market/synthetic/test_events.py`; split `tests/market/test_regimes.py` so the `build_schedule` tests go to `tests/market/synthetic/test_schedule.py`. `tests/market/test_check.py` keeps only the count and date-mismatch unit tests (the round-level and implied-moment tests move).

**Interfaces:**
- Consumes: everything as it exists today.
- Produces (all names unchanged unless stated):
  - `pm_traitbench.market.output.ProcessOutput`
  - `pm_traitbench.market.seed.SeedMarket` with a new field `fills: Mapping[str, int] = field(default_factory=dict)` placed last; `MarketRows`; `market_rng(root_seed) -> RngFor`; `to_rows(market) -> MarketRows`
  - `pm_traitbench.market.check`: `CheckMetric`, `CheckReport`, `family_indices(market, instruments) -> dict[Family, np.ndarray]`, `count_metrics(market, instruments) -> list[CheckMetric]`, `check_error(misses) -> MarketCheckError`
  - `pm_traitbench.market.synthetic.build.build_seed(config, seed, instruments, shocks, axis) -> SeedMarket` (was `generate_seed`)
  - `pm_traitbench.market.synthetic.check.check_market(market, instruments, config) -> CheckReport`
  - `pm_traitbench.market.synthetic.schedule.build_schedule(config, seed, timeline) -> list[RegimeSpan]`
  - `pm_traitbench.market.synthetic.events`: `sample_events`, `SampledEvents`, `EVENT_TARGETS`, `EventJumps`, `build_jumps`
  - `pm_traitbench.market.synthetic.processes.common`: `ProcessInputs`, `unit_student_t`, `log_grid_step`, `nearest_level`, `round_log_gap` (it re-imports `ProcessOutput` from `market.output` so existing imports keep working)

- [ ] **Step 1: Move files with `git mv`, then fix every import** in `src` and `tests`. `grep -rn "pm_traitbench.market" src tests` must show no reference to a moved path.
- [ ] **Step 2: Run** `uv run pytest -q -W error::RuntimeWarning`. Expected: 624 passed, the same count as before the move; `uv run ruff check` clean (import sorting will need `ruff check --fix` on touched files).
- [ ] **Step 3: Add one test** in `tests/market/test_seed.py`: `SeedMarket` accepts and defaults `fills` to an empty mapping, and `to_rows` ignores it.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `refactor: split market package into shared, synthetic and real`

---

### Task 2: Pilot seeds and real seed config

**Files:**
- Modify: `src/pm_traitbench/config.py` (`PopulationConfig`, new `RealSeedSpec`, `MarketRealConfig`, `MarketConfig.real`, `Config` cross-checks)
- Modify: `src/pm_traitbench/sampling/population.py:36-42`
- Modify: `src/pm_traitbench/market/stage.py` (seed iteration)
- Modify: `configs/demo.yaml`
- Test: `tests/test_config.py`, `tests/sampling/test_population.py`, `tests/market/test_stage.py`

**Interfaces:**
- Produces in `config.py`:
  - `PopulationConfig.pilot_market_seeds: tuple[str, ...] = ("R1",)` with basis `design`, note "the pilot runs on a real historical market so it can finish before the synthetic full split". Replaces `pilot_market_seed_count`, which is deleted. Validation as for `market_seeds`: non-empty, no repeats, no blank names.
  - `RealSeedSpec(BaseModel, frozen, extra forbid)`: `window_start: date`, `regime_starts: tuple[tuple[Regime, date], tuple[Regime, date], tuple[Regime, date]]`, `basis: Basis`, `note: str` (min length 1). Validators: `window_start.weekday() == 0`; `regime_starts[0][1] == window_start`; dates strictly ascending; each regime exactly once.
  - `MarketRealConfig`: `seeds: dict[str, RealSeedSpec]` default `{"R1": RealSeedSpec(window_start=date(2018, 6, 4), regime_starts=((RANGE, 2018-06-04), (RISK_OFF, 2018-10-01), (RISK_ON, 2018-12-26)), basis="design", note="a calm range (+7%), the Q4 2018 selloff (-19%) and the 2019 rebound (+12%) give the pilot seed the range, risk_off, risk_on order; the year is less memorable than 2020 or 2022")}`.
  - `MarketConfig.real: MarketRealConfig`. Validator on `MarketConfig`: no seed name in both `seeds` and `real.seeds`.
  - `Config` cross-check (replaces the old "market.seeds keys cover population.market_seeds"): every name in `population.pilot_market_seeds` and `population.market_seeds` is in `market.seeds` or `market.real.seeds`; the error names the missing seed. Every real seed's regime dates must fall inside `[window_start, window_start + n_weeks weeks)`.
  - Module constants in `config.py` next to `MarketRealConfig`, with a one-line reason each: `REAL_FILL_LIMIT = 5` (a longer gap than a holiday week means a broken series), `REAL_FETCH_BUFFER_DAYS = 10` (so the first axis day can be filled from earlier values), `REAL_REQUEST_INTERVAL_S = 1.0`, `REAL_RETRIES = 5`.
- `build_population`: pilot split uses `pop.pilot_market_seeds`, full split uses `pop.market_seeds`.
- `market/stage.py`: `referenced_seeds(config) -> list[str]` = pilot seeds followed by full seeds, without repeats; `run` iterates it. A seed in `config.market.real.seeds` raises `StageIOError(f"real market seed '{seed}' needs a raw cache; run fetch-market first")` for now (Task 7 replaces this branch). The synthetic universe is built only if a synthetic seed is referenced.
- `configs/demo.yaml`: `population.pilot_market_seeds: [A]` for now (Task 7 sets it to `[R1]`); comment says the demo pilot runs on the synthetic seed until a cache exists. Keep the other overrides.

- [ ] **Step 1: Write failing tests:**
  - `test_config.py`: defaults (`pilot_market_seeds == ("R1",)`, `market.real.seeds["R1"].window_start == date(2018,6,4)`); rejections with `match=`: `pilot_market_seed_count` now an unknown key; empty pilot seeds; a pilot seed defined nowhere; a name in both `market.seeds` and `market.real.seeds`; `window_start` not a Monday; regime starts out of order; first regime start not equal to `window_start`; a repeated regime; a regime start after the window end. Update `test_dump_with_basis_covers_every_leaf` expectations if the walker needs `RealSeedSpec` (it has a `basis` field, so the generic rule covers it).
  - `test_population.py`: pilot slots carry each of `pilot_market_seeds`, full slots each of `market_seeds`; replace the `pilot_market_seed_count` test.
  - `test_stage.py`: with the demo config the stage still writes six tables (pilot A); with a config whose pilot seed is `R1`, `run_stage` raises `StageIOError` matching "run fetch-market first" and writes nothing; `referenced_seeds` orders pilot first and dedupes.
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py tests/sampling/test_population.py tests/market/test_stage.py -q`. Expected: FAIL on the missing field and function names.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.** The end-to-end and CLI tests may need `pilot_market_seeds` overrides; make them minimal.
- [ ] **Step 5: Commit** - `feat: add pilot market seeds and real seed config`

---

### Task 3: Real registry, fetch and `fetch-market` command

**Files:**
- Create: `src/pm_traitbench/market/real/sources.py`, `src/pm_traitbench/market/real/fetch.py`
- Modify: `src/pm_traitbench/cli.py` (new subcommand), `src/pm_traitbench/tables/store.py` (`data_dir` property)
- Test: `tests/market/real/conftest.py`, `tests/market/real/test_sources.py`, `tests/market/real/test_fetch.py`, `tests/test_cli_stages.py` (or a new `tests/test_cli_fetch.py`), `tests/conftest.py` (the `--run-network` option and `network` marker)

**Interfaces:**
- Produces in `sources.py`:
  ```python
  Source = Literal["fred", "yahoo", "nasdaq"]

  @dataclass(frozen=True)
  class RealInstrument:
      instrument_id: str
      family: Family
      kind: InstrumentKind
      name: str
      currency: str
      source: Source
      series: str                      # FRED id or Yahoo ticker
      sector: str | None = None
      rating_band: RatingBand | None = None
      commodity_group: CommodityGroup | None = None
      spread_base: str | None = None   # FRED id subtracted to form a spread (credit only)

  REAL_INSTRUMENTS: tuple[RealInstrument, ...]   # 40 equities, RT-USD, 2 credit, 15 commodities, 6 USD FX pairs
  DERIVED_FX_PAIRS: tuple[str, ...] = ("EURGBP", "EURJPY", "AUDJPY")
  CURVE_SERIES: dict[Tenor, str] = {Y2: "DGS2", Y5: "DGS5", Y10: "DGS10", Y30: "DGS30"}
  CREDIT_DURATION_YEARS = 13.0        # Moody's seasoned indices hold maturities of 20 years or more
  REFERENCE_EQUITY = "SPY"            # Yahoo ticker for beta, abnormal returns and the driver
  EARNINGS_TICKERS: tuple[str, ...]   # the 40 equity tickers in registry order
  def fred_series() -> list[str]      # every FRED id needed: DGS2/5/10/20/30, DAAA, DBAA, 6 DEX ids
  def yahoo_tickers() -> list[str]    # SPY, 40 equities, 15 futures
  ```
  Binding registry values: equity ids `EQ-R001..EQ-R040` in this ticker order: AAPL MSFT INTC CSCO JNJ PFE MRK UNH JPM BAC WFC GS AMZN HD MCD NKE PG KO PEP WMT XOM CVX COP SLB BA CAT HON UNP APD ECL NEM SHW NEE DUK SO D VZ T DIS CMCSA; sector `sector_01` for the first four, `sector_02` for the next four, and so on to `sector_10`; name equals the id; currency `USD`. Credit: `CR-R-IG-001` (`DAAA`, band `AA`) and `CR-R-IG-002` (`DBAA`, band `BBB`), both `spread_base="DGS20"`, sector `sector_01`, name equals id, currency `USD`. Curve: `RT-USD` (`family` rates, `kind` sovereign_curve, source fred, series `DGS10` as the representative). Commodities: `CM-<code>` with the synthetic names and groups, Yahoo tickers `CL=F BZ=F HO=F RB=F GC=F SI=F PL=F HG=F ZW=F ZC=F ZS=F SB=F KC=F CT=F CC=F` for crude, brent, heating_oil, gasoline, gold, silver, platinum, copper, wheat, corn, soybeans, sugar, coffee, cotton, cocoa. FX: `FX-EURUSD DEXUSEU`, `FX-GBPUSD DEXUSUK`, `FX-USDJPY DEXJPUS`, `FX-AUDUSD DEXUSAL`, `FX-USDCHF DEXSZUS`, `FX-USDCAD DEXCAUS`, currency = quote currency, name = pair.
- Produces in `fetch.py`:
  ```python
  Opener = Callable[[str], bytes]
  def urlopen_bytes(url: str) -> bytes          # urllib, browser-like User-Agent, 30 s timeout
  def cache_dir(data_dir: Path) -> Path          # data_dir / "raw" / "market"
  def fred_url(series: str, start: date, end: date) -> str
  # https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>&cosd=<start>&coed=<end>
  def yahoo_url(ticker: str, start: date, end: date) -> str
  # https://query1.finance.yahoo.com/v8/finance/chart/<ticker>?period1=<unix start>&period2=<unix end + 1 day>&interval=1d
  def nasdaq_url(day: date) -> str
  # https://api.nasdaq.com/api/calendar/earnings?date=<YYYY-MM-DD>
  def fetch_range(config: Config) -> tuple[date, date]
  # union over real seeds of (burn-in start minus REAL_FETCH_BUFFER_DAYS weekdays, window end)
  def fetch_all(config: Config, data_dir: Path, *, force: bool = False, opener: Opener = urlopen_bytes) -> Manifest

  @dataclass(frozen=True)
  class ManifestEntry: path: str; url: str; retrieved_at: str; sha256: str; size: int
  @dataclass(frozen=True)
  class Manifest:
      window: tuple[str, str]
      entries: tuple[ManifestEntry, ...]
      def to_json(self) -> str / classmethod from_json

  class RawCache:
      @classmethod
      def open(cls, data_dir: Path) -> "RawCache"   # loads manifest, verifies every sha256
      def fred(self, series: str) -> dict[date, float]         # skips "." values
      def yahoo(self, ticker: str) -> dict[date, float]        # adjclose by date (UTC date of timestamp)
      def nasdaq_symbols(self, day: date) -> set[str]          # symbols reporting that day
      @property
      def manifest(self) -> Manifest
  ```
  File layout: `fred/<series>.csv`, `yahoo/<ticker>.json` (ticker `=` kept as is), `nasdaq/<YYYY-MM-DD>.json`, `manifest.json`. Nasdaq is fetched for every weekday from the first real seed's `window_start` to the window end (not the burn-in). Requests are spaced `REAL_REQUEST_INTERVAL_S` apart (sleep is injectable too: `sleeper: Callable[[float], None] = time.sleep`). On HTTP 429 or 5xx (urllib `HTTPError`) retry up to `REAL_RETRIES` times with backoff `interval * 2**attempt`, then raise `StageIOError` naming the series and the status. Any other `HTTPError` or `URLError` raises `StageIOError` at once. Validation before writing a file: FRED body starts with `observation_date,`; Yahoo JSON has `chart.result[0].timestamp` and `indicators.adjclose[0].adjclose` of equal length; Nasdaq JSON has `data.rows` (a list, possibly empty). Malformed bodies raise `StageIOError` naming the series. Without `force`, an existing file whose manifest sha256 still matches is kept and not refetched; with `force` all are refetched. The manifest is rewritten atomically (tmp then `os.replace`) after all fetches.
  `RawCache.open` raises `StageIOError("no raw market cache at <dir>; run fetch-market first")` when the manifest is missing and `StageIOError` naming the path on a sha256 mismatch or a missing file.
- `DataStore.data_dir` property.
- CLI: `fetch-market` subparser with `--config`, `--data-dir` (default `data`), `--force`; `main` routes it to `fetch_all` and prints a one-line summary (`fetched N files into <dir>`). Stage subcommands are unchanged. `build_parser` keeps rejecting duplicate stage names.
- Test infrastructure: `tests/conftest.py` adds `--run-network` and a `network` marker that skips unless the flag is passed (register the marker in `pyproject.toml`). `tests/market/real/conftest.py` provides `fake_cache(tmp_path, config) -> Path` fixture factory: writes a complete cache for the config's real seeds' fetch range with deterministic random walks (`np.random.default_rng(0)`), one US holiday gap (a weekday with no FRED and no Yahoo row), Yahoo timestamps at 13:30 UTC, and Nasdaq files where each equity ticker reports once per calendar quarter (one of them on a Saturday); then writes a valid manifest through `fetch.py`'s own manifest writer so hashes match. It also exposes `fake_opener(files: dict[str, bytes])`.

- [ ] **Step 1: Write failing tests:**
  - `test_sources.py`: counts per family (40, 1, 2, 15, 6), 10 sectors of 4, id formats, `fred_series()` has 13 ids, `yahoo_tickers()` has 56, no duplicate ids or series.
  - `test_fetch.py`: URLs match the binding forms; `fetch_range` starts 60 + 10 weekdays before `window_start` and ends at the window end; `fetch_all` with a fake opener writes every file and a manifest whose sha256 values match the files; second call without `force` performs no requests (opener call count 0); with `force` refetches all; a fake opener raising `HTTPError(429)` twice then succeeding gives one file and 3 calls with sleeper called with backoff values; five 503s raise `StageIOError` naming the series; malformed FRED/Yahoo/Nasdaq bodies raise naming the series; `RawCache.open` fails on a missing manifest with "run fetch-market first", on a tampered file naming its path; `fred()` skips `.`; `yahoo()` maps timestamps to dates; `nasdaq_symbols()` returns the set.
  - CLI test: `main(["fetch-market", "--data-dir", tmp])` with the opener monkeypatched to the fake returns 0 and creates the manifest; a bad URL path returns exit code 1 via `StageIOError`.
- [ ] **Step 2: Run** `uv run pytest tests/market/real -q`. Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add real market registry and fetch-market command`

---

### Task 4: Real universe and events

**Files:**
- Create: `src/pm_traitbench/market/real/universe.py`, `src/pm_traitbench/market/real/events.py`
- Test: `tests/market/real/test_universe.py`, `tests/market/real/test_events.py`

**Interfaces:**
- Consumes: `REAL_INSTRUMENTS`, `RawCache`, `SimAxis`, `RealSeedSpec`, `CalendarEvent`, `generated_rows`, `row_sort_key`, `Config`.
- Produces in `universe.py`:
  ```python
  def real_axis_dates(spec: RealSeedSpec, axis: SimAxis, calendar_start: date) -> list[date]
  # real_date(t) = spec.window_start + (axis.dates[t] - calendar_start), one per axis day
  def build_real_universe(config: Config, cache: RawCache, axis: SimAxis) -> list[Instrument]
  ```
  Order: equities, credit, curve, commodities, FX (6 USD pairs then the 3 derived pairs, `FX-EURGBP`, `FX-EURJPY`, `FX-AUDJPY`, family fx, kind fx_pair, currency = quote). Equity `beta` = OLS slope of the equity's daily log return on SPY's over the union of the real seeds' real axis dates (values aligned by real date, both series forward-filled by the alignment helper of Task 5, but implement the fill here as a small shared function `aligned_series(values: dict[date, float], dates: Sequence[date]) -> tuple[np.ndarray, int]` returning the array and the longest fill run, raising `StageIOError` naming the series when day 0 has no value within the buffer or the longest run exceeds `REAL_FILL_LIMIT`). Round beta to 4 decimals. Credit `duration_years = CREDIT_DURATION_YEARS`, `rating_band` from the registry, `sector` from the registry. Commodities `expiry_rule = config.market.universe.expiry_rule`. Put `aligned_series` in `market/real/align.py` (create it) so both universe and build use it.
- Produces in `events.py`:
  ```python
  FOMC_DATES: tuple[date, ...]     # 2018-06-13, 2018-08-01, 2018-09-26, 2018-11-08, 2018-12-19, 2019-01-30, 2019-03-20, 2019-05-01 (Federal Reserve meeting calendar)
  WASDE_DATES: tuple[date, ...]    # 2018-06-12, 2018-07-12, 2018-08-10, 2018-09-12, 2018-10-11, 2018-11-08, 2018-12-11, 2019-02-08, 2019-03-08, 2019-04-09, 2019-05-10 (USDA WASDE archive; no January 2019 report because of the government shutdown)
  NFP_DATES: tuple[date, ...]      # 2018-07-06, 2018-08-03, 2018-09-07, 2018-10-05, 2018-11-02, 2018-12-07, 2019-01-04, 2019-02-01, 2019-03-08, 2019-04-05, 2019-05-03 (BLS Employment Situation schedule)

  @dataclass(frozen=True)
  class RealEvent: instrument_id: str | None; event: EventType; day: int   # axis index

  def real_event_days(instruments, cache, spec, axis, calendar_start) -> list[RealEvent]
  # earnings: for each horizon weekday d, Nasdaq symbols reporting on real_date(d) that are registry tickers;
  #   a reporting date on a non-axis day (a weekend) moves to the next axis weekday; a holiday weekday is an axis day and stays
  # cb_meeting: FOMC dates that fall in the real horizon, for RT-USD
  # inventory_report: every horizon Wednesday, per energy commodity
  # crop_report: WASDE dates in the horizon, per agriculture commodity
  # macro_print: NFP dates in the horizon, instrument None
  # no rows for burn-in days
  def surprise_rows(events, output: ProcessOutput, spy_log_return: np.ndarray, betas: Mapping[str, float],
                    y10_bp: np.ndarray, axis, seed, config) -> list[CalendarEvent]
  ```
  Binding surprise formulas (`jump_size` from `config.market.events[event].jump_size`, `clip` to [-1, 1]; positive = good for the price):
  - earnings: `(r_i[t] - beta_i * r_SPY[t]) / jump_size` with log returns from `output.prices`
  - cb_meeting: `-(y10_bp[t] - y10_bp[t-1]) / jump_size`
  - inventory_report, crop_report: `r_commodity[t] / jump_size`
  - macro_print: `(r_SPY[t] / (market_vol / sqrt(252))) / jump_size`
  `affected` is the instrument's family value or `"all"` for macro. Rows sorted with `row_sort_key`. A `drawn_counts(rows) -> dict[EventType, int]` helper counts rows per type (rating types map to 0).

- [ ] **Step 1: Write failing tests** (fixture cache):
  - `test_universe.py`: 67 instruments in order; ids and kinds; every equity has a beta near the fixture's known slope (build the fixture equities as `beta * spy + noise` with known betas so the test can assert within 0.05); credit duration 13; derived FX pairs present with quote currency; `real_axis_dates` maps `calendar.start` to `window_start` and preserves weekday; `aligned_series` fills a gap and reports the longest run, and raises on a run above the limit and on a missing day 0.
  - `test_events.py`: every fixed date list is ascending and inside 2018-06-04..2019-05-31; earnings rows only for registry tickers, a report on the fixture's holiday weekday stays on that day, a weekend date (hand-built) moves to Monday, none in burn-in; one cb row per FOMC date in the horizon; Wednesdays for each energy commodity; WASDE rows for agriculture only; NFP rows with `instrument_id None` and `affected "all"`; surprise sign and clipping per type on hand-built arrays (a +10% earnings day with beta 1 and SPY +1% gives `clip(0.09/0.05) = 1.0`; a 10Y up 4bp FOMC day gives -0.5); `drawn_counts` has zeros for rating types.
- [ ] **Step 2: Run** `uv run pytest tests/market/real -q`. Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement.** The implementer verifies the WASDE list against the USDA archive (a web lookup is allowed for this) and records the check in the report.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add real market universe and event calendar`

---

### Task 5: Real seed builder

**Files:**
- Create: `src/pm_traitbench/market/real/build.py`
- Modify: `src/pm_traitbench/market/real/align.py` (if helpers are needed beyond Task 4's)
- Test: `tests/market/real/test_build.py`

**Interfaces:**
- Consumes: `aligned_series`, `real_axis_dates`, `real_event_days`, `surprise_rows`, `drawn_counts`, `generated_rows`, `event_day_indices`, `build_consensus`, `RegimeLookup`, `regime_path`, `SeedMarket`, `ProcessOutput`, `COMMODITIES`, `USD_PAIR`, `FX_PAIRS`, `stream`.
- Produces:
  ```python
  def build_seed(config: Config, seed: str, instruments: Sequence[Instrument],
                 cache: RawCache, axis: SimAxis) -> SeedMarket
  def real_schedule(spec: RealSeedSpec, axis: SimAxis, calendar_start: date, seed: str) -> list[RegimeSpan]
  # date_start = mapped regime start; date_end = day before the next mapped start, or the last horizon date
  ```
  Binding rebasing (axis day 0 is the burn-in start; every series keeps its real log changes):
  - equities: `p = p_real * start / p_real[0]`, `start = exp(uniform(log lo, log hi))` on `stream(root, "market", "real", "level", instrument_id)` with `levels.equity_price_range`
  - commodities: `M1 = p_real * start_code / p_real[0]` with `start` from `COMMODITIES`; `M_k = M1 * (1 + slope_g * (k-1)/12)` for k = 2..12, `price = M1`
  - FX: for each non-USD currency `v_real = ±log P_real` (plus for XXXUSD, minus for USDXXX); `v = v_real - v_real[0] + v_start` where `v_start` is from `levels.fx_start`; pair price `exp(v_base - v_quote)` for all 9 pairs
  - curves: `shift = curve_start["USD"][10Y index] - y10_real[0]`; each tenor `max(y_real + shift, yield_floor_pct)`
  - credit: `s = s_real_bp * base / s_real_bp[0]`, `s_real_bp = 100 * (yield - DGS20)`, `base = credit_base_spread_bp[band]`; price from 100 with `p_t = p_{t-1} * (1 - D * ((s_t - s_{t-1}) + 100 * (y5_t - y5_{t-1})) / 10000)` using the rebased USD 5Y
  - driver: `z = r_SPY / std(r_SPY)` with `r_SPY` the SPY daily log return on the axis (`z[0] = 0`)
  - `path = regime_path(axis, RegimeLookup(schedule, spec.regime_starts[0][0]), config)`
  - calendar = `surprise_rows(...)` + `generated_rows(...)` + consensus flips, sorted
  - `fills` = longest fill run per series id (instrument id, or `SPY`, or the FRED id for DGS20)
  - `drawn_events = drawn_counts(surprise rows)`

- [ ] **Step 1: Write failing tests** (fixture cache, demo-sized synthetic config not needed): row counts via `to_rows` (prices 66 x 260, curves (4 + 12 x 15) x 260, consensus 67 x 260, regimes 3); every rebased series' log changes equal the real log changes to 1e-12 (equities, commodities M1, USD pairs, credit spreads); day-0 levels as configured (equity within `equity_price_range`, crude 72, EURUSD 1.10, USD 10Y 4.1, AA spread 50); crosses consistent; curve shape preserved (tenor differences equal the real ones); credit price falls on a day when both spread and 5Y rise; `z` has unit sd and `z[0] == 0`; schedule dates equal the mapped regime starts and tile the horizon; burn-in maps to the first regime; `fills` reports the holiday gap as 1 for FRED-sourced series; every calendar event date is a horizon date; the same inputs give identical output twice; seeds `R1` and a second real seed with the same window differ only in flip rows.
- [ ] **Step 2: Run.** Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: build real market seeds from the raw cache`

---

### Task 6: Real data-quality check

**Files:**
- Create: `src/pm_traitbench/market/real/check.py`
- Test: `tests/market/real/test_check.py`

**Interfaces:**
- Consumes: `SeedMarket`, `CheckMetric`, `CheckReport`, `count_metrics`, `check_error`, `family_indices`, `REAL_FILL_LIMIT`.
- Produces: `check_real_market(market: SeedMarket, instruments: Sequence[Instrument], config: Config) -> CheckReport`.
  Pass/fail metrics (regime `None`, family from the instrument, `metric` names in quotes):
  - `"finite"`: every price, spread and curve level finite; realised = count of non-finite values, target 0
  - `"positive"`: prices and spreads > 0; yields >= `yield_floor_pct`
  - `"fill_run"`: per series in `market.fills`, realised = longest run, target `REAL_FILL_LIMIT`, passed when realised <= target
  - the shared `count_metrics` (expiry and positioning dates, flips per instrument, drawn event counts)
  - `"earnings_per_quarter"`: per equity, at most one earnings row per calendar quarter of mapped dates
  Reported only (`passed=True`, tolerance 0): per regime span and family, `"vol"` realised annualised sd of the family index and `"corr"` with `z`, using `family_indices` and the same span slicing as the synthetic check (changes over `a..b`, correlated with `z[a..b]`), target set equal to realised.
  A miss raises through `check_error`.

- [ ] **Step 1: Write failing tests:** passes on the fixture build; a NaN injected into one price series (use `dataclasses.replace` on the `SeedMarket` output) fails naming `finite` and the series; `fills` set above the limit fails naming `fill_run`; a moved expiry date fails; a second earnings row in the same quarter fails; the report has vol and corr entries for every regime and family with `passed=True`; `to_dict` round-trips through `json.dumps`.
- [ ] **Step 2: Run.** Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add real market data-quality check`

---

### Task 7: Stage dispatch, demo config, README, real run

**Files:**
- Modify: `src/pm_traitbench/market/stage.py`, `configs/demo.yaml`, `README.md`
- Test: `tests/market/test_stage.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `synthetic.build.build_seed`, `synthetic.check.check_market`, `real.build.build_seed`, `real.check.check_real_market`, `build_real_universe`, `RawCache.open`, `DataStore.data_dir`, `referenced_seeds`.
- Produces: `run` builds the synthetic universe if any referenced seed is synthetic and the real universe (opening the cache first) if any is real; instruments = synthetic list + real list, with a shared id (commodities, FX, `RT-USD`) kept once when the two `Instrument` rows are equal and raising `MarketCheckError(f"instrument '{id}' differs between real and synthetic universes")` otherwise. Each seed: real -> `real.build.build_seed` + `check_real_market`; synthetic -> as now. Extras: `{"check": {seed: report}, "raw_manifest": {"window": [...], "files": n}}` (the manifest key only when a real seed was built). Every seed is built and checked before any write.
- `configs/demo.yaml`: `pilot_market_seeds: [R1]`, `market_seeds: [A]`, the small synthetic universe; comment that the demo needs `fetch-market` first.
- README: `fetch-market` usage before `market` when a real seed is configured; the sources list (FRED for Treasuries, Moody's and FX; Yahoo Finance chart API for equities, futures and SPY; Nasdaq earnings calendar; Federal Reserve, USDA and BLS dates); raw data is not redistributed and the manifest records what was fetched; a "Real market" subsection with the disguise rule and the limitations from the design: USD rates only; IG credit only, two index series without issuer noise; commodity M2-M12 scripted and continuous front months jump at their own roll dates, unlike the third-Friday expiry rows; holiday fills; EIA holiday shifts ignored; surprises from price reaction; the disguise hides levels, names and dates but not the return pattern. Also the config recipe for running the full split on real seeds.

- [ ] **Step 1: Write failing tests:** with the fixture cache and a config `pilot [R1], full [A]` (small synthetic universe): six tables written, instruments contain both `EQ-R001` and `EQ-0001` and exactly one `CM-CRD`; prices have both seeds; `run_metadata/market.json` has `check.R1`, `check.A` and `raw_manifest`; a missing cache with a real seed referenced fails before any write; a tampered cache fails naming the file; the demo config with the fixture cache runs through `main([...])` and returns 0; a synthetic-only config writes no `raw_manifest`.
- [ ] **Step 2: Run.** Expected: FAIL (the interim `StageIOError` branch and missing keys).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Real run.** From the worktree: `uv run pm-traitbench fetch-market --data-dir data`, then `uv run pm-traitbench sample --data-dir data` and `uv run pm-traitbench market --data-dir data` at default config. Record in the report: fetch wall time and file count, any retries, stage wall time, row counts per table, and R1's reported vol and corr per regime and family. If the real data fails a data-quality check, report the metric and stop; do not loosen a limit. `data/` stays untracked.
- [ ] **Step 6: Commit** - `feat: dispatch market seeds to real or synthetic builders`
