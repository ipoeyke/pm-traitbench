**Tier:** heavy
**Escalation threshold:** n/a (heavy)

# Design: real market for the pilot

Date: 2026-09-23. Builds on `docs/specs/2026-09-22-market-design.md` (the synthetic market stage, built on branch `feat/market-stage-alt`). Branch for this work: `feat/real-market-pilot`.

## Scope

The pilot split uses a real historical market instead of a synthetic one, so the pilot can finish sooner. The synthetic stage stays for the full split. Both sources write the same six `market/` tables, and no consumer can tell them apart.

In scope: a restructure of `market/` into shared, `synthetic/` and `real/`; config changes for pilot seeds and real seeds; a `fetch-market` CLI command with a manifest-verified raw cache; a real-market registry, builder and data-quality check; per-seed dispatch in the market stage; tests; README.

Out of scope: CFTC positioning data, high-yield credit, non-USD curves, any consumer of the tables, changes to the synthetic model.

## Repo rule that binds all code

No file in the repo may mention `docs/`, a spec or a plan. Docstrings state the rule itself. Citing a public data source or paper as the basis of a value is allowed. Raw market data is never committed.

## Decisions taken

1. **Disguise, keep returns.** Real daily returns are kept; dates are remapped onto the configured calendar, equities and credit get abstract ids, and every series is rebased to the synthetic starting levels. Reason: a model under test may recognise a real period from levels, names or dates and know what came next.
2. **Window.** R1 is 2018-06-04 to the end of the configured horizon (52 weeks), with regimes range from 2018-06-04, risk_off from 2018-10-01, risk_on from 2018-12-26. Reason: the phases match seed A's order (calm range +7%, Q4 selloff -19%, rebound +12%) and the year is less memorable than 2020 or 2022.
3. **Fetch plus local cache.** A CLI command downloads raw series into `<data-dir>/raw/market/` (gitignored) with a manifest of URL, retrieval time and sha256. The stage builds offline from the cache and refuses a mismatched checksum. Reason: Yahoo's terms and ICE copyrights forbid redistribution; the manifest keeps runs reproducible and auditable.
4. **Narrow to real.** Only instruments with a free daily source are included; nothing idiosyncratic is synthesised. Commodity tenors M2-M12 are scripted from the synthetic slopes, the one exception, because the synthetic stage already documents its curve as scripted.
5. **Separate pilot seeds.** `population.pilot_market_seeds` replaces `pilot_market_seed_count`; the full split keeps `market_seeds`. The market stage builds every referenced seed, dispatching by where the seed is defined.
6. **Credit is IG only.** FRED serves ICE BofA OAS only from 2023-09, so the 2018 window uses Moody's Aaa and Baa daily yields minus the 20Y Treasury: two instruments, bands AA and BBB, no high yield.
7. **Surprises come from price reaction**, uniform across event types, so a calendar row always agrees with the price series it sits on.
8. **Symmetric builders.** `synthetic/build.py` and `real/build.py` each expose `build_seed(...) -> SeedMarket`; everything downstream of a `SeedMarket` is shared.

## Package layout

```
src/pm_traitbench/market/
  __init__.py
  stage.py          # MARKET_STAGE: per-seed dispatch, writes the six tables
  axis.py           # SimAxis, build_axis (unchanged)
  constants.py      # commodity and FX tables, USD_PAIR, CREDIT_BAND_ORDER,
                    #   largest_remainder, day counts (unchanged)
  regimes.py        # RegimeLookup, RegimePath, regime_path, constant_path
  output.py         # ProcessOutput (moved from processes/common.py)
  calendar.py       # RngFor, row_sort_key, third_friday, generated_rows,
                    #   event_day_indices (sampling moves out)
  consensus.py      # unchanged
  seed.py           # SeedMarket, MarketRows, to_rows, market_rng (from generate.py)
  check.py          # CheckMetric, CheckReport, date and count checks, raise helper
  synthetic/
    __init__.py
    universe.py     # build_universe (moved)
    schedule.py     # build_schedule from boundary_weeks (split from regimes.py)
    drivers.py      # moved
    events.py       # sample_events, SampledEvents, EVENT_TARGETS, EventJumps,
                    #   build_jumps (split from calendar.py)
    build.py        # build_seed (was generate_seed)
    check.py        # implied moments, vol, corr, round-level checks (split)
    processes/      # common.py (ProcessInputs, unit_student_t, grid helpers),
                    #   equities, rates, credit, commodities, fx (moved)
  real/
    __init__.py
    sources.py      # registry of real instruments and reference series
    fetch.py        # download, manifest, cache verification
    universe.py     # Instrument rows from the registry, betas from the cache
    events.py       # real event dates and price-reaction surprises
    build.py        # build_seed: cache -> aligned, disguised SeedMarket
    check.py        # data-quality checks; realised moments reported only
tests/market/            # shared modules
tests/market/synthetic/  # moved tests
tests/market/real/       # new tests, fixture cache in conftest.py
```

Dependency direction: `real/` and `synthetic/` depend on the shared modules and on `config`, `rng`, `timeline`, `enums`, `tables`; neither depends on the other; only `stage.py` touches the store. The restructure is a pure move, committed first with the suite green.

## Config

### `population`

- `pilot_market_seeds: tuple[str, ...] = ("R1",)` (basis design: the pilot runs on a real market). Replaces `pilot_market_seed_count`. Non-empty, no repeats, no blanks.
- `market_seeds: tuple[str, ...] = ("A", "B", "C")` unchanged; it now names the full split's seeds only.
- `build_population` uses `pilot_market_seeds` for the pilot split and `market_seeds` for the full split.

### `market.real`

```
market.real.seeds: dict[str, RealSeedSpec]
RealSeedSpec:
  window_start: date            # a Monday; the real date that maps to calendar.start
  regime_starts: tuple[tuple[Regime, date], ...]
                                # (regime, first real date); the first equals window_start
```

Default: `R1: window_start 2018-06-04, regime_starts ((range, 2018-06-04), (risk_off, 2018-10-01), (risk_on, 2018-12-26))`, basis design with the reason from decision 2.

Validation: `window_start` is a Monday; `regime_starts` dates strictly ascending, the first equal to `window_start`, each regime exactly once, all inside the window (`window_start` plus `calendar.n_weeks` weeks). Other real settings, fixed in code with a note: `fill_limit = 5` consecutive filled days, `fetch_buffer_days = 10`, `request_interval_s = 1.0`, `retries = 5`.

### `Config` cross-checks

- Every seed in `pilot_market_seeds` and `market_seeds` is defined in exactly one of `market.seeds` (synthetic) or `market.real.seeds`.
- A name defined in both is rejected.
- `market.seeds` and `market.real.seeds` may define seeds nobody references (they are not built).

### Running the full split on real markets

This is config only, no code change: define more real seeds (`R2`, `R3`, each with its own window and regime starts), set `population.market_seeds` to them, run `fetch-market` again (it merges the new windows into the cache), then `sample` and `market`. Choosing windows whose phases give the wanted regime orders is the only work. Coverage limits the years: Yahoo futures history and the Nasdaq earnings calendar are thin before about 2008, and every registry ticker must be listed throughout the window. Betas pool across the real seeds' axes by design.

### `configs/demo.yaml`

Pilot `R1`, full `A`, one PM per cell, the small synthetic universe as now. The demo market run needs a fetched cache; `tests/market/test_stage.py` uses the fixture cache.

## Real universe (`real/sources.py`, `real/universe.py`)

The registry is a fixed table; config does not choose instruments. 67 instruments plus one reference series.

| Family | Instruments | Source and series | Ids and facts |
|---|---|---|---|
| Equities | 40 US large caps, 4 per sector: AAPL MSFT INTC CSCO; JNJ PFE MRK UNH; JPM BAC WFC GS; AMZN HD MCD NKE; PG KO PEP WMT; XOM CVX COP SLB; BA CAT HON UNP; APD ECL NEM SHW; NEE DUK SO D; VZ T DIS CMCSA | Yahoo chart API, adjusted close | `EQ-R001..EQ-R040` in registry order; sector `sector_01..sector_10` in the order the groups are listed; `beta` from the cache |
| Rates | USD curve 2Y, 5Y, 10Y, 30Y | FRED DGS2, DGS5, DGS10, DGS30 | `RT-USD`, same facts as synthetic |
| Credit | Moody's Aaa and Baa seasoned yields minus 20Y Treasury, in bp | FRED DAAA, DBAA, DGS20 | `CR-R-IG-001` (band AA), `CR-R-IG-002` (band BBB); sector `sector_01`; `duration_years` 13.0 (guess: the indices hold maturities of 20 years or more); currency USD |
| Commodities | crude, brent, heating_oil, gasoline; gold, silver, platinum; copper; wheat, corn, soybeans, sugar, coffee, cotton, cocoa | Yahoo continuous front months `CL=F BZ=F HO=F RB=F GC=F SI=F PL=F HG=F ZW=F ZC=F ZS=F SB=F KC=F CT=F CC=F` | `CM-<code>` from the synthetic table; identical facts, so shared |
| FX | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD; crosses EURGBP, EURJPY, AUDJPY derived | FRED DEXUSEU, DEXUSUK, DEXJPUS, DEXUSAL, DEXSZUS, DEXCAUS | `FX-<pair>`, identical facts, so shared |
| Reference | SPY | Yahoo | not an instrument; used for beta, abnormal returns and `z` |

Beta: slope of an instrument's daily log return on SPY's over the union of the real seeds' axes, rounded to 4 decimals. The real universe is built once from the cache; with several real seeds the betas pool their axes.

Registry entries carry: `instrument_id`, `family`, `kind`, `name`, `currency`, static tags, `source` (`fred`, `yahoo`, `nasdaq`), `series` (id or ticker), and for derived series the formula key (`spread_over_dgs20`).

Instrument names: equities and credit use the id as name (as synthetic does); commodities and FX keep the real generic names.

## Event dates (`real/events.py`)

Fixed lists in code, each with the public source named in a comment:

- FOMC: 2018-06-13, 2018-08-01, 2018-09-26, 2018-11-08, 2018-12-19, 2019-01-30, 2019-03-20, 2019-05-01 (Federal Reserve meeting calendar).
- WASDE: 2018-06-12, 2018-07-12, 2018-08-10, 2018-09-12, 2018-10-11, 2018-11-08, 2018-12-11, 2019-02-08, 2019-03-08, 2019-04-09, 2019-05-10 (USDA WASDE archive; no January 2019 report because of the government shutdown). The implementer verifies this list against the archive before committing.
- NFP (macro print): 2018-07-06, 2018-08-03, 2018-09-07, 2018-10-05, 2018-11-02, 2018-12-07, 2019-01-04, 2019-02-01, 2019-03-08, 2019-04-05, 2019-05-03 (BLS Employment Situation release schedule).
- EIA weekly petroleum status: every Wednesday in the window; holiday shifts are ignored.
- Earnings: SEC EDGAR 8-K filings with item 2.02 (Results of Operations), from the free submissions API, requested with a declared User-Agent (name and contact email) as SEC's fair-access policy asks. An acceptance time before 16:00 New York time maps to that trading day, later to the next trading day. (The Nasdaq earnings calendar was tried first and dropped: it only answers browser-like clients.)
- Contract expiry and positioning reports: the shared `generated_rows`.
- Rating actions: none.

Dates before `window_start` (burn-in) produce no rows, matching the synthetic stage.

## Fetch (`real/fetch.py`, CLI `fetch-market`)

`pm-traitbench fetch-market --config PATH --data-dir PATH [--force]`. It is a CLI subcommand, not a `Stage`, because it writes raw files rather than tables. `cli.py` gains it alongside the stage subcommands.

- For each real seed: the fetch range is the burn-in start minus `fetch_buffer_days` weekdays through the window end. Ranges of several seeds are merged per series.
- The cache lives inside the same data dir as the tables, `data/` by default for every command: `<data-dir>/raw/market/<source>/<series>.<ext>` (`csv` for FRED, `json` for Yahoo and Nasdaq per date). `<data-dir>/raw/market/manifest.json` lists each file's relative path, URL, `retrieved_at`, `sha256` and byte size, plus the `window` fetched.
- Without `--force`, existing files in the manifest are kept; with it, everything is refetched.
- HTTP: `urllib` with a browser-like user agent, one request per `request_interval_s`, up to `retries` attempts on 429 and 5xx with exponential backoff, then a `StageIOError` naming the series. Malformed responses (not CSV, missing chart result, missing rows) raise `StageIOError` naming the series. The opener is injectable for tests.
- `RawCache.open(data_dir)` loads the manifest and verifies every sha256; a missing manifest or a mismatch raises `StageIOError` with "run fetch-market first" or the mismatched path.
- The Yahoo response gives `timestamp` and `indicators.adjclose[0].adjclose`; FRED CSV gives `observation_date,<id>` with `.` for missing; Nasdaq gives `data.rows[].symbol`.

## Build (`real/build.py`)

`build_seed(config, seed, instruments, cache, axis) -> SeedMarket`.

Axis: the caller's `SimAxis` (calendar dates). The real axis is the same number of weekdays ending at the window end: `real_date(t) = window_start + (axis.dates[t] - calendar.start)`, a constant day offset, so weekdays are preserved. Sources are read for those real dates.

Alignment: a real weekday with no value takes the previous value. More than `fill_limit` consecutive fills, or no value on axis day 0 (searching back through the buffer), fails the build with the series named.

Rebasing (all on axis day 0; every series keeps its real log changes):

- Equities: `p = p_real * start / p_real[0]`, `start = exp(U(log lo, log hi))` on `stream(root, "market", "real", "level", instrument_id)` with the configured `equity_price_range`.
- Commodities: to the synthetic start table (`COMMODITIES`).
- FX: currency values `v_c = ±log(P_real)` rebased so each USD pair equals `fx_start` on day 0; crosses from the rebased values.
- Curves: one additive shift `curve_start[USD][10Y] - y10_real[0]` applied to all four tenors, then floored at `yield_floor_pct`.
- Credit: `s = s_real * base / s_real[0]` with `base = credit_base_spread_bp[band]`; price from 100 through the synthetic linear mark `p_t = p_{t-1} (1 - D (Δs + 100 Δy5)/10000)` with the rebased 5Y.
- Commodity curve: the Yahoo front month is `M1`; `M_k = M1 * (1 + slope_g (k-1)/12)` for k = 2..12 from `commodity.curve_slope`, no noise; `price = M1`.
- `SeedMarket` gains `fills: Mapping[str, int]` (longest fill run per series, default empty) so the check can read it; synthetic seeds leave it empty.

Surprises (revised after the first real run, where dividing a whole day's move by the synthetic event size capped half the inventory rows at ±1): `clip(move / (3 * sd), -1, 1)`, where `sd` is the sample sd of that series' daily moves over the seed's full axis, so a 1-sd day scores about 0.33 and only moves beyond 3 sd reach ±1. The moves:

- earnings: `r_i - beta_i * r_SPY` (log returns);
- cb_meeting: `-(Δy10 in bp)`;
- inventory_report, crop_report: the commodity's log return;
- macro_print: `r_SPY / (market_vol / sqrt(252))`.

Rows carry `affected` as in synthetic. `drawn_events` counts rows per type; `EventJumps` is not used (the moves are already in the prices).

Driver: `z = r_SPY / sd(r_SPY)` over the full axis. `RegimePath` from `market.regimes` via the mapped spans, burn-in in the first regime. Schedule: `RegimeSpan(seed, regime, mapped start, day before the next start or the horizon end)`.

Consensus: `build_consensus` unchanged, on `event_day_indices` of the real rows.

## Check (`real/check.py`)

`check_real_market(market, instruments, config) -> CheckReport` using the shared `CheckMetric` and `CheckReport`:

Pass/fail metrics: finite and positive prices and spreads; yields at or above the floor; fill count per series at most `fill_limit` (from the builder's fill log carried on the `SeedMarket` as `fills: dict[str, int]`, the longest run); exact expiry and positioning dates and flip counts (shared checks); every event date on its mapped horizon day; one earnings row per equity per quarter at most.
Reported only (`passed=True`, tolerance 0): realised annualised vol and correlation with `z` per regime and family, using the shared index definitions from `synthetic/check.py` moved to the shared `check.py` (`family_indices`).

A miss raises `MarketCheckError` through the shared raise helper.

## Stage (`market/stage.py`)

- Seeds to build: `pilot_market_seeds` followed by `market_seeds`, in that order, without repeats.
- Instruments: the synthetic universe (only if a synthetic seed is referenced) plus the real universe (only if a real seed is referenced), concatenated; the shared ids (commodities, FX, `RT-USD`) are added once. A conflicting duplicate id (same id, different facts) raises `MarketCheckError`.
- Real seeds need `RawCache.open(data_dir)`; the store exposes `data_dir` for it.
- Every seed is built and checked before any table is written; extras `{"check": {seed: report}}` as now, plus `{"raw_manifest": manifest summary}` when a real seed was built.

## README

- Usage: `fetch-market` before `market` when a real seed is configured.
- Sources: FRED (Treasuries, Moody's, FX), Yahoo Finance chart API (equities, futures, SPY), Nasdaq earnings calendar, Federal Reserve, USDA, BLS. Raw data is not redistributed; the manifest records what was fetched.
- Limitations: USD rates only; IG credit only, two index series with no issuer noise; commodity M2-M12 scripted, and continuous front months jump at their own roll dates, which differ from the third-Friday expiry rows; holiday fills; EIA holiday shifts ignored; surprises from price reaction; the disguise hides levels, names and dates but not the return pattern.

## Testing

No test needs the network by default.

- `tests/market/real/conftest.py`: a fixture that writes a fake cache (FRED CSV, Yahoo JSON, Nasdaq JSON per date, manifest) for a short window with deterministic random-walk data, including a US holiday gap and one weekend earnings date.
- `test_sources.py`: registry counts (40, 1, 2, 15, 9), id format, sector labels, one source per instrument.
- `test_fetch.py`: injectable opener; manifest fields and sha256; keep-versus-force; retry on 429 and 5xx then failure naming the series; malformed CSV and JSON failures; `RawCache.open` on a missing manifest and on a mismatch.
- `test_universe.py`: betas from the fixture; names and currencies.
- `test_events.py`: fixed lists in the window; earnings filtered and weekend-shifted; surprise sign and clipping per type; no burn-in rows.
- `test_build.py`: weekday mapping; holiday fill and the fill limit; day-0 rebasing per family; returns unchanged after rebasing; crosses consistent; curve shape preserved; credit price falls when spread and 5Y rise; `z` unit sd; spans at mapped dates; row counts per table.
- `test_check.py`: passes on the fixture; fails naming the series on a NaN, a fill breach, a moved expiry.
- `tests/market/test_stage.py`: pilot R1 plus synthetic A from the fixture cache; union instruments; missing cache and mismatch fail before any write; the demo config path with the fixture.
- `tests/test_config.py`: `pilot_market_seeds`; the both-defined and undefined seed rejections; `regime_starts` validation.
- `tests/sampling/test_population.py`: pilot slots carry `R1`.
- One test marked `network`, skipped without `--run-network`: fetches DGS10 and SPY for a week and checks shape.
- The moved synthetic tests are unchanged apart from imports.

## Delivery order

1. Restructure `market/` (pure move, suite green).
2. Config and population changes; stage dispatch with only synthetic seeds still working.
3. Real registry, fetch and CLI command.
4. Real events, build, check; stage integration; README.
5. A real `fetch-market` into `data/`, then `sample` and `market` at default config; record wall time, row counts and R1's reported realised moments. `data/` is gitignored.
