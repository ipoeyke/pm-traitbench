# Market Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pipeline stage 2, `market`: a synthetic, regime-scripted market per configured market seed, written as six tables under `market/`, checked against its own regime parameters.

**Architecture:** Pure numpy simulation on a daily axis (60 burn-in weekdays plus the 260-weekday horizon). A seed-independent universe and driver shocks are built once. Per seed, a regime path plus the calendar drive five family processes, then consensus. A check compares realised moments with model-implied targets and fails the stage on a miss. Only `market/stage.py` touches the `DataStore`.

**Tech Stack:** Python 3.13, numpy, pydantic v2, pyarrow, pytest, ruff, uv.

## Global Constraints

- Worktree: `/Users/ianpoey/code/github/pm-traitbench-market-alt`, branch `feat/market-stage-alt`. Run every command from there with `uv run`.
- No file in the repo (code, docstrings, comments, tests, README, commit messages) may mention `docs/`, a spec or a plan. Docstrings state the rule itself. Citing a published paper or a named public data series (for example "FRED DGS10") as the basis of a parameter is allowed.
- Never commit anything under `docs/`.
- Commit messages: one Conventional Commits subject line, no body, then a blank line and the trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. No other trailer.
- Plain dash `-`, never an em dash, in all text. No filler words ("genuinely", "actually", "truly", "really").
- Docstrings concise; inline comments 1-2 lines. Match surrounding style: frozen pydantic models with `extra="forbid"`, `Field(description=...)` on every row-model field, `json_schema_extra={"basis": ..., "note": ...}` on every config leaf.
- Every config leaf must carry `basis` (`sourced`, `design` or `guess`) and a non-empty `note`; `Config.dump_with_basis()` must keep working.
- Randomness only through `pm_traitbench.rng.stream(root_seed, *keys)`. Market streams are `stream(config.seed.root, "market", *keys)`. Seed-independent draws never include the market seed in their key; seed-dependent draws (surprises, flip dates) always do.
- Lint and tests green after every task: `uv run ruff check && uv run ruff format --check && uv run pytest -q`.
- A realised-moment check failing at default config is a design problem, not a tolerance to loosen: if it happens, stop and report the numbers (status BLOCKED) instead of changing a threshold.

## Decisions made while planning (binding, differ from or refine the design)

1. **Correlation check is model-implied.** The check computes each family index's model-implied correlation `rho*` with the driver `z` and fails when `|realised - rho*| > corr_tolerance_se * (1 - rho*^2) / sqrt(n)`. Config field `market.check.corr_tolerance_se = 4.0` replaces `min_driver_corr`. Reason: at default loadings some index correlations are about 0.2, so an absolute floor of 0.2 fails half the time on 70-110 day spans.
2. **Round-level check is a family mean, counting crossings.** A test is a close-to-close step where the previous close is outside the band of every grid level and the current close is inside a band, or where the two closes lie outside bands on opposite sides of a grid level. The family mean of tests per instrument in the range span must be at least `min_round_level_tests`, for equities, commodities, FX pairs and sovereign 10Y yields. Reason: with daily closes, a move larger than the band skips it while the path still crosses the level; and a per-instrument minimum fails some low-priced name every run.
3. **Vol targets are model-implied per family index** (formulas in Task 12), since families with several vols (commodities, FX) have no single "vol field".
4. **Credit check index uses IG issuers only** (rating bands AA, A, BBB, loading 1 on the factor), so its vol and correlation are the factor's, not a mix with the HY loading.
5. **Regime config is three per-quantity maps** (`driver_mean`, `vol_multiplier`, `mean_reversion_kappa`, each `dict[Regime, float]` with its own basis), exposed as `RegimeParams` through `MarketRegimesConfig.params(regime)`. Reason: the three quantities have different bases and the basis walker needs one basis per leaf.
6. **Surprise sign convention:** `surprise > 0` means good news for the instrument's price. Equities and commodities: log price up. Credit: spread tightens. Curves: bond prices up, so yields fall. Macro: risk-on (adds to `z`). `rating_downgrade` surprises are negative, `rating_upgrade` positive.
7. **Starting curve reproduced exactly:** `y_k = L + w_k S + o_k` with constant per-tenor offsets `o_k` (the fixed curvature) fitted so day 0 equals `curve_start`.
8. **Commodity round-level pull targets the deterministic front month** `spot * (1 + slope/12)`, so the published `price` (M1) anchors at round levels.
9. **FX pull acts on each non-USD currency through its USD pair** (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD); crosses have no own pull.
10. New helper modules beyond the design's layout: `market/axis.py` (simulation axis), `market/processes/common.py` (shared process types and helpers), `market/generate.py` (per-seed generation and row conversion). The "sign of mean return follows `driver_corr x driver_mean`" test lives in `tests/market/test_generate.py` because it needs every process.
11. Seed-equality test: two seeds with the same regime order produce identical prices, curves and consensus when event jump sizes are 0 and flips are off (surprises and flip dates are the only seed-specific draws).

---

### Task 1: Market enums, error and row models

**Files:**
- Modify: `src/pm_traitbench/enums.py`
- Modify: `src/pm_traitbench/errors.py`
- Modify: `src/pm_traitbench/tables/schema.py`
- Modify: `src/pm_traitbench/tables/specs.py`
- Test: `tests/tables/test_schema.py`, `tests/test_errors.py`

**Interfaces:**
- Produces (enums, all `StrEnum`, value = lowercase name unless noted):
  - `Family`: `EQUITIES="equities"`, `RATES="rates"`, `CREDIT="credit"`, `COMMODITIES="commodities"`, `FX="fx"`.
  - `InstrumentKind`: `EQUITY="equity"`, `CREDIT_ISSUER="credit_issuer"`, `SOVEREIGN_CURVE="sovereign_curve"`, `COMMODITY="commodity"`, `FX_PAIR="fx_pair"`.
  - `EventType`: `EARNINGS`, `RATING_DOWNGRADE`, `RATING_UPGRADE`, `CB_MEETING`, `INVENTORY_REPORT`, `CROP_REPORT`, `MACRO_PRINT`, `CONTRACT_EXPIRY`, `POSITIONING_REPORT`, `CONSENSUS_FLIP` (values snake_case of the name).
  - `StreetView`: `UNDERWEIGHT`, `NEUTRAL`, `OVERWEIGHT`. `Positioning`: `CROWDED_SHORT`, `NEUTRAL`, `CROWDED_LONG`.
  - `Tenor`: `Y2="2Y"`, `Y5="5Y"`, `Y10="10Y"`, `Y30="30Y"`, `M1="M1"` ... `M12="M12"`, declared in that order.
  - `CommodityGroup`: `ENERGY="energy"`, `INDUSTRIAL_METALS="industrial_metals"`, `PRECIOUS="precious"`, `AGRICULTURE="agriculture"`.
  - `RatingBand`: `AA="AA"`, `A="A"`, `BBB="BBB"`, `BB="BB"`, `B="B"` (declared best to worst).
  - `ExpiryRule`: `MONTHLY_THIRD_FRIDAY="monthly_third_friday"`.
  - Module constants in `enums.py`: `SOVEREIGN_TENORS: tuple[Tenor, ...] = (Y2, Y5, Y10, Y30)`, `FUTURES_TENORS: tuple[Tenor, ...] = (M1..M12)`, `NULL_SURPRISE_EVENTS: frozenset[EventType] = {CONTRACT_EXPIRY, POSITIONING_REPORT, CONSENSUS_FLIP}`, `MARKET_WIDE_EVENTS: frozenset[EventType] = {MACRO_PRINT, POSITIONING_REPORT}` (rows with null `instrument_id`), `HY_BANDS: frozenset[RatingBand] = {BB, B}`.
- Produces `MarketCheckError(PmTraitbenchError)` with `exit_code = 1`, docstring "Raised when a generated market misses its own regime targets."
- Produces row models in `tables/schema.py` (all frozen, `extra="forbid"`, every field with `description`), re-exported through `__all__` together with the new enums:
  - `Instrument`: `instrument_id: str`, `family: Family`, `kind: InstrumentKind`, `name: str`, `currency: str`, `sector: str | None`, `rating_band: RatingBand | None`, `commodity_group: CommodityGroup | None`, `duration_years: float | None`, `beta: float | None`, `expiry_rule: ExpiryRule | None`. Validator: the set of non-null optional fields equals exactly the set for `kind` - equity `{sector, beta}`; credit_issuer `{sector, rating_band, duration_years}`; sovereign_curve `{}`; commodity `{commodity_group, expiry_rule}`; fx_pair `{}`. Also `family` must match `kind` (equity-equities, credit_issuer-credit, sovereign_curve-rates, commodity-commodities, fx_pair-fx).
  - `Price`: `seed: str`, `date: datetime.date`, `instrument_id: str`, `price: float` (`gt=0`), `spread_bp: float | None`. Validator: `spread_bp` is non-null exactly when `instrument_id` starts with `"CR-"`, and positive when set.
  - `CurvePoint`: `seed`, `date`, `curve_id: str`, `tenor: Tenor`, `level: float`.
  - `ConsensusRow`: `seed`, `date`, `instrument_id: str`, `street_score: float` (`ge=-1, le=1`), `street_view: StreetView`, `positioning_pct: float` (`ge=0, le=100`), `positioning: Positioning`.
  - `CalendarEvent`: `seed`, `date`, `instrument_id: str | None`, `event: EventType`, `surprise: float | None` (in [-1, 1] when set), `affected: str`. Validators: `surprise` null exactly when `event in NULL_SURPRISE_EVENTS`; `instrument_id` null exactly when `event in MARKET_WIDE_EVENTS`; `affected` is a `Family` value or `"all"`, and is `"all"` exactly when `instrument_id` is null; `rating_downgrade` needs `surprise < 0`, `rating_upgrade` needs `surprise > 0`.
  - `RegimeSpan`: `seed`, `regime: Regime`, `date_start`, `date_end`; validator `date_start <= date_end`.
- Produces specs in `tables/specs.py`: `MARKET_INSTRUMENTS = TableSpec("market/instruments", Instrument, ("instrument_id",))`, `MARKET_PRICES` key `("seed", "date", "instrument_id")`, `MARKET_CURVES` key `("seed", "date", "curve_id", "tenor")`, `MARKET_CONSENSUS` key `("seed", "date", "instrument_id")`, `MARKET_CALENDAR` key `("seed", "date", "instrument_id", "event")`, `MARKET_REGIMES` key `("seed", "date_start")`, and `MARKET_TABLES: tuple[TableSpec, ...]` of the six in that order.

- [ ] **Step 1: Write failing tests** in `tests/tables/test_schema.py` (append) and `tests/test_errors.py`:
  - `Instrument` accepts one valid row per kind; rejects each kind with one extra optional field set and with one required optional field missing (parametrise); rejects a family/kind mismatch.
  - `Price` rejects `price <= 0`; rejects `spread_bp` on an `EQ-` id; rejects missing `spread_bp` on a `CR-` id.
  - `CalendarEvent`: rejects surprise on `contract_expiry`, missing surprise on `earnings`, positive surprise on `rating_downgrade`, negative on `rating_upgrade`, `instrument_id` set on `macro_print`, null `instrument_id` on `earnings`, `affected="all"` with an instrument, `affected="bonds"`, surprise 1.5.
  - `ConsensusRow` rejects `street_score=1.2` and `positioning_pct=-1`. `RegimeSpan` rejects start after end.
  - `Tenor` values in order are `2Y, 5Y, 10Y, 30Y, M1..M12`.
  - `MarketCheckError` is a `PmTraitbenchError` with exit code 1 (match the existing pattern in `tests/test_errors.py`).
  - Round-trip: `to_record` of a `CalendarEvent` with null `instrument_id` gives `None` in the dict, and every new model survives `columns()` introspection (import `pm_traitbench.tables.introspect.columns` and call it on each model).
- [ ] **Step 2: Run** `uv run pytest tests/tables/test_schema.py tests/test_errors.py -q`. Expected: FAIL with ImportError on the new names.
- [ ] **Step 3: Implement** per the Interfaces block.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market enums, row models and table specs`

---

### Task 2: Store subdirectories, null-first key sort, run-metadata extras

**Files:**
- Modify: `src/pm_traitbench/tables/store.py`
- Modify: `src/pm_traitbench/stages.py`
- Test: `tests/tables/test_store.py`, `tests/test_stages.py`

**Interfaces:**
- Consumes: `MARKET_CALENDAR`, `CalendarEvent` from Task 1.
- Produces:
  - `DataStore.write` creates `target.parent` (not only `data_dir`), so `market/prices` lands at `<data_dir>/market/prices.<ext>`. Sorting uses a key where each component `v` maps to `(0, "")` when `v is None` and `(1, v)` otherwise, so a null `instrument_id` sorts first within the key and never raises `TypeError`. Duplicate detection is unchanged.
  - `DataStore.write_run_metadata(stage_name: str, config: Config, extra: dict[str, Any] | None = None) -> Path`: merges `extra` into the top-level JSON (extra keys must not collide with the fixed keys; raise `ValueError` on collision).
  - `Stage.run` type becomes `Callable[[Config, DataStore], dict[str, Any] | None]`. `run_stage` passes the return value as `extra` to `write_run_metadata`. `run_stage` still returns `None`.
  - Output overrides key on the spec name including the subdirectory: `OutputConfig(tables={"market/prices": "parquet"})`.

- [ ] **Step 1: Write failing tests:**
  - `test_store.py`: write and read back a `MARKET_CALENDAR` table with rows that mix null and non-null `instrument_id` on the same date; file lands in `tmp_path/"market"/"calendar.jsonl"`; the null row comes first; the same with a parquet override `{"market/calendar": "parquet"}`; `write_run_metadata(..., extra={"check": {"A": 1}})` writes `"check"` into the JSON; a colliding extra key (`"stage"`) raises `ValueError`.
  - `test_stages.py`: a fake stage whose `run` returns `{"check": {"ok": True}}` gets that key in `run_metadata/<name>.json`; a stage returning `None` keeps today's metadata shape.
- [ ] **Step 2: Run** `uv run pytest tests/tables/test_store.py tests/test_stages.py -q`. Expected: FAIL (FileNotFoundError on the subdirectory, TypeError on sorting None, unexpected keyword `extra`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: support market subdirectory tables and run metadata extras`

---

### Task 3: Market config section

**Files:**
- Create: `src/pm_traitbench/market/__init__.py` (empty docstring module: `"""Stage 2: a synthetic, regime-scripted market per market seed."""`)
- Create: `src/pm_traitbench/market/constants.py`
- Modify: `src/pm_traitbench/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: enums from Task 1.
- Produces in `market/constants.py` (fixed code tables; config only chooses which entries are used):
  - `COMMODITIES: dict[CommodityGroup, tuple[CommoditySpec, ...]]` with frozen dataclass `CommoditySpec(name: str, code: str, start: float)`. Binding order and values:
    - energy: crude CRD 72, brent BRN 76, gasoil GSO 700, gasoline GSL 2.2, heating_oil HOL 2.4, coal COL 130
    - industrial_metals: copper CPR 9000, aluminium ALU 2400, nickel NKL 16000, zinc ZNC 2700
    - precious: gold GLD 2400, silver SLV 28, platinum PLT 950
    - agriculture: wheat WHT 600, corn CRN 450, soybeans SOY 1100, sugar SGR 20, coffee COF 250, cotton CTN 75, cocoa CCO 8000
  - `FX_PAIRS: dict[str, tuple[str, str]]` pair name to (base, quote): EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, EURGBP, EURJPY, AUDJPY.
  - `USD_PAIR: dict[str, str]` currency to its USD pair: EUR-EURUSD, GBP-GBPUSD, JPY-USDJPY, AUD-AUDUSD, CHF-USDCHF, CAD-USDCAD.
  - `SECTOR_LABEL = "sector_{:02d}"` style helper `sector_label(i: int) -> str` (1-indexed).
  - `HORIZON_DAYS_PER_YEAR = 260` (weekday sessions a year) and `ANNUALISATION_DAYS = 252` (trading-day convention the vols are quoted in).
- Produces in `config.py` (every leaf with basis and note copied from the tables below; notes cite the FRED series or paper named there):
  - `RegimeParams` frozen dataclass: `driver_mean: float`, `vol_multiplier: float`, `mean_reversion_kappa: float`.
  - `MarketUniverseConfig`: `n_equities: int = 80` (ge 0), `n_sectors: int = 10` (1..99), `n_credit_issuers: int = 48` (ge 0), `credit_band_shares: dict[RatingBand, float]` = AA 0.15, A 0.25, BBB 0.30, BB 0.20, B 0.10 (validator: all five keys, non-negative, sum to 1 within 1e-9, at least one IG band share > 0), `curves: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY")`, `commodities: dict[CommodityGroup, int]` = energy 6, industrial_metals 4, precious 3, agriculture 7 (validator: 0 <= n <= len of the group's code table), `fx_pairs: tuple[str, ...]` = the nine pairs above (validator: known names, no repeats), `equity_beta_range: tuple[float, float] = (0.6, 1.4)`, `credit_duration_range: tuple[float, float] = (3.0, 8.0)`, `expiry_rule: ExpiryRule = MONTHLY_THIRD_FRIDAY`. Ranges need `0 < lo <= hi`.
  - `MarketRegimesConfig`: `driver_mean: dict[Regime, float]` = range 0.00, risk_off -0.09, risk_on 0.06 (basis design); `vol_multiplier: dict[Regime, float]` = 1.0, 1.6, 0.95 (basis sourced: VIX medians 17.0 / 27.6 / 16.3 conditioned on trailing 60-day equity return, FRED VIXCLS and NASDAQCOM); `mean_reversion_kappa: dict[Regime, float]` = 0.05, 0.0, 0.0 (basis design, half-life about 14 days). Validators: each map has exactly the three regimes; `vol_multiplier > 0`; `0 <= kappa < 1`. Method `params(regime: Regime) -> RegimeParams`.
  - `EquityFamilyConfig`: `market_vol = 0.16`, `idio_vol = 0.25`. `RatesFamilyConfig`: `level_vol_bp = 90.0`, `slope_vol_bp = 60.0`, `driver_corr = 0.3`. `CreditFamilyConfig`: `factor_vol = 0.25`, `hy_vol_multiplier = 1.4`, `driver_corr = -0.5`, `asymmetry = 1.1` (ge 1), `issuer_vol = 0.15` (basis guess). `CommodityFamilyConfig`: `group_vol: dict[CommodityGroup, float]` = energy 0.40, industrial_metals 0.21, precious 0.15, agriculture 0.23; `driver_corr: dict[CommodityGroup, float]` = 0.15, 0.3, -0.1, 0.0; `curve_slope: dict[CommodityGroup, float]` = -0.05, 0.01, 0.03, 0.04; `group_share = 0.6` (basis guess, 0 < x < 1). `FxFamilyConfig`: `currency_vol: dict[str, float]` = EUR 0.09, GBP 0.09, JPY 0.10, AUD 0.12, CHF 0.10, CAD 0.08; `driver_corr: dict[str, float]` = EUR 0.1, GBP 0.2, JPY -0.1, AUD 0.3, CHF 0.0, CAD 0.25. `MarketFamiliesConfig`: `equity`, `rates`, `credit`, `commodity`, `fx` sub-models plus `student_t_df: int = 4` (ge 3, so unit-variance scaling exists). All vols `> 0`, all `driver_corr` in `[-1, 1]`, dict-valued fields must cover every key of their group/currency set.
  - `MarketLevelsConfig`: `equity_price_range: tuple[float, float] = (10.0, 400.0)`; `curve_start: dict[str, tuple[float, float, float, float]]` = USD (4.0, 3.9, 4.1, 4.4), EUR (2.4, 2.3, 2.5, 2.8), GBP (4.2, 4.0, 4.2, 4.6), JPY (0.4, 0.5, 1.0, 2.0) for 2Y, 5Y, 10Y, 30Y in percent; `credit_base_spread_bp: dict[RatingBand, float]` = AA 50, A 70, BBB 105, BB 180, B 305 (basis sourced, ICE BofA OAS medians 2023-09 to 2026-09, FRED BAMLC0A2CAA, BAMLC0A3CA, BAMLC0A4CBBB, BAMLH0A1HYBB, BAMLH0A2HYB); `fx_start: dict[str, float]` = EURUSD 1.10, GBPUSD 1.28, USDJPY 150.0, AUDUSD 0.66, USDCHF 0.88, USDCAD 1.36; `yield_floor_pct: float = 0.0`.
  - `EventSpec` model with its own `basis` and `note` fields (like `BiasSpec`): `per_year: float` (ge 0), `jump_size: float` (ge 0), `placement: Literal["grid", "poisson"]`, `jitter_days: int = 0` (ge 0). Validator: for `grid`, `per_year > 0` and `HORIZON_DAYS_PER_YEAR / per_year > 2 * jitter_days` (keeps jittered dates unique).
  - `MarketConfig.events: dict[EventType, EventSpec]` defaults: `earnings` 4, 0.05, grid, jitter 5 (sourced, Dubinsky, Johannes, Kaeck and Seeger 2019); `rating_downgrade` 0.2, 0.20, poisson (guess; direction from Hand, Holthausen and Leftwich 1992); `rating_upgrade` 0.1, 0.20, poisson (guess); `cb_meeting` 8, 8.0 (bp), grid (sourced, eight FOMC meetings; Gurkaynak, Sack and Swanson 2005); `inventory_report` 52, 0.012, grid (sourced, WTI Wednesday sd 3.15% vs 2.93%, FRED DCOILWTICO); `crop_report` 12, 0.03, grid (sourced, WASDE; Adjemian 2012); `macro_print` 12, 1.0 (driver sd), grid (design). Validator: keys are exactly these seven sampled types.
  - `MarketConsensusConfig`: `street_window_days: int = 20` (guess), `positioning_window_multiple: int = 3`, `revision_weekday: int = 2` (0=Monday; Wednesday), `view_threshold: float = 0.25` (0 < x < 0.5, so twice it stays inside [-1, 1]), `flips_per_instrument_year: float = 2.0` (ge 0), `positioning_thresholds: tuple[float, float] = (20.0, 80.0)` (0 <= lo < hi <= 100), `report_weekday: int = 4` (Friday; sourced, CFTC COT release day). Weekdays in 0..4.
  - `MarketCheckConfig`: `vol_tolerance: float = 0.30` (gt 0), `corr_tolerance_se: float = 4.0` (gt 0; design: about 1 in 16,000 false alarms per test for a normal estimate), `min_round_level_tests: float = 3.0` (ge 0), `round_level_band: float = 0.002` (0 < x < 0.05).
  - `MarketConfig`: `seeds: dict[str, tuple[Regime, Regime, Regime]]` = A (range, risk_off, risk_on), B (risk_on, range, risk_off), C (risk_off, risk_on, range); `boundary_weeks: tuple[int, int] = (16, 30)`; `burn_in_days: int = 60` (ge 0); sub-sections `universe`, `regimes`, `families`, `levels`, `events`, `consensus`, `check`. Validators: every seed lists each regime exactly once; `boundary_weeks` strictly increasing; `levels.curve_start` keys cover `universe.curves`; `families.fx.currency_vol` covers every non-USD currency used by `universe.fx_pairs`; `levels.fx_start` covers the USD pair of each such currency.
  - `Config.market: MarketConfig`. `Config` cross-checks: `market.seeds` keys cover `population.market_seeds` (error names the missing seed); `1 <= boundary_weeks[0] < boundary_weeks[1] <= calendar.n_weeks - 1`.

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`:
  - Defaults: spot-check one value per sub-section equals the binding default above (for example `market.families.credit.asymmetry == 1.1`, `market.events[EventType.EARNINGS].jitter_days == 5`, `market.regimes.params(Regime.RISK_OFF) == RegimeParams(-0.09, 1.6, 0.0)`).
  - Rejections (each raises `ValidationError` directly, and one YAML case raises `ConfigError`): seed with a repeated regime; seed with two regimes; `population.market_seeds` containing `"D"` not in `market.seeds`; `boundary_weeks` (30, 16), (0, 16), (16, 52) with 52 weeks; band shares summing to 0.9; `commodities.energy = 7`; unknown FX pair; `vol_tolerance = 0`; `corr_tolerance_se = -1`; grid event with `jitter_days` too large; `events` missing `macro_print`.
  - YAML override `market: {universe: {n_equities: 20}}` keeps sibling defaults.
  - Update `test_dump_with_basis_covers_every_leaf`: its local walker must accept any model that has a `basis` field (not only `BiasSpec`) and dicts of such models, mirroring `_has_basis_field` in `config.py`; assert `"market.families.credit.asymmetry"` and `"market.events.earnings"` are in the dumped paths.
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -q`. Expected: FAIL (no `market` attribute / import errors).
- [ ] **Step 3: Implement.** Keep the new models in `config.py` next to the existing sections. Notes must read as facts ("realised vol of daily 10Y yield changes 1990-2026 is 92bp a year, FRED DGS10"), not references to a design.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market config section`

---

### Task 4: Simulation axis and instrument universe

**Files:**
- Create: `src/pm_traitbench/market/axis.py`
- Create: `src/pm_traitbench/market/universe.py`
- Test: `tests/market/test_axis.py`, `tests/market/test_universe.py` (test dirs have no `__init__.py`)

**Interfaces:**
- Consumes: `Config`, `MarketConfig`, `Timeline`, `Instrument`, enums, `market/constants.py`.
- Produces in `axis.py`:
  ```python
  @dataclass(frozen=True)
  class SimAxis:
      dates: tuple[date, ...]   # burn-in weekdays then horizon weekdays, ascending
      n_burn: int
      @property
      def n_days(self) -> int: ...
      @property
      def horizon(self) -> slice: ...          # slice(n_burn, n_days)
      def index(self, day: date) -> int: ...   # KeyError-free: raise ValueError for a date not on the axis

  def build_axis(timeline: Timeline, burn_in_days: int) -> SimAxis
  ```
  Burn-in dates walk back Mon-Fri from the day before `timeline.start`. Horizon dates are `timeline.weekdays_in_weeks(1, n_weeks)`.
- Produces in `universe.py`: `build_universe(config: Config, rng: np.random.Generator) -> list[Instrument]`, called by the stage with `stream(root, "market", "universe")`. Deterministic order: equities, credit, curves, commodities, FX.
  - Equities: `EQ-0001..`, name `"Equity 0001"`, currency `"USD"`, sector round-robin `sector_01, sector_02, ...` over `n_sectors`, `beta ~ U(equity_beta_range)`.
  - Credit: band counts by largest remainder of `n_credit_issuers * share` (ties broken by band order AA..B), issuers ordered AA, A, BBB then BB, B. IG bands get `CR-IG-001..`, HY bands `CR-HY-001..` (separate counters), name `"Issuer IG 001"` / `"Issuer HY 001"`. Sector uniform over the sector labels, currency uniform over `universe.curves`, `duration_years ~ U(credit_duration_range)`.
  - Curves: `RT-<CCY>`, name `"<CCY> sovereign curve"`, currency `<CCY>`.
  - Commodities: per group in `CommodityGroup` order, the first `n` specs of the group's table: `CM-<CODE>`, name the commodity name, currency `"USD"`, `commodity_group`, `expiry_rule` from config.
  - FX: `FX-<PAIR>` in config order, name the pair, currency the quote currency.
  - Draw order inside the one generator is fixed: all equity betas, then credit sectors, currencies, durations.

- [ ] **Step 1: Write failing tests:**
  - `test_axis.py`: default config axis has `60 + 260` dates, all weekdays, strictly increasing, no gap except weekends; `dates[n_burn] == calendar.start`; last burn-in date is the Friday before start; `burn_in_days=0` gives the horizon only; `index` of a weekend date raises `ValueError`.
  - `test_universe.py`: default counts per family (80, 48, 4, 20, 9 = 161); ids unique and matching `^EQ-\d{4}$`, `^CR-(IG|HY)-\d{3}$`, `^RT-[A-Z]{3}$`, `^CM-[A-Z]{3}$`, `^FX-[A-Z]{6}$`; band counts at 48 issuers equal `(7, 12, 14, 10, 5)` for AA..B (48 x shares = 7.2, 12, 14.4, 9.6, 4.8; floors sum to 46; the two largest remainders are B 0.8 and BB 0.6); IG ids only on AA/A/BBB; betas and durations within range; same generator seed gives identical output; demo-sized config (20, 5, 12) gives 20 + 12 + 4 + 20 + 9.
- [ ] **Step 2: Run** `uv run pytest tests/market -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market simulation axis and instrument universe`

---

### Task 5: Regime schedule and driver shocks

**Files:**
- Create: `src/pm_traitbench/market/regimes.py`
- Create: `src/pm_traitbench/market/drivers.py`
- Test: `tests/market/test_regimes.py`, `tests/market/test_drivers.py`

**Interfaces:**
- Consumes: `SimAxis`, `Config`, `MarketRegimesConfig.params`, `RegimeSpan`, `stream`.
- Produces in `regimes.py`:
  ```python
  def build_schedule(config: Config, seed: str, timeline: Timeline) -> list[RegimeSpan]
  class RegimeLookup:
      def __init__(self, spans: Sequence[RegimeSpan], burn_in_regime: Regime) -> None: ...
      def regime(self, day: date) -> Regime: ...   # burn-in_regime before the first span; ValueError after the last
  @dataclass(frozen=True)
  class RegimePath:
      regimes: tuple[Regime, ...]        # one per axis day
      driver_mean: np.ndarray            # float, per axis day
      vol_multiplier: np.ndarray
      kappa: np.ndarray
  def regime_path(axis: SimAxis, lookup: RegimeLookup, config: Config) -> RegimePath
  def constant_path(regime: Regime, n_days: int, config: Config) -> RegimePath   # for tests and long-axis checks
  ```
  Spans: regime 1 covers weeks `1..b1`, regime 2 `b1+1..b2`, regime 3 `b2+1..n_weeks`; `date_start` is the Monday of the first week, `date_end` the Friday of the last week. `burn_in_regime` is the seed's first regime.
- Produces in `drivers.py`:
  ```python
  @dataclass(frozen=True)
  class DriverShocks:
      common: np.ndarray                              # u, standard normal, length n_days
      groups: Mapping[CommodityGroup, np.ndarray]     # one standard-normal vector per group
  def draw_shocks(root_seed: int, n_days: int) -> DriverShocks
  def seed_driver(shocks: DriverShocks, path: RegimePath, macro: np.ndarray) -> np.ndarray
  def daily_vol(annual_vol: float, path: RegimePath) -> np.ndarray   # annual_vol * vol_multiplier / sqrt(252)
  ```
  Streams: `stream(root, "market", "driver", "common")` and `stream(root, "market", "driver", group.value)`, each drawn as one whole vector. `seed_driver` returns `z = u + driver_mean + macro` (macro is the per-day sum of `surprise x jump_size` for `macro_print`, zeros elsewhere). Positive `z` is a risk-on day.

- [ ] **Step 1: Write failing tests:**
  - `test_regimes.py`: with default calendar, seed A spans are range 2026-01-05..2026-04-24, risk_off 2026-04-27..2026-07-31, risk_on 2026-08-03..2027-01-01; B and C follow their orders on the same dates (compute expected dates from the week arithmetic in the test, and assert these literal ones for A); spans tile the horizon with no gap; burn-in dates map to the seed's first regime; a date after the horizon raises; `regime_path` arrays match `params(regime)` on sampled days.
  - `test_drivers.py`: `draw_shocks` is identical for two calls with the same root and differs for another root; group vectors are pairwise different and each differs from `common`; `seed_driver` adds driver mean and macro exactly.
- [ ] **Step 2: Run** `uv run pytest tests/market -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market regime schedule and driver shocks`

---

### Task 6: Event calendar

**Files:**
- Create: `src/pm_traitbench/market/calendar.py`
- Test: `tests/market/test_calendar.py`

**Interfaces:**
- Consumes: `SimAxis`, `Instrument`, `CalendarEvent`, `EventSpec`, `Config`, `stream`.
- Produces:
  ```python
  RngFor = Callable[..., np.random.Generator]   # rng_for(*keys) -> stream(root, "market", *keys)

  EVENT_TARGETS: dict[EventType, Callable[[Instrument], bool]]
  # earnings: equities; rating_downgrade/upgrade: credit issuers; cb_meeting: sovereign curves;
  # inventory_report: energy commodities; crop_report: agriculture commodities; macro_print: none (market-wide)

  @dataclass(frozen=True)
  class SampledEvents:
      rows: list[CalendarEvent]
      drawn: dict[EventType, int]          # count drawn per sampled type

  def sample_events(instruments: Sequence[Instrument], axis: SimAxis, config: Config,
                    rng_for: RngFor, seed: str) -> SampledEvents
  def generated_rows(instruments: Sequence[Instrument], axis: SimAxis, seed: str) -> list[CalendarEvent]
  def third_friday(year: int, month: int) -> date

  @dataclass(frozen=True)
  class EventJumps:
      by_instrument: Mapping[str, np.ndarray]   # per axis day, sum of surprise x jump_size
      macro: np.ndarray
      def for_instrument(self, instrument_id: str) -> np.ndarray   # zeros when absent
  def build_jumps(rows: Sequence[CalendarEvent], axis: SimAxis, config: Config) -> EventJumps
  def event_day_indices(rows: Sequence[CalendarEvent], axis: SimAxis) -> dict[str, set[int]]
  # instrument_id -> axis indices of its sampled events (macro and generated rows excluded)
  ```
  Placement works on horizon indices `0..H-1` (H = number of horizon days) and converts to axis indices, so burn-in never has events:
  - Count scale `years = H / 260`.
  - `grid`: `count = round(per_year * years)`, `spacing = H / count`, `offset ~ U[0, spacing)`, day `k` = `floor(offset + k * spacing) + jitter_k`, `jitter_k` uniform integer in `[-jitter_days, jitter_days]`, clipped to `[0, H-1]`.
  - `poisson`: `count ~ Poisson(per_year * years)` capped at H, days = sorted `rng.choice(H, count, replace=False)`.
  - Date draws: `rng_for("events", target_key, event.value)`, where `target_key` is the instrument id or `"macro"`. Seed-independent.
  - Surprises: `rng_for(seed, "surprise", target_key, event.value)`, one vector for that instrument-event in date order. Magnitude `m ~ Beta(2, 2)`; two-sided sign uniform over {-1, +1}; `rating_downgrade` is `-m`, `rating_upgrade` `+m`.
  - `affected` is the instrument's family value, or `"all"` for macro.
  - `generated_rows`: for each commodity, one `contract_expiry` on `third_friday` of every calendar month whose third Friday is a horizon date (`affected="commodities"`); one `positioning_report` on every horizon Friday (`instrument_id=None`, `affected="all"`). Surprise null.
  - `build_jumps` uses only sampled types; the value added on an event's axis day is `surprise * events[event].jump_size`, in the family's unit (log return for equities and commodities, relative spread for credit, bp for curves, driver sd for macro). Processes convert sign and units (sign convention: positive surprise is good news for the price).

- [ ] **Step 1: Write failing tests** (use the default universe and a long horizon where rates matter: a `Config` with `calendar.n_weeks = 520` and `boundary_weeks` inside it):
  - Per-type mean counts per instrument-year within 10% of `per_year` for every sampled type (rating actions pooled over all credit issuers).
  - Earnings per equity: exactly `round(4 * years)` rows, dates unique per instrument.
  - Sign agrees with label for all rating rows; all surprises in [-1, 1]; two-sided types have both signs.
  - `contract_expiry` rows land on third Fridays, one per commodity per month; `positioning_report` rows on Fridays, one per horizon Friday.
  - No row dated before `calendar.start`.
  - Same root, seeds A and B: identical `(date, instrument_id, event)` sets, different surprise vectors.
  - `drawn` counts equal row counts per type.
  - `build_jumps`: on a hand-built two-row input, the jump vector has `surprise * jump_size` on the right axis day and zeros elsewhere; `macro` vector filled from `macro_print` only.
- [ ] **Step 2: Run** `uv run pytest tests/market/test_calendar.py -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market event calendar`

---

### Task 7: Process scaffolding and equities

**Files:**
- Create: `src/pm_traitbench/market/processes/__init__.py` (docstring only)
- Create: `src/pm_traitbench/market/processes/common.py`
- Create: `src/pm_traitbench/market/processes/equities.py`
- Test: `tests/market/test_equities.py`, `tests/market/test_process_common.py`

**Interfaces:**
- Consumes: `SimAxis`, `RegimePath`, `daily_vol`, `EventJumps`, `MarketConfig`, `Instrument`, `RngFor`.
- Produces in `common.py`:
  ```python
  @dataclass(frozen=True)
  class ProcessInputs:
      instruments: tuple[Instrument, ...]          # full universe; each process filters its own
      axis: SimAxis
      path: RegimePath
      z: np.ndarray
      group_shocks: Mapping[CommodityGroup, np.ndarray]
      jumps: EventJumps
      market: MarketConfig
      rng_for: RngFor

  @dataclass
  class ProcessOutput:
      prices: dict[str, np.ndarray] = field(default_factory=dict)            # full axis
      spreads: dict[str, np.ndarray] = field(default_factory=dict)           # credit only, bp
      curves: dict[tuple[str, Tenor], np.ndarray] = field(default_factory=dict)
      def merge(self, other: "ProcessOutput") -> "ProcessOutput": ...        # ValueError on a repeated key

  def unit_student_t(rng: np.random.Generator, df: int, size) -> np.ndarray   # t(df) / sqrt(df / (df - 2))
  def log_grid_step(price: np.ndarray | float) -> np.ndarray | float           # 10 ** floor(log10 p) / 2
  def nearest_level(value, step) -> ...                                        # round(value / step) * step
  def round_log_gap(price, step) -> ...                                         # log p - log nearest_level(p, step)
  ```
  Instrument noise is drawn per purpose as one whole vector on `rng_for("noise", instrument_id, purpose)`; adding an instrument never changes another's draws. Starting levels that need a draw use `rng_for("level", instrument_id)`.
- Produces in `equities.py`: `simulate(inputs: ProcessInputs) -> ProcessOutput` (prices only). Binding recursion, per equity `i` and axis day `t >= 1` (day 0 is the start price `p_0 ~ LogUniform(equity_price_range)`):
  ```
  sigma_m = daily_vol(market_vol, path); sigma_i = daily_vol(idio_vol, path)
  r_t = beta_i * sigma_m[t] * z[t] + sigma_i[t] * t_i[t] - kappa[t] * round_log_gap(p_{t-1}, log_grid_step(p_{t-1})) + jump_i[t]
  p_t = p_{t-1} * exp(r_t)
  ```
  `jump_i` is `EventJumps.for_instrument(id)` (log return, sign as drawn). `t_i` is `unit_student_t(rng_for("noise", id, "idio"), df, n_days)`. Vectorise across instruments, loop over days.

- [ ] **Step 1: Write failing tests:**
  - `test_process_common.py`: `unit_student_t` sample variance within 5% of 1 on 200k draws for df 5 (df 4 has infinite fourth moment, so its sample variance converges slowly; use df 5 here); `log_grid_step(72) == 5`, `(7.2) == 0.5`, `(250) == 50`; `nearest_level(73, 5) == 75`; `merge` raises on duplicate key.
  - `test_equities.py` (build inputs by hand: default config, axis of 20 years via `build_axis(Timeline(date(2026,1,5), 1040), 0)`, `constant_path`, `z = u + driver_mean` from a fixed generator, zero jumps): one price series per equity, all positive and finite; realised annualised vol of the equal-weight index within 10% of `mult * sqrt(mean_beta^2 * market_vol^2 + idio_vol^2 / n)` for each regime's constant path; correlation of index returns with `z` positive and above 0.9; mean-reversion pull: with kappa > 0 (range) the mean absolute `round_log_gap` over the path is smaller than with kappa forced to 0 on the same draws; an earnings jump of `+0.05` on day `t` raises the log return on that day by exactly 0.05 relative to a run with zero jumps; adding an equity to the universe leaves the other equities' paths unchanged.
- [ ] **Step 2: Run** `uv run pytest tests/market -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market process scaffolding and equity process`

---

### Task 8: Rates and credit processes

**Files:**
- Create: `src/pm_traitbench/market/processes/rates.py`
- Create: `src/pm_traitbench/market/processes/credit.py`
- Test: `tests/market/test_rates.py`, `tests/market/test_credit.py`

**Interfaces:**
- Consumes: `ProcessInputs`, `ProcessOutput`, helpers from Task 7, `SOVEREIGN_TENORS`, `HY_BANDS`.
- Produces `rates.simulate(inputs) -> ProcessOutput` (curves only, keys `(curve_id, tenor)` for the four sovereign tenors) and `credit.simulate(inputs, rates: ProcessOutput) -> ProcessOutput` (prices and spreads for credit issuers).
- Binding rates model, per curve (all yields in percent, vols converted with `bp / 100`):
  ```
  w = (-0.5, -0.15, 0.15, 0.5) for 2Y, 5Y, 10Y, 30Y          # Diebold and Li 2006 loadings, curvature fixed
  L_0 = mean(start); S_0 = sum(w * (start - L_0)) / sum(w^2); o = start - (L_0 + w * S_0)
  sigma_L = daily_vol(level_vol_bp / 100, path); sigma_S = daily_vol(slope_vol_bp / 100, path); c = rates.driver_corr
  raw10_{t-1} = L_{t-1} + 0.15 * S_{t-1} + o_10Y
  dL = sigma_L * (c * z + sqrt(1 - c^2) * e_L) - kappa * (raw10_{t-1} - nearest_level(raw10_{t-1}, 0.25)) - jump_bp / 100
  dS = sigma_S * e_S
  y_k = max(L + w_k * S + o_k, yield_floor_pct)
  ```
  `e_L`, `e_S` on `rng_for("noise", curve_id, "level")` and `("noise", curve_id, "slope")`. `jump_bp` is the curve's cb_meeting jump vector (positive surprise lowers yields).
- Binding credit model:
  ```
  sigma_F = daily_vol(factor_vol, path); c = credit.driver_corr; a = asymmetry
  shock = c * z + sqrt(1 - c^2) * e_F                       # e_F on rng_for("noise", "credit_factor")
  dlogF = sigma_F * (shock * (a if shock > 0 else 1 / a) - m)   # positive = widening
  m = (a - 1 / a) / sqrt(2 * pi)   # mean of the scaled standard-normal shock; centring keeps skew without adding drift
  logF_0 = 0
  x_i: AR(1) log issuer noise, x_0 = 0,
       x_t = phi * x_{t-1} + daily_vol(issuer_vol, path)[t] * e_i[t] - kappa[t] * (log s_{t-1} - log nearest_level(s_{t-1}, 10)) - jump_i[t]
       phi = 0.98 (module constant: half-life about 34 trading days, so issuer news persists for weeks)
  h_i = hy_vol_multiplier for BB and B, else 1
  s_i,t = base_spread_bp[band_i] * exp(h_i * logF_t + x_i,t)
  y5 = rates.curves[("RT-" + currency_i, Tenor.Y5)]
  p_0 = 100; p_t = p_{t-1} * (1 - D_i * ((s_t - s_{t-1}) + 100 * (y5_t - y5_{t-1})) / 10000)
  ```
  `nearest_level(s, 10)` must not return 0: floor it at 10bp. `e_i` on `rng_for("noise", id, "issuer")`. `jump_i` is the rating-event relative-spread jump (positive surprise tightens). Output `prices[id] = p`, `spreads[id] = s`.

- [ ] **Step 1: Write failing tests** (20-year hand-built inputs as in Task 7):
  - Rates: four tenor series per curve, none below the floor (use a curve starting near 0 to force the floor); day-0 curve equals `curve_start` exactly; realised annualised sd of daily 10Y changes (bp) within 10% of `level_vol_bp * mult` per constant regime path (ignore floor effect by testing USD); correlation of 10Y changes with `z` positive; range kappa reduces mean distance of the 10Y to the nearest 25bp level versus kappa 0; a +1 surprise cb jump of 8bp lowers every tenor by 0.08 on that day relative to a no-jump run.
  - Credit: spreads positive, prices positive; mean absolute widening step of `logF` larger than mean absolute tightening step (asymmetry); IG equal-weight log spread change vol within 10% of the Task 12 credit formula (centred asymmetric shock plus issuer noise `issuer_vol^2 / n_IG`); correlation of IG log spread change with `z` negative; HY issuers' log spread vol greater than IG's; bond price falls on a day when spread and 5Y both rise.
- [ ] **Step 2: Run.** Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add rates and credit processes`

---

### Task 9: Commodity and FX processes

**Files:**
- Create: `src/pm_traitbench/market/processes/commodities.py`
- Create: `src/pm_traitbench/market/processes/fx.py`
- Test: `tests/market/test_commodities.py`, `tests/market/test_fx.py`

**Interfaces:**
- Consumes: `ProcessInputs`, `ProcessOutput`, helpers from Task 7, `COMMODITIES`, `FX_PAIRS`, `USD_PAIR`, `FUTURES_TENORS`.
- Produces `commodities.simulate(inputs) -> ProcessOutput` (prices = M1 and curves `(instrument_id, M1..M12)`) and `fx.simulate(inputs) -> ProcessOutput` (prices per pair).
- Binding commodity model, commodity `i` in group `g` (spot start from `COMMODITIES`):
  ```
  sigma = daily_vol(group_vol[g], path); c = driver_corr[g]; share = group_share
  front_det_{t-1} = spot_{t-1} * (1 + slope_g / 12)
  r = sigma * (c * z + sqrt(1 - c^2) * (sqrt(share) * group_shocks[g] + sqrt(1 - share) * t_i))
      - kappa * round_log_gap(front_det_{t-1}, log_grid_step(front_det_{t-1})) + jump_i
  spot_t = spot_{t-1} * exp(r)
  M_k,t = spot_t * (1 + slope_g * k / 12) * exp(0.002 * n_k,t),  k = 1..12
  price = M_1
  ```
  `t_i` on `("noise", id, "idio")` (unit Student-t), `n` shape `(n_days, 12)` on `("noise", id, "curve")`.
- Binding FX model: for each non-USD currency `c` used by the configured pairs, `v_c` = log USD value of one unit of `c`, `v_USD = 0`. Starts from `fx_start` of `USD_PAIR[c]` (XXXUSD: `v = log P`; USDXXX: `v = -log P`).
  ```
  anchor = USD_PAIR[c]; P_anchor = exp(v_c) if anchor starts with c else exp(-v_c)
  step = 1.0 if anchor's quote is "JPY" else 0.01
  gap = log P_anchor - log nearest_level(P_anchor, step); pull = -kappa * gap * (+1 if anchor starts with c else -1)
  dv_c = sigma_c * (cc * z + sqrt(1 - cc^2) * e_c) + pull
  pair BASEQUOTE price = exp(v_base - v_quote)
  ```
  `e_c` on `("noise", "FX-" + c, "value")`. No FX events.

- [ ] **Step 1: Write failing tests** (20-year hand-built inputs):
  - Commodities: 12 curve rows per day per commodity and `price == M1`; positive; curve sign: `M12 > M1` on nearly every day (at least 99%) for groups with positive slope and `M12 < M1` for energy, under every constant regime path; group index realised vol within 10% of the implied value (formula in Task 12) per regime; sign of index correlation with `z` matches the sign of the group-weighted loading; range pull reduces mean `round_log_gap` of M1 versus kappa 0.
  - FX: `EURGBP == EURUSD / GBPUSD` and `EURJPY == EURUSD * USDJPY` and `AUDJPY == AUDUSD * USDJPY` to 1e-12 relative on every day; day-0 USD pairs equal `fx_start`; each currency's realised vol vs USD within 10% of `currency_vol * mult`; AUD correlation with `z` positive, JPY negative; a pair subset without any JPY pair still runs.
- [ ] **Step 2: Run.** Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add commodity and fx processes`

---

### Task 10: Consensus and positioning

**Files:**
- Create: `src/pm_traitbench/market/consensus.py`
- Test: `tests/market/test_consensus.py`

**Interfaces:**
- Consumes: `SimAxis`, `Instrument`, `ProcessOutput`, `MarketConfig`, `CalendarEvent`, `ConsensusRow`, `RngFor`, `event_day_indices` output.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ConsensusResult:
      rows: list[ConsensusRow]            # horizon days only, one per instrument per day
      flips: list[CalendarEvent]          # consensus_flip rows, surprise None, affected = family
      drawn_flips: dict[str, int]         # instrument_id -> flips drawn
      street_score: dict[str, np.ndarray] # full axis, for tests and the check
      positioning_pct: dict[str, np.ndarray]

  def ema_weight(half_life: float) -> float           # 1 - 2 ** (-1 / half_life)
  def consensus_signal(instrument: Instrument, output: ProcessOutput) -> np.ndarray
  # log price; for sovereign curves minus the 10Y yield (percent), so up means bond prices up
  def signal_vol(instrument: Instrument, market: MarketConfig) -> float   # annual vol of the signal, no regime multiplier
  def build_consensus(instruments: Sequence[Instrument], axis: SimAxis, output: ProcessOutput,
                      event_days: Mapping[str, set[int]], market: MarketConfig,
                      rng_for: RngFor, seed: str) -> ConsensusResult
  ```
  `signal_vol`: equities `sqrt((beta * market_vol)^2 + idio_vol^2)`; credit `duration * base_spread_bp[band] * factor_vol * h / 10000`; curves `level_vol_bp / 100`; commodities `group_vol[group]`; FX pairs `sqrt(vol_base^2 + vol_quote^2)` with USD vol 0.
- Binding recursion, per instrument on the full axis (`W = street_window_days`, `P = positioning_window_multiple * W`):
  ```
  trend_X(t) = clip((s_t - s_{max(0, t - X)}) / (signal_vol * sqrt(X / 252)), -1, 1)
  street: score_0 = 0; update days = event_days[id] | {t : weekday == revision_weekday} | flip days
    flip day:  score_t = -sgn(score_{t-1}) * 2 * view_threshold   (sgn(0) = +1)
    update:    score_t = score_{t-1} + ema_weight(W) * (trend_W(t) - score_{t-1})
    otherwise: score_t = score_{t-1}
  positioning: pct_0 = 50; on report_weekday days:
    target = 50 + 40 * tanh(2 * trend_P(t)); pct_t = clip(pct_{t-1} + ema_weight(P / 5) * (target - pct_{t-1}), 0, 100)
    otherwise pct_t = pct_{t-1}
  labels: overweight if score > view_threshold, underweight if score < -view_threshold, else neutral;
          crowded_long if pct > hi, crowded_short if pct < lo, else neutral
  ```
  Day 0 counts as a revision or report day only if its weekday matches. Flip count per instrument `~ Poisson(flips_per_instrument_year * H / 260)` capped at H, dates uniform without replacement over horizon days, on `rng_for(seed, "flips", instrument_id)`. Consensus covers every instrument including curves.

- [ ] **Step 1: Write failing tests** (hand-built `ProcessOutput` with simple deterministic paths - a steady uptrend and a flat series - on a short axis with burn-in, plus the default universe for flip statistics):
  - Stepping: score changes only on update days; positioning changes only on report days.
  - Flip: on a flip day the score equals `-sgn(prev) * 0.5`; flip rows equal `drawn_flips` per instrument and land on flip days; flips never in burn-in.
  - Positioning lags: after a flip the positioning percentile keeps its prior direction for at least the next report day (flips do not touch positioning).
  - Uptrend: score and positioning rise above thresholds and labels become overweight and crowded_long; labels agree with thresholds on every row.
  - Day one: on a series trending through burn-in, the first horizon row has a non-zero score.
  - Curves: a falling 10Y gives a positive score.
  - Rows: one per instrument per horizon day; seed A and seed B with the same output differ only in flip placement.
- [ ] **Step 2: Run.** Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market consensus and positioning`

---

### Task 11: Per-seed generation and row conversion

**Files:**
- Create: `src/pm_traitbench/market/generate.py`
- Test: `tests/market/test_generate.py`

**Interfaces:**
- Consumes: everything from Tasks 4-10.
- Produces:
  ```python
  @dataclass(frozen=True)
  class SeedMarket:
      seed: str
      axis: SimAxis
      schedule: list[RegimeSpan]
      path: RegimePath
      z: np.ndarray
      output: ProcessOutput            # merged equities, rates, credit, commodities, fx; full axis
      calendar: list[CalendarEvent]    # sampled + generated + flips
      drawn_events: dict[EventType, int]
      consensus: ConsensusResult

  @dataclass(frozen=True)
  class MarketRows:
      prices: list[Price]
      curves: list[CurvePoint]
      consensus: list[ConsensusRow]
      calendar: list[CalendarEvent]
      regimes: list[RegimeSpan]

  def market_rng(root_seed: int) -> RngFor          # lambda *keys: stream(root_seed, "market", *keys)
  def generate_seed(config: Config, seed: str, instruments: Sequence[Instrument],
                    shocks: DriverShocks, axis: SimAxis) -> SeedMarket
  def to_rows(market: SeedMarket) -> MarketRows     # horizon days only
  ```
  `generate_seed` order: schedule, `RegimeLookup` with the seed's first regime for burn-in, `regime_path`, `sample_events`, `generated_rows`, `build_jumps`, `seed_driver`, processes (equities, rates, credit with rates output, commodities, fx), merge, `build_consensus`, calendar = sampled + generated + flips. Prices get `spread_bp` only for credit issuers. Floats are written as numpy floats converted to Python `float`.

- [ ] **Step 1: Write failing tests** (demo-sized universe: `n_equities 20, n_sectors 5, n_credit_issuers 12`):
  - Row counts per seed: prices `(n_instruments - n_curves) * 260`, curves `(4 * 4 + 12 * n_commodities) * 260`, consensus `n_instruments * 260`, regimes 3; all dates within the horizon.
  - Seed equality (decision 11): config with seeds `A` and `B` both `(range, risk_off, risk_on)`, all event `jump_size = 0`, `flips_per_instrument_year = 0`: prices, curves and consensus rows identical apart from `seed`.
  - Seeds with different orders differ in prices.
  - Drift signs on a long axis: for each family index (definitions in Task 12), under a constant `risk_off` path over 20 years, the mean daily index return has the sign of `driver_corr x driver_mean` (equities: positive loading; rates 10Y: sign of `rates.driver_corr`; credit spread: sign of `credit.driver_corr`); under `risk_on` the opposite. Build `ProcessInputs` directly in the test with `constant_path`; do not add a production parameter for it.
  - Determinism: two calls with the same inputs give identical rows.
- [ ] **Step 2: Run.** Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add per-seed market generation`

---

### Task 12: Realised-moment check

**Files:**
- Create: `src/pm_traitbench/market/check.py`
- Test: `tests/market/test_check.py`

**Interfaces:**
- Consumes: `SeedMarket`, `Instrument`, `Config`, `MarketCheckError`, helpers `log_grid_step`, `nearest_level`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class CheckMetric:
      seed: str; regime: Regime | None; family: Family; metric: str   # "vol", "corr", "round_level_tests", "count:<name>"
      target: float; realised: float; tolerance: float; passed: bool
  @dataclass(frozen=True)
  class CheckReport:
      seed: str
      metrics: tuple[CheckMetric, ...]
      def to_dict(self) -> dict[str, Any]     # JSON-safe
  def family_indices(market: SeedMarket, instruments: Sequence[Instrument]) -> dict[Family, np.ndarray]
  def implied_moments(family: Family, params: RegimeParams, instruments: Sequence[Instrument],
                      config: Config) -> tuple[float, float]   # (annual vol target, corr with z)
  def check_market(market: SeedMarket, instruments: Sequence[Instrument], config: Config) -> CheckReport
  ```
  Family indices (daily series of changes on the axis, index `t` = change from `t-1` to `t`):
  - equities: equal-weight mean of log price changes.
  - rates: equal-weight mean over curves of 10Y changes in bp.
  - credit: change in log of the equal-weight mean spread over IG issuers (AA, A, BBB).
  - commodities: equal-weight mean of log M1 changes.
  - fx: equal-weight mean over non-USD currencies of `dv_c`, recovered from each currency's USD pair price (`+dlog P` for XXXUSD, `-dlog P` for USDXXX).

  Binding implied moments (annual vol `V`, correlation `rho`) with multiplier `m`:
  - Each index is `sum_j b_j z + (independent terms)`. Compute common loading `B` and independent variance `I` (annualised units), then `V = m * sqrt(B^2 + I)`, `rho = B / sqrt(B^2 + I)`.
  - equities: `B = mean(beta) * market_vol`, `I = idio_vol^2 / n`.
  - rates: `B = level_vol * c`, `I = level_vol^2 * (1 - c^2) / n + 0.15^2 * slope_vol^2 / n` (in bp).
  - credit: `k = sqrt((a^2 + a^-2) / 2 - (a - 1/a)^2 / (2 * pi))` (sd of the centred asymmetric shock); `B = factor_vol * k * c`, `I = factor_vol^2 * k^2 * (1 - c^2) + issuer_vol^2 / n_IG`.
  - commodities: `N` commodities; `B = mean_i(vol_g(i) * c_g(i))`; `I = sum_g (n_g / N)^2 * vol_g^2 * (1 - c_g^2) * share + sum_i vol_g(i)^2 * (1 - c_g(i)^2) * (1 - share) / N^2`.
  - fx: `n` currencies; `B = mean(vol_c * c_c)`; `I = sum_c vol_c^2 * (1 - c_c^2) / n^2`.

  Checks, per regime span (horizon days only; the first horizon day's change uses the last burn-in day):
  - vol: realised annualised sd (`sd * sqrt(252)`) within `vol_tolerance` relative of `V`.
  - corr: `|realised - rho| <= corr_tolerance_se * (1 - rho^2) / sqrt(n_days)`; skip (no metric) where `B == 0`.
  - round-level tests in the range span only, family mean per instrument `>= min_round_level_tests`, for equities (price, grid `log_grid_step`), commodities (M1, grid `log_grid_step`), FX pairs (grid 1.0 for JPY-quoted pairs else 0.01) and curves (10Y yield, grid 0.25). Band half-width = `round_level_band * level`. Grid levels are positive only: a nearest level of 0 (a 10Y yield near the floor) is replaced by one step, so the band never collapses to zero width. A test (decision 2): previous close outside every band and current close inside a band, or both closes outside bands with a grid level strictly between them. Use the grid step of the current close; skip the check when the seed has no range span (a seed always has one given validation, so assert it).
  - counts (regime `None`): calendar rows per sampled type equal `drawn_events`; flip rows per instrument equal `drawn_flips`; each commodity has exactly one `contract_expiry` per month whose third Friday is in the horizon; one `positioning_report` per horizon Friday.
  - On any miss raise `MarketCheckError` naming seed, regime, family, metric, target and realised value of the first miss plus the total miss count, for example `market check failed for seed A, regime risk_off, family credit, metric vol: target 0.4040, realised 0.6210 (2 misses)`.

- [ ] **Step 1: Write failing tests:**
  - Default config, full default universe, all three seeds: `check_market` passes (no raise) and every metric has `passed=True`. If this fails at default config, stop and report the failing metrics (see Global Constraints).
  - Rigged target: generate seed A with default config, then call `check_market` with a config whose `risk_off` vol multiplier is 3.0: raises `MarketCheckError` with message containing `seed A`, `regime risk_off`, `metric vol`.
  - Rigged sign: config with `rates.driver_corr = -0.3` against a market generated at +0.3 raises naming `family rates` and `metric corr`.
  - Count miss: drop one `positioning_report` row from the market's calendar (build a modified `SeedMarket` with `dataclasses.replace`) and assert the raise names the count metric.
  - Round-level test counter unit test on a hand-made price series: crossing counted, entry counted, staying inside the band not counted twice, move entirely between two levels not counted.
  - `implied_moments` equities matches the hand formula for a 2-instrument universe.
  - `to_dict` round-trips through `json.dumps`.
- [ ] **Step 2: Run.** Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market realised-moment check`

---

### Task 13: Market stage, CLI registration, demo config, README

**Files:**
- Create: `src/pm_traitbench/market/stage.py`
- Modify: `src/pm_traitbench/pipeline.py`
- Modify: `configs/demo.yaml`
- Modify: `README.md`
- Test: `tests/market/test_stage.py`, `tests/test_cli_stages.py` (extend if it enumerates stages)

**Interfaces:**
- Consumes: `build_universe`, `build_axis`, `draw_shocks`, `generate_seed`, `to_rows`, `check_market`, `MARKET_TABLES` and the six specs, `Stage`.
- Produces:
  ```python
  def run(config: Config, store: DataStore) -> dict[str, Any]
  MARKET_STAGE = Stage(number=2, name="market",
                       help="simulate prices, curves, consensus, calendar and regimes per market seed",
                       run=run, reads=(), writes=MARKET_TABLES)
  ```
  `run`: universe once (`stream(root, "market", "universe")`), axis once, shocks once; for each seed in `config.population.market_seeds` in order: `generate_seed`, `check_market` (raises on a miss before anything is written), `to_rows`; concatenate; write instruments, prices, curves, consensus, calendar, regimes; return `{"check": {seed: report.to_dict()}}`.
  `pipeline.STAGES = (SAMPLE_STAGE, MARKET_STAGE)`.
  `configs/demo.yaml` gains `market: {universe: {n_equities: 20, n_sectors: 5, n_credit_issuers: 12}}` with a one-line comment in the file's style.
  README: add the `market` command to Usage (`uv run pm-traitbench market --config configs/demo.yaml --data-dir data`), what it writes (six tables under `data/market/`), a note recommending `output.tables` parquet overrides for `market/prices`, `market/curves`, `market/consensus` on the full run, and a "Market model" section listing each simplification with its cost and why it is acceptable, covering: no sector factor; one common driver; a regime is a driver mean and a vol multiplier; no vol clustering inside a regime; rates are level and slope only; credit is one factor times rating base times issuer noise (no defaults, static rating band, issuer vol a guess); commodities are spot plus a scripted curve, no seasonality, natural gas excluded; FX is zero-drift per currency, no carry; consensus is a lagged moving average plus scripted flips, credit positioning is a survey analogue; one sampled event type per family; no holidays or intraday data; sourcing (sourced values checked against named FRED series or papers on 2026-09-22, OAS medians from a three-year calm window, guessed values listed); the check (vol and model-implied correlation per regime and family, event counts, round-level tests; drift is not checked because its standard error over a 14-22 week span exceeds the drift). The README must not mention docs, specs or plans.

- [ ] **Step 1: Write failing tests** in `tests/market/test_stage.py`:
  - Demo config via `load_config(Path("configs/demo.yaml"))` into `tmp_path`: `run_stage(MARKET_STAGE, ...)` writes the six files under `tmp_path/"market"`; each reads back through the store without duplicate-key errors; `run_metadata/market.json` has `"check"` with key `"A"`.
  - Parquet override for `market/prices` writes `market/prices.parquet`.
  - Second run without `--force` raises `StageIOError`; with force succeeds.
  - End-to-end CLI: `main(["market", "--config", "configs/demo.yaml", "--data-dir", str(tmp_path)])` returns 0.
  - A config whose check would fail (rig by monkeypatching `check_market` in `pm_traitbench.market.stage` to raise `MarketCheckError`) makes the CLI return exit code 1 and writes no market files.
  - `pipeline.STAGES` names are `("sample", "market")`.
- [ ] **Step 2: Run** `uv run pytest tests/market/test_stage.py -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement**, then run the CLI once by hand on the demo config into a scratch directory and on the default config (all three seeds) and record wall time and row counts in the report.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add market stage and cli registration`
