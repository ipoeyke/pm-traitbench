**Tier:** heavy
**Escalation threshold:** n/a (heavy)

# Design: market stage

Date: 2026-09-22. Source of requirements: `docs/pm-dataset-plan.md`, sections 2, 8, 9 (stage 2). Builds on the foundation of `docs/specs/2026-09-20-foundation-sampling-design.md`.

## Scope

Second sub-project. Delivers **stage 2, market**: a synthetic, regime-scripted market for every configured market seed, written as six tables under `market/`. Pure numpy, no model calls. Independent of stage 1: both read only config, so either can run first.

In scope: instrument universe, regime schedule per seed, shared daily drivers, one price process per family (equities, rates, credit, commodities, FX), event calendar with scripted price effects, consensus and positioning series, burn-in, a realised-moment check, config section, row models, table specs, a store change for the `market/` subdirectory, CLI registration, tests.

Out of scope: the engine's reads of the market (`days_to_expiry`, curve trades, roll logic), idea-scope rules, Gate 1 trigger counting, holidays, intraday data, any realism classifier test.

## Repo rule that binds all code

Unchanged from the foundation spec: no file in the repo may mention `docs/`, a spec, or a plan. Docstrings state the rule itself. Citing a published paper or a named public data series as the basis of a parameter is allowed.

## Decisions taken

1. All four families plus FX, all configured seeds, in this stage. Seeds absent from `population.market_seeds` are not generated, so the demo config (seed A only) stays small.
2. New table `market/instruments.jsonl`: static per-instrument facts (family, kind, sector, rating band, commodity group, duration, beta, expiry rule), no seed column, shared by all seeds. Plan deviation: section 8 lists five `market/` files; this adds a sixth.
3. Commodities are one instrument per commodity, not one per contract month. `prices.price` is the front month; the futures curve `M1..M12` lives in `curves`; expiry follows a monthly rule stored on the instrument. The engine derives `days_to_expiry` from the rule and the date.
4. Sovereign curves are instruments too (`kind = sovereign_curve`, ids `RT-USD` etc.). They have no `prices` row; their tenors live in `curves`. Consensus, calendar and signposts key on one `instrument_id` for every family.
5. `prices` is wide: `seed, date, instrument_id, price, spread_bp`. A rule's `field` is a column name.
6. `curves` has one shape for both uses: `seed, date, curve_id, tenor, level`. `level` is a yield in percent for sovereign tenors (`2Y, 5Y, 10Y, 30Y`) and a futures price for commodity tenors (`M1..M12`).
7. Consensus carries a continuous score and a derived label for both the street view and positioning. Street view updates on the instrument's event days plus one weekly revision day; positioning updates only on weekly report days. Both step between updates. Rows are written daily for every instrument with consensus, so no consumer forward-fills.
8. Calendar events are inputs to the price process, not decoration. Each event row carries a signed `surprise` in [-1, 1]; the price effect is `surprise x jump_size[event]` with `jump_size` always positive. One-sided events carry the sign in the label (`rating_downgrade`, `rating_upgrade`) and the generator draws the surprise with that sign, so a row's sign always agrees with its label. There are no guidance, auction or OPEC events: a disappointing result is an `earnings` row with a large negative surprise, and curves and energy already have one sourced dated event each.
9. Every dated mechanism appears in `calendar.jsonl`: sampled events, generated `contract_expiry` and `positioning_report` rows, and `consensus_flip` rows written by the consensus module. Regime boundaries stay in `regimes.jsonl` only.
10. Burn-in of 60 trading days before `calendar.start` with the first regime's parameters and no events, rows dropped, so day one already has lagged history behind prices, street score and positioning.
11. The three seeds share every idiosyncratic and driver draw; seed-specific draws are only surprises and flip dates. Regime order is the only structural difference between seeds, as the plan requires.
12. The stage checks its own realised moments against the regime parameters and fails on a miss. Realised drift is not checked: over a 14-22 week regime span its standard error is larger than the drift itself.
13. Credit spreads get an asymmetry knob: widening shocks are scaled up and tightening shocks scaled down, because daily OAS changes have positive skew (mean widening step about 1.1 times the mean tightening step), and slow tightening is what a disposition-biased credit PM sits through.
14. Equities and credit issuers are abstract (numbered ids, sector and rating tags) because single names carry memorable stories. Commodities and currencies keep real generic names (crude, gold, USD curve, EURUSD): a PM cannot talk about "curve 3" or "pair 7", FX quoting conventions and cross consistency come for free with real currencies, and a unit is not a story, so with round starting levels and seeded paths there is nothing to memorise.
15. Model forms cite the canonical paper; magnitudes cite a recent paper or a named public series in each config `note`. No further literature search: the check here and Gate 1 later are the tests of whether a magnitude is wrong for this dataset.
16. A regime is two numbers, `driver_mean` and `vol_multiplier`, plus the round-level pull. Every family's drift per regime follows from its correlation with the common driver, so no drift is a config field. Loadings, vols and commodity curve slopes are constants across regimes. `sourced` means checked on 2026-09-22 against the named FRED series or taken from the named paper; the computation is not in the repo. Figures that could not be checked are tagged guess with the series to check named in the note. Natural gas is left out of the energy group: Henry Hub spot vol is above 100% a year, an outlier that one group vol cannot carry.

## Package layout

```
src/pm_traitbench/
  enums.py            # + Family, InstrumentKind, EventType, StreetView, Positioning
  config.py           # + MarketConfig and sub-sections; Config cross-checks
  errors.py           # + MarketCheckError
  tables/
    schema.py         # + Instrument, Price, CurvePoint, ConsensusRow, CalendarEvent, RegimeSpan
    specs.py          # + MARKET_INSTRUMENTS, MARKET_PRICES, MARKET_CURVES, MARKET_CONSENSUS,
                      #   MARKET_CALENDAR, MARKET_REGIMES
    store.py          # write creates the table's parent directory; run metadata takes extras
  market/
    __init__.py
    stage.py          # MARKET_STAGE: number 2, "market"; reads (), writes the six specs
    universe.py       # build_universe(config, rng) -> list[Instrument]
    regimes.py        # build_schedule(config, seed, timeline) -> list[RegimeSpan]; RegimeLookup
    drivers.py        # shared daily shocks (common, per commodity group)
    calendar.py       # sample_events(...), generated rows, jump application helpers
    processes/
      __init__.py
      equities.py rates.py credit.py commodities.py fx.py
    consensus.py      # street score, positioning, flips
    check.py          # realised moments vs targets; raises MarketCheckError
  pipeline.py         # STAGES = (SAMPLE_STAGE, MARKET_STAGE)
tests/market/         # one test module per market module, plus test_stage.py
configs/demo.yaml     # unchanged population; market sizes reduced for speed
```

Dependency direction: `market` depends on `config`, `rng`, `timeline`, `enums`, `tables`. Nothing depends on `market`. Only `market/stage.py` touches the store. `processes/*` depend on `drivers`, `regimes`, `calendar` (for event effects) and nothing else in `market`.

## Time axis

`Timeline` gives the 260 weekday dates of the horizon. The stage prepends `burn_in_days` weekdays before `calendar.start` (walking back Mon-Fri) to form the simulation axis of `T = burn_in_days + 260` days. Every process runs on the full axis; `RegimeLookup` returns the first regime of the seed for burn-in dates; the calendar has no rows in burn-in; output rows are filtered to `date >= calendar.start`.

## Config

New section `market:` on `Config`, all fields with `basis` and `note`, frozen, `extra="forbid"`.

### `market.universe`

| Field | Default | Basis |
|---|---|---|
| `n_equities` | 80 | design |
| `n_sectors` | 10 | design: sector is a static tag for exclusion rules only; there is no sector factor in the price process |
| `n_credit_issuers` | 48 | design |
| `credit_band_shares` | AA 0.15, A 0.25, BBB 0.30, BB 0.20, B 0.10 | guess: an IG-heavy mix in the spirit of the ICE BofA US Corporate and High Yield indices, HY share raised so the HY book has enough names; index fact sheets are the series to check |
| `curves` | USD, EUR, GBP, JPY | design |
| `commodities` | energy 6, industrial_metals 4, precious 3, agriculture 7 (names fixed in code: crude, brent, gasoil, gasoline, heating_oil, coal; copper, aluminium, nickel, zinc; gold, silver, platinum; wheat, corn, soybeans, sugar, coffee, cotton, cocoa) | design |
| `fx_pairs` | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, EURGBP, EURJPY, AUDJPY | design |
| `equity_beta_range` | (0.6, 1.4) uniform | guess: about the 10th-90th percentile of single-name betas in a large-cap index; not verified here |
| `credit_duration_range` | (3.0, 8.0) uniform, years | guess: HY index duration about 4, IG about 7, so the span covers both; index fact sheets are the series to check |
| `expiry_rule` | `monthly_third_friday` (only value in this build) | design |

Instrument ids: `EQ-0001..`, `CR-IG-001..` for AA/A/BBB, `CR-HY-001..` for BB/B, `RT-USD`, `CM-CRD` (three-letter code per commodity, fixed in code), `FX-EURUSD`. Equity and credit sectors are ten abstract labels (`sector_01..sector_10`); each credit issuer gets a sector and a currency drawn uniformly from the curve currencies.

### `market.seeds` and boundaries

| Field | Default | Basis |
|---|---|---|
| `seeds` | `{A: [range, risk_off, risk_on], B: [risk_on, range, risk_off], C: [risk_off, risk_on, range]}` | design, plan section 2 |
| `boundary_weeks` | (16, 30): regime 1 is weeks 1-16, regime 2 weeks 17-30, regime 3 weeks 31-52 | design |
| `burn_in_days` | 60 | design, longest consensus lag |

Validation: every seed lists exactly three regimes and each regime once (every PM sees every regime); `boundary_weeks` strictly increasing and inside `1..n_weeks - 1`; `Config` cross-check: `market.seeds` keys cover `population.market_seeds`.

### `market.regimes`

Regimes are defined by two numbers each, not by a parameter set per family. Everything else per family is a constant.

| Field | range | risk_off | risk_on | Basis |
|---|---|---|---|---|
| `driver_mean` | 0.00 | -0.09 | 0.06 | design: the regime definition, expressed as the equity index return over the span: `-0.09 x 0.16 x sqrt(252)` is about -23% annualised, `+0.06` about +15%; every other family's drift follows from its loading |
| `vol_multiplier` | 1.0 | 1.6 | 0.95 | sourced: VIX median 2000-2026 conditioned on the trailing 60-day equity return (FRED VIXCLS, NASDAQCOM): 27.6 when below -10%, 16.3 when above +5%, 17.0 in between; one ratio applied to every family |
| `mean_reversion_kappa` | 0.05 | 0.0 | 0.0 | design: round-level pull in range only; half-life about 14 days, long enough to be retested, short enough to matter in 16 weeks |

Stored as one `RegimeParams(driver_mean, vol_multiplier, mean_reversion_kappa)` per regime, keyed by `Regime`.

### `market.families` (constants across regimes, annualised)

| Field | Default | Basis |
|---|---|---|
| `equity.market_vol` | 0.16 | sourced: S&P 500 realised vol 2016-2026 18.1%, 15.2% excluding 2020 (FRED SP500); VIX median 17.6 1990-2026 |
| `equity.idio_vol` | 0.25 | sourced: Campbell, Lettau, Malkiel and Xu (2023) |
| `rates.level_vol_bp` | 90 | sourced: realised vol of daily 10Y yield changes 1990-2026 is 92bp a year (FRED DGS10) |
| `rates.slope_vol_bp` | 60 | sourced: realised vol of daily 2s10s changes 2000-2026 is 62bp a year (FRED T10Y2Y) |
| `rates.driver_corr` | 0.3 | sourced: correlation of daily 10Y yield changes with equity returns 0.34 for 2000-2020 and 0.26 for 2000-2026 (FRED DGS10, NASDAQCOM); positive means yields fall when equities fall, the post-2000 regime of Campbell, Pflueger and Viceira (2020); it turned negative in 2022 |
| `credit.factor_vol` | 0.25 | sourced: annualised vol of daily log OAS changes 2023-2026, IG 0.23 (FRED BAMLC0A0CM), HY 0.33 (BAMLH0A0HYM2) |
| `credit.hy_vol_multiplier` | 1.4 | sourced: HY over IG log-OAS vol, 0.33 / 0.23, applied to the factor for BB and B issuers |
| `credit.driver_corr` | -0.5 | sourced: correlation of daily log OAS changes with S&P 500 returns 2023-2026, IG -0.43, HY -0.62 (FRED) |
| `credit.asymmetry` | 1.1 | sourced: mean widening step over mean tightening step of daily log OAS 2023-2026 is 1.04-1.11 across bands, skew 0.5-0.9 (FRED); widening shocks scaled by this, tightening by its reciprocal |
| `credit.issuer_vol` | 0.15 | guess: issuer-level OAS data is not public; set to about half the index vol so the factor dominates in stress |
| `commodity.group_vol` | energy 0.40, industrial_metals 0.21, precious 0.15, agriculture 0.23 | sourced: WTI daily 2000-2026 47%, 37% excluding 2020 (FRED DCOILWTICO); copper monthly 2000-2026 21% (PCOPPUSDM); wheat 25% and maize 21% monthly (PWHEAMTUSDM, PMAIZMTUSDM); precious is a guess at gold's usual 15%, no free daily series found |
| `commodity.driver_corr` | energy 0.15, industrial_metals 0.3, precious -0.1, agriculture 0.0 | sourced: WTI vs S&P 500 daily 2016-2026 0.15; copper vs NASDAQ monthly 2000-2026 0.30; wheat -0.01 and maize 0.06 monthly (FRED); precious is a guess at slightly negative |
| `commodity.curve_slope` | energy -0.05, industrial_metals 0.01, precious 0.03, agriculture 0.04 | sourced: average annual roll yield per commodity, Erb and Harvey (2006), Gorton and Rouwenhorst (2006); negative is backwardation |
| `commodity.group_share` | 0.6 | guess: share of single-commodity variance explained by its group index, about 0.5-0.7; not verified here |
| `fx.currency_vol` | EUR 0.09, GBP 0.09, JPY 0.10, AUD 0.12, CHF 0.10, CAD 0.08 | sourced: realised vol of daily USD rates 2000-2026 (FRED DEXUSEU, DEXUSUK, DEXJPUS, DEXUSAL, DEXSZUS, DEXCAUS) |
| `fx.driver_corr` | EUR 0.1, GBP 0.2, JPY -0.1, AUD 0.3, CHF 0.0, CAD 0.25 | sourced: correlation of each currency's daily return vs USD with S&P 500 returns 2016-2026: 0.11, 0.21, -0.09, 0.30, 0.02, 0.24 (FRED); the yen safe-haven effect is weaker than folklore |
| `student_t_df` | 4 | sourced: S&P 500 daily return kurtosis 2016-2026 is about 20 (FRED SP500); a t(4) has infinite kurtosis, t(5) about 9, so 4 errs on the fat side |

`driver_corr` is used directly as the loading on the common driver `z`; with independent residual noise the realised correlation equals it, which is what the check measures. Drift per family per regime is `driver_corr x vol x driver_mean x sqrt(252)` annualised and is not a config field.

### `market.levels` (starting values, seed-independent)

Starting levels do not affect dynamics; they are design choices set to round values near 2024-2025 levels so that round-level grids and dialogue read naturally.

| Field | Default | Basis |
|---|---|---|
| `equity_price_range` | LogUniform(10, 400) | design |
| `curve_start` | USD (4.0, 3.9, 4.1, 4.4); EUR (2.4, 2.3, 2.5, 2.8); GBP (4.2, 4.0, 4.2, 4.6); JPY (0.4, 0.5, 1.0, 2.0) for 2Y, 5Y, 10Y, 30Y, percent | design |
| `credit_base_spread_bp` | AA 50, A 70, BBB 105, BB 180, B 305 | sourced: ICE BofA OAS medians 2023-09 to 2026-09 by rating (FRED BAMLC0A2CAA, BAMLC0A3CA, BAMLC0A4CBBB, BAMLH0A1HYBB, BAMLH0A2HYB; only three years are downloadable), a calm window; the Moody's Baa minus 10Y median 1990-2026 is 214bp (DBAA, DGS10) for scale |
| `commodity_start` | fixed per commodity in code (crude 72, gold 2400, copper 9000, wheat 600, ...) | design |
| `fx_start` | EURUSD 1.10, GBPUSD 1.28, USDJPY 150, AUDUSD 0.66, USDCHF 0.88, USDCAD 1.36 | design |
| `yield_floor_pct` | 0.0 | design |

### `market.events`

Rates are calendar facts unless marked; jump sizes are published event-day statistics. Each family has at most one sampled event type, and only rating actions are one-sided.

| Event | Family | Per instrument-year | Unit of jump | `jump_size` | Sign rule | Basis |
|---|---|---|---|---|---|---|
| `earnings` | equities | 4, quarterly with +-5 day jitter | log return | 0.05 | two-sided | sourced: Dubinsky, Johannes, Kaeck and Seeger (2019) report option-implied and realised earnings-day moves of about 5% for large caps |
| `rating_downgrade` / `rating_upgrade` | credit | 0.2 / 0.1 | relative spread | 0.20 | label | guess: rates are a reading of agency one-year migration tables, not verified here; the spread reaction direction is from Hand, Holthausen and Leftwich (1992), its size a guess |
| `cb_meeting` | curves | 8 | bp on level | 8 | two-sided | sourced: eight scheduled FOMC meetings a year; Gürkaynak, Sack and Swanson (2005) measure 10Y responses to meeting-day surprises of several bp |
| `inventory_report` | energy | 52, weekly | log return | 0.012 | two-sided | sourced: WTI daily sd on Wednesdays (EIA day) 3.15% vs 2.93% other days 2000-2026 (FRED DCOILWTICO); the incremental sd is about 1.2% |
| `crop_report` | agriculture | 12, monthly | log return | 0.03 | two-sided | sourced: monthly WASDE report; Adjemian (2012) quantifies the announcement effect on grain futures at a few percent on report day |
| `macro_print` | all (`instrument_id` null) | 12, monthly | added to the common driver for the day | 1.0 driver sd | two-sided | design: one big print a month, sized as one ordinary driver shock |
| `contract_expiry` | commodities | monthly, from `expiry_rule` | none | none | null surprise | design |
| `positioning_report` | all (`instrument_id` null) | weekly, Friday | none | none | null surprise | sourced: CFTC COT release cadence |
| `consensus_flip` | all with consensus | `flips_per_instrument_year` | none (effect is on the score) | none | null surprise | design |

Surprise magnitude ~ Beta(2, 2) mapped to [0, 1] (hump in the middle, so most surprises are moderate and extremes rare), signed uniformly for two-sided events and by the label for one-sided ones. Event dates are drawn once per instrument (seed-independent); surprises per seed. `affected` is the instrument's family, or `all` for macro and positioning rows.

### `market.consensus`

Both series are exponential moving averages of a trailing-return signal, updated on their own days; one window per series is the only shape parameter.

| Field | Default | Basis |
|---|---|---|
| `street_window_days` | 20 | guess: one month of price is what a sell-side revision reacts to; also the EMA half-life |
| `positioning_window_multiple` | 3 | design: positioning builds over a quarter, three times the street view's month; the positioning window is this times `street_window_days` |
| `revision_weekday` | Wednesday | design |
| `view_threshold` | 0.25 (`street_view` overweight above, underweight below the negative) | design |
| `flips_per_instrument_year` | 2 | design: Gate 1 needs about 14 herding opportunities per PM-year; a PM holds or enters about 10 instruments over a year, so 2 flips each gives about 20 |
| `positioning_threshold` | 20 / 80 (`crowded_short` below, `crowded_long` above) | design |
| `report_weekday` | Friday | design: the CFTC publishes the Commitments of Traders report on Fridays, so positioning steps on Fridays |

Flip magnitude is fixed at twice `view_threshold` (design: past the label, not full conviction). No noise terms: the series step on event and report days and their randomness comes from prices and flips.

### `market.check`

| Field | Default | Basis |
|---|---|---|
| `vol_tolerance` | 0.30 relative | design |
| `min_driver_corr` | 0.20 absolute, where `driver_corr` is non-zero | design |
| `min_round_level_tests` | 3 per instrument in the range regime | design |
| `round_level_band` | 0.002 relative | design |

## Universe (`universe.py`)

`build_universe(config, rng) -> list[Instrument]`, stream `("market", "universe")`. Deterministic order: equities, credit, curves, commodities, FX. Assigns sectors round-robin, betas and durations uniform, credit bands by shares (largest remainder so counts sum), currencies uniform. Commodity and FX entries come from fixed tables in the module; config controls only which groups and pairs are included.

## Regimes and drivers

`build_schedule(config, seed, timeline)` returns three `RegimeSpan` rows from `market.seeds[seed]` and `boundary_weeks`: `date_start` is the Monday of the span's first week, `date_end` the Friday of its last week. `RegimeLookup(spans, burn_in_regime)` maps any axis date to a regime.

`drivers.py` draws, on the full axis, seed-independent standard-normal shocks: `u` (common risk driver, positive = risk-on day) and one per commodity group. Streams `("market", "driver", name)`, one whole vector per stream. Per seed and date the driver is `z = u + driver_mean(regime)`; `macro_print` events add `surprise x 1.0` on their date. Per-family daily vol is `vol x vol_multiplier(regime) / sqrt(252)`.

## Processes

Each `simulate(universe, axis, regimes, events, params, rng_for) -> ProcessOutput` where `rng_for(*keys)` yields `stream(root, "market", *keys)`; instrument noise is keyed `("noise", instrument_id)`, drawn as one vector, so adding an instrument changes no other. Regime values (`driver_mean`, `vol_multiplier`, `kappa`) are looked up per date. All processes apply the event jumps of their instruments on the event date after the diffusion step. Below, `sigma` means the family's daily vol after the regime multiplier and `c` the family's `driver_corr`; residual noise `e` is standard normal and `t` Student-t(df) scaled to unit variance.

**Equities.** Per name `i`, log return
`r = beta_i * sigma_m * z + sigma_i * t_i - kappa * (log p - log round(p))`,
`round(p)` the nearest grid level (grid step `10^floor(log10 p) / 2`), `kappa` non-zero in range only. Price `p_t = p_{t-1} exp(r)`. Output `Price(price, spread_bp=None)`. The market factor is `sigma_m z` itself, so equities define the driver's sign.

**Rates.** Per curve, level `L` and slope `S` in percent:
`dL = sigma_L * (c * z + sqrt(1 - c^2) * e_L)` (with `c > 0`, yields fall on risk-off days), `dS = sigma_S * e_S`, tenor yield `y_k = L + w_k S` with `w = (-0.5, -0.15, 0.15, 0.5)` for 2Y, 5Y, 10Y, 30Y (Diebold and Li 2006 form, curvature fixed). Range regime pulls the 10Y toward the nearest 25bp level through `L`. Yields floored at `yield_floor_pct`. Event jumps hit `L`. Output `CurvePoint` rows; no `Price` row.

**Credit.** One credit factor `F` per seed (log process): `d log F = sigma_F * a(shock) * shock`, `shock = c * z + sqrt(1 - c^2) * e_F`, `a` = `asymmetry` when the shock widens (positive `d log F`) and `1 / asymmetry` when it tightens. Issuer spread `s_i = base_band * F^(h_i) * exp(x_i)` with `h_i = hy_vol_multiplier` for BB and B issuers and 1 otherwise, and `x_i` an AR(1) lognormal issuer noise at `issuer_vol`. Range regime pulls `s_i` toward the nearest 10bp. Bond price `p_t = p_{t-1} * (1 - D_i * (delta s_i + delta y_5Y) / 10000)`, start 100, `y_5Y` from the issuer's currency curve. Output `Price(price, spread_bp)`.

**Commodities.** Per commodity in group `g`, spot log return `r = sigma_g * (c_g * z + sqrt(1 - c_g^2) * (sqrt(share) e_g + sqrt(1 - share) t_i)) - kappa * (log p - log round(p))`. Curve `M_k = spot * (1 + slope_g * k / 12) * exp(0.002 * n_k)` with small per-tenor noise so the curve is not a straight line. `price` = `M1`. Output `Price(price, spread_bp=None)` and 12 `CurvePoint` rows per day.

**FX.** One log value per non-USD currency against USD: `dv_c = sigma_c * (c_c * z + sqrt(1 - c_c^2) * e_c) - kappa * (...)`; pair prices from the currency ratio in the pair's quoting convention, so crosses are arbitrage-consistent. Round grid 0.01 (1.0 for JPY quotes). Output `Price(price, spread_bp=None)`.

## Calendar (`calendar.py`)

`sample_events(universe, timeline, config, rng_for, seed)` produces the sampled rows: dates per instrument on the seed-independent stream `("events", instrument_id)`; surprises on `(seed, "surprise", instrument_id)`. Quarterly and monthly events are placed on a regular grid with jitter; rate-based ones are Poisson counts placed uniformly. `generated_rows` adds `contract_expiry` (third Friday of each month, per commodity) and `positioning_report` (each Friday). Consensus appends `consensus_flip` rows. Dates outside the horizon (burn-in) are never generated. The module also exposes `jumps_for(instrument_id, seed)` used by the processes.

## Consensus (`consensus.py`)

Per seed, per instrument with consensus (all instruments, including curves), on the full axis, with `lam(W) = 1 - 2^(-1/W)` the EMA weight for half-life `W`:

- `trend_t = clip(ret_{W}(t) / (sigma_family * sqrt(W/252)), -1, 1)` where `ret_W` is the trailing `W`-day log return of `price` (10Y yield change, sign flipped, for curves), `W = street_window_days`.
- On update days (the instrument's event days, the weekly revision weekday, and flip days): `score_t = score_{t-1} + lam(W) * (trend_t - score_{t-1})`; on a flip day, `score_t = -sign(score_{t-1}) * 2 * view_threshold`, and a `consensus_flip` calendar row is written. Other days hold. Start `score = 0` at burn-in start.
- `positioning_pct`: on report days `target = 50 + 40 * tanh(2 * trend_{P})`, `P = positioning_window_multiple x street_window_days`, `pct = pct_{t-1} + lam(P / 5) * (target - pct_{t-1})` (half-life in reports, five trading days apart), clipped to [0, 100]; other days hold. Start 50 at burn-in start.
- Labels from thresholds. Flip dates on `(seed, "flips", instrument_id)`; no other seed-dependent draw.

## Tables

Row models in `tables/schema.py`, every field with a `description`:

- `Instrument`: `instrument_id, family (Family), kind (InstrumentKind), name, currency, sector, rating_band, commodity_group, duration_years, beta, expiry_rule`. Validator: the set of non-null optional fields matches `kind` (equity: sector, beta; credit_issuer: sector, rating_band, duration_years; sovereign_curve: none; commodity: commodity_group, expiry_rule; fx_pair: none).
- `Price`: `seed, date, instrument_id, price, spread_bp`. `price > 0`; `spread_bp` null unless a credit issuer.
- `CurvePoint`: `seed, date, curve_id, tenor, level`. `tenor` a `Tenor` StrEnum (`2Y, 5Y, 10Y, 30Y, M1..M12`).
- `ConsensusRow`: `seed, date, instrument_id, street_score, street_view, positioning_pct, positioning`. Validator: `street_score` in [-1, 1], `positioning_pct` in [0, 100]. Label-threshold agreement is not validated on the row because thresholds are config; `test_consensus.py` covers it.
- `CalendarEvent`: `seed, date, instrument_id, event (EventType), surprise, affected`. Validator: null `surprise` only for `contract_expiry, positioning_report, consensus_flip`; sign agrees with a one-sided label.
- `RegimeSpan`: `seed, regime, date_start, date_end`, `date_start <= date_end`.

New enums in `enums.py`: `Family(equities, rates, credit, commodities, fx)`, `InstrumentKind`, `EventType`, `StreetView(underweight, neutral, overweight)`, `Positioning(crowded_short, neutral, crowded_long)`, `Tenor`.

Specs in `tables/specs.py` (name, key): `market/instruments (instrument_id)`, `market/prices (seed, date, instrument_id)`, `market/curves (seed, date, curve_id, tenor)`, `market/consensus (seed, date, instrument_id)`, `market/calendar (seed, date, instrument_id, event)`, `market/regimes (seed, date_start)`. A null `instrument_id` sorts first within the key.

Store: `DataStore.write` creates `target.parent`; `write_run_metadata(stage_name, config, extra: dict | None = None)` merges `extra` into the JSON. Output overrides key on the spec name (`output.tables: {market/prices: parquet}`).

Sizes at default config, per seed: prices 157 x 260 = 40,820 rows; curves (16 + 240) x 260 = 66,560; consensus 161 x 260 = 41,860; calendar about 1,700; regimes 3. Three seeds about 452k rows, about 47 MB JSONL; parquet with zstd about 5-8 MB. The README will recommend parquet for `market/prices`, `market/curves`, `market/consensus` on the full run.

## Check (`check.py`)

`check_market(seed, rows, schedule, config) -> CheckReport`, run per seed after generation:

- Realised annualised vol of each family index (equal-weight log returns; for rates the 10Y level change; for credit the log spread of the factor proxy, the equal-weight issuer spread) per regime span, within `vol_tolerance` of the regime's target: the family's vol field times `vol_multiplier` (equities: `sqrt(market_vol^2 + idio_vol^2 / n)` times the multiplier).
- Sign and size of the correlation of each family index with `z` per regime: correct sign and `|corr| >= min_driver_corr` where `driver_corr` is non-zero.
- Counts: sampled events per type equal the drawn counts; flips per instrument equal the drawn counts; every commodity has one `contract_expiry` per month; one `positioning_report` per Friday.
- Round-level tests: in the range span, each equity, commodity, FX pair and 10Y level enters the `round_level_band` around a grid level from outside at least `min_round_level_tests` times.

A miss raises `MarketCheckError` naming seed, regime, family, metric, target and realised value. The report (realised numbers per seed, regime, family) goes into `run_metadata/market.json` under `"check"`.

## Stage and CLI

`MARKET_STAGE = Stage(number=2, name="market", help="simulate prices, curves, consensus, calendar and regimes per market seed", run=run, reads=(), writes=(six specs))`. `run` builds the universe once, then per seed in `config.population.market_seeds`: schedule, calendar, drivers, five processes, consensus, check; concatenates rows; writes the six tables; passes the check reports to `write_run_metadata` through the stage runner (`run_stage` gains an optional `extra` return: `stage.run` may return a dict, which the runner forwards). CLI: `pm-traitbench market --config configs/demo.yaml --data-dir data [--force]`.

`configs/demo.yaml` adds `market.universe` overrides: `n_equities: 20, n_sectors: 5, n_credit_issuers: 12`, commodities and FX unchanged, so the demo market runs in seconds and still exercises every family.

## Errors

`MarketCheckError(PmTraitbenchError)`. Config errors through `ConfigError` as now. A `Family` and `AssetClass` are distinct enums: `rates_credit` PMs trade `rates` and `credit` instruments; the mapping lives in the engine, not here.

## Testing

`tests/market/`:

- `test_universe.py`: counts per family, id uniqueness and format, band shares sum, kind/field validator.
- `test_regimes.py`: seed A/B/C spans match the plan table dates; burn-in dates map to the first regime; config rejects a seed missing a regime; the sign of every family's mean return follows `driver_corr x driver_mean` on a long axis.
- `test_drivers.py`: same vectors for any seed; changing one stream leaves others equal.
- One module per process: output shape (rows per day), positivity, realised vol on a 20-year axis within 10% of target per regime, sign of driver correlation, range-only round-level pull, credit asymmetry (mean absolute widening step larger than tightening), commodity curve sign per regime and group, FX cross consistency (EURGBP = EURUSD / GBPUSD), yield floor.
- `test_calendar.py`: rates per type on a long axis, sign agrees with label, expiry on third Fridays, reports on Fridays, no burn-in rows, seed-independent dates with seed-dependent surprises.
- `test_consensus.py`: stepping (values change only on update days), positioning lags score after a flip, flip rows land in the calendar, flip magnitude equals twice the threshold, label thresholds, day-one score non-zero after burn-in.
- `test_check.py`: passes on default config; fails with the right message when a regime's vol target is rigged.
- `test_stage.py`: demo config writes six files under `market/`, keys unique, `run_metadata/market.json` carries the check report; identical rows across two seeds with identical regime order; end-to-end CLI run.
- `tests/tables/test_store.py`: subdirectory write and read; extras in run metadata.
- `tests/test_config.py`: market defaults, seed coverage cross-check, boundary validation, tolerance signs.

## Sources to record in the README

The freeze stage's README lists every FRED series used to check a magnitude, in the citation form "Federal Reserve Bank of St. Louis, FRED series DGS10, retrieved 2026-09-22", alongside the papers. The series ids are the ones named in the config notes above; the README section is generated from `Config.dump_with_basis()` by collecting `sourced` notes, so the list cannot drift from the config.

## Limitations to record in the README

The repo README gets a "Market model" section when this stage lands, and the freeze-stage dataset README repeats it. Each item names the simplification, what it costs, and why it is acceptable for what the dataset measures.

Simplifications of the variables:

- No sector factor in equities: a stock's return is market plus own noise, so two names in one sector are no more correlated than two names in different sectors. A sector basket diversifies like any basket. No bias fingerprint measures sector co-movement.
- One common driver for cross-family co-movement: correlations between families run through one shock with fixed loadings, so pairwise structure outside it (oil vs the Canadian dollar, gold vs real yields) is absent and correlations do not rise in a crisis beyond the vol multiplier. The engine reads single instruments; allocation decisions only need the regime sign pattern.
- A regime is two numbers, a driver mean and a vol multiplier: drifts, loadings and commodity curve slopes do not change shape with the regime. Regime order is still the only difference between seeds, which is what the drift-vs-regime confound needs.
- No volatility clustering inside a regime: vol is constant within a span and jumps at the boundary, so stops cluster less than in a real drawdown. The vol multiplier is coarse clustering; event jumps add dated spikes.
- Rates are level and slope only: no curvature dynamics, no front end pinned by a policy floor, no negative yields. PM rules key on 10Y levels and 2s10s, both of which live in level and slope.
- Credit is one factor times a rating base times issuer noise: no default jumps, no rating migration except the sampled rating events (the static `rating_band` never changes), no spread floor. Issuer noise is a guess because issuer-level data is not public.
- Commodities are a spot process with a scripted curve: the futures curve is spot times a fixed average roll yield per group, not derived from storage costs or inventories, and there is no seasonality. Natural gas is excluded because its vol does not fit one group value. Roll yield and calendar spreads exist, which is what the roll rule and curve-trade expressions need.
- FX is a zero-drift random walk per currency: no carry, no intervention, no rate-differential drift. FX is a hedge sleeve and one multi-asset expression; the right vol and safe-haven sign are enough.
- Consensus is a lagged moving average of price plus scripted flips: the street has no information of its own, and positioning is a percentile of the same signal on a slower clock. Credit positioning has no public counterpart and is a survey analogue. Herding needs a signal that is not the PM's own and that sometimes flips without cause; both exist.
- Event calendar is one sampled event type per family plus scheduled rows: no guidance changes, auctions or OPEC decisions; a disappointing result is an earnings row with a large negative surprise. Signposts key on the labels that remain.
- No holidays, no intraday data, 260 weekday sessions a year.

Sourcing: magnitudes tagged `sourced` were checked against FRED series or taken from the named paper on 2026-09-22 (retrieval date in the README); the OAS medians come from a three-year calm window because only three years are downloadable; guessed values (issuer vol, rating-action rate and size, street window) are listed with what would verify them.

## Plan edits to make (local docs only)

Section 8: add `market/instruments.jsonl`; note `calendar.jsonl` carries flip, expiry and report rows; consensus columns. Section 2: curves as instruments, one instrument per commodity, burn-in. Section 9 stage 2 paragraph: the check and its exclusions (no drift check).
