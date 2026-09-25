# pm-traitbench

This repo builds the pipeline that generates PM-TraitBench, a synthetic dataset of portfolio manager
behavioural traits for evaluating a copilot's behavioural memory.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run pm-traitbench fetch-market --config configs/demo.yaml --data-dir data
uv run pm-traitbench sample --config configs/demo.yaml --data-dir data
uv run pm-traitbench market --config configs/demo.yaml --data-dir data
uv run pm-traitbench engine --config configs/demo.yaml --data-dir data
uv run pm-traitbench gate1 --config configs/demo.yaml --data-dir data
```

The `sample` stage writes four tables to `data`: `personas`, `traits`,
`rules` and `drift_events`. The `market` stage writes six tables under
`data/market`: `instruments`, `prices`, `curves`, `consensus`, `calendar`
and `regimes`, simulating one market per configured market seed. The
`engine` stage runs the behaviour loop for every non-multi-asset PM and
writes five tables: `ideas`, `ledger`, `rule_events` and `position_days`,
plus `rules` rewritten with every idea-scope rule the run created alongside
the mandate- and self-imposed rules `sample` already wrote; a rerun replaces
the previous run's idea-scope rules rather than adding to them.
`position_days` is a hidden table in full. `ideas` and `ledger` also carry
hidden columns: the PM's own signal, forecast and interval, its street-view
context, conviction and size rank on `ideas`, and each order's bias flag and
driving rule on `ledger`. Hidden data is generator provenance for checking
the engine itself and is never shown to a system under test. The `gate1`
stage recovers each direct-asset PM's eight planted biases from the engine's
ledger and pools them per asset class, writing `gate1_pm` and `gate1_cells`;
it exits 1 when a synthetic recovery test fails, leaving both tables and its
run metadata on disk either way. Pass `--force` to overwrite a table that
already exists. Run `uv run pm-traitbench --help` for the full command list.

`fetch-market` only needs to run first when the config references a real
market seed, as the default and demo configs both do for their pilot seed;
a config with only synthetic seeds can skip it. See "Real market" below.

On a full run, `market/prices`, `market/curves` and `market/consensus` are
by far the largest tables; set `output.tables` in the config to override
their format to `parquet` while leaving the rest as `jsonl`:

```yaml
output:
  tables:
    market/prices: parquet
    market/curves: parquet
    market/consensus: parquet
```

## Market model

The market simulates five instrument families (equities, rates, credit,
commodities, FX) driven by one common risk factor `z` plus per-family and
per-instrument noise, over a schedule of three regimes (range, risk-off,
risk-on) that each market seed cycles through in its own order. It trades
off realism for a model whose moments can be checked in closed form:

- No sector factor: equities load on `z` through a single beta plus
  idiosyncratic noise, with no separate sector-level co-movement.
- One common driver: every family's exposure to broad risk-on/risk-off
  moves runs through the same `z`, not a family-specific factor.
- A regime sets a driver mean, a vol multiplier, and a mean-reversion
  speed that is zero outside the range regime: switching regimes shifts
  `z`'s average level and scales every family's volatility, and in range
  it also adds a pull back toward nearby round levels.
- No vol clustering inside a regime: volatility is constant within a
  regime span, with no GARCH-style clustering on top of the regime switch.
- Rates are level and slope only: the sovereign curve is two factors, not
  a richer term-structure model, with fixed offsets giving the other tenors.
  Level vol is set per currency, and published yields are floored at
  `yield_floor_pct`, so a low-yield curve can sit at the floor.
- Credit is one factor times a rating base times issuer noise: no issuer
  defaults, a static rating band per issuer for the whole run, and issuer
  volatility is a guess since issuer-level OAS series are not public.
  The factor's asymmetric shock is centred, so it adds skew to spread moves
  without adding drift.
- Commodities are spot plus a scripted curve: the term structure follows a
  fixed roll yield by commodity group, with no seasonality and natural gas
  excluded from the simulated universe.
- FX is zero-drift per currency: each currency's log value against USD has
  no drift term, and there is no carry from interest-rate differentials.
- Consensus is a lagged moving average plus scripted flips: the street
  view tracks a trend-following average of recent price moves with
  occasional random direction flips, and both the street and positioning
  half-lives are measured in trading days, not update counts. Positioning
  is built the same way for every instrument, a slower-moving percentile
  score for crowding; credit is the one family with no public positioning
  data to check it against, so its score is a survey analogue rather than
  a proxy for a reported one.
- One sampled event class per instrument (earnings for equities, rating
  actions for credit issuers, central-bank meetings for curves, inventory
  reports for energy, crop reports for agriculture) plus a market-wide
  macro print, tied to no family, that adds directly to the common driver
  `z`; industrial metals, precious metals and FX carry no idiosyncratic
  event of their own. Contract expiries and the weekly positioning report
  are separate, deterministic calendar rows, not sampled events.
- No holidays or intraday data: the axis is every weekday, one close per
  instrument per day, with no exchange holiday calendar.

Wherever a parameter is drawn from real data, its config note names the
source: sourced values were checked against named FRED series or papers on
2026-09-22, and the credit rating-band spread medians use a calm three-year
window. Where no reliable public source exists, the config says so and
marks the value a guess: the investment-grade/high-yield credit mix and
credit duration ranges, the equity beta range, credit issuer volatility,
the commodity group-versus-idiosyncratic variance share, the rating-action
event frequencies, the consensus window and flip rate, and rates level
vol (USD is sourced, but a single field basis cannot mark only the other
currencies a guess).

Before writing its tables, the market stage checks each generated seed
against what the config implies. For every regime and family it compares
the realised annualised volatility of an equal-weight family index
against its model-implied value within a relative tolerance, and the
index's realised correlation with `z` against its model-implied
correlation within 4 standard errors of the estimate in Fisher-z space,
which keeps that error band from collapsing to nearly nothing as the
correlation approaches +-1; credit's index is the investment-grade
issuers' spread, and rates' is the mean 10Y yield change, not a price.
Sampled event types and consensus flips are checked
by count, against how many the config implies were drawn; contract
expiries and the weekly positioning report, both deterministic rather
than sampled, are checked against their exact dates instead. Round-level
tests (a close crossing into or through the band around a round level)
are counted as a family mean per instrument, for equities, commodities,
FX pairs and sovereign 10Y yields, during the range regime, with level
crossings between two closes also counted as a test. Drift is not
checked: a drift check over one regime span would pass or fail mostly by
chance, because the standard error of drift over 14-22 weeks exceeds the
drift itself.

### Real market

A real market seed (the population's pilot seed by default) replays actual
history instead of simulating it. Its raw data comes from:

- FRED: Treasury yields (2Y/5Y/10Y/20Y/30Y), Moody's seasoned corporate
  bond yields for the credit spread, and daily FX rates.
- Yahoo Finance's chart API: equity closes, continuous front-month futures
  for commodities, and the SPY reference series.
- SEC EDGAR: 8-K filings with item 2.02 (Results of Operations), the
  earnings-date source. Requests declare a User-Agent with a contact
  email, set in `market.real.sec_user_agent`.
- Fixed public dates: FOMC (Federal Reserve), WASDE (USDA) and Employment
  Situation (BLS) release schedules.

`fetch-market` writes this raw data under `<data-dir>/raw/market`, gitignored
and never committed or redistributed; its manifest records the URL, retrieval
time and hash of every file fetched, not the data itself. The market stage
refuses to run against an incomplete cache.

A real seed has no model target to check against, so its check is mostly
structural: every price, spread and curve value must be finite and positive
(a yield against its floor instead), no series may sit on a forward-filled
value for longer than the fetch's fill limit, and calendar row counts are
checked against the seed's own drawn events and flips, plus exact expiry
and positioning dates. Each equity also needs `earnings_coverage`: at
least 3 earnings rows a year (scaled to a shorter horizon), since
companies report quarterly and fewer rows mean a missing or unparsed
filing history. These run before anything realised is computed, since a
moment over bad data cannot be trusted. Once they pass, each regime span
and family's realised annualised volatility and correlation with `z` are
reported for information, not checked against a target, and a family
index with zero variance over a span fails as `flat`; the days a yield
curve tenor spent at its floor, and each equity's smallest gap in days
between earnings rows (`earnings_spacing`), are likewise reported only.

Every real series keeps its historical day-over-day change (log for
prices; level for yields, which are then floored); only its starting level
is rebased onto the same configured ranges a synthetic seed uses. Its
instrument id is the fixed anonymised registry (`EQ-R001`, not a ticker),
and every date is remapped onto the simulated calendar's own axis. This
disguise hides levels, names and dates, but not the return pattern itself,
which is real history. Credit prices and commodity M2-M12 are derived from
these series rather than fetched directly. An event's surprise is its own
series' daily move scaled by that series' own historical daily sd and
passed through `tanh`, so ordinary days stay small, a large outlier still
ranks above a smaller one instead of saturating to the same score, and
the value approaches but never reaches +-1; an earnings surprise instead
scales the equity's abnormal return (its own move net of `beta` times
SPY's), so a market-wide move on the same day is not mistaken for the
earnings reaction.

Limitations from the model:

- Only a USD sovereign curve is real; EUR/GBP/JPY curves stay synthetic-only.
- Credit is investment-grade only: two index series (Moody's Aaa- and
  Baa-equivalent proxies), with no high-yield tier and no issuer-level noise.
- A commodity's M2-M12 tenors are scripted from its M1 level, not
  independently fetched, and its continuous M1 series rolls at Yahoo's own
  contract-roll dates, which do not line up with the calendar's third-Friday
  contract-expiry rows.
- A missing observation (a holiday) forward-fills from the prior value, up
  to a capped run length; a longer gap fails the market stage.
- The inventory report event fires on every Wednesday; if that commodity's
  series was closed that day (filled, with no source value), the report
  moves to the next trading day with one. A real event row can also score
  zero surprise on an ordinary day when its source repeats the previous
  close.
- An event's surprise is priced from the seed's own realised price or yield
  reaction, not from a reported consensus-versus-actual figure.
- An earnings date is the filing date of the results 8-K. A company that
  published results before filing is shifted by up to a day, and off-cycle
  2.02 filings (pre-announcements) also count as earnings events.

The real window a seed replays is recorded in the run metadata and the
config itself, for operators only; never pass either to a model under test.

To run the full split on real data rather than just the pilot, point
`population.market_seeds` at a real seed too:

```yaml
population:
  market_seeds: [R1]
  pilot_market_seeds: [R1]
```

Add further real seeds under `market.real.seeds`, each with its own
`window_start`, `regime_starts`, and the `basis` and `note` fields every
config leaf requires. A seed's window (`window_start` through the horizon
it implies) must fall inside the FOMC/WASDE/NFP date coverage, 2018-06-04
to 2019-05-31; replaying a later or longer window needs those date lists
extended first.

## Engine model

The behaviour engine walks each PM through its horizon day by day, turning
its persona, traits and rules into ideas, orders and rule responses. It
trades off realism for a model whose biases are each one legible formula:

- A deterministic daily loop: `step` marks every open position, resolves
  today's rule triggers and the discretionary block, enters new ideas, and
  closes out the whole book on the horizon's last day. Prices, rules and
  biases read only data up to the day the loop is on; the own signal below
  is the one deliberate exception, since it peeks at the forward move by
  design.
- The own signal is a noisy peek at the forward move: `skill * z + sqrt(1 -
  skill^2) * n`, where `z` is the realised forward move (in bullish units,
  scaled by its own forward-window sd) and `n` is standard noise. `skill`
  is one fixed constant for every PM, not a trait.
- The thesis move is the unscaled signal times the series' realised vol
  (`bullish_sign * own_signal * sd`), extrapolation-blended into the
  stated forecast; the stated interval, though, is centred on the honest
  conditional mean of the forward move given the signal, so a neutral PM's
  realised coverage matches its stated coverage regardless of what its
  forecast says.
- One rule per bias parameter, each collapsing to a formula: loss aversion
  is a cut/hold/add softmax over `{-lambda|z|, r, r(1+f) - lambda|z|f}`,
  where `r` is the forecast still to come in z units,
  `side * bullish_sign * (forecast - (level - entry)) / sd` with `sd` taken
  at entry; disposition is a daily sell hazard scaled by `sqrt(D)` at
  a gain and `1/sqrt(D)` at a loss; anchoring blends the exit level
  `(1-rho)*target + rho*anchor`; extrapolation blends the forecast
  `(1-theta)*thesis_move + theta*trailing_move`, and entry and side follow
  the same theta-blend of the own signal and the trailing move, in z-units
  normalised to unit variance; herding follows the
  street with probability `herding_weight` when it conflicts with the
  PM's own side; overconfidence narrows the stated interval and inflates
  entry size by the same z-score ratio, `z(0.8) / z(coverage)`; conviction
  blends the size rank `round((1-m)*conviction + m*u)` toward a
  uniform(1,5) draw; exit deficiency makes a fired rule's response miss
  with probability `e` (adding instead of missing, at a loss, when loss
  aversion is also active).
- A day's fired rules resolve in one precedence order: exclusions and the
  mandate cap first (pre-trade constraints that never reach this table),
  then exits and signposts, then rolls, then trims and targets, then
  holds, with a no-add breach ranked last.
- Expression forms are per asset class, steered by a matching preference
  trait when one is stated: equities trade outright or pair
  (`pair_vs_outright`); sovereign rates trade outright or curve
  (`duration_expression`); long/short credit trades outright only;
  commodities directional trades outright only, and curve-and-spread
  trades outright or calendar spread (`curve_trade_expression`).
- A commodity roll is P&L-neutral on the constant-maturity curves: it
  retags legs one contract month out and re-bases `rolled_offset`, never
  shifting `entry_level`, `stop_level` or `target_level`.
- No transaction costs: every ledger row prices at the raw market level,
  with no spread, slippage or commission anywhere in the loop.
- Multi-asset PMs are skipped for now: no adapter routes a multi-asset
  idea's legs across asset classes, so the engine stage lists them under
  `skipped` rather than running them.

Limitations from the model:

- The real-seed universe is small: two credit issuers and one sovereign
  curve, not a representative cross-section. On a real seed, long/short
  credit PMs revisit the same two issuers, so `max_positions` never binds,
  and sovereign-rates PMs trade one curve's tenors only.
- Signposts are grammar-only: each is one of three condition kinds (one
  event type, one level held for a window, one relative move against
  peers), with no qualitative signposts. An idea with no peer besides
  itself (a sovereign curve, or a credit issuer alone in its rating band)
  carries no relative signpost.
- Skill is one fixed constant, not a per-PM trait: every PM's own signal
  carries the same edge over the forward move.
- Multi-asset PMs are absent from every output table, not merely
  under-weighted: the engine stage skips them rather than simulating a
  cross-asset book.

## Gate 1

Gate 1 recovers each direct-asset PM's eight planted bias parameters from the
engine's own ledger, pools the recovered statistics per asset class, and
blocks the pipeline only when the pooled synthetic comparison over the full
horizon fails or has too few PMs to judge. One estimator per parameter:

| Parameter | Statistic | Opportunity unit | Direction |
| --- | --- | --- | --- |
| `loss_aversion_lambda` | share of loss-side opportunities where the PM adds instead of cutting | loss-side, untriggered position-days | higher recovers more strongly |
| `disposition_ratio` | proportion-of-gains-realised over proportion-of-losses-realised (Odean 1998) | sell-day position-days | higher recovers more strongly |
| `anchoring_rho` | share of discretionary exits landing inside a band around the anchor | discretionary exits | higher recovers more strongly |
| `extrapolation_theta` | share of entries chasing a trailing move already past one horizon-sd | entries | higher recovers more strongly |
| `herding_weight` | share of entries on the street's side, among entries with a non-neutral street view | entries with a non-neutral street view | higher recovers more strongly |
| `overconfidence_coverage` | share of entries whose realised move lands inside the stated interval | entries | lower recovers more strongly |
| `conviction_size_miscalibration` | one minus the rank correlation of entry sizing and stated conviction | entries with a stated conviction | higher recovers more strongly |
| `exit_deficiency` | share of non-overridden fired rules left unacted on or added to | non-overridden rule firings | higher recovers more strongly |

Each parameter is estimated over the full horizon (split `all`) for every
direct-asset PM, and over further splits: a neutral trait over all three
regimes, an active trait over each regime that boosts it, and a drifted trait
over its before and after windows only. The `all`-split estimates
are then pooled into a neutral baseline and an active mean per asset class,
over every synthetic seed together (seed group `synthetic`, seed group kind
`synthetic_pool` - the one that blocks the pipeline) and again per seed on
its own; a drifted PM's `all`-split estimate is dropped from every pooled
comparison, since it mixes two different trait values. On the default
population, each synthetic seed's own cell holds 12 PMs per asset class, of
which only 1-11 are neutral for any given parameter; pooling the three
synthetic seeds together is what gives the pooled `synthetic` cell enough
neutral PMs for a usable baseline, about 36 PMs per asset class in all.

A regime split compares the active PMs boosted in that regime with every
neutral PM's rows for the same regime, since a neutral trait is estimated
over all three regimes. A before or after split compares each drifted PM's
rows for that window with the neutral PMs' full-horizon (`all`-split) rows
instead of a regime-style shared date set, since a drift window differs per
PM. Every split beyond `all` is report-only, the same as the per-seed rows.

A cell passes when the active mean sits on the stronger side of the neutral
mean (per the parameter's own direction) with the neutral standard deviation
no more than half the gap between them (`gap_fraction`, default 0.5), and
the planted-versus-recovered rank correlation clears `min_rank_corr` (default
0.5); a cell with fewer than `min_pms` (default 5) neutral or active PMs is
`insufficient` rather than judged. `floor_se` (default 2.0, standard
deviations above or below the neutral mean) and the `active_share_past_floor`
it produces are reported for re-centring the marginals, not part of the pass
rule. `anchor_band_k` (default 0.1, in horizon-vols) is the anchoring
estimator's own band width, not a pass-rule knob either. Only the pooled
`synthetic`/`all` cell (seed group `synthetic`, kind `synthetic_pool`) blocks
the pipeline: every per-seed cell, synthetic or real, is reported but never
blocks, and every split beyond `all` is report-only for the same reason.

Four parameters also carry an opportunity-count minimum (`n_min`): exit
deficiency 7, loss aversion 16, herding 14, anchoring 29 - each the
observation count at which a neutral PM's binomial standard error is a
quarter of the gap between the neutral and active centres (herding's pair was
derived for the conflict-follow rate, not the agreement statistic gate 1
estimates, and its minimum is applied to the latter as an approximation). The
`count_shortfall` warning compares each seed's 10th-percentile PM on the
estimator's own opportunity count (`Estimate.n`) against `n_min`, rather than
an engine counter. On the default run anchoring falls short on every seed and
asset class (the 10th-percentile PM has 2-9 discretionary exits away from the
target against 29), exit deficiency on rates and credit for seeds A and R1,
and herding on rates and credit for R1 only.

Gate 1 writes two tables: `gate1_pm`, one row per PM/parameter/split keyed on
`(pm_id, param, split)`, and `gate1_cells`, one row per seed group/asset
class/parameter/split keyed on `(seed_group, asset_class, param, split)`.

On the default root, exit deficiency and overconfidence pass on all three
asset classes, herding passes on rates/credit and commodities but fails on
equities, and the other five parameters fail on all three:

- `exit_deficiency` passes: its miss probability acts directly on the
  fired-rule response the estimator reads, with no rule precedence or
  softmax layer between the trait and the observable.
- `overconfidence_coverage` passes: overconfidence rescales the stated
  interval by the same z-score ratio the inside-share estimator reads, again
  a direct readout of the trait.
- `herding_weight` on equities: the neutral spread is about half the gap,
  just past the 0.5 limit - a marginal fail, with no mechanism claimed.
- `loss_aversion_lambda` fails: a higher lambda makes cutting worse but also
  penalises adding against holding, so the two effects offset, the add rate
  stays flat, and cut is rarely chosen at softmax temperature 1.
- `disposition_ratio` fails: the planted multiplier sqrt(D), about 1.1 at
  the professional centre, acts on a 0.03 base sell hazard and is swamped by
  rule-triggered sales.
- `anchoring_rho` fails: exits at the anchored level are rare beside
  hazard-driven discretionary exits.
- `extrapolation_theta` fails: the forecast never reaches entry direction or
  target (the side follows the own signal, and the target follows the stop
  and the reward-to-risk draw), so theta leaves no public trace.
- `conviction_size_miscalibration` fails the gap test: the neutral spread
  across PMs is wider than half the active-neutral gap, although its rank
  correlation clears 0.5.

This pattern holds beyond the default root: sweeping the default config over
99 root seeds (`seed.root`, every other knob fixed), exit deficiency and
overconfidence pass on 93-99% of roots per asset class; herding sits at the
threshold, its median neutral sd running 0.48-0.51 of the gap, and passes on
about half the roots; the other five parameters fail on nearly every root,
with conviction-size miscalibration passing on at most 5%.

Limitations from the model:

- The pooled synthetic baseline hides seed-level effects: a bias that only
  shows up on one market seed is averaged away in the pooled `synthetic`
  cell that blocks the pipeline.
- Herding is measured as agreement with the street's non-neutral view, not
  as a PM crossing its own conflicting signal: a PM who follows the street
  in a conflict ends on the street's side, so the conflict itself cannot be
  seen in public data.
- Disposition's realised share counts any sell-day cut, trim or exit,
  rule-triggered or discretionary alike, not only a PM's own voluntary
  realisation.
- Anchoring's recovered rate depends on the configured band width
  (`anchor_band_k`) around the anchor, not a model-free distance.
- The pass-rule thresholds were probed against one run of the default
  population, not validated across many.
- `active_share_past_floor` is reported for every cell but never gates a
  verdict.
- Every split beyond `all` (by regime, and before/after a drift event) is
  report-only and never gates a verdict.
- Gate 1 blocks the pipeline at the default configuration: five of the
  eight parameters fail on every asset class, as the verdict pattern above
  shows.

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format
```
