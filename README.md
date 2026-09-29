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
uv run pm-traitbench plan --config configs/demo.yaml --data-dir data
uv run pm-traitbench dialogue --config configs/demo.yaml --data-dir data
uv run pm-traitbench validate --config configs/demo.yaml --data-dir data
uv run pm-traitbench gate2 --config configs/demo.yaml --data-dir data
uv run pm-traitbench probes --config configs/demo.yaml --data-dir data
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
`position_days` and `skeletons` are hidden tables in full. `ideas` and
`ledger` also carry hidden columns: the PM's own signal, forecast and
interval, its street-view context, conviction and size rank on `ideas`,
whether the entry chased a trend that had already run, and each order's bias
flag and driving rule on `ledger`. Hidden data is generator provenance for
checking the engine itself and is never shown to a system under test. The
`gate1` stage recovers each direct-asset PM's eight planted biases from the
engine's ledger and pools them per asset class, writing `gate1_pm` and
`gate1_cells`; it exits 1 when a blocking row fails or a non-report-only
parameter has no blocking row - one row per non-report-only parameter,
pooled over every direct asset class of the synthetic seeds. Per-class and
report-only rows are judged but never block.
Both tables and its run metadata land on disk either way. The `plan` stage
plants trait signals on dated sessions and writes `signals` and `skeletons`,
never blocking on a shortfall. The `dialogue` stage narrates every planted
session and writes `sessions` and the hidden `dialogue_logs`. The `validate`
stage checks every narrated session against the ledger, the leakage rule and
the forbidden set, regenerating or dropping a session that keeps failing,
and writes the hidden `validation` table while rewriting `sessions` and
`dialogue_logs` in place. The `gate2` stage asks one strong model, per PM,
which traits the validated dialogue shows and writes `gate2_traits`,
`gate2_signals`, `gate2_pm` and `gate2_cells`; it exits 1 when a blocking
row fails - one one-sided Fisher exact test per bias parameter and one
pooled Poisson-binomial test over every held preference, at `gate2.alpha`.
A blocking row below `gate2.min_class` PMs either side is `insufficient`
and reported but never blocks. The `probes` stage writes `probes`, one row per question with ground truth at
each checkpoint of a PM's schedule, calling no model; `answer`, the option
sources and the supporting signal ids are hidden columns. Pass `--force` to
overwrite a table that already exists. Run `uv run pm-traitbench --help` for the full command list.

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
  cuts a losing position with probability `0.05 / lambda` and, when adding
  is allowed, adds with probability `min(0.5, 0.1 * max(lambda - 1, 0))`,
  else holds, so lambda above 1 raises the add hazard and lowers the cut
  hazard; disposition is a daily sell hazard scaled by `sqrt(D)` at
  a gain and `1/sqrt(D)` at a loss; anchoring fixes one round-level anchor
  per idea, 40% of the way from entry to target, and exits there instead of
  at target with probability rho, drawn once at entry; extrapolation blends
  the forecast
  `(1-theta)*thesis_move + theta*trailing_move`, and entry and side follow
  the same theta-blend of the own signal and the trailing move, in z-units
  normalised to unit variance; herding follows the
  street with probability `herding_weight` when it conflicts with the
  PM's own side; overconfidence narrows the stated interval and inflates
  entry size by the same z-score ratio, `z(0.8) / z(coverage)`, and entry
  size divides the mandate cap's risk step by a headroom of 2.5 so that
  inflation has room below the cap; conviction sizes at the stated
  conviction rank with probability `1-m`, else at a uniform(1,5) draw;
  exit deficiency makes a fired rule's response miss
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
engine's own ledger, pools the recovered statistics per asset class and, for
the synthetic pool, once more across every direct asset class together. Only
that cross-class row blocks the pipeline, when it fails or has too few PMs to
judge over the full horizon. One estimator per parameter:

| Parameter | Statistic | Opportunity unit | Direction |
| --- | --- | --- | --- |
| `loss_aversion_lambda` | share of loss-side opportunities where the PM adds instead of cutting | loss-side, untriggered position-days | higher recovers more strongly |
| `disposition_ratio` | proportion-of-gains-realised over proportion-of-losses-realised (Odean 1998) | sell-day position-days | higher recovers more strongly |
| `anchoring_rho` | share of anchor crossings the PM exits on that same day | anchor crossings before the last horizon date | higher recovers more strongly |
| `extrapolation_theta` | share of entries chasing a trailing move already past one horizon-sd | entries | higher recovers more strongly |
| `herding_weight` | share of entries on the street's side, among entries with a non-neutral street view | entries with a non-neutral street view | higher recovers more strongly |
| `overconfidence_coverage` | share of entries whose realised move lands inside the stated interval | entries | lower recovers more strongly |
| `conviction_size_miscalibration` | one minus the rank correlation of the lead leg's entry risk and stated conviction | entries with a stated conviction | higher recovers more strongly |
| `exit_deficiency` | share of non-overridden fired rules left unacted on or added to | non-overridden rule firings | higher recovers more strongly |

Each parameter is estimated over the full horizon (split `all`) for every
direct-asset PM, and over further splits: a neutral trait over all three
regimes, an active trait over each regime that boosts it, and a drifted trait
over its before and after windows only. The `all`-split estimates
are then pooled into a neutral baseline and an active mean per asset class,
over every synthetic seed together (seed group `synthetic`, seed group kind
`synthetic_pool`) and again per seed on its own; a drifted PM's `all`-split
estimate is dropped from every pooled comparison, since it mixes two
different trait values. For the synthetic pool, those same `all`-split
estimates are pooled once more across every direct asset class together,
into a cross-class row (`asset_class` null) - the one row that blocks the
pipeline. On the default population, each synthetic seed's own cell holds 12
PMs per asset class, of which only 1-11 are neutral for any given parameter;
pooling the three synthetic seeds together gives each per-asset-class
`synthetic` cell about 36 PMs, and pooling the three direct asset classes
together again gives the cross-class row about 108 PMs in all.

A regime split compares the active PMs boosted in that regime with every
neutral PM's rows for the same regime, since a neutral trait is estimated
over all three regimes. A before or after split compares each drifted PM's
rows for that window with the neutral PMs' full-horizon (`all`-split) rows
instead of a regime-style shared date set, since a drift window differs per
PM. Every split beyond `all` is report-only, the same as the per-seed rows.

A cell is judged by one of two pass rules, chosen by whether its parameter is
in `population_params`. The default, per-PM rule passes when the active mean
sits on the stronger side of the neutral mean (per the parameter's own
direction) with the neutral standard deviation no more than half the gap
between them (`gap_fraction`, default 0.5), and the planted-versus-recovered
rank correlation over active PMs clears `min_rank_corr` (default 0.4); the
combined correlation is reported as `rank_corr`. The population rule passes
when the active mean exceeds the neutral mean, in the parameter's own
direction, by at least `min_pop_z` (default 3.0) standard errors of the
difference (each side's own sample standard deviation), with no rank
condition; it suits a parameter whose per-PM opportunity
count is limited by how many decisions one PM makes in a year, so a single
PM's own estimate is too noisy to judge even though the pooled population
carries a signal.
`population_params` (default `herding_weight`, `conviction_size_miscalibration`,
`disposition_ratio`, `anchoring_rho`) names the parameters judged this way;
every other parameter keeps the per-PM rule. A cell with fewer than `min_pms`
(default 5) neutral or active PMs is `insufficient` rather than judged,
whichever rule applies. `floor_se` (default 2.0, standard deviations above or
below the neutral mean) and the `active_share_past_floor` it produces are
reported for re-centring the marginals, not part of either pass rule.

Only the synthetic pool's cross-class row (seed group `synthetic`, kind
`synthetic_pool`, split `all`, `asset_class` null) of a parameter not in
`report_only_params` blocks the pipeline: every per-asset-class `synthetic`
cell, every per-seed cell (synthetic or real), and every split beyond `all`
are reported but never block, and a report-only parameter's cross-class row
is reported but never blocks either, whatever its verdict. `report_only_params`
defaults to `herding_weight`, `disposition_ratio` and `anchoring_rho`:
disposition's pooled population z is about 1 (median 1.0 over 12 roots) at
the sourced 1.2 centre, and anchoring's pooled z has median 3.9 but falls
below 3 on 2 of 12 roots; herding passes pooled on all 12 roots but stays
report-only because the trend-built street view couples it to extrapolation,
so a pass does not isolate it.

Four parameters also carry an opportunity-count minimum (`n_min`): exit
deficiency 7, loss aversion 16, herding 14, anchoring 29 - each the
observation count at which a neutral PM's binomial standard error is a
quarter of the gap between the neutral and active centres (herding's pair was
derived for the conflict-follow rate, not the agreement statistic gate 1
estimates, and its minimum is applied to the latter as an approximation). The
`count_shortfall` warning compares each seed's 10th-percentile PM on the
estimator's own opportunity count (`Estimate.n`) against `n_min`, rather than
an engine counter. The anchoring opportunity is an anchor crossing; on the
default run anchoring falls short of its minimum on every seed and asset
class, and exit deficiency and herding both fall short on rates and credit
for the real seed R1.

Gate 1 writes two tables: `gate1_pm`, one row per PM/parameter/split keyed on
`(pm_id, param, split)`, and `gate1_cells`, one row per seed group/asset
class/parameter/split keyed on `(seed_group, asset_class, param, split)`,
where a null `asset_class` is the synthetic pool's cross-class row. Each
`gate1_cells` row records which rule judged it (`test`), the per-PM gap and
rank checks (`gap_ok`, `rank_ok`), and, for a population-tested parameter,
the standard-error z (`pop_z`) and whether it cleared its minimum (`pop_ok`).

Blocking rows are pooled over every direct asset class, synthetic seeds:
exit deficiency, extrapolation, loss aversion and overconfidence use the
per-PM rule; conviction uses the population rule. On the default root and on
12 further root seeds (20260301-20260312) all five pass on every root, so
Gate 1 exits 0. Over the 12 roots: per-PM neutral-sd-over-gap medians
0.18-0.30 (max 0.41) and rank correlations 0.71-0.90; conviction population z
median 7.7 (min 5.6). Those rank correlations are the combined (neutral and
active together) measure; the default population's result under the
active-only measure is not yet recorded.

Report-only rows: herding passes pooled on 12 of 12 roots (z median 6.7, min
5.5) but stays report-only for the coupling reason above; anchoring passes on
10 of 12 (z median 3.9, min 2.8); disposition on 1 of 12 (z median 1.0).

Per-asset-class rows are reported only; over the 12 roots herding passed 26
of 36 class rows, overconfidence 31 (plus 2 insufficient), conviction 32,
which is why the blocking verdict pools.

Before these rule changes, on the default root only exit deficiency and
overconfidence passed per PM on all three asset classes, herding passed on
two, and loss aversion, disposition, anchoring, extrapolation and conviction
failed everywhere. The causes were that the forecast never reached entry
side or target, lambda had no measurable effect under a linear value
function, the conviction estimator double-counted two-leg ideas and the size
ladder tied the top ranks at the cap, and anchors moved every day so
anchored exits were rare.

Limitations from the model:

- The pooled synthetic baseline hides seed-level effects: a bias that only
  shows up on one market seed is averaged away in the cross-class row that
  blocks the pipeline.
- Pooling across asset classes hides asset-class-level effects the same way:
  a bias recoverable in only one direct asset class is averaged away in the
  cross-class row, and only the per-asset-class `synthetic` cell (report-only)
  would show it.
- Herding is measured as agreement with the street's non-neutral view, not
  as a PM crossing its own conflicting signal: a PM who follows the street
  in a conflict ends on the street's side, so the conflict itself cannot be
  seen in public data.
- Disposition's realised share counts any sell-day cut, trim or exit,
  rule-triggered or discretionary alike, not only a PM's own voluntary
  realisation.
- The size headroom of 2.5 matches the overconfidence size factor at the
  active prior's centre (coverage about 0.4); PMs planted with lower coverage
  still hit the cap on their top size ranks (on the default run about half of
  active overconfident PMs have rank 5 clipped and 8.4% of all entries sit at
  the cap), which ties conviction ranks for those PMs and blocks their
  loss-aversion adds (recorded as holds).
- Anchoring's recovered rate is hits over crossings, not rho alone: a
  crossing inside the minimum holding period, or on a day a rule exit acts,
  counts as an opportunity without a hit, so the statistic sits below rho
  plus the background sell hazard.
- Rounding the anchor to the nearest round level can place it close to entry
  (10th percentile about 0.1-0.26 of the way to target by asset class), so
  some anchored ideas exit soon after entry.
- The pass-rule thresholds were checked on the default root and 12 further
  roots, not a wider sweep or other configs.
- `active_share_past_floor` is reported for every cell but never gates a
  verdict.
- Every split beyond `all` (by regime, and before/after a drift event) is
  report-only and never gates a verdict.
- The population rule certifies a shift in the population, not that one PM's
  value can be read back: conviction and the report-only parameters are
  judged this way.
- A report-only parameter's cross-class row can fail or stay insufficient
  without blocking the pipeline, so a planted bias in `herding_weight`,
  `disposition_ratio` or `anchoring_rho` can ship unverified at the default
  population.

## Signal plan

The `plan` stage reads `personas`, `traits`, `drift_events` and the engine's
four tables (`ideas`, `ledger`, `rule_events`, `position_days`), plus the
engine's run metadata for the multi-asset PMs it must skip. It writes two
tables: `signals`, one row per planted trait signal keyed on
`(pm_id, signal_id)`, and `skeletons`, one row per dated session keyed on
`(pm_id, session_id)`.

A `signals` row names the trait, the session and date it lands on, how it
expresses the trait (`mode`: revealed, stated or contradiction) and its
valence (confirm or retracted). `third_party_value` is set only when
`ownership` is a colleague or client instead of the PM, and holds the value
that third party is attributed with; it is null for the PM's own signals.
`claim_session_id` is set only on a contradiction, naming the earlier session
where the PM stated a view the later session's carrier then contradicts; it
is null for every other mode.

`skeletons` is a hidden table in full: it is the narrator's whole input for
turning a session into the PM's own words, so a system under test is never
shown which trait a stance plants or which traits and preference params the
session must stay silent on.

A revealed signal only exists because the engine itself recorded, on some
dated row, that the planted bias drove the decision; the plan never invents
a decision the engine did not take. Each planted bias's carrier evidence:

| Bias parameter | Carrier source | Engine action(s) behind it |
| --- | --- | --- |
| `loss_aversion_lambda` | `ledger`, `position_days` | `add`, `add_before_trigger`, `hold` |
| `disposition_ratio` | `position_days` | `realise_gain_early`, `hold_loser` |
| `anchoring_rho` | `position_days` | `exit_at_anchor` |
| `extrapolation_theta` | `ideas` | `chased_trend` |
| `herding_weight` | `ledger` | `followed_street` |
| `overconfidence_coverage` | `ledger` | `oversized` |
| `conviction_size_miscalibration` | `ledger` | `mis_sized` |
| `exit_deficiency` | `rule_events` | `acked_no_action`, `added` |
| `exit_deficiency` | `ledger`, `position_days` | `added`, `late_roll` |

Every revealed stance line is keyed by the specific engine action behind its
carrier, not just the bias it plants, so the line drawn always matches what
the engine actually logged that day. A signal with no carrier to point at
(too few qualifying rows, or the window already used up) is dropped rather
than backfilled with another mode. An expression preference's revealed
signals need a carrier only where its own pool has one dated inside that
signal's segment; a segment left without a carrier (a drift across a mapped
and an unmapped form) falls back to an advisor-reaction signal instead of
being dropped. Signals that need no carrier - stated and third-party rows,
retractions - pack into existing sessions up to
`plan.max_signals_per_session` (default 2), never two stances of the same
trait or more than one advisor-violation stance in a session. A drift note
instead anchors at its own event: it sits on the first trading day on or
after the event date, joining an existing session there if one has room and
opening a new one otherwise, and never lands on a later date.

A shortfall never fails the run: too few revealed signals placed against
their planted quota, a drifted trait's segment with fewer confirming
signals (its placed confirm signals plus the drift notes that land in it)
than `plan.drift_min_per_side`, or the session cap that could not be met for
lack of free trading days are all warnings in `run_metadata/plan.json`, in
PM order. Multi-asset PMs are skipped, the same PMs the engine stage already
skipped, and are listed under `skipped` in the plan's own run metadata.

## Dialogue

The `dialogue` stage reads `personas`, `rules`, `traits`, `drift_events`, the
engine's `ideas`, `ledger` and `position_days`, the plan's `skeletons`, and
the market's `instruments`, `prices`, `curves`, `consensus` and `calendar`.
A PM with no skeleton (the same multi-asset PMs the earlier stages already
skip) is skipped here too; the plan's own `skipped` list is copied into this
stage's run metadata, not used to decide the skip itself. The stage narrates
every skeleton into a two-agent session and writes two tables: `sessions`,
the public transcript of text-only turns, and `dialogue_logs`, a hidden
table carrying each session's voice and, per turn, its directive, its
`mentions`, any tool calls, the model that produced it, the cache keys of
the requests behind it, and its token usage.

Both agents are `claude-opus-5-5` at low reasoning effort. The advisor
never sees the PM's persona, rules, ideas or plan; it is instructed to
answer market questions only through five tools, each capped to dates on or
before the session's own date, rather than from memory - whether it
actually follows that instruction is checked by a later stage, not
enforced here. The narrator is one model across every PM, so no model
choice can leak a signal; each PM instead draws one voice, independently of
every trait and preference, that colours every session it narrates. Each
session's turn count is drawn per session kind and then raised just enough
to fit every stance the skeleton schedules; a stance and the opening line
reach the narrator, and a scripted advisor violation reaches the advisor,
each as a system message partway through the conversation, never folded
into the first turn.

Every request goes through a response cache under `<data-dir>/cache/llm/`,
keyed on the session id plus the full request body - so two sessions whose
requests happen to render identically never share a cached reply - which
doubles as the run's resume manifest: a crash, a rejected reply or widening
`dialogue.pm_filter` to cover more PMs is a rerun with `--force` that only
calls the API for the turns still missing from the cache. Credentials
(`ant auth login` or `ANTHROPIC_API_KEY`) are needed only on a cache miss,
so a fully cached rerun works offline. The advisor's system prompt is
`src/pm_traitbench/catalogues/advisor_prompt.md` unless
`dialogue.advisor_prompt_path` names another file. Like every other stage,
it writes both tables or neither. `dialogue.token_budget` caps fresh input
plus output tokens spent in one run of the stage; it is a soft stop (an
already in-flight batch of sessions can overshoot it by up to
`dialogue.max_concurrency` calls) and a resumed run's count starts back at
zero, so it caps each run's own spend, not a cumulative total. Before a
full run, measure cost with `dialogue.pm_filter` restricted to 2-3 PMs and
set `dialogue.token_budget` accordingly. Checking whether a session
actually holds up (its `mentions` against the ledger, leakage, forbidden
traits) is a later stage's job, not this one's.

## Gate 2

The `gate2` stage asks whether a strong model can recover every planted
trait from the validated dialogue alone. It needs validate's run
metadata at or after dialogue's, or it refuses to run. For each PM it sends
one full-context call over every session's transcript in date order, with
the mandate, the PM-scope rules, the bias vocabulary and the catalogue's
candidate preferences for that PM's asset class - never a trait id,
value, stance line, signal mode, or which sessions carry which signal.
The model answers per bias whether it is active, and per candidate
preference whether it is held and at what value, citing its sessions.

The stage writes four tables: `gate2_traits` (truth against prediction
per PM per parameter), `gate2_signals` (one row per surviving signal,
joined to its recovery and classification), `gate2_pm` (one row per PM)
and `gate2_cells` (the pooled verdicts). Nine rows block: a Fisher exact
test per bias parameter and a pooled Poisson-binomial test over held
preferences against chance, at `gate2.alpha`. A class under
`gate2.min_class` PMs either side is `insufficient`: it is reported but
never blocks, since a trait active on too few PMs says nothing about
whether narration carries it. Only a `fail` row exits 1. At pilot size
(about 12 PMs) a pass needs near-perfect recovery on every bias; the full
split is where the gate has power. Every other row - by kind, mode,
held-versus-not, asset class, typicality, drift, and one per preference
parameter - is report-only. A
second pass classifies each stated signal as bias or preference from the
session transcript, the mandate, the rules and the session's ledger rows,
with no trait vocabulary. Cross-PM 5-gram
containment is reported and warned above `gate2.overlap_warning`, never
blocking. A failing row is fixed by editing the plan stage's signal-mode
weights and rerunning stages 5-8. Caching and credentials match the
dialogue stage.

## Probes

The `probes` stage writes questions with deterministic ground truth for
evaluating a copilot's memory of a PM. It needs validate's run metadata at
or after dialogue's, like `gate2`, and calls no model: every answer key comes
from `traits`, `drift_events`, `rules`, `signals` and the engine's own
decision functions, and every question comes from an authored bank.

**Checkpoints.** Each PM with sessions has its own schedule, one date per
label, each the last trading day of a week: `week4`, `week13`, `week52` (the
timeline's last week), `pre_drift` (the week before each drift event),
`post_drift` (`probes.post_drift_weeks` weeks after it) and `regime_shift`
(the week after each regime boundary of the PM's market seed). A
week outside the timeline is dropped. When labels share a date the earliest
of `pre_drift`, `post_drift`, `regime_shift`, `week4`, `week13`, `week52`
wins, since the drift labels are the rarer and are what drift analysis slices
on. The checkpoint still covers every label that landed on its date, and
governance rows follow the date, not the winning label. A checkpoint with no session on or before it is skipped and listed in run
metadata under `skipped_checkpoints`.

**Context.** The context at a checkpoint is every surviving session dated on
or before it, so it is derived by filter. `context_chars` is the one derived
column: the length of the same transcript rendering Gate 2 counts. A signal
counts only when its session survives.

**Types and answer keys.** A positive probe on a trait with no supporting
signal in context is not emitted and is counted under `skipped_probes`.

- `trait_presence`: a two-option multiple choice (yes, no). Yes for an active
  bias and each held preference value; no for a dormant or never-active bias,
  an old value replaced by an update, a value only a third party attributed,
  and `probes.presence_never_held` never-held catalogue values.
- `trait_mcq`: a multiple choice on the action the PM would take in a
  hypothetical situation (per active bias) or on the held value (per
  preference), each with an open twin whose answer is the correct option's
  text. Option sources are `current`, `pre_update`, `stated_profile` (an
  anti-typical PM's neutral value for a bias its self-description
  contradicts), `third_party` (preference values) and `none` distractors.
- `in_situ`: an open request per active trait. The key is `comply: honour
  <value>` for a preference, `counteract: ...` for a bias, or `decline: ...`
  for loss aversion and overconfidence on odd-indexed checkpoints, where the
  request breaches the mandate risk cap by `probes.decline_excess_pct` points.
- `routine_question`: `probes.routine_per_checkpoint` open questions per
  checkpoint; the key is the PM's communication formats with a supporting
  signal in context, and no intrusion.
- `governance`: at `post_drift` and `week52` dates of a drifting PM, a
  question with a false premise about a trait that went dormant or changed;
  the key rejects the premise and gives the date and the current value.

**Closed-form MCQ rule.** The situation is hypothetical, on the PM's universe
at the checkpoint date. The action for each option value is the engine's most
likely outcome in closed form, called with the parameters for that value and
consuming no random draws. The bias-typical action is chosen when the
probability of the bias-typical outcome is at or above 0.5, else the default.
Loss aversion instead takes the most likely of add, hold and cut over the
horizon, ties going to add, then hold.

| Bias | Probability of the bias-typical outcome |
|---|---|
| `loss_aversion_lambda` | add first, hold throughout or cut first over H sessions, from the cut, add and no-add-rule hazards |
| `disposition_ratio` | a sale within H sessions, from the disposition hazard |
| `anchoring_rho` | rho |
| `exit_deficiency` | e |
| `herding_weight` | w |
| `extrapolation_theta` | the blended forecast against the entry threshold |
| `overconfidence_coverage` | the size factor, bucketed at `probes.overconfidence_size_edges` |
| `conviction_size_miscalibration` | 0.8 times m |

The horizon H for the two hazard biases is the lower middle of the window of
horizons at which the active median reads bias-typical and the neutral median
does not, up to `probes.max_horizon`; at the defaults it is 34 for loss
aversion and 14 for disposition, and it is recorded under `mcq_horizon`. The
question states H. When the window is empty the bias emits no MCQ.

**Collapse.** Sources that map to the same action collapse into one option,
keeping the first of `current`, `pre_update`, `stated_profile`. Free slots
take the engine's other actions from the bank in order, and options are
shuffled with the PM's keyed stream. An MCQ is emitted whenever the current
action is defined. A PM holding the `no_add_before_trigger` rule can add
before a trigger only after a breach drawn at its exit deficiency, so with a
neutral exit deficiency its loss-aversion MCQ answer is usually "cut" even
when loss aversion is active.

**Bank.** Question wording lives in the authored bank `probes.yaml`, loaded
and checked with the other catalogues: per bias, presence, MCQ,
in-situ, decline, governance and action lines; per preference group,
presence, MCQ, in-situ and governance lines; routine lines per asset
class. The loader checks slots and rejects a bias line that names the trait,
since the question must not leak the label the corpus hides.

**Config.** `probes.post_drift_weeks` (4, design), `presence_never_held` (3,
guess), `routine_per_checkpoint` (2, guess), `disposition_progress` (0.5,
design), `extrapolation_thesis_sd` and `extrapolation_trailing_sd` (design),
`overconfidence_size_edges` (design), `max_horizon` (60, design),
`situation_attempts` (10, design), `loss_depth`, `anchor_approach`,
`conviction_rating` and `decline_excess_pct` (design). Each carries its basis
and note in the config.

**Hidden columns.** `answer`, `source_a` to `source_d` and
`supporting_signal_ids` are hidden: they are ground truth and provenance for
scoring and slicing, never shown to a system under test.

## Development

```sh
uv run pytest -n auto
uv run ruff check
uv run ruff format
```

Tests marked `network` are skipped by default because they reach real
endpoints. Pass `--run-network` to run them:

```sh
uv run pytest -m network --run-network
```

There are two: one fetches a real FRED series, and one runs a single
four-turn dialogue session against the live Anthropic API to confirm the
dialogue stage's requests (structured output, advisor tools, mid-conversation
system messages) work on the configured model. The second needs Anthropic
credentials (`ant auth login` or `ANTHROPIC_API_KEY`) and spends a few cents
of tokens; run it once before a paid `dialogue` run.
