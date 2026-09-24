**Tier:** heavy
**Escalation threshold:** n/a (heavy)

# Design: behaviour engine stage

Date: 2026-09-24. Source of requirements: `docs/pm-dataset-plan.md`, sections 1.3, 3, 3.1, 4 (Gate 1 inputs), 5 (probe re-run), 8, 9 (stage 3). Builds on `docs/specs/2026-09-20-foundation-sampling-design.md`, `docs/specs/2026-09-22-market-design.md` and `docs/specs/2026-09-23-real-market-design.md`.

## Scope

Third sub-project. Delivers **stage 3, engine**: a deterministic daily loop per PM over the PM's market seed that turns sampled traits and rules into trade ideas, a ledger, rule events and hidden per-day provenance. Pure Python and numpy, no model calls, no text beyond templates.

In scope: engine core (`step`), own-signal model, condition-grammar evaluator, conflict precedence, eight bias rules, three expression adapters (equities, rates_credit, commodities), idea generation with idea-scope rules and signposts, thesis and outcome templates, five output tables (`ideas`, idea-scope rows appended to `rules`, `ledger`, `rule_events`, `position_days`), engine config section, opportunity counts in run metadata, tests.

Out of scope, each a later sub-project or spec:

- **Multi-asset PMs.** The engine skips them (no rows written; skipped ids listed in run metadata). Sleeves, `allocations.jsonl`, rebalance bands and sleeve drawdown rules are a separate adapter spec.
- **Asset-class-specific biases** (style drift, duration creep, carry chasing, roll-yield neglect). Stage 1 samples only the eight canonical parameters and `traits.jsonl` carries no row for them, so Gate 1 could not recover them. The bias registry keyed by param name is the hook for adding one later.
- **Gate 1 itself** (stage 4). The engine reports opportunity counts; pass or fail is Gate 1's.
- **Probe generation** (stage 9). The engine exposes the pure `step` function the probe generator will call; no probe code here.
- Transaction costs, financing, intraday data.

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec or a plan. Docstrings state the rule itself. Citing a published paper or a named public series as the basis of a value is allowed.

## Decisions taken

1. **Decompose: direct books first, multi-asset later.** Equities, rates_credit and commodities share one idea lifecycle; multi-asset is allocation weights, not ideas, and the plan leaves its sleeve design open. Gate 1 on three direct classes gives early signal before sleeve logic exists.
2. **Own signal is a noisy peek at the forward move.** `own_signal = skill * true_move / sd_H + sqrt(1 - skill^2) * noise`, unit variance, correlation `skill` with the realised H-day move. Chosen over a feature-based signal (unknown dispersion, collides with the extrapolation bias, skill uncontrolled) and a pure random signal (every PM loses, outcomes uninformative). The peek never leaves the engine except as a hidden column.
3. **Thesis expected move is unscaled**: `own_signal * sd_H`, not `skill * own_signal * sd_H`. The scaled version is the Bayesian conditional forecast, but it makes every target tiny (about 2% on crude for a 1.5-sd view), so target rules almost never fire and the disposition statistic loses its reference. Unscaled is the view as a PM states it. Design choice; Gate 1 calibrates extrapolation through the engine anyway.
4. **Hidden provenance: a seventh engine table `position_days` plus hidden columns on `ideas`.** Gate 1 needs loss-side untriggered position-days (loss aversion opportunities), every open position's gain or loss on sell days (Odean), anchor level at each exit, own signal versus street view at entry (herding conflict), stated interval per idea (overconfidence) and conviction rank versus size rank. `ledger.bias_flag` alone cannot carry these. Same rule as `bias_flag`: stripped from every view given to a system under test. Plan deviation: section 8 lists no such table.
5. **Expression preferences plant only where the universe has a form.** Fixed form set per asset class; each preference value maps to one form or to null (stated only). See the expression-preference table under Adapters. Alternative rejected: all expression preferences stated only, which loses Gate 1's one preference fingerprint and leaves rates PMs never trading curves.
6. **Architecture: pure daily step, adapters as data, bias rules as a registry.** `step(state, day, params, adapter, rng) -> (state, DayOutput)` with no I/O, so a probe generator can call it on a hand-built one-position state per option source. Rejected: a stateful PM agent object (cloning for probe re-runs, drift and regime overrides leak into object state) and a numpy-vectorised loop (bias logic is branchy per position, precedence is sequential, and the population is at most 180 PMs by 260 days by about 10 positions).
7. **No transaction costs.** P&L is mark-to-market in the adapter's unit. A cost model would add a guessed parameter with no fingerprint.
8. **Opportunity counts come from the engine, verdicts from Gate 1.** The loop tallies per PM per fingerprint and writes them to run metadata; no pass or fail here.
9. **Ledger rows carry `tenor`.** Curve legs (rates tenors, commodity months) are tenors of one instrument, not instruments, so a multi-leg idea is one ledger row per leg with the same `trade_idea_id` and a `tenor`. Plan deviation: section 3's column list has no `tenor`.
10. **Idea-scope rules are appended to `rules`.** The stage rewrites the table with stage 1's PM-scope rows unchanged plus the idea rows; the stage runner gains an `appends` contract that checks the existing rows survive.
11. **`rule_events.response` gains `overridden`** for a trigger that lost the precedence contest (plan section 3: a hold rule overridden by a stop is logged, never counted as a breach).

## Package layout

```
src/pm_traitbench/
  enums.py                 # + RuleResponse (acted, acked_no_action, added, overridden),
                           #   Expression (outright, pair, curve, calendar_spread),
                           #   Side (buy, sell)
  config.py                # + EngineConfig; Config cross-checks
  errors.py                # + EngineError
  stages.py                # Stage gains `appends: tuple[TableSpec, ...]`; run_stage checks them
  pipeline.py              # + ENGINE_STAGE
  tables/
    schema.py              # + Idea, LedgerRow, RuleEvent, PositionDay
    specs.py               # + IDEAS, LEDGER, RULE_EVENTS, POSITION_DAYS, HIDDEN_COLUMNS
  catalogues/
    signposts.yaml         # signpost text templates per asset class and kind
    theses.yaml            # thesis and outcome templates per asset class and expression
    models.py, loader.py   # + SignpostCatalogue, ThesisCatalogue
  engine/
    __init__.py
    stage.py               # ENGINE_STAGE number 3 "engine"
    market_view.py         # MarketView: one seed in memory; field lookups, trailing stats
    state.py               # PmState, Position, Leg, IdeaDraft
    params.py              # EffectiveParams per day: base, drift, regime multiplier
    own_signal.py          # own signal, forecast, interval, conviction bucket
    rules_eval.py          # condition grammar evaluator with window run counters
    precedence.py          # action precedence and the conflict table
    biases/
      __init__.py          # BIAS_RULES registry keyed by param name
      loss_aversion.py disposition.py anchoring.py extrapolation.py
      herding.py overconfidence.py conviction.py exit_deficiency.py
    adapters/
      __init__.py          # adapter_for(asset_class)
      base.py              # Adapter protocol: universe, forms, fields, levels, anchors, peers
      equities.py rates_credit.py commodities.py
    ideas.py               # arrival, candidate, form, direction, sizing, idea-scope rules
    step.py                # step(state, day, params, adapter, rng) -> (state, DayOutput)
    loop.py                # run_pm: days -> rows and opportunity counts
    templates.py           # thesis and outcome rendering from the catalogues
tests/engine/              # one test module per engine module, fixtures in conftest.py
```

Dependency direction: `engine/` depends on `config`, `rng`, `enums`, `tables`, `catalogues`, `market.axis`, `market.regimes` (regime lookup) and `market.synthetic.processes.common` (round-level grid helpers, to be moved to a shared `market/levels.py` so the engine does not import a synthetic module). Nothing in `market/` or `sampling/` imports `engine/`.

## Enums and row models

`Side`: `buy | sell`. `Expression`: `outright | pair | curve | calendar_spread`. `RuleResponse`: `acted | acked_no_action | added | overridden`. `Action` is unchanged; `roll` and `hold` already exist.

`Idea` (table `ideas`, key `pm_id, trade_idea_id`; id pattern `ti_\d{3,}`, per PM sequence):

| Column | Type | Note |
|---|---|---|
| pm_id, trade_idea_id | str | |
| instrument_id | str | the instrument every leg belongs to (pair: the long leg; the short leg's id is on its ledger rows and in `legs`) |
| expression | Expression | |
| side | Side | direction of the idea on its level series (buy = expects the level series to rise) |
| legs | list of `{instrument_id, tenor, side, weight}` | one entry per leg; tenor null for equities and credit |
| entry_date, exit_date | date, date or null | exit null while open at horizon end |
| entry_level, target_level, stop_level | float | in the adapter unit of the level series |
| thesis | str | rendered template |
| outcome | str or null | rendered template, null while open |
| own_signal, forecast, interval_lo, interval_hi | float | hidden |
| street_view_at_entry | StreetView | hidden |
| conflict, followed_street | bool | hidden; `followed_street` null when no conflict |
| conviction | int 1-5 | hidden here, public on the ledger as `stated_conviction` |
| size_rank | int 1-5 | hidden |

`LedgerRow` (table `ledger`, key `pm_id, date, trade_idea_id, instrument_id, tenor, side`): plan columns `date, trade_idea_id, instrument_id, instrument_type, side, size, risk_amount, price_or_yield, stated_conviction, bias_flag, rule_id` plus `tenor` (str or null). `size` is in the mandate's risk unit and always positive; `side` carries direction. `risk_amount` is notional in currency. `bias_flag` is `param:reason` or null; `rule_id` is the rule breached by this row, or null.

`RuleEvent` (table `rule_events`, key `pm_id, rule_id, trade_idea_id, date_fired`): `response: RuleResponse`, `response_date: date`. `response_date` equals `date_fired` for every response in this build (the PM responds the day the trigger fires); the column exists because the plan's schema has it and a later build may add delay.

`PositionDay` (table `position_days`, key `pm_id, date, trade_idea_id`), hidden in full:

| Column | Note |
|---|---|
| pnl_unit | `(level_now - entry_level) * (+1 if side == buy else -1)`, in the adapter unit; positive means the position is winning whatever its direction (the level series is already sign-flipped for yields and spreads) |
| pnl_z | `pnl_unit / sd_H`, the unit-free P&L every bias formula uses |
| pnl_state | `gain | loss | flat` (flat when `abs(pnl_z) < 1e-9`) |
| sessions_held | int |
| triggers_fired | count of idea-scope and PM-scope triggers fired on this idea so far |
| trigger_pending | bool, a trigger fired today |
| action | `none | hold | add | cut | trim | exit | roll` |
| bias_flag | as on ledger, or null; also set on hold days (`disposition:hold_loser`, `loss_aversion:hold`) that write no ledger row |
| anchor_level, effective_exit_level | floats or null |

`HIDDEN_COLUMNS: dict[str, tuple[str, ...]]` in `specs.py`: `ledger -> (bias_flag, rule_id)`, `ideas -> (own_signal, forecast, interval_lo, interval_hi, street_view_at_entry, conflict, followed_street, conviction, size_rank)`, `position_days -> all`. `rule_id` on the ledger is hidden because it names a breach. Later stages strip through this one registry.

## Config: `engine`

Every field carries `basis` and `note` like the rest of the config; every value below is a guess unless stated, and each exists because Gate 1 needs a knob to turn.

| Field | Default | Note |
|---|---|---|
| `horizon_days` | 20 | H, forward window of the own signal and the trailing window for vol and extrapolation |
| `skill` | 0.15 | correlation of own signal with the realised move; a small positive edge |
| `arrival_rate` | 0.6 candidate attempts per day | the one knob that sets ideas per year; a scratch simulation at plan time gave about 48 ideas per PM-year (10th percentile 39) at 0.6, inside the 30-80 target |
| `rr_range` | (1.5, 3.0) | target distance as a multiple of stop distance, plan section 1.3 |
| `signpost_k` | 1.0 | level and relative signposts sit at `k * sd_H` from entry |
| `preferred_form_weight` | 0.7 | share of ideas in the preferred expression when a mapped preference is held; uniform over forms otherwise |
| `softmax_tau` | 1.0 | temperature of the loss-side action draw, in `pnl_z` units |
| `base_hazard` | 0.03 | daily discretionary sell hazard at zero progress; about one in 33 days |

Cross-check on `Config`: `horizon_days` at most a quarter of the horizon in trading days.

**Fixed constants in code** (`engine/constants.py`, each with a one-line basis note; design choices, not knobs, because Gate 1 has no reason to turn them and every extra knob is a sensitivity axis):

| Constant | Value | Note |
|---|---|---|
| `ENTRY_THRESHOLD` | 1.0 | minimum `abs(own_signal)` to enter; one sd of view |
| `CONVICTION_CUTS` | (1.0, 1.4, 1.8, 2.3) | `abs(own_signal)` edges for buckets 1-5 |
| `SIGNPOSTS_PER_IDEA` | 2 or 3, uniform | plan section 1.3 |
| `ADD_FRACTION` | 0.5 | an add is half the original size |
| `RISK_STEPS` | (0.2, 0.4, 0.6, 0.8, 1.0) | size at rank 1-5 as a fraction of the mandate cap, linear |
| `NO_ENTRY_LAST_SESSIONS` | 5 | every idea gets at least a week of life |
| `DV01_PER_MILLION` | 2Y 190, 5Y 450, 10Y 800, 30Y 1700; credit `duration_years * 100` | approximate DV01 of a par bond per 1m notional |
| contract multipliers | per commodity, in `market/constants.py` next to the start prices | exchange contract specs |

## Market view

`MarketView.build(store_rows, seed, instruments, axis)` loads one seed's `prices`, `curves`, `consensus`, `calendar` and `regimes` into arrays indexed by `(instrument_id, tenor)` and day index on the horizon axis (the engine never sees burn-in rows; trailing windows that would reach before day 0 are truncated to the days available, and the first `horizon_days` days therefore use shorter windows). One view per seed, shared by every PM on that seed.

Lookups the adapters use:

- `level(series_key, t)`: the level series of a leg or a form (see adapters).
- `trailing_move(series_key, t, H)`, `sd_H(series_key, t)`: trailing H-day move and the sample sd of daily moves over the trailing 60 days scaled by `sqrt(H)`, floored at a small positive value.
- `forward_move(series_key, t, H)`: the move from `t` to `min(t + H, last_day)`.
- `street_view(instrument_id, t)`, `positioning(instrument_id, t)`.
- `events(instrument_id, t)`: event types on that day for that instrument, plus market-wide rows.
- `event_types(instrument_id)`: event types that occur for that instrument anywhere on the seed (signpost catalogue keys on these, so a rule keyed to an event type absent on the seed never exists).
- `regime(t)`.
- `days_to_expiry(instrument_id, t)`: sessions until the next `contract_expiry` calendar row for the instrument.
- `peer_move(instrument_id, t, H)`: mean trailing move of the adapter's peer group (sector, rating band, commodity group), for relative signposts.
- `round_level(series_key, level)`: nearest grid level, same grid as the market stage's range pull: half-decade for prices (`log_grid_step`), 0.25% for yields, 10bp for spreads.

## Adapters

`Adapter` protocol (`adapters/base.py`):

```
asset_class: AssetClass
def universe(instruments, rules) -> list[str]            # candidate instrument ids after exclusion rules
def forms(sub_style) -> tuple[Expression, ...]
def build_legs(form, instrument_id, side, view, rng) -> list[Leg]
def series_key(legs) -> SeriesKey                          # the level series the idea is judged on
def unit(form) -> str                                      # pct, bp
def position_fields(position, view, t) -> dict[str, float | int | str]   # every rule field
def size_to_risk(size_pct_book, legs, view, t) -> (size, risk_amount)     # risk unit and notional
def anchors(position, view, t) -> list[float]              # candidate anchor levels
def peer_group(instrument_id) -> str
def round_step(series_key) -> float
```

| | Equities | Rates and credit | Commodities |
|---|---|---|---|
| Universe | kind equity, minus `sector` exclusions | sovereign_rates: curve instruments; long_short_credit: credit issuers minus `rating_band` exclusions | kind commodity minus `commodity_group` exclusions |
| Forms | outright, pair | sovereign_rates: outright (10Y), curve (2s10s or 5s30s); long_short_credit: outright | commodity_futures_directional: outright; curve_and_spread: outright, calendar_spread (M1 versus M4, M5 or M6) |
| Level series | log price times 100 (pct); pair: log ratio times 100 | yield in percent times 100 (bp) for a tenor; curve: long tenor minus short tenor; credit: `spread_bp` | log price times 100; calendar spread: `(M1 - Mk) / M1 * 100` |
| Bullish sign | up | yield and spread down (sign flipped so buy = expects the flipped series to rise); curve buy = steepener | up |
| Rule fields | `pnl_from_entry` (pct), `sector`, `n_positions`, `sessions_held`, `size_pct_book`, `triggers_fired`, `target_hit`, `price` | `adverse_yield_move_bp`, `adverse_spread_move_bp`, `rating_band`, `spread_bp`, plus the shared fields | `pnl_from_entry`, `commodity_group`, `days_to_expiry`, `price`, plus shared |
| Size to risk | `size = size_pct_book` (pct_nav), `risk_amount = book * size_pct_book / 100` | notional as above; `size` = DV01 in currency = notional / 1m times `DV01_PER_MILLION[tenor]` (credit: `duration_years * 100`); a curve trade is DV01-neutral, so `size` is the DV01 per leg | notional as above; `size` = contracts = notional / (price times contract value), rounded down to at least 1 |
| Loss reference | entry level | entry yield or spread | entry price, roll-adjusted |
| Anchors | entry level, nearest round price (range regime), trailing 60-day high | entry level, nearest round yield 0.25% or spread 10bp (range regime), trailing 60-day yield high | entry level, nearest round price (range regime), trailing 60-day high |
| Peer group | sector mean | curve: the curve's 10Y; credit: rating-band mean | commodity group mean |

`pnl_from_entry` for a rates position is the DV01-weighted move in bp expressed as a percent of notional (`bp move * dv01 / notional * 100`) so the shared `pnl_z` and `pnl_state` are defined on every asset class; the rates stop rules read `adverse_yield_move_bp` directly.

### Expression preferences to forms

The catalogue's expression preferences map to a form only where the universe supports one; every other value is stated only (dialogue) and leaves no ledger fingerprint. Mapping is a fixed table in `adapters/__init__.py`:

| Preference param | Value | Form |
|---|---|---|
| `duration_expression` | steepeners over outright duration | curve |
| | outright duration over curve trades | outright |
| | butterflies over outright duration | null (no butterfly form) |
| `pair_vs_outright` | express the view as a pair trade | pair |
| | express the view as an outright position | outright |
| | express the view as a basket versus the index | null (no index instrument) |
| `curve_trade_expression` | express curve views as calendar spreads | calendar_spread |
| | spread ratios, butterfly spreads | null |
| `hedge_instrument`, `futures_vs_etf`, `fx_hedge_expression`, `credit_index_vs_single_name` | all values | null |

A mapped preference tilts the form draw to `preferred_form_weight` on its form; the remainder is split evenly over the other forms of the sub-style. A PM with no mapped preference draws forms uniformly. A `curve_and_spread` commodity PM whose preference maps to null still draws calendar spreads at the uniform rate.

Pair legs: the long leg is the candidate, the short leg is the same-sector name with the most opposite own signal on the day; both legs are entered at equal notional. Curve legs are DV01-neutral: the long-tenor leg's notional is scaled by the DV01 ratio. Calendar spread legs are equal contracts.

## Own signal, forecast, conviction (`own_signal.py`)

For a candidate series key at day `t`, horizon H:

```
z            = forward_move / sd_H                          # standardised realised move
own_signal   = skill * z + sqrt(1 - skill^2) * n,  n ~ N(0, 1), stream(root, "engine", pm_id, "signal", t, attempt)
thesis_move  = own_signal * sd_H                            # decision 3: unscaled
forecast     = (1 - theta) * thesis_move + theta * trailing_move   # extrapolation, plan 3.1
interval     = forecast ± z_c * sd_H * sqrt(1 - skill^2),  z_c = Phi^-1((1 + c) / 2)
conviction   = 1 + number of CONVICTION_CUTS below abs(own_signal), capped at 5
```

`c` is `overconfidence_coverage` (0.8 neutral by the marginal's definition, so a neutral PM states the honest 80% interval). `theta` is `extrapolation_theta`. Both are the day's effective values. When `t + H` runs past the last day the forward move is truncated and `sd_H` is scaled to the days available; no entries in the last 5 sessions (design, so every idea has at least a week of life).

## Idea generation (`ideas.py`)

Per day, after monitoring:

1. `n_attempts ~ Poisson(arrival_rate)` on `stream(root, "engine", pm_id, "arrival", t)`.
2. Per attempt: stop if `n_positions >= max_positions` rule level (when the PM has one) or the day is inside the last 5 sessions. Draw a candidate uniformly from `universe` minus instruments already held in an open idea. Draw the form from `forms(sub_style)`: weight `preferred_form_weight` on the preferred form when the PM holds a mapped expression preference, remainder split evenly, else uniform. Build legs, series key, own signal.
3. Enter only if `abs(own_signal) >= ENTRY_THRESHOLD`.
4. Direction: `side = buy if own_signal > 0 else sell`. Herding: if `street_view` is not neutral and disagrees with `side`, `conflict = true`; with probability `w` (`herding_weight`) the side flips to the street's and `followed_street = true`, `bias_flag = herding:followed_street` on the entry rows. `own_signal` keeps its sign on the idea row.
5. Sizing: `size_rank = round((1 - m) * conviction + m * u)`, `u ~ U(1, 5)` continuous, clipped to 1-5 (`m` = `conviction_size_miscalibration`; `m = 0` means rank equals conviction). `size_pct_book = cap_level * RISK_STEPS[size_rank - 1] * (z_0.8 / z_c)`, then clipped to `cap_level` (the mandate cap is a rule and is never breached). `bias_flag` on the entry rows: `overconfidence:oversized` when `c` is active and the factor exceeded 1, `conviction:mis_sized` when `m` is active and `size_rank != conviction`; when several apply they are joined with `;` in a fixed order (herding, overconfidence, conviction).
6. Levels: `stop_level` from the PM stop rule applied to `entry_level` in the form's unit (pct rules on price series, bp rules on yield and spread series; a pct stop on a rates PM cannot occur because the rule catalogue keys the stop field on sub-style). `target_level = entry_level + side * rr * stop_distance`, `rr ~ U(rr_range)`. Both are already in vol terms through the stop rule's own scale; the plan's "scaled by trailing volatility" is met by the signpost `k * sd_H` levels, not by moving the PM's stated stop.
7. Idea-scope rules (appended to `rules`, ids continue the PM's `r_NN` sequence, `scope = idea`, `source = self`): `stop` (`field` = the series field, `op` and `level` = `stop_level`, `action = exit`), `target` (`action = target`, `level = target_level`), and `SIGNPOSTS_PER_IDEA` signposts drawn without replacement from: (a) event, `field = event`, `op = ==`, `level` = an event type from `event_types(instrument_id)`, `window 1`; (b) level held, `field` = series field, level `entry ± signpost_k * sd_H` on the adverse side, `window ~ U{3, 4, 5}`; (c) relative, `field = relative_move_pct` (adapter: own trailing H move minus peer trailing H move), adverse beyond `signpost_k * sd_H`, `window 1`. (a) is skipped when the instrument has no event type on the seed (credit on a real seed). `text` from `catalogues/signposts.yaml`, keyed by asset class and kind, with slots for level, unit, window and event name.
8. Ledger rows: one per leg, `side` per leg, `stated_conviction = conviction`, `price_or_yield` = the leg's level on the day, `rule_id` null. `ideas` row with the thesis rendered from `catalogues/theses.yaml` (keyed by asset class and expression; slots: instrument name, tenor pair, entry level, target, expected move, horizon).

## Daily monitoring (`step.py`)

Order within `step` for one PM on day `t`:

1. **Params.** `EffectiveParams.for_day(t)` (below).
2. **Mark.** For every open position compute the position fields, `pnl_unit`, `pnl_z`, `pnl_state` and `progress = pnl_unit / abs(target_level - entry_level)` (used by the hazard below, not stored: it is arithmetic on stored columns).
3. **Triggers.** `rules_eval` evaluates every rule with a market or position condition against every open idea: PM-scope stop, trim, roll, and the idea's own stop, target and signposts. `min_holding_period` is evaluated as a condition too (`sessions_held < level`) but only produces an event row on a day another trigger wins against it (step 4); on every other day it is a silent constraint on discretionary actions (step 7). `max_positions`, the mandate cap and `no_add_before_trigger` never fire; they bound entries and adds. A rule with `window > 1` keeps a run counter per `(trade_idea_id, rule_id)` in state; it fires when the counter reaches `window`. Idea-scope rules fire once. The PM-scope stop and trim fire once per position size (they re-arm after an add or a trim). A `rule_events` row is written per firing.
4. **Precedence.** Actions of the day's firings are ordered `exclude | cap` > `exit` (stop, signpost) > `roll` > `trim_half | target` > `hold`. The highest is the effective trigger; every other firing that day gets `response = overridden`. `hold` never wins: when an exit, roll or trim fires inside the holding period the hold rule gets an event row with `response = overridden`, so the ledger and the transcript can both say the PM broke their own holding rule for a stated reason without Gate 1 counting a breach. Full conflict table:

| Fires together | Winner | Loser logged |
|---|---|---|
| stop, signpost | stop (both exit; the stop is the PM's stated rule) | signpost overridden |
| stop or signpost, roll | exit | roll overridden |
| stop or signpost, target or trim | exit | target overridden |
| stop or signpost, hold | exit | hold overridden |
| roll, target or trim | roll, then the trim executes on the rolled position the same day | none |
| roll, hold | roll | hold overridden |
| target, trim | trim (target is the level, trim is the action) | none, one event each with `acted` |
| target or trim, hold | trim (plan order: trims before holds) | hold overridden |
| two signposts | one event each; one exit | none |

5. **Response** (`exit_deficiency.py`) to the effective trigger: with probability `1 - e` the action executes, `response = acted`. With probability `e`: if `loss_aversion_lambda` is active and the position is at a loss, `response = added`, an add of `ADD_FRACTION` of the original size, ledger row with `rule_id` = the fired rule and `bias_flag = exit_deficiency:added`; otherwise `response = acked_no_action`. A breached idea-scope trigger does not re-fire; the PM-scope stop re-arms only after the position changes size, so a breach is not re-tested daily.
6. **Discretionary decisions** on positions with no trigger pending today, in this order per position; the first that acts ends the position's day:
   - **Anchored exit** (`anchoring.py`): `effective_exit = (1 - rho) * target_level + rho * anchor`, anchor = the candidate from `anchors()` nearest to the current level on the target side, round level only in the range regime. If the day's level has reached `effective_exit`, exit; `bias_flag = anchoring:exit_at_anchor` when `rho` is active (Gate 1 defines its own band for the rate statistic; the engine only records provenance). At neutral `rho` this is the plain target exit through a level the target rule also fires on, so it rarely acts before the target rule.
   - **Loss side** (`loss_aversion.py`), when `pnl_state == loss`: values in `pnl_z` units, `V_cut = -lambda * abs(pnl_z)`, `V_hold = forecast_remaining / sd_H` (the forecast from entry minus the move so far), `V_add = V_hold * (1 + ADD_FRACTION) - lambda * abs(pnl_z) * ADD_FRACTION`; softmax at `softmax_tau` on `stream(root, "engine", pm_id, "loss_side", t, trade_idea_id)`. `add` is blocked when the PM holds `no_add_before_trigger` and `triggers_fired == 0` unless a breach draw at `e` passes; a passed breach writes the ledger add row with `rule_id` = the no-add rule and `bias_flag = loss_aversion:add_before_trigger`. An unblocked add writes `bias_flag = loss_aversion:add` when lambda is active. A cut is an exit with `bias_flag` null. A hold writes `position_days.bias_flag = loss_aversion:hold` when lambda is active.
   - **Sell hazard** (`disposition.py`): `h = base_hazard * (1 + max(progress, 0))`, then `h * sqrt(D)` at a gain and `h / sqrt(D)` at a loss, clipped to [0, 1]. A draw below `h` sells: at a gain, exit with `bias_flag = disposition:realise_gain_early` when `D` is active and `progress < 1`; at a loss, exit with `bias_flag` null. A no-sell day at a loss writes `position_days.bias_flag = disposition:hold_loser` when `D` is active (the loss-side branch above runs first, so this only applies when it chose hold).
7. **Constraints checked on every action**: `min_holding_period` blocks a discretionary exit or trim before `sessions_held` reaches the level (the position holds; no event row, since the rule did not fire, it constrained); the mandate cap and `max_positions` bound adds and entries.
8. **Roll** (commodities): the roll rule fires at `days_to_expiry <= level`. `acted`: sell the M1 leg at M1 and buy at the Mk price of the same instrument where k is the month after expiry (M2 on the curve table), two ledger rows with `tenor` M1 and M2; the position's `entry_level` is shifted by `M2 - M1` so `pnl_from_entry` is continuous, and after the expiry row the leg is tracked as M1 again. Breach: `acked_no_action`; on the expiry day the position is force-rolled with the same two rows and `bias_flag = exit_deficiency:late_roll`. A PM without a roll rule rolls on the expiry day with no event and no flag.
9. **Idea generation** (above).
10. **Position days** row per open position (including those closed today, with the day's action).
11. **Close-out**: on the last horizon day every open position is exited at the close with `bias_flag` null and `outcome` rendered with "open at horizon end"; the idea's `exit_date` is the last day. Design: a ledger that ends with open positions leaves Gate 1's disposition count incomplete.

`DayOutput` is `ledger_rows, rule_events, position_days, new_ideas, closed_ideas, idea_rules, opportunities` (counts per fingerprint for the day).

## Effective parameters (`params.py`)

`EffectiveParams.for_day(t)` returns one value per bias param:

1. Base: `traits.value` (inactive biases carry their sampled neutral value, already in the table).
2. Drift, by date: `update` replaces the value with `to`; `dormant` sets the param to the median of its neutral marginal in `biases.params[param].neutral`; `revive` restores the value in force before the dormant event.
3. Regime: multiply by the trait's multiplier for the seed's regime on `t`; on the logit scale for params on [0, 1] (`anchoring_rho, extrapolation_theta, herding_weight, overconfidence_coverage, conviction_size_miscalibration, exit_deficiency`), plain for `loss_aversion_lambda` and `disposition_ratio`. `overconfidence_coverage` is lower-is-stronger, so its multiplier is applied to `1 - c` on the logit scale.

`active` per param comes from `traits.active` and is what decides whether a `bias_flag` is written; the formula runs on every PM regardless.

## Opportunity counts

The loop tallies per PM: `loss_side_untriggered_days`, `triggers_fired`, `conflict_entries`, `exits`, `sell_day_position_days`, `ideas`, `entries_after_run` (entries where trailing move exceeded `sd_H`). Run metadata extras: `{"ideas_per_pm": {pm_id: n}, "opportunities": {pm_id: {...}}, "skipped": [pm_id, ...]}`. Gate 1 reads these against `n_min`.

## Stage (`engine/stage.py`)

`ENGINE_STAGE`: number 3, name `engine`, reads `PERSONAS, TRAITS, RULES, DRIFT_EVENTS` and the six market tables, writes `IDEAS, LEDGER, RULE_EVENTS, POSITION_DAYS`, appends `RULES`. `run`: load stage 1 tables; group PMs by seed; build one `MarketView` per seed; for each PM in `pm_id` order: skip multi_asset, else validate every PM rule `field` against the adapter's field set (an unknown field raises `EngineError` naming the rule before any day runs), then `run_pm`. Rows are collected and written once at the end.

`run_stage` change: `Stage.appends` tables must exist before the run (like `reads`), may exist as outputs, and after the run every pre-existing row must be present unchanged in the written table (compared as records), else `StageIOError`.

## Errors

`EngineError(PmTraitbenchError)`, exit code 1: unknown rule field for the adapter, a position size going negative, a cap exceeded after sizing, an idea closed twice, a `tenor` missing on a curve leg. Each message names `pm_id`, the date and the idea or rule.

## Catalogues

`signposts.yaml`: per asset class, per kind (`event`, `level`, `relative`), 2-3 templates with slots `{level}`, `{unit}`, `{window}`, `{event}`, `{peer}`. `theses.yaml`: per asset class, per expression, 2-3 thesis templates with slots `{name}`, `{tenors}`, `{entry}`, `{target}`, `{move}`, `{unit}`, `{horizon}`, and outcome templates keyed by `win | loss | open` with `{pnl}`, `{unit}`, `{closer}` (the rule param or `discretionary`). Drafted with a model once, human-edited, versioned with the code, like the stage 1 banks. `check_catalogue` extends to require a template for every (asset class, expression) the adapters expose and every event type in `EventType`.

## Testing

`tests/engine/conftest.py`: a small fixture market (3 equities in one sector, `RT-USD`, 2 credit issuers, 2 commodities with expiry rows, 60 horizon days, one consensus flip, one earnings and one cb_meeting row) built directly as row lists, plus a fixture PM per asset class with neutral traits and a full rule set.

- `test_market_view.py`: field lookups, trailing and forward moves, truncation at the edges, `days_to_expiry`, `event_types`, round levels per family.
- `test_own_signal.py`: over 20,000 draws the correlation of `own_signal` with `z` is within 0.02 of `skill`; interval coverage at `c = 0.8` within 0.02 of 0.8; conviction buckets monotone in `abs(own_signal)`.
- `test_rules_eval.py`: every op; window counters reset when the condition breaks; string levels (`sector`, `event`); idea-scope rules fire once; PM stop re-arms after an add.
- `test_precedence.py`: every pair in the conflict table, including the deferred trim.
- `tests/engine/biases/test_*.py`: each rule on a fixed one-position state, neutral versus extreme value, on a fixed stream: lambda 1 versus 4 raises the add share; D 1 versus 3 raises gain-side sells relative to loss-side; rho 0 versus 1 moves the exit to the anchor; theta 0 versus 1 moves the forecast to the trailing move; w 0 versus 1 goes from never to always following the street on conflicts; c 0.8 versus 0.3 enlarges size; m 0 versus 1 drops the rank correlation to about 0; e 0 versus 1 goes from every trigger acted to none.
- `test_adapters.py`: per asset class, universe after exclusions, forms per sub-style, legs and DV01 neutrality, level series and bullish sign, every rule field in the catalogue for that asset class resolves, size conversion.
- `test_ideas.py`: threshold gate, no entries in the last 5 sessions, cap never exceeded, stop and target on the right side of entry, signposts skip the event kind when the instrument has none, idea rule ids continue the sequence.
- `test_params.py`: update, dormant, revive by date; logit-scale multiplier stays in (0, 1); coverage multiplier lowers `c`.
- `test_step.py`: determinism (same inputs, same output); one day with a stop firing produces one event, one ledger row and a closed idea; a roll produces two rows and a continuous P&L.
- `test_loop.py`: neutral fixture PMs over 60 days: every ledger row belongs to an idea, every idea-scope rule to an idea, closed ideas have zero size, breach rate near `e`, opportunity counts match a recount from the tables.
- `test_stage.py`: runs on the fixture through the store, all tables written, PM-scope rules unchanged in `rules`, run metadata extras present, multi-asset PMs skipped and listed, unknown rule field fails before any write.
- `tests/test_stages.py`: the `appends` contract, including a modified pre-existing row failing.
- `tests/test_end_to_end.py`: extended through stage 3 on the demo config with the fixture market cache.

## Refinements fixed at plan time

- Level series are quoted, not flipped, and carry a `bullish_sign`; `pnl_unit = bullish_sign * side_sign * (level_now - entry_level)`. Ledger leg sides follow `side_sign * bullish_sign * coeff * leg_bullish`, so a steepener buy books a 2Y buy and a 10Y sell.
- The PM-scope `stop_loss` rule is applied only through each idea's `stop` row, so a stop produces one event. `trim_at_target`, `min_holding_period` and `roll_before_expiry` are evaluated daily.
- An idea `target` row firing alone exits the whole position; with `trim_at_target` present the trim wins and half comes off.
- `min_holding_period` writes an event row only when overridden by an exit, roll or trim; otherwise it silently blocks discretionary exits.

- Thesis templates carry a `{side}` slot (`long`/`short`, `steepener`/`flattener`, `long the front`/`short the front`) and never a directional verb; signpost templates are direction-neutral; outcome templates take `{closer}` as a rendered phrase (`the stop`, `the target`, `a signpost`, `the trim`, `the roll`, `my call`, `the year end`). Found at review: hard-coded direction words contradicted short ideas and non-target closes.

## Limitations to record in the README

- On real seed R1 the credit universe is two index-level issuers and the rates universe one curve, so long_short_credit PMs on the pilot revisit the same two instruments and their `max_positions` rule never binds. Sovereign_rates PMs trade one curve's tenors only.
- Signposts are limited to the condition grammar: one event type, one held level, one relative move. No qualitative signposts.
- The own signal peeks at the future with a fixed skill; PM skill is not a trait and is the same for every PM.
- Multi-asset PMs produce no engine rows until the multi-asset adapter sub-project.

## Delivery order

1. Enums, row models, table specs, `HIDDEN_COLUMNS`, `EngineConfig`, `EngineError`, the `appends` contract, the shared `market/levels.py` move.
2. `MarketView`, `state.py`, `params.py`, `own_signal.py`.
3. `rules_eval.py`, `precedence.py`, the bias registry with all eight rules.
4. Adapters: equities first, then rates_credit, then commodities.
5. `ideas.py`, catalogues, `templates.py`.
6. `step.py`, `loop.py`, `stage.py`, pipeline registration, README.
7. Run stages 1-3 at default config on `data/`; record wall time, ideas per PM per asset class, opportunity counts against the plan's `n_min` table, and any cell below `n_min` as an input to the Gate 1 sub-project.
