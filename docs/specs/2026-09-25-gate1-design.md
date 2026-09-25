**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** none (the engine spec's decision 8, "opportunity counts come from the engine, verdicts from Gate 1", is what this spec delivers)

# Design: Gate 1, parameter recovery from the ledger

Date: 2026-09-25. Source of requirements: `docs/pm-dataset-plan.md`, sections 1.1 (floors), 3.1 (Gate 1 statistic column), 9 (stage 4 row and paragraph, n_min table, verification paragraph). Builds on `docs/specs/2026-09-24-engine-design.md`.

## Scope

Fourth sub-project. Delivers **stage 4, gate1**: a deterministic pass over the engine's tables that computes one statistic per bias parameter per PM, aggregates it per asset class into a neutral baseline and an active separation, decides pass or fail per (asset class, parameter) on synthetic seeds, reports the same numbers for real seeds without blocking, and writes two tables plus run metadata. Pure Python and numpy, no model calls.

In scope: `gates/gate1/` package (inputs, eight estimators, aggregate, verdict, stage), two output tables, a `gate1` config section, a `Gate1Error`, a `DataStore.read_run_metadata` helper, a series-from-idea helper in the engine, README section, tests.

Out of scope: multi-asset PMs (no engine output yet); rescaling the section 1.1 marginals after calibration (a human config edit made after reading the report); Gate 2; changing market or engine knobs when a cell fails (the report says so, the fix is a separate change).

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself. Citing a published paper or a named public series as the basis of a value is allowed.

## Decisions taken

1. **Neutral baseline per asset class, pooled across synthetic seeds.** A (seed, asset class) cell holds 12 PMs and 1-11 neutral PMs per parameter, so a per-cell standard deviation has no meaning. Synthetic seeds share the instrument universe and every idiosyncratic draw and differ only in regime order, so their neutral PMs are one population (about 36 PMs per asset class, 20-25 neutral per parameter). Per-cell rows are still written, as non-blocking breakdowns. Rejected: per-cell verdicts (noise), and pooling across asset classes (the plan's reason for per-asset-class gating, a yield versus a price as the loss reference, applies to the noise too).
2. **Pass rule = separation plus recovery.** Pass when (a) the neutral standard deviation is at most `gap_fraction` (0.5) of the gap between the active and neutral means of the statistic, so the active mean sits past the floor (neutral mean plus `floor_se` = 2 neutral standard deviations, in the strengthening direction), and (b) the Spearman correlation between planted value and recovered statistic over neutral plus active PMs is at least `min_rank_corr`. (b) is what "planted values must come back near what was planted" means once scales differ, and rank correlation is scale-free so it applies to definition and spirit rows alike. The share of active PMs past the floor is reported, not gated: the active marginals overlap a 2-sd floor by construction (about a fifth of Beta(4, 5) exit-deficiency draws sit below a 0.27 floor), so the share measures the marginal, not the estimator, and it is the input for re-centring the marginal as section 1.1 of the dataset plan describes. The plan's quarter-gap rule is not used on the cross-PM spread: that rule bounds one PM's sampling error, while the neutral spread across PMs also carries the neutral marginal's own spread (Beta(2, 10) herding has sd 0.11 against a 0.41 gap), so a quarter-gap test would fail a perfect estimator. Rejected: separation only (a constant-offset estimator would pass) and recovery with a slope test (needs the calibration done first for spirit rows).
3. **Opportunity-count shortfall is a warning, the measured neutral error is the test.** The plan's n_min values are checked per synthetic cell against the 10th-percentile PM and recorded as `count_shortfall`, but a cell fails only on the measured tests in decision 2. This settles the herding shortfall (10th-percentile PM has 6-12 conflicts against n_min 14) by pooling, as the plan allows, and applies one rule to every fingerprint. Market consensus knobs are tuned only if the pooled test fails.
4. **Two tables plus run metadata.** `gate1_pm` (one row per PM per parameter per split) and `gate1_cells` (one row per seed group per asset class per parameter per split), queryable and diffable like every stage output. Run metadata carries the failure list, warnings and thresholds. Rejected: run metadata only (about 1,500 nested entries nobody can inspect) and a Markdown report (no other stage writes prose, not machine-checkable).
5. **Splits are report-only; calibration is reported, not applied.** Regime rows (for PMs whose trait carries a multiplier above 1 on that regime) and before or after rows (for PMs with a drift event on that trait) reuse the same estimator over a date filter and never enter a verdict. Extrapolation gets a pooled calibration statistic beside its literature target (anchoring's correlation form is geometric, see Aggregation, and loss aversion's rate is its own scale); rescaling the config marginal is done by hand afterwards. Rejected: dropping splits (half the pilot is drift PMs) and splits in the verdict (half the opportunities, double the noise).
6. **README stays the living doc.** A "Gate 1" section beside "Engine model". Rejected: creating `ARCHITECTURE.md` now (a second description of the same components to keep in sync).
7. **Architecture: estimator registry mirroring the engine's bias registry.** `inputs.py` builds one `PmInputs` per PM, `estimators/<param>.py` each expose `estimate(inputs, split) -> Estimate`, `aggregate.py` and `verdict.py` are pure functions over lists of estimates, `stage.py` wires them. Rejected: one `gates/gate1.py` module (about 700 lines and Gate 2 will want the aggregation code) and a pandas pipeline (repo is numpy and pydantic throughout).
8. **Herding is measured as the agreement rate with a non-neutral street view.** The plan's wording, "recompute the conflict from the public consensus and the public side", cannot work: a follower's public side equals the street side, so a conflict is invisible in public data. Among entries whose street view at entry is non-neutral, the share whose public side matches it rises monotonically with `w`, and the neutral baseline absorbs the natural agreement share. The hidden `conflict` and `followed_street` columns give the direct estimate, used only in a test.
9. **Denominators are cross-checked against the engine's counters.** The loss-side, conflict, entries-after-run and sell-day counts Gate 1 recomputes must equal the engine's run metadata per PM; a mismatch raises. This guards both stages against a drift in definitions.
10. **Drift PMs leave the whole-year sets.** A PM with a drift event on a parameter is neither neutral nor active for that parameter's whole year; it gets before and after rows only.
11. **`insufficient` blocks.** Fewer than `min_pms` neutral or active PMs with a value is a blocking verdict on synthetic seeds: a parameter nobody can measure has not passed.
12. **Real seeds are recognised by config**, membership in `config.market.real.seeds`, the same test the market stage uses.

## Package layout

```
src/pm_traitbench/gates/__init__.py
src/pm_traitbench/gates/gate1/__init__.py
src/pm_traitbench/gates/gate1/inputs.py        PmInputs and its builder
src/pm_traitbench/gates/gate1/splits.py        Split and the date filters
src/pm_traitbench/gates/gate1/estimators/__init__.py   ESTIMATORS registry keyed by param
src/pm_traitbench/gates/gate1/estimators/{exit_deficiency,loss_aversion,disposition,anchoring,extrapolation,herding,overconfidence,conviction}.py
src/pm_traitbench/gates/gate1/n_min.py         the plan's n_min table as constants with derivation
src/pm_traitbench/gates/gate1/aggregate.py     per-group aggregation
src/pm_traitbench/gates/gate1/verdict.py       verdict rules
src/pm_traitbench/gates/gate1/stage.py         GATE1_STAGE
```

Touched: `stages.py` (`Stage.verdict`, called by `run_stage` after metadata), `engine/series.py` (series-from-idea helper), `tables/schema.py`, `tables/specs.py`, `tables/store.py` (`read_run_metadata`), `config.py` (`Gate1Config`), `enums.py` (`Gate1Verdict`, `Gate1Split`, `SeedGroupKind`), `errors.py` (`Gate1Error`), `pipeline.py`, `README.md`.

## Inputs (`inputs.py`)

`PmInputs` is frozen: `pm_id`, `seed`, `asset_class`, `is_real_seed`, `traits` (by param: planted value, active flag, regime multipliers), `drift_dates` (by param: the trait's drift event dates in order, empty if none), `ideas` (with a rebuilt `Series` per idea), `position_days`, `rule_events`, `entry_risk` (by idea: sum of `risk_amount` on the entry date's ledger rows), `sell_dates` (dates with any position-day action in `cut`, `trim`, `exit`), `view` (the seed's `MarketView`), `adapter`, `horizon_days`, `regime_spans`, and `engine_counts` (the PM's run-metadata opportunity dict).

The builder groups the tables by PM once, builds one `MarketView` per seed exactly as the engine stage does, and calls `adapter_for(asset_class, sub_style, horizon_days)`.

**Series from an idea** (`engine/series.py`, new `series_for_idea(idea, adapter) -> Series`): `bullish_sign` depends only on the form (`adapter.series(form, ())` gives it), and each public `Leg` maps back to a `LegRef` with `coeff = weight * sign`, where `sign = side_sign * bullish_sign * leg_bullish * leg_side_sign` inverts the engine's `leg_side` rule. A round-trip test builds legs through each adapter and checks `series_for_idea` returns an equal `Series`.

## Splits (`splits.py`)

`Gate1Split` is an enum in `enums.py`: `all`, `regime_range`, `regime_risk_off`, `regime_risk_on`, `before`, `after`. A split resolves to a set of dates: all horizon dates; the seed's regime spans for that regime; for `before`, dates strictly before the trait's first drift event; for `after`, dates from the first drift event up to but excluding the second one if there is one (an update has one event, so `after` runs to the horizon end; a dormant and revive pair has two, so `after` is the dormant window). Every estimator filters by the date that carries its observation: entry date for idea-level statistics, row date for position-day statistics, `date_fired` for rule events. A PM gets a regime row for a parameter only when that trait's multiplier on the regime is above 1, and before or after rows only when it has a drift event on the trait.

## Estimators

Each module exposes `estimate(inputs: PmInputs, split: Split) -> Estimate(value: float | None, n: int)` and `HIGHER_IS_STRONGER: bool`. `value` is `None` when `n` is 0 or a ratio is undefined. The registry maps the eight `BIAS_PARAMS` names to modules and is checked complete by a test.

| Parameter | Statistic | n | Stronger |
|---|---|---|---|
| exit_deficiency | share of rule events with response `acked_no_action` or `added` among events not `overridden` | non-overridden events | higher |
| loss_aversion_lambda | add rate: position-days with `pnl_state = loss`, `trigger_pending = false`, action not `roll`, and date before the horizon's last day (the engine's loss-side opportunity set), share with action `add` | those position-days | higher |
| disposition_ratio | Odean (1998) PGR over PLR on position-days whose date is a sell date: realised gain = sold row at a gain, paper gain = held row at a gain, likewise for losses; `flat` rows excluded; `None` if either PGR or PLR has a zero denominator or PLR is 0 | sell-day position-days | higher |
| anchoring_rho | among discretionary exits (action `exit` and no rule event for that idea with response `acted` on that date) whose `anchor_level` differs from the target, share whose tracked exit level lies within `anchor_band_k` horizon-vols of `anchor_level`; tracked exit level = `entry_level + bullish_sign * side_sign * pnl_unit`, horizon-vol = `view.sd_h(series, t_exit, horizon_days)` | such exits | higher |
| extrapolation_theta | share of entries where `bullish_sign * side_sign * trailing_move > sd_h`, with `trailing_move = view.trailing_move(series, t_entry, horizon_days)` and `sd_h = view.sd_h(series, t_entry, horizon_days)` | ideas | higher |
| herding_weight | agreement rate: among entries whose public street view at the entry date for the primary instrument is `overweight` or `underweight`, share whose public side is buy under overweight or sell under underweight | non-neutral-street entries | higher |
| overconfidence_coverage | share of ideas with `interval_lo <= realised <= interval_hi`, `realised = view.forward_move(series, t_entry, horizon_days)` (the same truncation near the horizon end the engine used when stating the interval) | ideas | lower |
| conviction_size_miscalibration | one minus Spearman correlation between `entry_risk` and `stated_conviction` over the PM's ideas; `None` when either has no variation | ideas | higher |

Cross-checks in the `all` split, raising `Gate1Error` with the PM id and both counts on a mismatch: loss-side n equals `loss_side_untriggered_days`; extrapolation numerator equals `entries_after_run`; ideas n equals `ideas`; disposition n equals `sell_day_position_days`; the count of hidden `conflict` rows equals `conflict_entries`.

## Aggregation (`aggregate.py`)

Groups: `seed_group` is `synthetic` (every seed in `config.market.seeds` pooled) or one real seed name; breakdown rows use a single synthetic seed name as `seed_group` and never block. Per (seed group, asset class, parameter, split):

- Sets: neutral = trait `active` false, active = true, excluding for the `all` split any PM with a drift event on the parameter; PMs whose value is `None` are dropped and counted in `n_missing`.
- `neutral_mean`, `neutral_sd` (sample standard deviation), `active_mean`.
- `floor = neutral_mean + floor_se * neutral_sd` when higher is stronger, minus otherwise; `active_share_past_floor` = share of active PMs on the strengthening side of the floor.
- `rank_corr`: Spearman correlation of planted value against statistic over neutral plus active PMs. No sign adjustment: every statistic moves the same way as its planted value (coverage falls as planted coverage falls), so a correct recovery is positive for all eight.
- `gap_ok`: `neutral_sd <= gap_fraction * |active_mean - neutral_mean|`.
- `count_p10`, `count_ok`: for a single-seed group (a synthetic seed or a real seed) and the `all` split, the 10th percentile (numpy default interpolation) of the fingerprint's run-metadata count over the cell's PMs against `n_min`; `None` for the pooled group and for parameters without an n_min.
- `calibration`: extrapolation only, Pearson correlation of `bullish_sign * side_sign` against `trailing_move / sd_h` pooled over the group's entries (literature 0.60, Bloomfield and Hales 2002); `None` elsewhere. Anchoring's correlation form is not reported: correlating exit and anchor distances from the target is dominated by the geometry (both scale with the target distance; the probe gives 0.97-1.00 for neutral and active alike).

## Verdict (`verdict.py`)

For each aggregate row: `insufficient` when `n_neutral < min_pms` or `n_active < min_pms`; otherwise `pass` when `gap_ok` and `rank_corr >= min_rank_corr` (a `None` rank correlation, from a constant statistic, fails), else `fail`. `rank_ok` is `rank_corr >= min_rank_corr`. `count_shortfall` is true when `count_ok` is false on the row's own cell or, for a pooled row, on any of its single-seed cells; it is a warning, never a failure. `blocking` is true only for the pooled synthetic group in the `all` split. `Gate1Error` listing every blocking row whose verdict is not `pass` is raised after both tables and the run metadata are written (see Stage).

## Tables

`gate1_pm` (key `pm_id`, `param`, `split`), `split` a `Gate1Split`: `seed`, `asset_class`, `statistic` (float or None), `n`, `planted`, `active`, `drifted`.

`gate1_cells` (key `seed_group`, `asset_class`, `param`, `split`): `seed_group_kind` (`synthetic_pool`, `synthetic_seed`, `real_seed`), `n_neutral`, `n_active`, `n_missing`, `neutral_mean`, `neutral_sd`, `active_mean`, `floor`, `active_share_past_floor`, `rank_corr`, `gap_ok`, `rank_ok`, `count_p10`, `count_ok`, `count_shortfall`, `calibration`, `verdict`, `blocking`. Float fields and `count_ok` are None when undefined. Failure reasons are the boolean columns rather than a list column: every reason is one of a closed set, and the table layer stores no list of scalars.

Neither table has hidden columns: `planted` and `active` are ground truth already held by `traits`, and both tables are gate output, never shown to a system under test.

Run metadata extras: `failed` (list of `asset_class/param` blocking failures), `warnings` (list of `seed/asset_class/param: count_shortfall`), `thresholds` (the `gate1` config section).

## Config: `gate1`

| Knob | Default | Basis |
|---|---|---|
| `anchor_band_k` | 0.1 | design; horizon-vols around the anchor counted as "at the anchor"; at 0.1 neutral PMs sit at 0.14-0.24 on the default run, the dataset plan's "about 0.20" baseline |
| `floor_se` | 2.0 | design; the dataset plan's two-standard-error floor |
| `gap_fraction` | 0.5 | design; neutral sd at most half the gap puts the active mean past the 2-sd floor |
| `min_rank_corr` | 0.5 | design; under no recovery a rank correlation on about 33 PMs has sd about 0.18, so 0.5 is near three standard errors (false pass about 0.3%) |
| `min_pms` | 5 | design; the smallest set with a usable standard deviation |

Validation: `anchor_band_k > 0`, `floor_se > 0`, `0 < gap_fraction <= 1`, `0 <= min_rank_corr < 1`, `min_pms >= 3`.

The n_min table (`n_min.py`) is a constant, not config: exit_deficiency 7, loss_aversion 16, herding 14, anchoring 29 (the ceiling of the formula; the dataset plan's table rounds 28.4 down to 28), keyed to the run-metadata count names (`triggers_fired`, `loss_side_untriggered_days`, `conflict_entries`, `exits`), with the derivation `16 * p0 * (1 - p0) / (p1 - p0)^2` and its p0, p1 pairs in the comment.

## Probe evidence (default run, synthetic pool, whole year)

Every estimator was run on the committed default run before planning. All engine counters matched the recomputed denominators for every PM once the loss-side set excluded roll days and the horizon's last day. Neutral sd over gap (`gapq`, passes at 0.5 or below) and rank correlation (`rho`, passes at 0.5 or above):

| Parameter | equities gapq, rho | rates_credit gapq, rho | commodities gapq, rho | Expected verdict |
|---|---|---|---|---|
| exit_deficiency | 0.16, 0.88 | 0.24, 0.87 | 0.32, 0.96 | pass |
| overconfidence_coverage | 0.20, 0.95 | 0.36, 0.96 | 0.18, 0.66 | pass |
| herding_weight | 0.51, 0.80 | 0.35, 0.70 | 0.39, 0.86 | fail on equities (gap, marginal), pass elsewhere |
| conviction_size_miscalibration | 1.85, 0.51 | 5.35, 0.50 | 0.58, 0.54 | fail (gap) |
| loss_aversion_lambda | 2.47, 0.21 | 3.99, 0.12 | 5.31, 0.10 | fail |
| disposition_ratio | 3.15, 0.23 | 19.7, -0.14 | 1.96, 0.29 | fail |
| anchoring_rho | band rate shows no separation at k 0.1, 0.25 or 0.5 | | | fail |
| extrapolation_theta | 2.25, 0.18 | 25.2, 0.04 | 0.95, 0.21 | fail |

Alternative estimators (hold-or-add rate, discretionary-only Odean ratio, gain share of discretionary exits, exit fraction between anchor and target, target distance against trailing move) fail the same way, so the failures sit in the engine rules, not the estimators. Causes found in the engine code: the forecast never reaches entry direction or target (side follows the own signal, targets follow the stop and the reward-to-risk draw), so extrapolation has no public fingerprint; the add value subtracts lambda times the added loss, so higher lambda makes adding less attractive and the add rate stays flat while cut is rarely chosen at temperature 1; the disposition multiplier sqrt(1.2) acts on a 0.03 base hazard and is swamped by triggered sales; anchored exits are rare beside hazard exits. Gate 1 therefore blocks at default config. Fixing the engine rules is the next sub-project; Gate 1 is the instrument that measures the fix.

## Stage (`gates/gate1/stage.py`)

`GATE1_STAGE = Stage(number=4, name="gate1", reads=(PERSONAS, TRAITS, DRIFT_EVENTS, *ENGINE_TABLES, *MARKET_TABLES), writes=(GATE1_PM, GATE1_CELLS), verdict=raise_on_failures)`. `run`: read the engine's run metadata (`store.read_run_metadata("engine")`, new; raises `Gate1Error` if absent), build inputs for every non-multi-asset PM (PMs listed in the engine's `skipped` are skipped here too), estimate every (PM, parameter, split), aggregate, decide, write both tables, return extras.

The stage runner gains an optional `verdict: Callable[[dict[str, Any]], None] | None` on `Stage`, called by `run_stage` after the run metadata is written. Gate 1's raises `Gate1Error` when the extras' `failed` list is non-empty, so a blocked run leaves its tables and its metadata on disk and the CLI still exits non-zero. Raising inside `run` would skip the metadata write. Runtime is a few seconds on the default population.

## Errors

`Gate1Error(PmTraitbenchError)`, exit code as the other stage errors. Raised for: missing engine run metadata, a PM missing one of the eight bias traits, a denominator mismatch, and blocking verdict failures (after writing). Estimators never raise on thin data.

## Testing

- **Estimators** on hand-built rows: every trigger acked gives 1.0; two gain sales and one paper loss give a known PGR over PLR, PLR of zero gives None; an exit at the anchor is inside the band and one at the target outside; agreement on 3 of 4 non-neutral entries gives 0.75; coverage 2 of 5 gives 0.4; sizes monotone in conviction give 0 and constant conviction gives None; each estimator honours a split filter.
- **Series round trip** through each adapter and form.
- **Aggregate and verdict** on hand-built estimate lists: clean separation passes; active mean within two neutral sds fails with `gap_ok` false; a statistic uncorrelated with the planted value fails with `rank_ok` false; a coverage-like statistic that falls with its planted value passes with positive `rank_corr`; four neutral PMs gives `insufficient`; a real group has `blocking` false; a count shortfall sets `count_shortfall` without failing; drifted PMs leave the whole-year sets; the floor sits below the neutral mean for coverage.
- **Registry** covers `BIAS_PARAMS` exactly.
- **Integration** on the engine's test fixture market and PMs (engine stage then gate1 stage into a temporary store): every denominator cross-check passes, both tables are written, the run metadata carries `failed`, `warnings` and `thresholds`, and a forced rerun gives byte-identical tables. `data/` is not tracked, so no test reads the default run; the probe evidence above is re-checked by running the stage on the default run once at the end of execution.
- **End to end** through the CLI on the demo config: `gate1` after `engine` exits 1 (one PM per cell gives `insufficient`) with both tables and `run_metadata/gate1.json` on disk; the help text lists `gate1`.
- **Stage and CLI**: missing engine tables gives the stage I/O error; `min_rank_corr` 1.0 is rejected by config validation; a blocking failure exits non-zero with both tables and the run metadata on disk; the runner calls `verdict` only after metadata is written.

## Limitations

- **Pooled baseline hides seed effects.** A regime order that weakens one fingerprint on one seed is averaged in. Cost accepted: per-seed rows are still written, and the alternative is a baseline of 1-11 PMs.
- **Herding is measured as agreement, not as following.** Agreement mixes natural concordance with following; the neutral baseline absorbs the first. Cost: a lower gap than the direct estimator would show. Accepted because the direct estimator needs a hidden column.
- **Disposition counts triggered sales as sales.** A stop honoured at a loss raises PLR for every PM. Cost: the neutral ratio sits below 1 rather than at 1. Accepted because Odean counts every sale and the baseline is measured, not assumed.
- **Anchoring uses a band rate, not the correlation, for the verdict.** The correlation form needs about 87 exits per PM. The correlation is reported pooled per group for calibration only.
- **Thresholds are design choices** probed on one default run; a later population change may need them re-probed. The report carries the thresholds used.
- **Active share is not gated.** An estimator that separates the means but leaves most active PMs inside the neutral band passes. Accepted because the share is a property of the active marginal; it is reported for re-centring.
- **Gate 1 blocks at default config** on five parameters (probe evidence above). The pipeline stops at stage 4 until the engine rules are strengthened in a separate change.
- **Splits are not gated.** A drift PM whose behaviour did not change is not caught here; Gate 2 and the validator see it later.
- **Multi-asset PMs are absent**, as in the engine.

## Living docs impact

`README.md`: usage gains the `gate1` subcommand line; a new "Gate 1" section after "Engine model" describes the estimators table, the pooled neutral baseline, the verdict rules and knobs, the two tables, and that real seeds never block; "Limitations" gains the points above.

`docs/pm-dataset-plan.md` (committed with the docs): the stage 4 paragraph and the section 3.1 herding and anchoring cells gain the decisions above (pooled synthetic baseline per asset class; pass on the half-gap and rank-correlation tests with active share reported; herding measured as agreement with a non-neutral street view because a conflict is invisible in public data; opportunity shortfalls are warnings; the default-config verdict pattern from the probe).

## Corrections during execution

- **Count shortfall uses the estimator's own opportunity count.** `count_p10` is the 10th percentile of each cell member's full-horizon `Estimate.n`, not the engine run-metadata counter: the engine's `exits` and `conflict_entries` are not what the anchoring and herding estimators divide by (anchoring's estimator sees about 5 exits at p10 against `exits` p10 34). `N_MIN` is keyed by param only. Herding's minimum was derived for the follow rate and is applied to the agreement statistic's count as an approximation. Decided by the user after the final review.
- **Split cells get a neutral baseline.** Regime multipliers and drift events exist only on active traits, so as first specified every split cell had no neutral PM. Neutral PMs are now estimated over every regime's dates and form the regime cells' neutral set; before and after cells compare the drifted PMs with the neutral PMs' full-horizon rows, since drift windows differ per PM. A drifted parameter gets before and after rows only, no regime rows. Decided by the user after the final review.
- **The gap test also requires the active mean on the strengthening side** of the neutral mean, so a large gap in the wrong direction fails.
- **Conviction's `n`** counts the ideas with a stated conviction (an idea without an entry-date ledger row is skipped).
- **A market seed may not be named `synthetic`**, the pooled group's `seed_group` value.
- **Verdict stability over 99 root seeds** of the default config: exit deficiency passed on 93-99% of roots per asset class and overconfidence on 93-99%; herding sits at the threshold (median neutral sd 0.48-0.51 of the gap) and passed on about half; loss aversion, disposition, anchoring and extrapolation failed on nearly every root and conviction passed on at most 5%. One of 100 roots failed earlier, in the market stage's correlation check.
