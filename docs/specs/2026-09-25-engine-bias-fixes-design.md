**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** docs/specs/2026-09-24-engine-design.md (the extrapolation, loss-aversion, conviction and anchoring rules and the sizing ladder); docs/specs/2026-09-25-gate1-design.md (the anchoring and conviction estimators, the single per-PM pass rule, `anchor_band_k`)

# Design: engine bias-rule fixes and population tests in Gate 1

Date: 2026-09-25. Source of requirements: the Gate 1 results on the default run and the investigation that followed (recorded below), `docs/pm-dataset-plan.md` sections 3.1 and 9 (stage 4). Builds on the engine and Gate 1 specs named above.

## Scope

Fifth sub-project. Strengthen the engine rules whose planted values leave no trace in the ledger (extrapolation, loss aversion, conviction-size miscalibration, anchoring), and change Gate 1 so that the parameters limited by how many decisions one PM makes in a year are judged across PMs rather than per PM. Record the per-PM failures before the change so they are not lost.

In scope: four engine rule changes and one sizing change; two Gate 1 estimator changes; a population test in Gate 1 with a per-parameter test kind and a report-only list; config changes in `engine` and `gate1`; README and dataset-plan updates; a multi-root verification sweep.

Out of scope: disposition's engine rule (unchanged; its planted centre and sell hazard are untouched); scaling the population (the decision is deferred until the full population size is set); herding's market coupling (the street view stays a trend average); multi-asset PMs.

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, a task number or a coordinator. Docstrings state the rule itself. Citing a published paper is allowed.

## Baseline: per-PM results before this change

Recorded so that the per-PM failures stay on file after the population tests replace some verdicts. Default run (root 20260105), pooled synthetic seeds, full horizon. `gapq` is the neutral standard deviation over the active-neutral gap (passes at 0.5 or below); `rho` is the planted-to-recovered Spearman correlation (passes at 0.5 or above).

| Parameter | equities gapq, rho | rates_credit gapq, rho | commodities gapq, rho | Verdict |
|---|---|---|---|---|
| exit_deficiency | 0.16, 0.88 | 0.24, 0.87 | 0.32, 0.96 | pass |
| overconfidence_coverage | 0.20, 0.95 | 0.36, 0.96 | 0.18, 0.66 | pass |
| herding_weight | 0.51, 0.80 | 0.35, 0.70 | 0.39, 0.86 | fail equities, pass others |
| conviction_size_miscalibration | 1.85, 0.51 | 5.35, 0.50 | 0.58, 0.54 | fail |
| loss_aversion_lambda | 2.47, 0.21 | 3.99, 0.12 | 5.31, 0.10 | fail |
| disposition_ratio | 3.15, 0.23 | 19.7, -0.14 | 1.96, 0.29 | fail |
| anchoring_rho | 151.8, 0.06 | 4.18, 0.08 | 3.74, 0.10 | fail |
| extrapolation_theta | 2.25, 0.18 | 25.2, 0.04 | 0.95, 0.21 | fail |

Over 99 root seeds of the default config: exit deficiency and overconfidence passed on 93-99% of roots per asset class; herding passed on about half (median neutral sd 0.48-0.51 of the gap); conviction passed on at most 5%; loss aversion, disposition, anchoring and extrapolation failed on nearly every root.

## Investigation findings

Each cause was confirmed by changing the engine in a scratch copy and rerunning the engine and Gate 1 on the default data, then on 12 further root seeds.

1. **Extrapolation.** The forecast (the theta blend of thesis move and trailing move) never reaches the ledger: entry is gated on the own signal, the side follows the own signal, and the target follows the stop distance times a random reward-to-risk draw.
2. **Loss aversion.** With a linear prospect value function, lambda shifts the cut, hold or add choice only in proportion to the loss, and the median loss-side day has a loss of 0.3 horizon-sd against a PM forecast of about 1.1 sd, so the choice barely moves. The add value also subtracted lambda times the added loss, offsetting the effect. Adds were further blocked by the mandate cap (action `none`) and by the no-add-before-trigger rule, which 70% of PMs hold and which admits an add only at the exit-deficiency breach rate.
3. **Conviction.** The estimator summed risk over an idea's legs, double-counting pairs and calendar spreads (each leg books the full risk). The sizing ladder `min(cap, cap * step * factor)` clipped the top steps for overconfident PMs (factor up to about 2.5), tying their sizes. The blend `round((1 - m) * conviction + m * U(1, 5))` also compresses the planted effect (planted 0.5 recovers as about 0.27).
4. **Anchoring.** The anchors (entry, a range-regime round level, a trailing extreme) move every day, so the blended exit level drifts and is rarely reached; hazard sales and rule exits end most ideas first. The band estimator (within `k` horizon-vols of the anchor) also misses exits that overshoot the anchor by a day's move (about 0.22 horizon-vol).
5. **Disposition.** Not a rule defect: a PM makes about 30 discretionary sales a year, and the Odean ratio from that many has a log standard error near 0.37 against a 0.18 log gap between 1.0 and 1.2.
6. **Herding.** After the extrapolation fix, trend-following PMs agree with the trend-built street view whatever their herding weight: a neutral herder's agreement rate has rank correlation 0.72 with its planted extrapolation theta, and more data does not shrink that spread.
7. **More data per PM does not rescue herding, anchoring or disposition.** A two-year horizon (median 127 ideas per PM) left their neutral spread nearly unchanged; the spread comes from cross-bias contamination, attenuation and the neutral marginals, not sampling.

## Decisions taken

1. **Extrapolation: entry and side follow the normalised forecast.** `fz = ((1 - theta) * own_signal + theta * trail_z) / sqrt((1 - theta)^2 + theta^2)`, with `trail_z = bullish_sign * trailing_move(series, t, horizon_days) / sd_h`. An idea is entered when `abs(fz) >= ENTRY_THRESHOLD` and its side is buy when `fz > 0`. The normalisation keeps the entry rate from falling as theta rises (the blend of two unit normals has sd below 1). Stated conviction still comes from the own signal, and herding resolves a conflict between this side and the street. This is the dataset plan's own wording, "the forecast feeds entry and sizing", which the engine had not implemented. Probe: per-PM pass on all three asset classes on 35 of 36 (root, asset class) cells, population z 3.5-9.2.
2. **Loss aversion: a hazard construction replaces the softmax over prospect values.** On a loss-side day with no trigger fired, the PM cuts with probability `LOSS_CUT_HAZARD / lambda` and, when adding is allowed, adds with probability `min(LOSS_ADD_CAP, LOSS_ADD_SLOPE * max(lambda - 1, 0))`; otherwise it holds. Constants: `LOSS_CUT_HAZARD = 0.05`, `LOSS_ADD_SLOPE = 0.1`, `LOSS_ADD_CAP = 0.5`, so a neutral PM (lambda about 1.1) adds on about 1% of permitted loss days and an active one at the 2.0 centre on about 10%. The basis changes from "definition for the value function" to "spirit, calibrated": the linear value function over a belief-dominated gamble gives lambda no measurable effect at the losses the engine produces. `engine.softmax_tau` is removed. Probe: per-PM pass on 36 of 36 cells.
3. **Conviction: mixture sizing, a size ladder with headroom, and the lead leg's risk.** Size rank is the stated conviction with probability `1 - m` and a uniform draw from 1-5 with probability `m`, so the rank correlation falls about linearly in `m`. Size is `min(cap, cap * RISK_STEPS[rank - 1] * factor / SIZE_HEADROOM)` with `SIZE_HEADROOM = 2.5`, the overconfidence size factor at the active coverage centre 0.4 (`Z_80 / z(0.4)`, about 2.45), so a centre-planted PM's largest step stays under the cap and ranks do not tie. Every position is therefore smaller (at most 40% of the cap for a calibrated PM), which also leaves room for loss-aversion adds. The estimator reads the risk of the idea's lead leg (the ledger row matching `legs[0]` on the entry date) instead of the sum over legs. Probe: per-PM pass on 20 of 36 cells, population z 4.3-4.5 at the median and at least 3 on 75-92% of roots, so conviction moves to the population group (decision 6).
4. **Anchoring: one fixed round-level anchor per idea, taken with probability rho.** At entry the engine sets `anchor_level = round_step(series, entry + ANCHOR_FRACTION * (target - entry))` with `ANCHOR_FRACTION = 0.4`, kept only when strictly between entry and target (else none). With probability rho, drawn once at entry on its own stream, the idea is anchored: its discretionary exit fires on the first day the tracked level reaches the anchor (subject to the minimum holding period, as today). Otherwise the idea's discretionary exit level is its target. The adapters' `anchors` method, `standard_anchors`, the blend formula and `TRAILING_HIGH_DAYS` are removed if nothing else uses them. `position_days.anchor_level` carries the idea's anchor (anchored or not) and `effective_exit_level` the level the PM exits at. Round levels are salient prices (Northcraft and Neale, 1987); the mixture makes rho the share of ideas exited at the anchor, a "spirit, calibrated" construction.
5. **Anchoring estimator: crossing-day rate.** Among an idea's days from entry to exit, find the first day its tracked level (rebuilt as `entry_level + bullish_sign * side_sign * pnl_unit` from `position_days`) reaches `anchor_level`; the idea is an opportunity when such a day exists before the horizon's last day, and a hit when its discretionary exit (action `exit`, no acted rule event that day) is that day. The statistic is hits over opportunities, an estimate of rho plus the small background sell hazard. `gate1.anchor_band_k` is removed. Probe: rank correlation 0.3-0.6, population z median 0.8-2.5.
6. **Gate 1 gains a population test for the count-limited parameters.** `gate1.population_params` defaults to herding, conviction, disposition and anchoring. For those, a cell passes when `pop_z = direction * (active_mean - neutral_mean) / sqrt(neutral_sd^2 / n_neutral + active_sd^2 / n_active) >= gate1.min_pop_z` (default 3.0, about a 0.1% one-sided false pass) and the rank correlation is positive. Per-PM parameters keep the half-gap and rank-correlation rule. `insufficient` still applies below `min_pms`. The cell row gains `test` (`per_pm` or `population`), `pop_z` and `pop_ok`; `pop_z` is computed for every cell, whichever test decides.
7. **Disposition and anchoring are report-only for now.** `gate1.report_only_params` defaults to disposition and anchoring: their pooled synthetic full-horizon rows carry verdicts but `blocking` is false. Probe: disposition population z about 0.5 at the sourced 1.2 centre (1-1.9 even at an emulated 1.5 centre with double the sell hazard); anchoring median z 0.8-2.5, rates_credit worst. Both are limited by the number of PMs per asset class (about 12 active); z grows with the square root of that number, so the list is revisited when the full population size is set.
8. **Herding stays as is** in the engine and the estimator and is population-tested. Probe: population z median 3.4-5.5, at least 3 on 50-83% of roots, so herding can still block on some roots; the trend coupling in the market is left for a later change.
9. **Arrival rate and sell hazard are unchanged** (0.6 and 0.03): a longer or busier PM-year did not help the count-limited parameters, and the 30-80 ideas target stays.

## Config changes

`engine`: remove `softmax_tau`. `gate1`: remove `anchor_band_k`; add `min_pop_z: float = 3.0` (gt 0; basis design; note: about a 0.1% one-sided false pass per test), `population_params: tuple[str, ...]` (default herding_weight, conviction_size_miscalibration, disposition_ratio, anchoring_rho; basis design; note: parameters limited by how many decisions one PM makes a year), `report_only_params: tuple[str, ...]` (default disposition_ratio, anchoring_rho; basis design; note: too few PMs per asset class for the population test at the default population). Validation: both lists are subsets of `BIAS_PARAMS` with no repeats; `report_only_params` may include per-PM parameters.

## Tables

`Gate1CellRow` gains `test: Gate1Test`, `pop_z: float | None`, `pop_ok: bool`. `Gate1Test` is a new StrEnum (`per_pm`, `population`). `Position` gains `anchor_level: float | None` and `anchored: bool`. No other table changes; `position_days.anchor_level` and `effective_exit_level` keep their columns with the new meaning in decision 4.

## Verification

Engine and Gate 1 unit tests for each rule and estimator change; a Gate 1 stage test for report-only rows and population verdicts. Then a 12-root sweep of the default config (roots 20260301-20260312, as in the probe): expected per-PM pass rates are exit deficiency, extrapolation and loss aversion on at least 11 of 12 roots per asset class and overconfidence on at least 10; population passes for conviction on at least 9 of 12 and herding on at least 6. A root whose blocking verdicts all pass is counted; the report lists the rate per (parameter, asset class). Rates below these are reported, not tuned away.

## Limitations

- **Loss aversion's rule is a construction, not the prospect value function.** A reader of the dataset plan's "definition" basis would expect Tversky and Kahneman's value function; the hazard construction plants lambda where it can be read back. Cost accepted because the definitional form leaves no trace at the engine's loss sizes.
- **Disposition and anchoring do not block.** A dataset built at the default population can carry planted disposition and anchoring values that Gate 1 has not shown to be recoverable. Cost accepted until the population size is set; the probe numbers are in the README.
- **Herding stays coupled to extrapolation** through the trend-built street view; its population test can fail on some roots.
- **Smaller positions.** The headroom divisor shrinks every position to at most 40% of the cap for a calibrated PM; P&L is in the adapter's unit per position, so nothing downstream depends on absolute size.
- **Anchored exits respect the minimum holding period,** so crossings inside it attenuate the anchoring statistic.
- **Population tests pool across PMs,** so they certify that a planted bias shifts the population, not that any one PM's value can be read back.

## Living docs impact

`README.md`: the "Engine model" section's extrapolation, loss-aversion, conviction and anchoring rules and the sizing ladder; the "Gate 1" section's estimator table (anchoring, conviction), the pass rules (per-PM and population, report-only list, knobs), the default-config results and the limitations. `docs/pm-dataset-plan.md` (committed with the docs): section 3.1 rows for extrapolation, loss aversion, conviction and anchoring (formula, basis, statistic); the stage 4 paragraph (population tests, report-only list, new default-config results); the n_min anchoring row (opportunity is now an anchor crossing).

## Corrections during execution

- **Gate 1 blocks on one pooled row per parameter, and herding is report-only.** The 12-root sweep after the engine fixes met every per-parameter pass-rate expectation (exit deficiency 36 of 36 cells, extrapolation 35 plus 1 insufficient, loss aversion 35, overconfidence 31 plus 2 insufficient, conviction 32, herding 26), yet no root passed all 18 blocking cells (six blocking parameters times three asset classes), because a gate that needs every one of 18 tests to pass at about 90% each almost never passes. Herding caused 10 of the 20 failures. Decided by the user: herding joins `report_only_params` (default herding_weight, disposition_ratio, anchoring_rho), and the blocking verdict for each parameter is taken on a new pooled row over all direct asset classes of the synthetic seeds (`asset_class` null), with the same per-PM or population rule; per-asset-class rows are still written and judged but never block. Probe on the same 12 roots, raw pooling: every blocking parameter passed on 12 of 12 roots (per-PM gapq median 0.18-0.30, max 0.41; conviction population z median 7.7, min 5.6). Pooling after subtracting each class's neutral mean gave nearly the same numbers, so the simpler raw pooling is used. Cost: a parameter that separates on two asset classes and not on the third can pass; the per-class rows show it.
