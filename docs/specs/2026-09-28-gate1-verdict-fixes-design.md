**Tier:** light
**Escalation threshold:** 11 files
**Supersedes:** docs/specs/2026-09-25-gate1-design.md (the rank correlation test now runs within active PMs only, and a missing blocking row fails the gate); docs/specs/2026-09-25-engine-bias-fixes-design.md (its pass evidence was read with the combined rank correlation and is no longer the gate's measure)

# Design: Gate 1 verdict fixes

Date: 2026-09-28. Source: a statistical review of the Gate 1 tests as built. Two findings change what the gate measures; everything else the review raised is deferred (see Deferred).

## Scope

Two fixes to the Gate 1 verdict, no estimator or engine change:

1. **A missing blocking row fails the gate.** Today the only blocking rows are the synthetic pool's cross-class rows, one per non-report-only parameter. The pilot's PMs sit on the real seed R1, whose rows are all `insufficient` and never block, so the gate only decides anything because stage 1 also samples the synthetic full split. With `population.full_per_cell: 0` the failed list is empty and Gate 1 exits 0 having tested nothing.
2. **The rank correlation test runs within active PMs only.** Today Spearman runs over neutral and active PMs together. Planted values are bimodal (neutral marginal versus active marginal), so any statistic that separates the two groups scores a high rank correlation regardless of whether it orders PMs inside the active group. In simulation at 65 neutral and 35 active PMs with no information inside the active group, the combined rank at or above 0.5 passes about 90% of the time at the smallest gap the gap test allows, so the rank criterion adds almost nothing to the gap test. On the demo pool loss aversion has combined rho 0.80 and active-only rho 0.39.

Out of scope: the population z test's dependence on n (Welch t or a minimum effect size), centring per-class neutral means before pooling, held-out root seeds, the `n_min` reliability formula, exit deficiency's `added` overlap with loss aversion, before/after split noise. Each is a later change if the pilot shows it matters.

## Decisions

1. **Missing row is a fail named `all/<param>: missing`.** `blocking_failures` takes the `Gate1Config` and, after listing the blocking rows that did not pass, adds an entry for every parameter in `BIAS_PARAMS` that is not in `report_only_params` and has no blocking row. Rejected: raising a `Gate1Error` in `run` before writing (the tables are still useful, and the verdict hook is the one place a gate decides); making single-seed or real-seed rows blocking (the pilot's cells are 1-4 active PMs and would only ever be `insufficient`).
2. **Two rank columns, one gated.** `Gate1CellRow` gains `active_rank_corr` (Spearman between planted and recovered values over the cell's active PMs with a value; null below 3 such PMs or when either side is constant, as `rank_corr` is today). `rank_ok` reads `active_rank_corr`. `rank_corr` (combined) stays as a report column so the old and new measures sit side by side. The population rule drops its rank condition and rests on `pop_z` alone: that rule exists for parameters whose per-PM estimates are too noisy to order, so asking for a positive within-active correlation would fail the blocking conviction row on a coin flip, and the old combined-sign check was trivially met and guarded nothing. Rejected: replacing `rank_corr` in place (loses the comparison that shows why the change was needed); reporting only; keeping the combined sign check on the population rule (a check that cannot fail is noise in the row).
3. **`min_rank_corr` falls to 0.4.** The null standard error of a Spearman correlation over about 30 active PMs is about 0.18, so 0.4 is about two standard errors; the old 0.5 was set for a combined sample of about 33 and a different quantity. The threshold note in `Gate1Config` says so.
4. **A parameter that now fails is left failing.** Loss aversion may fail on the default population (active-only rho 0.39 on the demo pool, commodities only). That is the gate doing its job: the finding goes to an engine sub-project, and this change does not retune thresholds to keep the gate green. The plan's 12-root passage (section 9, Gate 1 paragraph) was read with the combined rank correlation; it is marked as such, and the rerun on the default population that would replace it is a separate step, not part of this change.
5. **Plan text stops claiming Gate 1 runs on the pilot PMs.** Section 6 says the 16 static pilot PMs are what Gate 1 measures on; they are not, since every real-seed row is report-only. Sections 6 and 9 say Gate 1 decides on the synthetic full population's engine output, which needs no model call, while the pilot's real-seed rows are reported.

## Implementation notes

Files to modify:

- `src/pm_traitbench/gates/gate1/_cell_stats.py`: `CellStats.active_rank_corr`; in `build_cell`, `active_rank_corr = _rank_corr([e.planted for e in active], [e.estimate.value for e in active])` next to the combined call.
- `src/pm_traitbench/gates/gate1/verdict.py`: `rank_ok` and `pop_ok` use `stats.active_rank_corr`; `blocking_failures(rows, knobs)` appends `all/<param>: missing` for every `param in BIAS_PARAMS` not in `knobs.report_only_params` without a row where `row.blocking` is true; result stays sorted.
- `src/pm_traitbench/gates/gate1/stage.py`: pass `config.gate1` to `blocking_failures`.
- `src/pm_traitbench/tables/schema.py`: `Gate1CellRow.active_rank_corr: float | None` after `rank_corr`, description "Rank correlation between planted strength and the recovered statistic over active PMs only; the value the rank check uses."; `rank_corr`'s description gains "over neutral and active PMs together; report-only".
- `src/pm_traitbench/config.py`: `Gate1Config.min_rank_corr` default 0.4, note "about two null standard errors of a Spearman correlation over about 30 active PMs".
- `README.md`: the Gate 1 paragraph's per-PM rule names the active-only rank correlation and the 0.4 default, and the gate's exit rule adds the missing-row case.
- `docs/pm-dataset-plan.md`: sections 6 and 9 per decisions 4 and 5.

Tests:

- `tests/gates/test_verdict.py`: copy `test_rank_corr_below_threshold_fails` for `active_rank_corr` (combined high, active low fails; combined low, active high passes); copy `test_blocking_failures_lists_only_blocking_non_pass_rows_sorted` for the missing-row case (rows for four of five non-report-only params give one `: missing` entry; a report-only param without a row gives none; an empty row list gives five). Command: `uv run pytest tests/gates/test_verdict.py -q`.
- `tests/gates/test_aggregate.py`: copy `test_rank_corr_is_positive_when_coverage_moves_with_its_planted_value` for `active_rank_corr` (needs three active PMs; a cell with two active PMs gives null). Command: `uv run pytest tests/gates/test_aggregate.py -q`.
- `tests/gates/test_stage.py`: `test_gate1_stage_writes_both_tables_and_raises_on_the_insufficient_default_thresholds` already covers the raise path; add an assertion that a config with `full_per_cell: 0` (pilot only) raises `Gate1Error` naming `: missing` rows. Command: `uv run pytest tests/gates/test_stage.py -q`.
- Full suite: `uv run pytest -q`.

No Gate 1 rerun on the default population is part of this change (decision 4).

## Living docs impact

- `README.md`, "Gate 1" section: per-PM rule reads "the planted-versus-recovered rank correlation over active PMs clears `min_rank_corr` (default 0.4); the combined correlation is reported as `rank_corr`"; the exit rule reads "exits 1 when a blocking row fails or a non-report-only parameter has no blocking row".
- `docs/pm-dataset-plan.md` section 6: "The 16 static PMs, 4 per asset class, are what Gates 1 and 2 measure recovery on" becomes: Gate 2 measures on them; Gate 1 decides on the synthetic full population's engine output, since every real-seed row is report-only, and reports the pilot's rows. Section 9 Gate 1 paragraph: the per-PM test's rank correlation runs within active PMs at 0.4; a non-report-only parameter with no blocking row fails; the 12-root passage gains one sentence saying its rank correlations are the combined measure and that the default-population result under the active-only measure is not yet recorded.

## Limitations

- **Loss aversion may block the pipeline** until an engine change raises its within-active recovery. Accepted: a gate that passes a parameter it cannot recover per PM is worse than a blocked pipeline.
- **The 0.4 threshold is a normal approximation** at about 30 active PMs; at the pilot's synthetic full split of about 100 non-drifted PMs the active set per parameter is 29-43, so it holds, but a smaller population makes the null wider. Accepted: `min_pms` still marks 4 or fewer as `insufficient`, and an exact test is deferred.
- **Deferred findings** (population z bar falling with n; class offsets inflating the neutral sd; in-sample root evidence; `n_min` reliability; exit deficiency's `added` overlap) stay as built. Accepted: none changes what a passing verdict means the way the two fixed here did.
