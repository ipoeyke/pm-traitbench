# Engine Bias Fixes and Gate 1 Population Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the planted extrapolation, loss-aversion, conviction and anchoring values leave a trace in the engine's ledger, and judge the count-limited parameters in Gate 1 with a population test, with disposition and anchoring report-only.

**Architecture:** Four local engine rule changes (entry gate, loss-side choice, sizing, anchored exit) and one sizing constant; two Gate 1 estimator changes; a per-parameter test kind in Gate 1's aggregation and verdict. No new modules except where a task names one.

**Tech Stack:** Python 3.13, numpy, scipy, pydantic v2, pytest, ruff, uv.

## Global Constraints

- Branch `feat/engine-bias-fixes` (already checked out) in `/Users/ianpoey/code/github/pm-traitbench`. Run every command with `uv run`.
- No file in the repo (code, docstrings, comments, tests, YAML, README, commit messages) may mention `docs/`, a spec, a plan, a task number or a coordinator. Docstrings state the rule itself and its domain reason. Citing a published paper is allowed.
- Never commit anything under `docs/` except in Task 7.
- Commit messages: one Conventional Commits subject line, a blank line, then exactly the trailer line the dispatch names. No body, no other trailer.
- Plain dash `-`, never an em dash. No filler words ("genuinely", "actually", "truly", "really"). Docstrings concise; inline comments 1-2 lines.
- Match surrounding style: frozen dataclasses for engine state, frozen pydantic rows with `Field(description=...)`, `json_schema_extra={"basis": ..., "note": ...}` on every config leaf; `Config.dump_with_basis()` keeps working. Every closed set is a StrEnum.
- Randomness only through the engine's `rng_for(purpose, t, ...)` streams; every new draw type gets its own purpose string that includes `t` and the attempt or idea id, so adding a draw never perturbs another stream.
- Engine purity: `step` and bias rules never mutate inputs; state is replaced with `dataclasses.replace`.
- Gate 1 stays deterministic; estimators never raise on thin data.
- Lint and tests green after every task: `uv run ruff check && uv run ruff format --check && uv run pytest -q`.
- A statistical test or sweep rate that misses its stated expectation is reported with numbers (status BLOCKED), never tuned away.

## Decisions made while planning (binding)

1. **Probe evidence.** A scratch copy with the four engine changes and the two estimator changes was run on the default data and on 12 roots (20260301-20260312). Per-PM pass counts over 36 (root, asset class) cells: exit deficiency 36, loss aversion 36, extrapolation 35 (1 insufficient), overconfidence 31 (2 insufficient), conviction 20. Population z (median; share of roots at z >= 3): herding 3.4-5.5 (50-83%), conviction 4.3-4.5 (75-92%), anchoring 0.8-2.5 (0-25%), disposition 0.4-0.5 (0%).
2. **Constants** (in `src/pm_traitbench/engine/constants.py`, each with a one-line reason comment): `LOSS_CUT_HAZARD = 0.05`, `LOSS_ADD_SLOPE = 0.1`, `LOSS_ADD_CAP = 0.5`, `SIZE_HEADROOM = 2.5`, `ANCHOR_FRACTION = 0.4`. Reasons: a neutral PM (lambda about 1.1) cuts on about 4.5% and adds on about 1% of loss days while an active one at the 2.0 centre adds on about 10%; `SIZE_HEADROOM` is the overconfidence size factor at the active coverage centre 0.4 (`Z_80 / z(0.4)`, about 2.45) so a centre-planted PM's largest step stays under the cap; `ANCHOR_FRACTION` puts the salient round level 40% of the way to the target, where most ideas that do not stop out reach it.
3. **Removed:** `EngineConfig.softmax_tau`, `Gate1Config.anchor_band_k`, the adapters' `anchors` method and `standard_anchors`, and `TRAILING_HIGH_DAYS` if nothing else uses it (grep first; keep anything still used).
4. **Stated conviction** still comes from the own signal (`conviction_bucket(own_signal)`); only entry and side move to the forecast.

---

### Task 1: Extrapolation - entry and side follow the normalised forecast

**Files:**
- Modify: `src/pm_traitbench/engine/ideas.py` (`attempt_entry`, the block after `draw_signal`)
- Modify: `README.md` ("Engine model" section: the extrapolation sentence)
- Test: `tests/engine/test_ideas.py`

**Interfaces:**
- Produces: a module-level function in `ideas.py`, `forecast_z(own_signal: float, trail_z: float, theta: float) -> float`, returning `((1 - theta) * own_signal + theta * trail_z) / sqrt((1 - theta) ** 2 + theta ** 2)`.
- Binding: in `attempt_entry`, `theta = params.value("extrapolation_theta")`, `trail_z = series.bullish_sign * view.trailing_move(series, t, config.engine.horizon_days) / draw.sd_h`, `fz = forecast_z(draw.own_signal, trail_z, theta)`. Return no idea when `abs(fz) < ENTRY_THRESHOLD`; the own side is buy when `fz > 0`, else sell. Everything after (herding, sizing, targets) is unchanged. `signal_sign` in `own_signal.py` is removed if nothing else uses it.

- [ ] **Step 1: Write failing tests**
  - `forecast_z` at theta 0 equals the own signal; at theta 1 equals `trail_z`; at theta 0.5 with own 1 and trail 1 equals `sqrt(2)`; it has unit variance over many normal draws (check `std` of `forecast_z(N, N, 0.5)` within 0.03 of 1 on 20,000 draws, a 4-sd band).
  - `attempt_entry` on the fixture market with a PM whose `extrapolation_theta` is 1.0 (override the trait) enters on the side of the trailing move whenever it enters, and with theta 0 on the side of the own signal (compare against `draw_signal` on the same stream).
  - Existing entry-side tests (`tests/engine/biases/test_entry_side.py`) still pass or are updated only where they asserted the own-signal side for a PM with nonzero theta.
- [ ] **Step 2: Run** `uv run pytest tests/engine -q`. Expected: FAIL with ImportError on `forecast_z`.
- [ ] **Step 3: Implement**; update the README sentence to say entry and side follow the theta blend of own signal and trailing move, normalised to unit variance.
- [ ] **Step 4: Tests green; full suite + lint green.**
- [ ] **Step 5: Commit** `feat: gate engine entries on the extrapolation-blended forecast`

---

### Task 2: Loss aversion - hazard construction

**Files:**
- Modify: `src/pm_traitbench/engine/biases/loss_aversion.py`, `src/pm_traitbench/engine/discretionary.py` (the call), `src/pm_traitbench/engine/constants.py`, `src/pm_traitbench/config.py` (remove `softmax_tau`), `README.md` ("Engine model": the loss-aversion sentence and any `softmax_tau` mention)
- Test: `tests/engine/biases/` (new `test_loss_aversion.py`), `tests/test_config.py`, `tests/engine/test_constants.py`

**Interfaces:**
- Produces: `choose(params: EffectiveParams, rng: np.random.Generator, add_allowed: bool) -> LossSideChoice` (the `pnl_z`, `forecast_remaining_z` and `config` parameters go; the caller stops computing `forecast_remaining_z`). `LossSideChoice` and the flags are unchanged.
- Binding: `lam = params.value("loss_aversion_lambda")`; `p_cut = LOSS_CUT_HAZARD / lam`; `p_add = min(LOSS_ADD_CAP, LOSS_ADD_SLOPE * max(lam - 1.0, 0.0))` when `add_allowed`, else 0; one uniform `u`: cut when `u < p_cut`, add when `u < p_cut + p_add`, else hold. Module docstring: lambda raises the add hazard above 1 and lowers the cut hazard, a calibrated construction because a linear prospect value over the PM's own forecast leaves lambda no measurable effect at typical loss sizes.

- [ ] **Step 1: Write failing tests**
  - Over 20,000 draws at lambda 1.0: add share 0 and cut share within 4 standard errors of 0.05; at lambda 2.0 with adds allowed: add share within 4 SE of 0.1 and cut share within 4 SE of 0.025; `add_allowed=False` gives add share 0; lambda 20 caps the add share at 0.5.
  - Flags: active trait gives `loss_aversion:hold` and `loss_aversion:add`; inactive gives none.
  - Config: `EngineConfig` has no `softmax_tau`; a YAML with `engine: {softmax_tau: 1}` is rejected (extra forbid).
  - Constants: the three loss constants exist with the plan's values.
- [ ] **Step 2: Run** `uv run pytest tests/engine tests/test_config.py -q`. Expected: FAIL (signature mismatch, missing constants).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests green; full suite + lint green.**
- [ ] **Step 5: Commit** `feat: plant loss aversion as add and cut hazards on losing days`

---

### Task 3: Conviction - mixture sizing and size headroom

**Files:**
- Modify: `src/pm_traitbench/engine/biases/conviction.py`, `src/pm_traitbench/engine/ideas.py` (the `size_pct_book` line), `src/pm_traitbench/engine/constants.py`, `README.md` ("Engine model": the conviction and sizing sentences)
- Test: `tests/engine/biases/` (new `test_conviction.py`), `tests/engine/test_ideas.py`

**Interfaces:**
- Produces: `size_rank(conviction, params, rng)` unchanged in signature. Binding: `m = params.value("conviction_size_miscalibration")`; `rank = int(rng.integers(1, 6))` when `rng.uniform() < m`, else `rank = conviction`; the flag is `conviction:mis_sized` when active and `rank != conviction`. Draw the uniform first, then the integer only when used.
- Binding sizing: `size_pct_book = min(cap_level, cap_level * RISK_STEPS[rank - 1] * factor / SIZE_HEADROOM)`.

- [ ] **Step 1: Write failing tests**
  - `size_rank` at m 0 always returns the conviction; at m 1 returns ranks spread over 1-5 (each within 4 SE of 0.2 on 20,000 draws); at m 0.5 returns the conviction on a share within 4 SE of 0.6 (0.5 kept plus 0.5 times 1/5 matching).
  - An entry for a calibrated PM (coverage 0.8, factor 1) at rank 5 has `size_pct_book == cap * 1.0 / 2.5`; a PM with coverage 0.4 at rank 5 stays at or under the cap and ranks 4 and 5 give different sizes.
- [ ] **Step 2: Run** `uv run pytest tests/engine -q`. Expected: FAIL on the new expectations.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests green; full suite + lint green.**
- [ ] **Step 5: Commit** `feat: size ideas by a conviction mixture with headroom under the cap`

---

### Task 4: Anchoring - fixed round-level anchor taken with probability rho

**Files:**
- Modify: `src/pm_traitbench/engine/state.py` (`Position`), `src/pm_traitbench/engine/ideas.py` (entry), `src/pm_traitbench/engine/biases/anchoring.py`, `src/pm_traitbench/engine/discretionary.py`, `src/pm_traitbench/engine/adapters/{base,equities,rates_credit,commodities}.py` (remove `anchors` and `standard_anchors`), `src/pm_traitbench/engine/constants.py`, `README.md` ("Engine model": the anchoring sentence)
- Test: `tests/engine/biases/` (new `test_anchoring.py`), `tests/engine/test_ideas.py`, `tests/engine/test_step.py`, adapter tests that exercised `anchors`

**Interfaces:**
- Produces: `Position.anchor_level: float | None = None`, `Position.anchored: bool = False` (defaults keep existing constructors valid). In `anchoring.py`: `entry_anchor(adapter: Adapter, series: Series, entry_level: float, target_level: float) -> float | None` returning `adapter.round_step(series, entry_level + ANCHOR_FRACTION * (target_level - entry_level))` when strictly between entry and target in the favourable direction, else None; `evaluate(pos: Position, level_now: float) -> AnchoredExit` where `AnchoredExit(anchor_level: float | None, effective_exit_level: float, reached: bool)`: `effective_exit_level` is `pos.anchor_level` when `pos.anchored`, else `pos.target_level`; `reached` is true only when `pos.anchored` and the tracked level has reached the anchor in the favourable direction. `flag(params)` unchanged.
- Binding entry: after the target is set, `anchor = entry_anchor(...)`; `anchored = anchor is not None and rng_for("anchor", t, attempt).uniform() < params.value("anchoring_rho")`; both stored on the new `Position`.
- Binding discretionary: `handle_discretionary` calls `anchoring.evaluate(position, level_now)`; an anchored exit fires when `reached and can_exit`, exactly where the blended exit fired before; the position day records `anchor_level` (the idea's anchor, anchored or not) and `effective_exit_level`.
- Remove `Adapter.anchors`, the three adapter implementations, `standard_anchors` and `TRAILING_HIGH_DAYS` (keep any symbol grep shows still used elsewhere).

- [ ] **Step 1: Write failing tests**
  - `entry_anchor` returns the round level between entry and target for a long price series and for a short rates-outright series (bearish series), and None when the rounded level falls on or beyond the target or on the entry side.
  - `evaluate`: an anchored position reaches at the anchor and not one tick short; an unanchored position never reaches and reports its target as `effective_exit_level`.
  - Entry: with `anchoring_rho` overridden to 1.0 every idea with a valid anchor is anchored; with 0.0 none is.
  - Step: an anchored position whose level crosses the anchor after the minimum holding period exits that day with action `exit` and the anchoring flag when active; its position day records `anchor_level`.
  - The idea-row, ledger and rule-event outputs of a fixture run stay valid (existing step and stage tests pass).
- [ ] **Step 2: Run** `uv run pytest tests/engine -q`. Expected: FAIL (missing fields and functions).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests green; full suite + lint green.**
- [ ] **Step 5: Commit** `feat: anchor exits to one round level per idea taken with probability rho`

---

### Task 5: Gate 1 estimators - conviction lead leg and anchoring crossing day

**Files:**
- Modify: `src/pm_traitbench/gates/gate1/inputs.py` (`entry_risk`), `src/pm_traitbench/gates/gate1/estimators/anchoring.py`, `src/pm_traitbench/config.py` (remove `Gate1Config.anchor_band_k`), `README.md` ("Gate 1" estimator table rows for anchoring and conviction, and the `anchor_band_k` sentences)
- Test: `tests/gates/test_inputs.py`, `tests/gates/estimators/test_anchoring.py`, `tests/test_config.py`

**Interfaces:**
- Binding `entry_risk`: the `risk_amount` of the idea's entry-date ledger row whose `instrument_id` equals `idea.legs[0].instrument_id` and, when `idea.legs[0].tenor` is set, whose `tenor` equals it; the first entry-date row when none matches; 0.0 when the idea has no entry-date row.
- Binding anchoring estimator (`estimate(inputs, days, knobs)`): for each idea with `entry_date in days`, take its position-day rows in date order; the anchor is the rows' `anchor_level` (skip the idea when None); the tracked level on a row is `idea.entry_level + bullish_sign * side_sign * row.pnl_unit` (`bullish_sign` from `inputs.series`); the crossing day is the first row after the entry date whose tracked level reaches the anchor in the direction of the target; the idea is an opportunity when a crossing day exists and is not `inputs.last_date`, and a hit when that row's action is `exit` and `(trade_idea_id, date)` is not in `inputs.acted`. `value` = hits over opportunities, `n` = opportunities. Module docstring: the share of anchor crossings the PM exits on, an estimate of rho plus the background sell hazard; round levels as salient anchors after Northcraft and Neale (1987).

- [ ] **Step 1: Write failing tests**
  - `entry_risk` of a two-leg equities pair equals one leg's risk, not the sum; a rates curve idea reads the leg matching `legs[0]`'s tenor.
  - Anchoring: an idea whose level crosses the anchor and exits that day is a hit; one that crosses and exits later is an opportunity without a hit; one that never crosses is not counted; a crossing on the last horizon date is not counted; an exit on the crossing day with an acted rule event is not a hit; a short bearish-series idea crosses in the target's direction (hand-check one case).
  - Config: `Gate1Config` has no `anchor_band_k`.
- [ ] **Step 2: Run** `uv run pytest tests/gates tests/test_config.py -q`. Expected: FAIL on the new expectations.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests green; full suite + lint green.**
- [ ] **Step 5: Commit** `feat: read conviction from the lead leg and anchoring from crossing days`

---

### Task 6: Gate 1 population test and report-only parameters

**Files:**
- Modify: `src/pm_traitbench/enums.py` (`Gate1Test`), `src/pm_traitbench/tables/schema.py` (`Gate1CellRow`), `src/pm_traitbench/config.py` (`Gate1Config`), `src/pm_traitbench/gates/gate1/_cell_stats.py`, `src/pm_traitbench/gates/gate1/aggregate.py`, `src/pm_traitbench/gates/gate1/verdict.py`, `README.md` ("Gate 1": pass rules, knobs, default-config results, limitations)
- Test: `tests/gates/test_aggregate.py`, `tests/gates/test_verdict.py`, `tests/gates/test_stage.py`, `tests/tables/` round trips, `tests/test_config.py`

**Interfaces:**
- Produces: `Gate1Test(StrEnum)`: `PER_PM = "per_pm"`, `POPULATION = "population"`. `Gate1CellRow` gains `test: Gate1Test`, `pop_z: float | None`, `pop_ok: bool` (described fields). `CellStats` gains `pop_z: float | None`. `Gate1Config` gains `min_pop_z: float = 3.0` (gt 0), `population_params: tuple[str, ...]` (default `("herding_weight", "conviction_size_miscalibration", "disposition_ratio", "anchoring_rho")`), `report_only_params: tuple[str, ...]` (default `("disposition_ratio", "anchoring_rho")`), each with basis `design` and the notes in the spec's config section; validation rejects names outside `BIAS_PARAMS` and repeats.
- Binding `pop_z`: `direction * (active_mean - neutral_mean) / sqrt(neutral_sd ** 2 / n_neutral + active_sd ** 2 / n_active)` with sample sds (`ddof=1`); None when either set has fewer than 2 values or the denominator is 0. `direction` is +1 when higher is stronger, else -1.
- Binding verdict: `test` is `POPULATION` when the param is in `population_params`, else `PER_PM`. `pop_ok = pop_z is not None and pop_z >= min_pop_z and rank_corr is not None and rank_corr > 0`. `insufficient` below `min_pms` as today; otherwise a population cell passes when `pop_ok`, a per-PM cell when `gap_ok and rank_ok`. `blocking` is true only for the pooled synthetic `all` row of a param not in `report_only_params`. `gap_ok`, `rank_ok` and `pop_z` are computed for every cell.

- [ ] **Step 1: Write failing tests**
  - `pop_z` on hand-built estimates equals the formula; None for one active PM; sign follows direction for coverage.
  - A population-test cell with `pop_z` 3.5 and rank correlation 0.2 passes although `gap_ok` is false; with `pop_z` 2.5 it fails; with negative rank correlation it fails.
  - A report-only param's pooled `all` row has `blocking` false and never appears in `blocking_failures`, whatever its verdict.
  - Config defaults and validation (unknown name, repeat).
  - Round trip of the new `Gate1CellRow` fields through jsonl and parquet.
  - Stage test: with report-only params set to all eight, a run whose cells are all `insufficient` does not raise.
- [ ] **Step 2: Run** `uv run pytest tests/gates tests/tables tests/test_config.py -q`. Expected: FAIL (missing enum, fields, config).
- [ ] **Step 3: Implement**; the README states the two pass rules, the three new knobs with defaults, which params are population-tested and which report-only and why (too few PMs per asset class at the default population).
- [ ] **Step 4: Tests green; full suite + lint green.**
- [ ] **Step 5: Commit** `feat: judge count-limited biases with a population test in gate 1`

---

### Task 7: Default run, 12-root sweep, README results and dataset plan

**Files:**
- Modify: `README.md` (default-config results in the "Gate 1" section and the engine limitations), `docs/pm-dataset-plan.md`

- [ ] **Step 1: Rerun the default data**: `uv run pm-traitbench engine --force` then `uv run pm-traitbench gate1 --force`. Record the pooled `all` verdicts, `pop_z` and rank correlation per (parameter, asset class).
- [ ] **Step 2: Run the 12-root sweep** (roots 20260301-20260312, default config otherwise, each in its own data dir with `data/raw` linked for the R1 seed): sample, market, engine, gate1. Report per (parameter, asset class) the pass count over roots. Expected from the probe: exit deficiency, extrapolation and loss aversion pass on at least 11 of 12 roots per asset class, overconfidence on at least 10, conviction on at least 9, herding on at least 6; disposition and anchoring are report-only. Any lower rate: stop and report (BLOCKED) with the table.
- [ ] **Step 3: README**: replace the default-config results and causes with the new results and sweep rates; keep the pre-fix per-PM results as one short paragraph ("before these rule changes ...") with the baseline numbers.
- [ ] **Step 4: Dataset plan**: section 3.1 rows for extrapolation (entry and side follow the normalised forecast; Gate 1 statistic unchanged), loss aversion (hazard construction, basis "spirit, calibrated", statistic add rate), conviction (mixture sizing and headroom; statistic from the lead leg; population test), anchoring (fixed round-level anchor with probability rho; crossing-day statistic; population test, report-only); the stage 4 paragraph (per-PM and population tests, report-only list, new results and sweep rates, the baseline per-PM results before the change); the anchoring n_min row (opportunity is an anchor crossing). Plain dash, no filler words.
- [ ] **Step 5: Commit** `docs: record engine bias fixes and population tests in the dataset plan and readme`
