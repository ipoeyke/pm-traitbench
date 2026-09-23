**Tier:** heavy
**Escalation threshold:** n/a (heavy)

# Design: foundation and sampling stage

Date: 2026-09-20. Source of requirements: `docs/pm-dataset-plan.md`, sections 1, 1.1-1.3, 8, 9 (stage 1).

## Scope

First of eleven sub-projects of the dataset pipeline. It delivers:

- **Foundation** shared by every later stage: config object, keyed random streams, row schemas, format-agnostic table store, week calendar, stage registry, CLI wiring.
- **Stage 1, sample**: writes `personas`, `traits`, `rules` (PM scope only), `drift_events`.
- **Catalogues**: full-size YAML drafts, human-edited later.

Out of scope: market, engine, idea-scope rules, asset-class-specific biases (deferred to the engine sub-project), signals, LLM client, dialogue, gates, probes, freeze, dataset card, data license.

Remaining sub-projects, each with its own spec: market; behaviour engine; Gate 1; signal plan; LLM client (one class over the Anthropic API and the Ollama API, per-role pinned model ids in config, LLM dependencies as an optional extra); dialogue and validator; Gate 2; probes; freeze.

## Repo rule that binds all code

This spec and the plan are local-only. No file in the repo may mention `docs/`, a spec, or a plan: not code, comments, docstrings, YAML, README, commit messages, or PR text. Docstrings state the rule or ordering itself. Citing a published paper as the basis of a parameter is allowed.

## Decisions taken

1. Tables are typed pydantic row models with a format-agnostic store. Three formats ship now: CSV, JSONL, parquet. No dataframe library in this sub-project.
2. Regime effects are three multiplier columns on `traits`: `mult_range`, `mult_risk_off`, `mult_risk_on`. Regime names follow the market layer (`range`, `risk_off`, `risk_on`); the plan's bear and bull map to `risk_off` and `risk_on`. Multipliers, not absolute values, so a drift update of the base value does not make them stale.
3. Multipliers are drawn per PM per trait. Knob `p_regime_cluster = 0.5` per active bias (guess). Fixed mapping: `loss_aversion_lambda` and `disposition_ratio` cluster in `risk_off`; `extrapolation_theta` and `herding_weight` in `risk_on`; `anchoring_rho` in `range`; `overconfidence_coverage`, `conviction_size_miscalibration`, `exit_deficiency` never cluster. Inactive biases never cluster.
4. Every PM has at least two active biases, by redrawing the activation vector (design choice), because the self-description needs the two strongest.
5. Dependencies added, all core: numpy, scipy, pydantic, pyyaml, pyarrow. Parquet support is not an optional extra. CLI stays on argparse.
6. Sampling modules are not number-prefixed. Order lives once, in `sampling/stage.py`. Pipeline stages are numbered in the CLI registry.
7. Output directory `data/` at repo root, gitignored.
8. A bias `update` event scales the distance from neutral, not the raw value: `to = neutral + f * (from - neutral)`, `f ~ Uniform(0.5, 0.75)`, `neutral` = median of the bias's neutral marginal. Confirmed by the user on 2026-09-20. This departs from the plan's worked example (disposition 1.8 to 1.1), which used raw multiplication.

## Package layout

```
src/pm_traitbench/
  cli.py              # argparse; subcommands built from the stage registry
  stages.py           # Stage dataclass and registry
  config.py           # Config object, YAML load, basis tags
  rng.py              # keyed child streams from one root seed
  calendar.py         # week number <-> date, weekdays
  tables/
    schema.py         # enums and row models
    specs.py          # TableSpec instances: PERSONAS, TRAITS, RULES, DRIFT_EVENTS
    formats.py        # TableFormat protocol, CsvFormat, JsonlFormat, ParquetFormat
    store.py          # DataStore
  catalogues/
    preferences.yaml  rules.yaml  mandates.yaml  self_descriptions.yaml
    models.py         # pydantic models per file
    loader.py         # load, validate, cache
  sampling/
    population.py  mandate.py  biases.py  preferences.py  rules.py  profile.py  drift.py
    stage.py          # runs the seven steps in order, writes four tables
configs/demo.yaml     # 12-PM config for smoke runs and CI
```

Dependency direction: `sampling` depends on `config`, `rng`, `calendar`, `tables`, `catalogues`. None of those depend on `sampling`. Only `sampling/stage.py` touches the store.

## Foundation

### config.py

- One frozen pydantic `Config`, nested: `seed`, `population`, `biases`, `preferences`, `rules`, `drift`, `calendar`, `output`.
- Defaults in code equal the plan's values. An optional YAML file overrides. Unknown keys are rejected (`extra="forbid"`).
- Every leaf field carries `basis` in `{sourced, design, guess}` and a one-line `note` through field metadata. `Config.dump_with_basis()` returns a flat list of `(path, value, basis, note)` for the later freeze stage.
- Validation: probabilities in [0, 1]; ranges ordered; the bias correlation matrix is symmetric, unit diagonal, positive definite.

Default values:

| Path | Value | Basis |
|---|---|---|
| `seed.root` | 20260105 | design |
| `population.asset_classes` | equities, rates_credit, commodities, multi_asset | design |
| `population.market_seeds` | A, B, C | design |
| `population.pilot_per_cell` | 1 | design |
| `population.full_per_cell` | 3 | design |
| `biases.p_active` | 0.35 | guess |
| `biases.min_active` | 2 | design |
| `biases.p_regime_cluster` | 0.5 | guess |
| `biases.regime_multiplier` | LogNormal median 1.15, sigma 0.10, truncated to [1.0, 1.5] | sourced centre, guess spread |
| `biases.marginals` | the neutral and active distributions and floors of plan section 1.1 | per row of that table |
| `biases.correlation` | see below | sourced and guess |
| `mandate.book_size` | LogUniform 50e6 to 2e9 | guess |
| `rules.max_risk_pct_choices` | 5, 8, 10, 15 | guess |
| `rules.n_self_rules` | 3 to 5 | guess |
| `preferences.n_per_pm` | 4 to 8 | guess |
| `drift.bias_update_weeks` | 18 to 30 | design |
| `drift.bias_update_remaining` | Uniform(0.5, 0.75) | sourced centre |
| `drift.p_preference_update` | 0.5 | guess |
| `drift.preference_update_weeks` | 8 to 45 | design |
| `drift.p_dormant_revive` | 0.5 | guess |
| `drift.dormant_weeks`, `drift.revive_weeks` | 32 to 40, 42 to 48 | design |
| `output.format` | `default` (JSONL for nested tables, CSV for flat); or `csv`, `jsonl`, `parquet` for all tables; per-table overrides under `output.tables` | design |
| `calendar.start` | 2026-01-05 | design |
| `calendar.n_weeks` | 52 | design |

Correlation matrix, bias order `loss_aversion_lambda, disposition_ratio, anchoring_rho, extrapolation_theta, herding_weight, overconfidence_coverage, conviction_size_miscalibration, exit_deficiency`. Non-zero pairs: lambda-disposition +0.42; overconfidence-herding -0.31; lambda-herding -0.08; extrapolation-herding +0.05 (all sourced); exit_deficiency-disposition +0.30; conviction_size_miscalibration-overconfidence_coverage -0.30 (both guess). All others 0.

### rng.py

`stream(root_seed: int, *keys: str | int) -> numpy.random.Generator`. Built on `numpy.random.SeedSequence`; string keys are hashed to stable integers with a fixed hash (not Python's salted `hash`). Keys used by sampling: `("pm", index, purpose)` with purpose in `mandate, activation, biases, regime, preferences, rules, profile, drift`. No module-level generator exists. Consequence: adding PMs or adding a draw in one purpose changes no other value.

### calendar.py

`week_start(week) -> date`, `weekdays_in_weeks(first, last) -> list[date]`, both from `calendar.start`. No holidays; the market is synthetic.

### tables/schema.py

Enums: `AssetClass`, `Split(pilot, full)`, `Typicality(typical, anti_typical)`, `Kind(bias, preference)`, `RuleSource(mandate, self)`, `RuleScope(pm, idea)`, `Op`, `Action`, `DriftEventType(update, dormant, revive)`, `Regime(range, risk_off, risk_on)`.

Row models, every field with a `description`:

- `Persona`: `pm_id, market_seed, split, mandate{asset_class, sub_style, book_size, risk_unit, benchmark}, stated_profile{self_description}, typicality`.
- `Trait`: `pm_id, trait_id, kind, param, value: float | str, active, mult_range, mult_risk_off, mult_risk_on`. Validator: bias rows have float `value` and all three multipliers set (default 1.0); preference rows have str `value`, `active` true, multipliers null.
- `Rule`: `pm_id, rule_id, source, scope, trade_idea_id, param, field, op, level: float | str, unit, window, action, text`. This sub-project writes `scope = pm` rows only, `trade_idea_id` null.
- `DriftEvent`: `pm_id, date, event, trait_id, from_value, to_value`, serialised under column names `from` and `to`. `dormant` and `revive` leave both null.

Note: whether a PM has drift is not a persona column. It is derivable from `drift_events`.

### tables/specs.py, formats.py, store.py

- `TableSpec(name, model, key: tuple[str, ...])`. No filename, no format.
- `TableFormat` protocol: `extension`, `write(records: list[dict], model: type[BaseModel], path)`, `read(path, model) -> list[dict]`. The format receives the row model, not just column names, so a typed format can derive its schema. Column order is model field order in every format.
  - `CsvFormat`: `\n` line endings, floats via `repr`, null as empty string, booleans as `true`/`false`. Rejects models with nested fields.
  - `JsonlFormat`: one object per line, keys in model order, compact separators.
  - `ParquetFormat` (pyarrow): Arrow schema derived from the model's field annotations: str, int, float, bool, date, enum as string, optional as nullable, nested model as struct.
- Mixed-type columns (`traits.value`, `rules.level`, `drift_events.from`/`to`, each number or text) are stored as strings in every format, floats via `repr`. Row models coerce on load from context: `Trait` by `kind`, `Rule` by `unit` being null or not, `DriftEvent` by numeric parse. One coercion path serves CSV and parquet.
- Stability promise per format: CSV and JSONL are byte-identical for the same rows. Parquet guarantees identical rows on round trip; bytes are not promised, since the file embeds the writer version.
- `DataStore(data_dir, output_config)` resolves a format per table from `output.format` and `output.tables`. With `default`: JSONL for `personas`, CSV for the other three. Methods `write(spec, rows)`, `read(spec)`, `exists(spec)`, `write_run_metadata(stage_name, config)`. `read` and `exists` locate a table by trying the configured format only; a table present in another format is reported in the error.
- `DataStore.write` validates rows, checks key uniqueness, sorts by key, then serialises to a temp file and renames. `read` validates on load. Format classes only serialise. Same rows give byte-identical files.
- Run metadata: `<data_dir>/run_metadata/<stage>.json` with resolved config, root seed, package version, git commit if available, UTC timestamp. It is the only output that is not byte-stable, and is excluded from identity checks.

### stages.py and cli.py

- `Stage(number, name, run, reads: list[TableSpec], writes: list[TableSpec])` and a registry.
- CLI: `pm-traitbench <stage> --config PATH --data-dir DIR [--force]`. Help lists stages with their numbers. Before running: all `reads` must exist; any existing `writes` without `--force` is an error.
- A stage computes every output in memory, then writes all tables, then run metadata.

## Sampling stage

Each module exposes one pure function `(config, catalogue, rng, earlier results) -> rows or values`.

Runtime order in `stage.py`: population, mandate, biases, preferences, rules, profile, drift. True dependencies: mandate needs population; rules and preferences need mandate; profile needs biases and typicality; drift needs biases and preferences.

1. **population.** Stratified grid. Pilot cells: asset class x market seed x typicality = 24, times `pilot_per_cell`, no drift. Full cells: the same x drift flag = 48, times `full_per_cell` = 144. Ids are assigned in deterministic grid order: pilot first (`pm_001`-`pm_024`), full after. Id width is three digits, widened if the population exceeds 999. `configs/demo.yaml` narrows the grid through config alone: two asset classes (equities, rates_credit), one market seed (A), `pilot_per_cell = 1`, `full_per_cell = 1`, giving 4 pilot and 8 full PMs, so drift is exercised.
2. **mandate.** Sub-style drawn uniformly from the asset class's entries in `mandates.yaml`; `risk_unit` and `benchmark` come from the sub-style; `book_size` LogUniform.
3. **biases.**
   - Activation: Bernoulli(`p_active`) per bias, redrawn as a whole vector until at least `min_active` are active, capped at 1,000 attempts, then an error.
   - Values: `z ~ MVN(0, R)`, `u = Phi(z)`, each `u_i` pushed through the inverse CDF of the active or neutral marginal. Floors and bounds are applied by truncation in quantile space: `u' = F(lo) + u (F(hi) - F(lo))`. No clipping.
   - Regime: for each active bias with a mapped regime, Bernoulli(`p_regime_cluster`); if set, the mapped regime's multiplier is drawn from the truncated LogNormal; all other multipliers are 1.0.
   - Trait ids `t_01`-`t_08` in the fixed bias order.
   - Also returns per-bias strength: the direction-adjusted quantile within the active marginal (`1 - q` for `overconfidence_coverage`, where lower means stronger).
4. **preferences.** Draw `n` in 4-8. Filter catalogue by asset class. Pick one param per group first, then fill the remainder uniformly without replacement. One value per param, uniform. Trait ids from `t_09`, ordered by catalogue order.
5. **rules.** `r_01` is the mandate cap (`source = mandate`, level from `max_risk_pct_choices`). Self rules: each catalogue param included with its share; then repair: `stop_loss` always present; at least one of `no_add_before_trigger` or `min_holding_period`; total self rules within 3-5, dropping or adding lowest-priority params by a seeded draw. Levels uniform within the asset class's range, rounded per catalogue `round_to`. `text` rendered from a template drawn from the bank.
6. **profile.** Takes the two strongest active biases. Typical PMs draw one phrasing each from the agree cells; anti-typical from the contradict cells. `self_description` joins the two phrasings with `", "`.
7. **drift.** Only PMs flagged drift. Always: one `update` on a uniformly chosen active bias, date uniform over weekdays in weeks 18-30, `to = neutral + f * (from - neutral)` with `f ~ Uniform(0.5, 0.75)`, where `neutral` is the median of that bias's neutral marginal. This moves the value toward no-bias for every parameter, including `overconfidence_coverage`, where the unbiased value is higher. With probability 0.5: one preference `update`, weeks 8-45, `to` drawn from the same param's other values. With probability 0.5: `dormant` in weeks 32-40 and `revive` in weeks 42-48, on a different active bias than the updated one (always possible since at least two are active).

Why distance from neutral (decision 8): raw multiplication by 0.5 sends `loss_aversion_lambda` 2.0 to 1.0, below its neutral 1.1; sends `overconfidence_coverage` 0.40 to 0.20, away from the calibrated 0.80, so the bias worsens; and sends `disposition_ratio` 1.8 to 0.9, a reverse disposition effect. Scaling the distance moves every parameter toward unbiased and never past it. The drop is smaller than under raw multiplication (1.8 becomes 1.4 to 1.6). If Gate 1 later shows the post-drift change is too small to detect, widen the range of `f`; do not return to raw multiplication.

## Catalogues

Shipped inside the package, read through `importlib.resources`. Full drafts: about 30 preference params with 2-4 values each across groups `communication, information, workflow, expression`, with asset-class-specific entries; six PM-scope rule params with level ranges per asset class, inclusion shares, and 2 or more text templates per asset class; two sub-styles per asset class; for each of the eight biases at least 3 agree and 3 contradict phrasings.

Load-time checks, all before sampling:

- Each group has at least one entry applicable to each asset class; each asset class has at least 8 applicable preference params.
- Each preference param has at least 2 values.
- Each rule param has a level range and at least one template for every asset class.
- Each bias has at least 2 agree and 2 contradict phrasings.
- Templates use only known slots (`{level}`, `{unit}`).

## Error handling

- Invalid config or catalogue: error names the file and field; exit code 2.
- Missing inputs, or existing outputs without `--force`: exit code 1 with the table names.
- Row validation failure at write: error names table, key, and field; no file is written.
- Capped rejection loops raise with the PM id and the config values involved.
- No silent repair of data anywhere. The rule-set repair in step 5 is part of the sampling definition, not error recovery.

## Testing

pytest, no network, CI-fast.

- **Foundation:** YAML override and unknown-key rejection; correlation matrix validation; pinned values for `rng.stream`; round trip for all three formats including null, bool, float, date, enum, nested struct (JSONL and parquet), and str-or-float columns; CSV rejects nested models; Arrow schema derivation per field type; duplicate key rejection; atomic write leaves no partial file; rewrite is byte-identical for CSV and JSONL, row-identical for parquet.
- **Samplers**, against a small fixture catalogue: grid counts exact; preferences 4-8 with all groups; rule constraints; trait id layout; multipliers only on mapped active biases and within [1.0, 1.5]; drift dates are weekdays inside their windows; updated value lies between `from` and neutral; dormant and updated biases differ.
- **Statistical**, 5,000 PMs at a fixed seed with loose tolerances: marginal medians; sign and rough size of the four sourced correlations among active-active pairs; mean active count consistent with `p_active` conditioned on at least two.
- **Shipped catalogues:** the load-time checks on the real YAML.
- **End to end:** CLI `sample` with `configs/demo.yaml` into a temp dir; four tables exist and validate; foreign keys resolve (`traits`, `rules`, `drift_events` to `personas`; `drift_events.trait_id` to `traits`); a second run is byte-identical; rerun without `--force` exits 1; the same run with `output.format: parquet` yields row-identical tables.
- **Stability:** raising `full_per_cell` changes no pilot row.

The two existing CLI smoke tests are replaced by the end-to-end test.

## Notes for later sub-projects

- Freeze: field descriptions generate a column reference; `dump_with_basis()` and run metadata feed the dataset README; choose a data license; dataset card.
- Engine: defines the asset-class-specific biases and adds them as new `traits` rows; no column change needed.

## Refinements made during planning (2026-09-21)

- The week helper module is `timeline.py`, not `calendar.py`, to avoid confusion with the stdlib module. Enums live in `enums.py` and are re-exported by `tables/schema.py`, because config needs them too.
- Mixed-type columns: CSV and parquet store them as strings; JSONL keeps native JSON types. On load, `Trait.value` is coerced by `kind`; `Rule.level` and drift `from`/`to` become floats when the text parses as a number. The catalogue check therefore forbids preference values that parse as numbers.
- `Action` gains `hold` for the minimum-holding-period rule.
- Population order puts the replicate loop outermost, so raising `full_per_cell` appends PMs and leaves existing ones unchanged.
- Every sampled float is rounded to 4 decimals. Book size is rounded to the nearest 1e6.
- The stage registry is a plain tuple `pipeline.STAGES`; stages are injected into the CLI for tests.
- Rule-set repair is fully specified: force-add a discipline rule if none, remove random non-mandatory rules above the maximum, add random rules below the minimum.

