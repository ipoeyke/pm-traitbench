# Foundation and Sampling Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared pipeline foundation (config, keyed RNG, typed tables, format-agnostic store, stage runner, CLI) and stage 1 `sample`, which writes `personas`, `traits`, `rules`, `drift_events`.

**Architecture:** Pure sampling functions take config, catalogue and a keyed random stream and return typed pydantic rows. One `DataStore` validates, sorts and serialises rows through pluggable CSV, JSONL and parquet formats. Stages are declared objects with `reads` and `writes`; the CLI builds one subcommand per stage.

**Tech Stack:** Python 3.13, uv, numpy, scipy, pydantic v2, pyyaml, pyarrow, argparse, pytest, ruff.

## Global Constraints

- **Local-only documents.** Everything under `docs/` (this plan, the spec) is never staged or committed. This overrides any skill instruction to commit plan or spec. Stage files by explicit path only; never `git add -A`, `git add .`, or `git commit -a`.
- **No pointers.** No file in the repo (code, comments, docstrings, YAML, README), no commit message, and no PR title or body may mention `docs/`, a spec, a plan, a task number, or a section of either. Docstrings state the rule itself. Citing a published paper as the basis of a parameter is allowed.
- **Branch.** All work on branch `feat/foundation-sampling`, created from `main` in Task 1. Open a PR to `main` when all tasks pass; PR body describes the change only.
- **Commits.** One Conventional Commits subject line, no body. Only trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. No session-link trailer.
- **Writing style in code and YAML.** Plain dash, never em dash. No filler or intensifier words ("really", "actually", "truly", "genuinely"). Docstrings concise; inline comments 1-2 lines.
- **Dependencies, all core (no extras):** numpy, scipy, pydantic, pyyaml, pyarrow. Add with `uv add`. No pandas or polars.
- **Determinism.** No module-level random generator, no `random` stdlib, no Python `hash()`. All randomness through `pm_traitbench.rng.stream`.
- **Quality gates for every task:** `uv run pytest`, `uv run ruff check`, `uv run ruff format --check` all pass before commit. Ruff line length 100. Tests use no network.
- **Python:** `requires-python >= 3.13`. Use `enum.StrEnum`, `X | Y` unions, frozen dataclasses.
- **pydantic models:** `ConfigDict(frozen=True, extra="forbid")` unless a task says otherwise.
- **Numeric rounding:** every sampled float written to a table is `round(x, 4)`.

## File Map

```
src/pm_traitbench/
  enums.py             T2   all StrEnum types
  errors.py            T1   exception hierarchy and exit codes
  rng.py               T1   keyed random streams
  timeline.py          T1   week <-> date
  distributions.py     T2   LogNormalSpec, BetaSpec, truncated ppf
  config.py            T2   Config, load_config, dump_with_basis
  tables/__init__.py   T3
  tables/schema.py     T3   enums, row models
  tables/introspect.py T4   ColumnInfo, columns()
  tables/formats.py    T4   CsvFormat, JsonlFormat, ParquetFormat, FORMATS
  tables/specs.py      T5   TableSpec, PERSONAS, TRAITS, RULES, DRIFT_EVENTS
  tables/store.py      T5   DataStore
  stages.py            T6   Stage, run_stage
  pipeline.py          T6   STAGES
  cli.py               T6   (modify)
  catalogues/__init__.py, models.py, loader.py   T7
  catalogues/*.yaml    T8
  sampling/__init__.py, population.py, mandate.py   T9
  sampling/biases.py   T10
  sampling/preferences.py, rules.py   T11
  sampling/traits.py, profile.py, drift.py   T12
  sampling/stage.py    T13
configs/demo.yaml      T13
tests/fixtures/catalogue/*.yaml   T7
```

Name note: the week helper is `timeline.py`, not `calendar.py`, to avoid shadowing confusion with the stdlib `calendar` module.

---

### Task 1: Project setup, errors, keyed RNG, timeline

**Files:**
- Modify: `pyproject.toml` (dependencies via `uv add`), `uv.lock`, `.gitignore` (append `data/` under a `# Pipeline output` comment)
- Create: `src/pm_traitbench/errors.py`, `src/pm_traitbench/rng.py`, `src/pm_traitbench/timeline.py`
- Test: `tests/test_rng.py`, `tests/test_timeline.py`, `tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `errors.py`: `class PmTraitbenchError(Exception)` with class attribute `exit_code: int = 1`. Subclasses: `ConfigError` (exit_code 2), `CatalogueError` (exit_code 2), `TableValidationError` (1), `StageIOError` (1), `SamplingError` (1).
  - `rng.py`: `def stream(root_seed: int, *keys: str | int) -> numpy.random.Generator`.
  - `timeline.py`: `@dataclass(frozen=True) class Timeline` with fields `start: datetime.date`, `n_weeks: int`; methods `week_start(week: int) -> date` and `weekdays_in_weeks(first: int, last: int) -> list[date]`.

- [ ] **Step 0: Branch and dependencies** — `git checkout -b feat/foundation-sampling`; `uv add numpy scipy pydantic pyyaml pyarrow`; append `data/` to `.gitignore`.

- [ ] **Step 1: Write failing tests**
  - `test_rng.py`:
    - same seed and keys give identical first 5 draws from two separate calls (proves statelessness).
    - different key tuples (`("pm", 1, "biases")` vs `("pm", 2, "biases")`, and vs `("pm", 1, "rules")`) give different draws.
    - string key hashing is stable across processes: run a subprocess with a different `PYTHONHASHSEED` that prints the first draw for `stream(7, "pm", 3, "drift")` and compare with the in-process value. Do not hardcode an expected number.
    - negative int key raises `ValueError`.
  - `test_timeline.py` with `Timeline(date(2026, 1, 5), 52)`: `week_start(1) == 2026-01-05`; `week_start(22) == 2026-06-01`; `week_start(0)` and `week_start(53)` raise `ValueError`; `weekdays_in_weeks(1, 1)` is the 5 dates Mon-Fri of week 1; `weekdays_in_weeks(18, 30)` has 65 dates, all `weekday() < 5`, first is `week_start(18)`; `first > last` raises `ValueError`; constructing with a non-Monday `start` raises `ValueError`.
  - `test_errors.py`: each subclass is a `PmTraitbenchError`; exit codes as listed.
  - Red phase: import errors for the missing modules.

- [ ] **Step 2: Run tests, verify they fail** — `uv run pytest tests/test_rng.py tests/test_timeline.py tests/test_errors.py -v`; expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Binding key rule for `rng.stream`:

```python
def _key_to_int(key: str | int) -> int:
    if isinstance(key, int):
        if key < 0:
            raise ValueError("integer keys must be non-negative")
        return key
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def stream(root_seed: int, *keys: str | int) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=root_seed, spawn_key=tuple(_key_to_int(k) for k in keys))
    return np.random.default_rng(ss)
```

  Why: keyed child streams mean adding a PM or adding a draw for one purpose changes no other sampled value. Python's `hash()` is salted per process, so it cannot be used.

- [ ] **Step 4: Run tests, full suite, ruff check, ruff format --check; all green.**
- [ ] **Step 5: Commit** — files: `pyproject.toml uv.lock .gitignore src/pm_traitbench/errors.py src/pm_traitbench/rng.py src/pm_traitbench/timeline.py tests/test_rng.py tests/test_timeline.py tests/test_errors.py`; message `feat: add keyed random streams, timeline and error types`.

---

### Task 2: Distributions and config

**Files:**
- Create: `src/pm_traitbench/enums.py`, `src/pm_traitbench/distributions.py`, `src/pm_traitbench/config.py`
- Test: `tests/test_distributions.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `errors.ConfigError`; `timeline.Timeline`.
- Produces `src/pm_traitbench/enums.py` (config needs `AssetClass` and `Regime`, and row models need the rest, so all enums live in one module with no other imports). All are `StrEnum`; values are the lowercase names unless given:
  - `AssetClass`: `equities, rates_credit, commodities, multi_asset`
  - `Split`: `pilot, full`; `Typicality`: `typical, anti_typical`; `Kind`: `bias, preference`
  - `RuleSource`: `mandate, self` (member name `SELF`); `RuleScope`: `pm, idea`
  - `Op`: `LE="<=", GE=">=", LT="<", GT=">", EQ="==", NE="!="`
  - `Action`: `exit, trim_half, no_add, exclude, cap, target, signpost, hold` (`hold` = stay in a position for a minimum period)
  - `DriftEventType`: `update, dormant, revive`; `Regime`: `range, risk_off, risk_on`
- Produces:
  - `distributions.py`:
    - `class LogNormalSpec(BaseModel)`: `kind: Literal["lognormal"] = "lognormal"`, `median: float` (> 0), `sigma: float` (> 0), `lo: float | None = None`, `hi: float | None = None`.
    - `class BetaSpec(BaseModel)`: `kind: Literal["beta"] = "beta"`, `a: float` (> 0), `b: float` (> 0), `lo`, `hi` as above.
    - Both expose `ppf(u: np.ndarray | float) -> np.ndarray | float`, `cdf(x) -> same`, `median_value() -> float` (median of the untruncated distribution).
    - `Distribution = Annotated[LogNormalSpec | BetaSpec, Field(discriminator="kind")]`.
  - `config.py`:
    - `BIAS_PARAMS: tuple[str, ...]` in this exact order: `loss_aversion_lambda, disposition_ratio, anchoring_rho, extrapolation_theta, herding_weight, overconfidence_coverage, conviction_size_miscalibration, exit_deficiency`.
    - `Basis = Literal["sourced", "design", "guess"]`.
    - `class BiasSpec`: `neutral: Distribution`, `active: Distribution`, `higher_is_stronger: bool`, `cluster_regime: Regime | None`, `basis: Basis`, `note: str`.
    - Section models and `class Config` with fields `seed, population, biases, mandate, rules, preferences, drift, calendar, output` (defaults below).
    - `Config.timeline() -> Timeline`.
    - `Config.dump_with_basis() -> list[BasisRow]` where `BasisRow` is a frozen dataclass `(path: str, value: Any, basis: Basis, note: str)`.
    - `def load_config(path: Path | None) -> Config`.

**Binding defaults** (each leaf field declared with `Field(default, json_schema_extra={"basis": ..., "note": ...})`):

| Path | Default | Basis |
|---|---|---|
| `seed.root` | `20260105` | design |
| `population.asset_classes` | all four `AssetClass` values in enum order | design |
| `population.market_seeds` | `("A", "B", "C")` | design |
| `population.pilot_per_cell` | `1` (>= 0) | design |
| `population.full_per_cell` | `3` (>= 0) | design |
| `biases.p_active` | `0.35` | guess |
| `biases.min_active` | `2` (0..8) | design |
| `biases.max_activation_attempts` | `1000` | design |
| `biases.p_regime_cluster` | `0.5` | guess |
| `biases.regime_multiplier` | `LogNormalSpec(median=1.15, sigma=0.10, lo=1.0, hi=1.5)` | sourced (Guiso, Sapienza and Zingales 2018 for the centre) |
| `biases.params` | dict below | per entry |
| `biases.correlation` | 8x8 below | sourced (Yee and Koh 2026) with two guessed pairs |
| `mandate.book_size_min`, `mandate.book_size_max` | `50e6`, `2e9` | guess |
| `rules.max_risk_pct_choices` | `(5.0, 8.0, 10.0, 15.0)` | guess |
| `rules.n_self_rules_min`, `rules.n_self_rules_max` | `3`, `5` | guess |
| `preferences.n_min`, `preferences.n_max` | `4`, `8` | guess |
| `drift.bias_update_weeks` | `(18, 30)` | design |
| `drift.bias_update_remaining` | `(0.5, 0.75)` | sourced (Feng and Seasholes 2005) |
| `drift.p_preference_update` | `0.5` | guess |
| `drift.preference_update_weeks` | `(8, 45)` | design |
| `drift.p_dormant_revive` | `0.5` | guess |
| `drift.dormant_weeks`, `drift.revive_weeks` | `(32, 40)`, `(42, 48)` | design |
| `calendar.start` | `date(2026, 1, 5)` | design |
| `calendar.n_weeks` | `52` | design |
| `output.format` | `"default"`; allowed `default, csv, jsonl, parquet` | design |
| `output.tables` | `{}`; maps table name to `csv, jsonl, parquet` | design |

`biases.params` (neutral; active; higher_is_stronger; cluster_regime):

| Param | Neutral | Active | Higher stronger | Regime |
|---|---|---|---|---|
| loss_aversion_lambda | LogNormal(1.1, 0.10) | LogNormal(2.0, 0.25, lo=1.5) | true | risk_off |
| disposition_ratio | LogNormal(1.0, 0.08) | LogNormal(1.2, 0.15, lo=1.2) | true | risk_off |
| anchoring_rho | Beta(2, 12) | Beta(9, 12) | true | range |
| extrapolation_theta | Beta(2, 10) | Beta(12, 8) | true | risk_on |
| herding_weight | Beta(2, 10) | Beta(7, 5) | true | risk_on |
| overconfidence_coverage | Beta(16, 4) | Beta(4, 6) | false | none |
| conviction_size_miscalibration | Beta(2, 10) | Beta(5, 5) | true | none |
| exit_deficiency | Beta(1, 15) | Beta(4, 5) | true | none |

Notes for `BiasSpec.note` cite the literature basis briefly (for example "Brown et al. 2024 meta-analysis mean 1.955"); `basis` is `sourced` for the first six, `guess` for the last two.

Correlation, in `BIAS_PARAMS` order, unit diagonal, symmetric, zero except: (lambda, disposition) +0.42; (overconfidence_coverage, herding_weight) -0.31; (lambda, herding) -0.08; (extrapolation, herding) +0.05; (exit_deficiency, disposition) +0.30; (conviction_size_miscalibration, overconfidence_coverage) -0.30.

- [ ] **Step 1: Write failing tests**
  - `test_distributions.py`: `ppf(0.5)` of untruncated specs equals `median_value()`; truncated `ppf` over `u = linspace(0.001, 0.999)` stays within `[lo, hi]` and is strictly increasing; truncated LogNormal with `lo` equal to the median has `ppf(0.0001) >= lo`; `cdf(ppf(u))` round trip for untruncated; `lo >= hi` raises validation error; discriminated union parses `{"kind": "beta", "a": 2, "b": 12}`.
  - `test_config.py`: `Config()` builds and matches a sample of table values; `biases.params` keys equal `BIAS_PARAMS` in order; YAML override of one nested leaf (`biases: {p_active: 0.5}`) keeps sibling defaults; override of one bias entry field (`biases: {params: {herding_weight: {cluster_regime: null}}}`) keeps the other seven; unknown key at top level and nested raises `ConfigError` naming the key; asymmetric, non-unit-diagonal, wrong-shape, and non-positive-definite correlation each raise; probability outside [0, 1] raises; week range with first > last raises; non-Monday `calendar.start` raises; missing file raises `ConfigError`; `load_config(None) == Config()`; `dump_with_basis()` returns one row per leaf, every row has a basis, paths include `biases.p_active` and `biases.params.herding_weight` (a `BiasSpec` is one row, its value the dumped spec), and no leaf is missing (compare against a recursive walk of `model_fields`).
  - Red phase: import errors.

- [ ] **Step 2: Run, verify failure.**

- [ ] **Step 3: Implement.**
  - Truncated ppf, binding: with `F` the untruncated cdf, `f_lo = F(lo)` or 0, `f_hi = F(hi)` or 1, `u2 = f_lo + clip(u, 1e-12, 1 - 1e-12) * (f_hi - f_lo)`, return untruncated `ppf(u2)`. Why: truncation keeps the distribution's shape above a floor; clipping would pile mass on the bound. Use `scipy.stats.lognorm(s=sigma, scale=median)` and `scipy.stats.beta(a, b)`.
  - `load_config`: read YAML (`yaml.safe_load`; empty file is `{}`), deep-merge onto `Config().model_dump(mode="python")` (dicts merge recursively, every other type replaces), then `Config.model_validate`. Wrap `OSError`, `yaml.YAMLError`, `pydantic.ValidationError` in `ConfigError` whose message includes the file path and pydantic's field locations.
  - `dump_with_basis`: walk fields recursively; a model that has its own `basis` field is a leaf; a dict of such models yields one row per key; any other leaf reads `json_schema_extra`. Raise `ConfigError` if a leaf has no basis.
  - Positive-definite check via `numpy.linalg.cholesky`.

- [ ] **Step 4: Run tests, suite, lint; green.**
- [ ] **Step 5: Commit** — explicit paths incl. `src/pm_traitbench/enums.py`; message `feat: add distributions and pipeline config`.

---

### Task 3: Table row models

**Files:**
- Create: `src/pm_traitbench/tables/__init__.py` (empty docstring-only), `src/pm_traitbench/tables/schema.py`
- Test: `tests/tables/test_schema.py`

**Interfaces:**
- Consumes: all enums from `pm_traitbench.enums` (`AssetClass, Split, Typicality, Kind, RuleSource, RuleScope, Op, Action, DriftEventType, Regime`); `schema.py` re-exports them via `__all__` so table code imports from one place.
- Produces, row models (every field has a `description`):
  - `Mandate`: `asset_class: AssetClass, sub_style: str, book_size: float, risk_unit: str, benchmark: str`
  - `StatedProfile`: `self_description: str`
  - `Persona`: `pm_id: str, market_seed: str, split: Split, mandate: Mandate, stated_profile: StatedProfile, typicality: Typicality`
  - `Trait`: `pm_id, trait_id, kind: Kind, param: str, value: float | str, active: bool, mult_range: float | None, mult_risk_off: float | None, mult_risk_on: float | None`
  - `Rule`: `pm_id, rule_id, source: RuleSource, scope: RuleScope, trade_idea_id: str | None, param: str, field: str, op: Op, level: float | str, unit: str | None, window: int (>= 1), action: Action, text: str`
  - `DriftEvent`: `pm_id, date: datetime.date, event: DriftEventType, trait_id, from_value: float | str | None = Field(alias="from"), to_value: float | str | None = Field(alias="to")`; model config adds `populate_by_name=True`.
  - `def to_record(row: BaseModel) -> dict[str, Any]`: `row.model_dump(mode="json", by_alias=True)`.
  - `def multiplier_field(regime: Regime) -> str`: `"mult_" + regime.value`.

**Binding validation rules:**
- `Trait`, before-validator: if `kind == bias` and `value` is a `str`, convert with `float(value)`; if `kind == preference`, `value` stays `str`. After-validator: bias rows require float `value` and all three multipliers non-null and `> 0` (the allowed range is a config matter, not a schema one); preference rows require `str` value, `active is True`, all three multipliers null.
- `Rule.level` and `DriftEvent.from_value`/`to_value`, before-validator: a `str` that parses with `float()` becomes a float; otherwise stays `str`. Why: CSV and parquet store these mixed columns as text, and numeric levels must come back as numbers.
- `Rule`: `scope == pm` requires `trade_idea_id is None`; `scope == idea` requires it set.
- `DriftEvent`: `update` requires both values non-null; `dormant` and `revive` require both null.
- Id patterns: `pm_id` matches `^pm_\d{3,}$`, `trait_id` `^t_\d{2,}$`, `rule_id` `^r_\d{2,}$`.

- [ ] **Step 1: Write failing tests** — one valid construction per model; each binding rule above has a passing and a failing case; `Trait` bias row built from all-string input (as a CSV reader would supply: `"2.6"`, `"true"`, `"1.0"`) validates to typed values; `Rule(level="energy")` stays `str`, `Rule(level="-15")` becomes `-15.0`; `DriftEvent` accepts both `from=` alias input and `from_value=` name input; `to_record` of a `DriftEvent` has keys `from`, `to` and an ISO date string; `to_record` of a `Persona` nests `mandate` as a dict; models are frozen (assignment raises); unknown field raises; every field of every row model has a non-empty `description` (loop over `model_fields`). Red phase: import error.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add table row models`.

---

### Task 4: Column introspection and table formats

**Files:**
- Create: `src/pm_traitbench/tables/introspect.py`, `src/pm_traitbench/tables/formats.py`
- Test: `tests/tables/test_introspect.py`, `tests/tables/test_formats.py`

**Interfaces:**
- Consumes: row models and `to_record` from Task 3; `errors.TableValidationError`.
- Produces:
  - `introspect.py`: `ColumnKind = Literal["str", "int", "float", "bool", "date", "mixed", "struct"]`; `@dataclass(frozen=True) class ColumnInfo: name: str; kind: ColumnKind; nullable: bool; struct: type[BaseModel] | None`; `def columns(model: type[BaseModel]) -> list[ColumnInfo]`. `name` is the alias if set, else the field name; order is model field order. Enums and `str` map to `"str"`; a union containing both `float` and `str` maps to `"mixed"`; `X | None` sets `nullable`; a nested `BaseModel` maps to `"struct"`. Any other annotation raises `TypeError` naming the model and field.
  - `formats.py`:
    ```python
    class TableFormat(Protocol):
        extension: str

        def write(self, records: list[dict[str, Any]], model: type[BaseModel], path: Path) -> None: ...
        def read(self, path: Path, model: type[BaseModel]) -> list[dict[str, Any]]: ...


    FORMATS: dict[str, TableFormat]  # keys "csv", "jsonl", "parquet"
    ```
    Records are the output of `to_record`. `read` returns dicts ready for `model.model_validate`. Formats never validate rows and never sort.

**Binding format rules:**
- **CSV:** extension `csv`. UTF-8, `newline=""` with `lineterminator="\n"`, `csv.QUOTE_MINIMAL`. Header row always written, even for zero records. Cell encoding: `None` -> empty string; `bool` -> `true`/`false`; `float` -> `repr(x)`; `mixed` float -> `repr(x)`, mixed str as is; everything else `str()`. Any `struct` column raises `TableValidationError` ("CSV cannot hold nested table <model>"). Read: every cell is `str`; for `nullable` columns an empty cell becomes `None`; header must equal the model's column names exactly, else `TableValidationError`.
- **JSONL:** extension `jsonl`. One `json.dumps(record, ensure_ascii=False, separators=(",", ":"))` per line, keys in column order, `"\n"` after every line. Native JSON types kept, including mixed columns. Zero records gives an empty file.
- **Parquet:** extension `parquet`. Arrow schema from `columns(model)`: `str` -> `pa.string()`, `int` -> `pa.int64()`, `float` -> `pa.float64()`, `bool` -> `pa.bool_()`, `date` -> `pa.date32()`, `mixed` -> `pa.string()` (floats via `repr`), `struct` -> `pa.struct` built recursively; `nullable` sets field nullability. Dates arrive as ISO strings in records and must be converted to `date` before building the table. Read returns dicts with `date` objects and `mixed` values as strings. Write with `pyarrow.parquet.write_table(table, path, compression="zstd")`. Zero records writes a valid empty table with the schema.
- Stability: for the same records, CSV and JSONL output is byte-identical across calls. Parquet promises equal rows on round trip only, because the file embeds the writer version.

- [ ] **Step 1: Write failing tests**
  - introspect: `columns(DriftEvent)` names are `pm_id, date, event, trait_id, from, to` with kinds `str, date, str, str, mixed, mixed` and the last two nullable; `columns(Persona)` has `mandate` as `struct` with `struct is Mandate`; `columns(Trait)` `value` is `mixed` non-nullable and multipliers are nullable `float`; an unsupported annotation (a local model with a `list[int]` field) raises `TypeError`.
  - formats, parametrised over all three where applicable, using rows built in the test (one bias `Trait`, one preference `Trait` whose value contains a comma and a double quote, a `Rule` with `level="energy"`, a `Rule` with `level=-15.0`, an `update` and a `dormant` `DriftEvent`, a `Persona`): write then read then `model_validate` equals the original rows. CSV with `Persona` raises `TableValidationError`. CSV header mismatch on read raises. CSV file bytes contain no `\r`. CSV and JSONL: two writes give identical bytes. Parquet: file schema has `value` as string, `date` as date32, `mandate` as struct. Zero-record round trip per format returns `[]`. `FORMATS` keys are exactly `csv, jsonl, parquet`.
  - Red phase: import errors.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add csv, jsonl and parquet table formats`.

---

### Task 5: Table specs and data store

**Files:**
- Create: `src/pm_traitbench/tables/specs.py`, `src/pm_traitbench/tables/store.py`
- Test: `tests/tables/test_store.py`

**Interfaces:**
- Consumes: `FORMATS`, `columns`, `to_record`, row models, `config.OutputConfig` (fields `format`, `tables`), `config.Config`, errors.
- Produces:
  - `specs.py`: `@dataclass(frozen=True) class TableSpec: name: str; model: type[BaseModel]; key: tuple[str, ...]` with property `nested: bool` (any `struct` column). Constants: `PERSONAS = TableSpec("personas", Persona, ("pm_id",))`, `TRAITS = TableSpec("traits", Trait, ("pm_id", "trait_id"))`, `RULES = TableSpec("rules", Rule, ("pm_id", "rule_id"))`, `DRIFT_EVENTS = TableSpec("drift_events", DriftEvent, ("pm_id", "date", "trait_id", "event"))`. Key entries are model field names (not aliases).
  - `store.py`:
    ```python
    class DataStore:
        def __init__(self, data_dir: Path, output: OutputConfig) -> None
        def format_name(self, spec: TableSpec) -> str
        def path(self, spec: TableSpec) -> Path
        def exists(self, spec: TableSpec) -> bool
        def write(self, spec: TableSpec, rows: Sequence[BaseModel]) -> Path
        def read(self, spec: TableSpec) -> list[BaseModel]
        def write_run_metadata(self, stage_name: str, config: Config) -> Path
    ```

**Binding rules:**
- Format resolution: `output.tables[spec.name]` if present; else `output.format` if not `"default"`; else `"jsonl"` when `spec.nested`, otherwise `"csv"`. Resolving `csv` for a nested table raises `ConfigError` naming the table.
- `path(spec)` is `data_dir / f"{spec.name}.{extension}"`.
- `write`: every row must be an instance of `spec.model`, else `TableValidationError`. Duplicate key raises `TableValidationError` naming table and key tuple. Rows are sorted by key tuple (field values; enums by value, dates naturally). Serialise to `path.with_name(path.name + ".tmp")`, then `os.replace` onto `path`. On any exception remove the temp file and leave any existing target untouched. Creates `data_dir` if missing.
- `read`: missing file raises `StageIOError` naming the table and expected path; if files for the same table exist in other extensions, the message lists them. Each record goes through `spec.model.model_validate`; failure raises `TableValidationError` naming table, 1-based row number, and pydantic's field errors. Duplicate keys on read also raise.
- `write_run_metadata`: writes `data_dir / "run_metadata" / f"{stage_name}.json`, indent 2, keys `stage`, `created_at` (UTC ISO 8601), `package_version` (`importlib.metadata.version("pm-traitbench")`), `git_commit` (output of `git rev-parse HEAD` run with `cwd` = the package source directory, `None` on any failure), `root_seed`, `config` (`config.model_dump(mode="json")`). This file is the one output that is not byte-stable.

- [ ] **Step 1: Write failing tests** (use `tmp_path`): default resolution gives `personas.jsonl`, `traits.csv`; `output.format="parquet"` gives parquet for all four; per-table override wins over global; `csv` for personas raises `ConfigError`; write returns path and file exists; rows given unsorted come back sorted by key; duplicate key raises and writes nothing; wrong model instance raises; a format write that raises midway (monkeypatch the format's `write` to create the temp file then raise) leaves no `.tmp` file and leaves a pre-existing target's bytes unchanged; read of missing table raises `StageIOError`, and mentions `traits.parquet` when that exists while csv is configured; a hand-corrupted CSV cell (`active` = `maybe`) raises `TableValidationError` mentioning `traits`, the row number and `active`; write twice gives identical bytes for csv and jsonl; round trip equality for all four specs in all legal formats; run metadata file has the listed keys, `root_seed` matches config, and `git_commit` is `None` or a 40-char hex string.
- [ ] **Step 2: Run, verify failure** (import errors).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add table specs and data store`.

---

### Task 6: Stage runner and CLI

**Files:**
- Create: `src/pm_traitbench/stages.py`, `src/pm_traitbench/pipeline.py`
- Modify: `src/pm_traitbench/cli.py` (replace body)
- Delete: `tests/test_cli.py` (its two smoke tests are superseded)
- Test: `tests/test_stages.py`, `tests/test_cli_stages.py`

**Interfaces:**
- Consumes: `Config`, `load_config`, `DataStore`, `TableSpec`, errors.
- Produces:
  - `stages.py`:
    ```python
    @dataclass(frozen=True)
    class Stage:
        number: int
        name: str
        help: str
        run: Callable[[Config, DataStore], None]
        reads: tuple[TableSpec, ...] = ()
        writes: tuple[TableSpec, ...] = ()

    def run_stage(stage: Stage, config: Config, store: DataStore, *, force: bool = False) -> None
    ```
  - `pipeline.py`: `STAGES: tuple[Stage, ...] = ()` (Task 13 adds the sample stage).
  - `cli.py`: `def build_parser(stages: Sequence[Stage]) -> argparse.ArgumentParser`; `def main(argv: list[str] | None = None, stages: Sequence[Stage] | None = None) -> int` (`None` means `pipeline.STAGES`). Entry point in `pyproject.toml` stays `pm_traitbench.cli:main`.

**Binding rules:**
- `run_stage` order: (1) every `reads` table must exist, else `StageIOError` listing the missing table names; (2) if any `writes` table exists and `force` is false, `StageIOError` listing them and mentioning `--force`; (3) call `stage.run(config, store)`; (4) verify every `writes` table now exists, else `StageIOError`; (5) `store.write_run_metadata(stage.name, config)`. A stage whose `run` raises leaves no run metadata.
- CLI: one subparser per stage, in `number` order, subparser help text `f"stage {number}: {help}"`. Each takes `--config PATH` (optional), `--data-dir PATH` (default `data`), `--force`. Top level keeps `--version`. No subcommand: print help, return 0. `PmTraitbenchError` is caught in `main`: print `error: <message>` to stderr and return the error's `exit_code`. Other exceptions propagate. Duplicate stage names or numbers passed to `build_parser` raise `ValueError`.

- [ ] **Step 1: Write failing tests** — build fake stages in tests whose `run` writes a `TRAITS` table of two rows through the store. `test_stages.py`: happy path writes table and metadata; missing read raises and `run` is never called (use a flag); existing output without force raises and file bytes unchanged; with force overwrites; stage that writes nothing raises after run; stage that raises leaves no metadata file. `test_cli_stages.py`: `main([], stages=fakes)` returns 0 and help lists `stage 1:`; `main(["fake", "--data-dir", tmp])` returns 0 and table exists; second run returns 1 and stderr starts with `error:`; with `--force` returns 0; `--config` pointing at a YAML with an unknown key returns 2; `--version` exits 0 (SystemExit); duplicate stage names raise `ValueError`; `main([])` with real `pipeline.STAGES` returns 0.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement; delete `tests/test_cli.py` with `git rm`.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add stage runner and stage subcommands`.

---

### Task 7: Catalogue models, loader, checks, test fixture catalogue

**Files:**
- Create: `src/pm_traitbench/catalogues/__init__.py`, `src/pm_traitbench/catalogues/models.py`, `src/pm_traitbench/catalogues/loader.py`
- Create: `tests/fixtures/catalogue/preferences.yaml`, `rules.yaml`, `mandates.yaml`, `self_descriptions.yaml`
- Create: `tests/conftest.py` (fixture `fixture_catalogue` returning `load_catalogue(Path("tests/fixtures/catalogue"))` resolved relative to the conftest file)
- Test: `tests/catalogues/test_loader.py`

**Interfaces:**
- Consumes: enums, `BIAS_PARAMS`, `CatalogueError`.
- Produces (`models.py`, all frozen, `extra="forbid"`):
  - `PreferenceGroup(StrEnum)`: `communication, information, workflow, expression`
  - `PreferenceEntry`: `param: str`, `group: PreferenceGroup`, `asset_classes: tuple[AssetClass, ...]` (required, non-empty), `values: tuple[str, ...]`
  - `RuleVariant`: `asset_class: AssetClass`, `sub_styles: tuple[str, ...] = ()` (empty means every sub-style), `field: str`, `op: Op`, `unit: str | None`, `level_min: float | None = None`, `level_max: float | None = None`, `round_to: float | None = None`, `level_choices: tuple[float | str, ...] = ()`, `action: Action`, `window: int = 1`, `templates: tuple[str, ...]`. Exactly one of (`level_min` and `level_max` and `round_to` all set) or (`level_choices` non-empty) or (neither, allowed only for the mandate cap, whose level comes from config).
  - `RuleEntry`: `param: str`, `share: float` in [0, 1], `mandatory: bool = False`, `discipline: bool = False`, `variants: tuple[RuleVariant, ...]`
  - `RuleCatalogue`: `mandate_cap: RuleEntry` (param `max_risk_pct`), `entries: tuple[RuleEntry, ...]`
  - `SubStyle`: `name: str`, `risk_unit: str`, `benchmark: str`
  - `Phrasings`: `agree: tuple[str, ...]`, `contradict: tuple[str, ...]`
  - `Catalogue`: `preferences: tuple[PreferenceEntry, ...]`, `rules: RuleCatalogue`, `sub_styles: dict[AssetClass, tuple[SubStyle, ...]]`, `self_descriptions: dict[str, Phrasings]`
  - Methods: `Catalogue.preferences_for(asset_class) -> tuple[PreferenceEntry, ...]` (catalogue order); `RuleEntry.variant_for(asset_class, sub_style) -> RuleVariant` (first variant whose `asset_class` matches and whose `sub_styles` is empty or contains `sub_style`; `CatalogueError` if none).
- Produces (`loader.py`):
  - `def load_catalogue(directory: Path | None = None) -> Catalogue` — `None` loads the packaged YAML via `importlib.resources.files("pm_traitbench.catalogues")`; result for `None` is cached (`functools.cache` on an inner no-arg function).
  - `def check_catalogue(catalogue: Catalogue, asset_classes: Sequence[AssetClass], n_preferences_max: int) -> None`.
  - `def render_template(template: str, level: float | str, unit: str | None) -> str` — slots `{level}` and `{unit}`; float level rendered with `format(level, "g")`; `unit` `None` renders as empty string.

**YAML file shapes** (top-level keys): `preferences.yaml`: `preferences: [ {param, group, asset_classes, values}, ... ]`. `rules.yaml`: `mandate_cap: {...RuleEntry}`, `entries: [...]`. `mandates.yaml`: `sub_styles: {equities: [{name, risk_unit, benchmark}, ...], ...}`. `self_descriptions.yaml`: `self_descriptions: {loss_aversion_lambda: {agree: [...], contradict: [...]}, ...}`.

**Binding checks in `check_catalogue`** (each failure raises `CatalogueError` naming file-level section and offending param):
1. For every asset class in `asset_classes` and every `PreferenceGroup`: at least one applicable entry.
2. For every asset class: applicable preference params >= `n_preferences_max`.
3. Preference `param` names unique; every entry has >= 2 values; values unique within an entry; no value parses with `float()` (why: text values share a stored column with numeric bias values, and a numeric-looking text would be read back as a number).
4. Rule `param` names unique; for every entry (and `mandate_cap`), every asset class, and every sub-style of that asset class: `variant_for` succeeds and the variant has >= 1 template.
5. At least one entry has `mandatory`; at least one has `discipline`.
6. Every template parses with `string.Formatter().parse` and uses only slots `level` and `unit`.
7. `level_min < level_max`; `round_to > 0`.
8. Every asset class in `asset_classes` has >= 1 sub-style; sub-style names unique within an asset class.
9. `self_descriptions` keys equal `set(BIAS_PARAMS)`; each has >= 2 `agree` and >= 2 `contradict`.

**Fixture catalogue** (small, valid for all four asset classes): 9 preference params (at least 2 per group; one `expression` entry limited to `rates_credit`; enough that each asset class has >= 8 applicable), so tests call `check_catalogue(..., n_preferences_max=8)`; all six rule params of Task 8's table with compact variants; two sub-styles per asset class; 2 agree and 2 contradict phrasings per bias.

- [ ] **Step 1: Write failing tests** — fixture catalogue loads and passes checks; `preferences_for` filters by asset class and keeps order; `variant_for` prefers the sub-style-specific variant when listed first and falls back to the generic one; `render_template("stop at {level}{unit}", -15.0, "%")` gives `stop at -15%`, and with `unit=None` no literal `None` appears; one test per check 1-9, each made by copying the fixture YAML into `tmp_path`, breaking one thing, and asserting `CatalogueError` with the param or section name in the message; malformed YAML and a missing file raise `CatalogueError`; unknown YAML key raises.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add catalogue models, loader and checks`.

---

### Task 8: Shipped catalogues

**Files:**
- Create: `src/pm_traitbench/catalogues/preferences.yaml`, `rules.yaml`, `mandates.yaml`, `self_descriptions.yaml`
- Modify: `pyproject.toml` only if the build backend does not already include non-Python package files (verify with the packaging test below; `uv_build` includes files inside the package directory by default)
- Test: `tests/catalogues/test_shipped.py`

**Interfaces:**
- Consumes: Task 7 loader and checks.
- Produces: packaged catalogue loadable by `load_catalogue()`.

**Content requirements** (drafts for later human editing; plain, professional buy-side voice; no real company, fund or person names; no em dash):

*mandates.yaml* — exactly these sub-styles:

| Asset class | name | risk_unit | benchmark |
|---|---|---|---|
| equities | equity_long_short | pct_nav | cash |
| equities | long_only_equity | pct_nav | broad_equity_index |
| rates_credit | sovereign_rates | dv01 | sovereign_index |
| rates_credit | long_short_credit | dv01 | agg |
| commodities | commodity_futures_directional | contracts | commodity_index |
| commodities | curve_and_spread | contracts | cash |
| multi_asset | global_macro | pct_nav_sleeve | cash |
| multi_asset | balanced_allocation | pct_nav_sleeve | sixty_forty |

*rules.yaml* — `mandate_cap`: param `max_risk_pct`, field `size_pct_book`, op `<=`, unit `pct`, action `cap`, one variant per asset class, no level range. Entries:

| param | share | flags | field, op, unit, action | Level ranges |
|---|---|---|---|---|
| stop_loss | 1.0 | mandatory | equities `pnl_from_entry <= pct exit`; sovereign_rates `adverse_yield_move_bp >= bp exit`; long_short_credit `adverse_spread_move_bp >= bp exit`; commodities `pnl_from_entry <= pct exit`; multi_asset `sleeve_drawdown <= pct exit` | equities -20..-10 round 1; sovereign 15..30 round 5; credit 25..50 round 5; commodities -15..-8 round 1; multi_asset -6..-3 round 0.5 |
| trim_at_target | 0.7 | | `target_hit == (unit null) trim_half`, `level_choices: [1]` | n/a |
| no_add_before_trigger | 0.6 | discipline | `triggers_fired == (unit null) no_add`, `level_choices: [0]` | n/a |
| min_holding_period | 0.4 | discipline | `sessions_held >= sessions hold` | equities 5..20 round 5; rates_credit 10..30 round 5; commodities 5..15 round 5; multi_asset uses `level_choices: [20]` (one rebalance cycle; a range needs min < max) |
| exclusion | 0.4 | | equities `sector != (null) exclude` choices: energy, financials, utilities, real_estate; rates_credit `rating_band != exclude` choices: ccc_and_below, single_b, subordinated_financials; commodities `commodity_group != exclude` choices: agriculture, energy, precious_metals; multi_asset `sleeve != exclude` choices: emerging_markets, commodities_sleeve, fx_sleeve | n/a |
| max_positions | 0.5 | | `n_positions <= positions cap` | equities 15..40 round 5; rates_credit 10..25 round 5; commodities 8..20 round 1; multi_asset `level_choices: [4, 5, 6]` |

Each variant has 2-3 templates in that asset class's language, using `{level}` and optionally `{unit}`, for example `stop at {level}% from entry`, `out if the spread moves {level}bp against me`.

*preferences.yaml* — about 30 params, 2-4 values each, every value a short instruction phrase that the copilot could follow and that does not change expected P&L or risk. Groups and required members:
- `communication` (all asset classes): `response_format`, `pushback_style`, `register`, `number_language`, `answer_ordering`, `hedging_language`, `length_on_routine_questions`.
- `information`: `positioning_context`, `carry_display`, `macro_commentary`, `calendar_citation`, `source_attribution`, `scenario_framing`, `peer_comparison` (all asset classes); plus `rating_agency_view` (rates_credit), `inventory_data_mention` (commodities).
- `workflow`: `pre_mortem_before_sizing`, `weekly_exposure_recap`, `end_of_day_summary`, `thesis_reminder_on_review`, `decision_journal_prompt` (all); `roll_reminder` (commodities), `refi_window_reminder` (rates_credit), `rebalance_recap` (multi_asset), `earnings_week_heads_up` (equities).
- `expression`: `hedge_instrument` (equities, rates_credit, multi_asset), `duration_expression` (rates_credit), `futures_vs_etf` (equities, commodities, multi_asset), `curve_trade_expression` (commodities), `pair_vs_outright` (equities), `fx_hedge_expression` (multi_asset), `credit_index_vs_single_name` (rates_credit).
Constraint to verify: every asset class has >= 8 applicable params and >= 1 per group.

*self_descriptions.yaml* — for each of the eight biases, 3 `agree` and 3 `contradict` first-person fragments, lowercase, no trailing period, 3-8 words, that read as something a PM would say about themselves and never name the bias. `agree` admits the behaviour (disposition: `i take profits early`); `contradict` claims the opposite virtue (`i let winners run`). For `overconfidence_coverage`, agree = confident in own forecasts; contradict = humble about ranges.

- [ ] **Step 1: Write failing tests** — `load_catalogue()` (packaged) succeeds; `check_catalogue(cat, list(AssetClass), n_preferences_max=8)` passes; preference param count between 28 and 34; every preference has 2-4 values; sub-style table matches exactly; rule params equal the six listed plus cap; every phrasing is lowercase, has no trailing period, 3-8 words; no YAML string in any of the four files contains an em dash or the banned filler words; packaging test: `importlib.resources.files("pm_traitbench.catalogues").joinpath("rules.yaml").is_file()`. Red phase: `CatalogueError` for missing files.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Write the four YAML files.**
- [ ] **Step 4: Tests, suite, lint green. Also run `uv build` and confirm the wheel lists the four YAML files (`unzip -l dist/*.whl | grep yaml`); then delete `dist/`.**
- [ ] **Step 5: Commit** — `feat: add preference, rule, mandate and self-description catalogues`.

---

### Task 9: Population grid and mandate sampling

**Files:**
- Create: `src/pm_traitbench/sampling/__init__.py`, `src/pm_traitbench/sampling/population.py`, `src/pm_traitbench/sampling/mandate.py`
- Test: `tests/sampling/test_population.py`, `tests/sampling/test_mandate.py`

**Interfaces:**
- Consumes: `Config`, `Catalogue`, `Mandate`, enums, `numpy.random.Generator`.
- Produces:
  - `@dataclass(frozen=True) class PmSlot: index: int; pm_id: str; split: Split; asset_class: AssetClass; market_seed: str; typicality: Typicality; drift: bool`
  - `def build_population(config: Config) -> list[PmSlot]`
  - `def sample_mandate(slot: PmSlot, config: Config, catalogue: Catalogue, rng: Generator) -> Mandate`

**Binding rules:**
- Pilot slots first, then full. Loop nesting, outermost to innermost: pilot = replicate (`range(pilot_per_cell)`), asset class, market seed, typicality (`typical` then `anti_typical`); full = replicate (`range(full_per_cell)`), asset class, market seed, typicality, drift (`False` then `True`). Pilot slots always have `drift=False`. Why replicate outermost: raising a per-cell count appends slots and leaves every existing slot's index unchanged.
- `index` is 1-based over the whole list. `pm_id = f"pm_{index:0{w}d}"` with `w = max(3, len(str(total)))`.
- Note the consequence: `pilot_per_cell` changes shift full indices; only `full_per_cell` growth is index-stable. That is accepted.
- Mandate: `sub_style` uniform over the asset class's sub-styles (`rng.integers`); `risk_unit` and `benchmark` from the sub-style; `book_size = exp(rng.uniform(log(min), log(max)))`, rounded to the nearest 1e6.

- [ ] **Step 1: Write failing tests** — defaults give 24 pilot + 144 full = 168 slots, ids `pm_001`..`pm_168` unique; every (asset class, seed, typicality) pilot cell has exactly 1; every full cell incl. drift has exactly 3; exactly half of full slots have drift; no pilot slot has drift; with `full_per_cell=4` the first 168 slots are equal to the default run's slots; restricted config (2 asset classes, 1 seed, 1 and 1) gives 4 + 8; zero counts give empty lists without error; 1,000+ slots widen ids to 4 digits. Mandate: sub-style belongs to the asset class; `risk_unit`/`benchmark` match the catalogue entry; book size within bounds and a multiple of 1e6; over 2,000 draws both sub-styles of an asset class appear and median book size is within a loose band around the log-uniform median (`sqrt(min*max)`, tolerance 25%).
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add population grid and mandate sampling`.

---

### Task 10: Bias sampling

**Files:**
- Create: `src/pm_traitbench/sampling/biases.py`
- Test: `tests/sampling/test_biases.py`

**Interfaces:**
- Consumes: `Config` (`biases` section), `BIAS_PARAMS`, `Regime`, `SamplingError`, distributions.
- Produces:
  - `@dataclass(frozen=True) class BiasDraw: param: str; value: float; active: bool; strength: float; multipliers: dict[Regime, float]` (`multipliers` always has all three regimes)
  - `def sample_biases(config: Config, rng_activation: Generator, rng_values: Generator, rng_regime: Generator) -> list[BiasDraw]` — 8 draws in `BIAS_PARAMS` order.

**Binding algorithm:**
1. Activation: `active = rng_activation.random(8) < p_active`; repeat the whole vector until `active.sum() >= min_active`; after `max_activation_attempts` failures raise `SamplingError` naming `p_active` and `min_active`.
2. Values: `z = rng_values.multivariate_normal(zeros(8), R, method="cholesky")`; `u = scipy.stats.norm.cdf(z)`; for each param, `dist = spec.active if active[i] else spec.neutral`; `value = round(float(dist.ppf(u[i])), 4)`. Why a Gaussian copula: it imposes the published cross-bias correlations while each bias keeps its own bounded, skewed marginal.
3. Strength: for active biases `q = spec.active.cdf(value)` (untruncated cdf), `strength = q if higher_is_stronger else 1 - q`; inactive biases get `0.0`.
4. Regime: iterate params in order and ALWAYS draw two uniforms `a, b = rng_regime.random(2)` per param (keeps the stream aligned whatever the activation pattern). If the bias is active, has a `cluster_regime`, and `a < p_regime_cluster`: that regime's multiplier is `round(float(regime_multiplier.ppf(b)), 4)`. Every other multiplier is `1.0`.

- [ ] **Step 1: Write failing tests** — returns 8 draws in order; with fixed streams two calls are equal; every PM has >= 2 active over 500 draws; `p_active=0.0, min_active=2` raises `SamplingError`; `min_active=0, p_active=0.0` gives all inactive with all multipliers 1.0 and strength 0.0; active values respect floors (`lambda >= 1.5`, `disposition >= 1.2`) and [0, 1] bounds; multipliers differ from 1.0 only on active biases and only in the mapped regime, and lie in [1.0, 1.5]; the three unmapped params never cluster; with `p_regime_cluster=1.0` every active mapped bias has its mapped multiplier `>= 1.0` and, across 200 draws, at least one is `> 1.0`; with `p_regime_cluster=0.0` every multiplier is 1.0. Statistical, 5,000 draws, streams `stream(1, "t", i, purpose)`, loose tolerances: share of draws with a given bias active lies in 0.36-0.44 (0.35 conditioned on at least two active is about 0.39); median of active `loss_aversion_lambda` in 1.9-2.3 (floor truncation lifts it above 2.0); median of active `anchoring_rho` within 0.05 of `Beta(9, 12)` median; Spearman correlation among PMs where both are active: lambda vs disposition > 0.2, overconfidence_coverage vs herding < -0.1; cluster rate among active mapped biases within 0.45-0.55; strength of `overconfidence_coverage` decreases as its value increases. Do not hardcode sampled values.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green. The statistical test must finish in under 20 seconds; if slower, vectorise inside the test helper, not in the sampler's public contract.**
- [ ] **Step 5: Commit** — `feat: add correlated bias sampling with regime multipliers`.

---

### Task 11: Preference and rule sampling

**Files:**
- Create: `src/pm_traitbench/sampling/preferences.py`, `src/pm_traitbench/sampling/rules.py`
- Test: `tests/sampling/test_preferences.py`, `tests/sampling/test_rules.py`

**Interfaces:**
- Consumes: `Catalogue`, `PreferenceGroup`, `render_template`, `Rule`, `Mandate`, `Config`, enums, `SamplingError`.
- Produces:
  - `@dataclass(frozen=True) class PreferenceDraw: param: str; value: str`
  - `def sample_preferences(asset_class: AssetClass, config: Config, catalogue: Catalogue, rng: Generator) -> list[PreferenceDraw]` — result in catalogue order.
  - `def sample_rules(pm_id: str, mandate: Mandate, config: Config, catalogue: Catalogue, rng: Generator) -> list[Rule]` — `r_01` is the mandate cap, self rules follow in catalogue order with ids `r_02`, `r_03`, ...

**Binding algorithms:**
- Preferences: `n = rng.integers(n_min, n_max + 1)`. Applicable entries = `catalogue.preferences_for(asset_class)`. For each `PreferenceGroup` in enum order pick one applicable entry uniformly. Fill the remaining `n - 4` uniformly without replacement from the unpicked applicable entries. One value per entry, uniform. If applicable entries < `n`, raise `SamplingError`.
- Rules:
  1. Cap: variant via `mandate_cap.variant_for(asset_class, sub_style)`; `level = float(rng.choice(max_risk_pct_choices))`; `source=mandate`.
  2. Inclusion, iterating entries in catalogue order and drawing one uniform each: included if `mandatory` or `u < share`.
  3. If no included entry has `discipline`, add one discipline entry chosen uniformly.
  4. While count > `n_self_rules_max`: remove one entry chosen uniformly among included entries that are not mandatory and are not the only included discipline entry.
  5. While count < `n_self_rules_min`: add one entry chosen uniformly among those not included.
  6. For each included entry in catalogue order: variant for (asset class, sub-style); level = `round_to * round(rng.uniform(level_min, level_max) / round_to)` clipped into `[level_min, level_max]`, or a uniform pick from `level_choices`; template uniform; `text = render_template(...)`. All self rules: `source=self`, `scope=pm`, `trade_idea_id=None`, `window` from the variant.
  Why repair steps exist: every PM needs an exit rule and a discipline rule so that breaches of each are observable later.

- [ ] **Step 1: Write failing tests** (fixture catalogue, loops over many seeds rather than pinned outputs) — preferences: count in 4-8; every group present; params unique; every param applicable to the asset class; values belong to the entry; order follows the catalogue; deterministic for a given stream; over 300 draws every count 4..8 occurs; a catalogue cut below `n` raises `SamplingError`. Rules: first rule is the cap with `source=mandate`, `level` in choices, id `r_01`; self rule count in 3-5; `stop_loss` always present; at least one discipline rule; ids sequential with no gaps; all `scope=pm` and `trade_idea_id is None`; numeric levels inside the variant range and multiples of `round_to`; exclusion level is a `str` from choices; `text` contains no `{`; a `sovereign_rates` PM gets the yield stop variant and a `long_short_credit` PM the spread one; over 2,000 draws the inclusion frequency of `trim_at_target` is above that of `min_holding_period` (shares 0.7 vs 0.4).
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add preference and rule sampling`.

---

### Task 12: Trait assembly, self-description, drift

**Files:**
- Create: `src/pm_traitbench/sampling/traits.py`, `src/pm_traitbench/sampling/profile.py`, `src/pm_traitbench/sampling/drift.py`
- Test: `tests/sampling/test_traits.py`, `tests/sampling/test_profile.py`, `tests/sampling/test_drift.py`

**Interfaces:**
- Consumes: `BiasDraw`, `PreferenceDraw`, `Trait`, `DriftEvent`, `Catalogue`, `Config`, `Timeline`, `Typicality`, `multiplier_field`.
- Produces:
  - `def build_traits(pm_id: str, biases: list[BiasDraw], preferences: list[PreferenceDraw]) -> list[Trait]` — biases `t_01`..`t_08` in given order, preferences from `t_09`; preference rows `active=True`, multipliers `None`.
  - `def sample_self_description(biases: list[BiasDraw], typicality: Typicality, catalogue: Catalogue, rng: Generator) -> str`
  - `def sample_drift(pm_id: str, traits: list[Trait], config: Config, catalogue: Catalogue, timeline: Timeline, rng: Generator) -> list[DriftEvent]` — called only for slots with `drift=True`.

**Binding rules:**
- Self-description: take active biases sorted by `strength` descending, ties broken by `BIAS_PARAMS` order; take the first two. For each, pick one phrasing uniformly from `agree` (typical) or `contradict` (anti-typical). Join with `", "`. Fewer than two active biases raises `SamplingError`.
- Drift, draws in this fixed order so streams stay aligned:
  1. Bias update: pick one active bias trait uniformly; date uniform from `timeline.weekdays_in_weeks(*bias_update_weeks)`; `f = rng.uniform(*bias_update_remaining)`; `neutral = config.biases.params[param].neutral.median_value()`; `to = round(neutral + f * (from - neutral), 4)`. Why distance from neutral: scaling the raw value would push loss aversion below its unbiased level and would move overconfidence coverage away from calibration; scaling the distance moves every bias toward unbiased and never past it.
  2. Draw `u1`; if `u1 < p_preference_update`: pick one preference trait uniformly; date from `preference_update_weeks`; `to` uniform among the catalogue entry's other values; `from` is the current value.
  3. Draw `u2`; if `u2 < p_dormant_revive`: pick one active bias trait uniformly from those other than the updated one; `dormant` date from `dormant_weeks`; `revive` date from `revive_weeks`; both with null values.
  Always draw `u1` and `u2`, even when the branch cannot apply (no preference traits, or only one active bias); in that case skip the branch.
- Returned list sorted by `(date, trait_id, event)`.

- [ ] **Step 1: Write failing tests** — traits: ids and order, bias multipliers copied from `BiasDraw.multipliers` into the right columns, preference rows valid. Profile: hand-built `BiasDraw` lists; typical draws only from `agree` cells of the two strongest, anti-typical only from `contradict`; tie broken by param order; inactive biases ignored even with high strength; one active bias raises. Drift (hand-built traits plus fixture catalogue, loop over 300 seeds): exactly one bias update always; its date is a weekday within weeks 18-30; `to` lies strictly between `from` and neutral for every one of the eight params (build a case per param, including `overconfidence_coverage` where `to > from`); `abs(to - neutral) / abs(from - neutral)` within [0.5, 0.75] up to rounding; preference update present in roughly half of seeds (0.4-0.6), `to != from`, `to` belongs to the same catalogue entry, date within weeks 8-45; dormant and revive appear together or not at all, on the same trait, which differs from the updated one; dormant within weeks 32-40, revive within 42-48, values null; with exactly one active bias no dormant event ever appears; output sorted; deterministic per stream.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests, suite, lint green.**
- [ ] **Step 5: Commit** — `feat: add trait assembly, self-description and drift sampling`.

---

### Task 13: Sample stage, demo config, end-to-end tests, README

**Files:**
- Create: `src/pm_traitbench/sampling/stage.py`, `configs/demo.yaml`
- Modify: `src/pm_traitbench/pipeline.py` (`STAGES = (SAMPLE_STAGE,)`), `README.md` (usage section)
- Test: `tests/sampling/test_stage.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass(frozen=True) class SampleResult: personas: list[Persona]; traits: list[Trait]; rules: list[Rule]; drift_events: list[DriftEvent]`
  - `def sample_all(config: Config, catalogue: Catalogue) -> SampleResult` — pure, no file access.
  - `def run(config: Config, store: DataStore) -> None`
  - `SAMPLE_STAGE = Stage(number=1, name="sample", help="sample personas, traits, rules and drift events", run=run, reads=(), writes=(PERSONAS, TRAITS, RULES, DRIFT_EVENTS))`

**Binding rules:**
- `run`: `catalogue = load_catalogue()`; `check_catalogue(catalogue, config.population.asset_classes, config.preferences.n_max)`; `result = sample_all(...)`; then write the four tables. Nothing is written if sampling raises.
- `sample_all` per slot, in this order, each with stream `stream(config.seed.root, "pm", slot.index, purpose)`: mandate (`"mandate"`), biases (`"activation"`, `"biases"`, `"regime"`), preferences (`"preferences"`), rules (`"rules"`), traits assembly, self-description (`"profile"`), drift (`"drift"`, only when `slot.drift`). The module docstring states this order and why: rules and preferences need the mandate; the self-description needs the two strongest biases; drift needs the final trait list.
- `configs/demo.yaml`: `population: {asset_classes: [equities, rates_credit], market_seeds: [A], pilot_per_cell: 1, full_per_cell: 1}` giving 4 pilot and 8 full PMs. A one-line comment says it is a small configuration for smoke runs.
- README: replace the Usage section with the `sample` command using the demo config and a sentence listing the four output tables and that `--force` overwrites. Keep Setup and Development sections. No mention of local documents.

- [ ] **Step 1: Write failing tests**
  - `test_stage.py` (uses `sample_all` with the fixture catalogue and a small config): persona count equals population size; every persona has 8 bias traits plus 4-8 preference traits; every trait/rule/drift `pm_id` exists in personas; every drift `trait_id` exists in that PM's traits; drift events exist only for drift slots and every drift slot has at least one; anti-typical and typical counts are equal; two calls are equal; raising `full_per_cell` leaves all pre-existing personas, traits, rules and drift events unchanged (compare by `pm_id`); changing `seed.root` changes trait values.
  - `test_end_to_end.py` (real CLI, packaged catalogue, `configs/demo.yaml`, `tmp_path`): `main(["sample", "--config", demo, "--data-dir", d])` returns 0; the four default files exist (`personas.jsonl`, others `.csv`) plus `run_metadata/sample.json`; reading back through `DataStore` validates and gives 12 personas; a second run into another directory gives byte-identical table files; rerun in the same directory returns 1; `--force` returns 0; a config adding `output: {format: parquet}` writes four `.parquet` files whose rows equal the CSV/JSONL run's rows after reading through the store; `pm-traitbench --help` output (via `build_parser(STAGES).format_help()`) contains `sample`.
  - Red phase: import error for `sampling.stage`, and unknown subcommand exit for the CLI test.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full suite, `uv run ruff check`, `uv run ruff format --check` green. Then a manual smoke run: `uv run pm-traitbench sample --config configs/demo.yaml --data-dir /tmp/pmtb-demo --force` and inspect `head` of each file for sane values. Then run the default population once: `uv run pm-traitbench sample --data-dir /tmp/pmtb-full --force` and confirm 168 personas.**
- [ ] **Step 5: Repo hygiene check before commit** — `git grep -n -iE "docs/|\bspec\b|the plan|per the plan|section [0-9]" -- src tests configs README.md` must return nothing relevant (the word "spec" inside identifiers such as `TableSpec`, `BiasSpec`, `LogNormalSpec` is fine). `git status --short` must show nothing staged under `docs/`.
- [ ] **Step 6: Commit** — `feat: add sample stage with demo config`.
- [ ] **Step 7: Push and open PR** — `git push -u origin feat/foundation-sampling`; `gh pr create --base main` with title `feat: add pipeline foundation and sample stage` and a body summarising: foundation pieces, the sample stage and its four tables, catalogues as drafts for human editing, test coverage, and how to run the demo. Body ends with the Claude Code attribution line. No mention of local documents. Wait for CI and report its result.
