# Signal Plan (Stage 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pipeline stage 5, `plan`: a deterministic pass that plants each PM's trait signals on dated sessions (carriers taken only from engine-flagged evidence), packs them into sessions, adds ledger-event and filler sessions, and writes `signals` plus a hidden `skeletons` table that is the stage 6 narrator's whole contract.

**Architecture:** A new `signals/` package in four pure passes: `carriers` (evidence pools per trait), `quotas` (how many signals of which mode per trait), `assemble` (dates and sessions), `skeleton` (stance text from a new `stances.yaml` bank). `stage.py` wires them per PM with keyed random streams. The engine gains one hidden `ideas` column, `chased_trend`, so extrapolation has carriers.

**Tech Stack:** Python 3.13, pydantic v2, numpy, PyYAML, pytest (+ pytest-xdist), ruff; uv for everything.

## Global Constraints

- Source of truth: `docs/specs/2026-09-25-signal-plan-design.md`. Read its "Decisions taken", "Carriers", "Quotas", "Assembly", "Skeleton" sections before your task.
- Repo rule: no file under `src/` or `tests/` may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings and comments state the rule itself and its domain reason. Citing a published paper is allowed.
- Code style follows the surrounding code: frozen pydantic models with `extra="forbid"` for table rows, frozen dataclasses for in-memory values, `StrEnum` for every closed set, full type hints, module docstring on every new module, line length 100, ruff rules `E, F, I, UP, B`.
- Commands: tests `uv run pytest <path> -n auto -q`; lint `uv run ruff check <paths>` and `uv run ruff format --check <paths>`. CI runs `ruff check`, `ruff format --check`, `pytest`.
- Determinism: every random draw comes from `pm_traitbench.rng.stream(config.seed.root, "plan", pm_id, <purpose>)`; never `random`, never an unseeded generator. Iterate dicts and sets in sorted order wherever order affects a draw. Two runs with the same config must write byte-identical tables.
- Dependency direction: `signals/` may import from `tables`, `enums`, `config`, `catalogues`, `errors`, `rng`, `market.axis` and `engine.adapters` (for `FORM_FOR_PREFERENCE` only). Nothing outside `signals/` imports `signals/` except `pipeline.py` and tests. `gates/` is never imported by `signals/`.
- Carriers come from engine-flagged evidence only; a shortfall is a warning in run metadata, never an error and never backfilled with another mode.
- Every session holds at most `max_signals_per_session` stances (default 2), no two stances of the same trait, and at most one `revealed_reaction` stance (so at most one advisor violation).
- Multi-asset PMs (listed in the engine's run metadata `skipped`) are skipped and listed in this stage's metadata.
- Living doc is `README.md` (no `ARCHITECTURE.md`). Each README change lands in the task that makes the old text wrong.

---

### Task 1: Engine persists `chased_trend` on ideas

**Files:**
- Modify: `src/pm_traitbench/tables/schema.py` (class `Idea`, around lines 441-500)
- Modify: `src/pm_traitbench/tables/specs.py` (`HIDDEN_COLUMNS["ideas"]`)
- Modify: `src/pm_traitbench/engine/ideas.py` (around lines 414-495: compute `entered_after_run` before building `idea_row`)
- Modify: `tests/gates/conftest.py` (`idea_row` default gains `chased_trend=False`)
- Modify: every other test that constructs `Idea(...)` directly: `tests/tables/test_store.py`, `tests/tables/test_formats.py`, `tests/tables/test_schema.py`, `tests/engine/adapters/test_series_for_idea.py` (add `chased_trend=False`)
- Test: `tests/engine/test_loop.py`, `tests/tables/test_specs.py`
- Modify: `README.md` (Engine usage paragraph that lists hidden `ideas` columns)

**Interfaces:**
- Produces: `Idea.chased_trend: bool` (required, non-null), description "Whether the entry is on the side of a trailing move already past one standard deviation of a horizon move." It is a hidden column: append `"chased_trend"` to `HIDDEN_COLUMNS["ideas"]`.
- The value is exactly `extrapolation.entered_after_run(trailing_move, draw.sd_h, side_sign, series.bullish_sign)`, the same call that today feeds `NewIdea.entered_after_run`. Keep `NewIdea.entered_after_run`, the function name and the `entries_after_run` counter unchanged; `NewIdea.entered_after_run` and `idea.chased_trend` must hold the same value (compute once, use for both).

- [ ] **Step 1: Write failing tests**
  - `tests/engine/test_loop.py::test_chased_trend_count_equals_entries_after_run_counter`: run the loop for a neutral equities PM on the fixture view (copy the setup of `test_opportunities_ideas_equals_len_ideas`); assert `sum(idea.chased_trend for idea in ideas) == opportunities["entries_after_run"]`. Red: `AttributeError`/validation error because `Idea` has no `chased_trend`.
  - `tests/tables/test_specs.py::test_chased_trend_is_a_hidden_ideas_column`: `"chased_trend" in HIDDEN_COLUMNS["ideas"]`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_loop.py tests/tables/test_specs.py -n auto -q`. Expected: the two new tests FAIL.
- [ ] **Step 3: Implement** the field, the hidden-column entry, the engine reorder, and the `chased_trend=False` default in every test builder listed above. README: in the paragraph listing hidden `ideas` columns ("the PM's own signal, forecast and interval, its street-view context, conviction and size rank on `ideas`"), add "whether the entry chased a trend that had already run".
- [ ] **Step 4: Run** `uv run pytest tests/engine tests/tables tests/gates -n auto -q` and `uv run ruff check src tests && uv run ruff format --check src tests`. All green.
- [ ] **Step 5: Commit** `git commit -m "feat: persist chased_trend on ideas for extrapolation carriers"`

---

### Task 2: Plan enums, row models, table specs and `PlanError`

**Files:**
- Modify: `src/pm_traitbench/enums.py`
- Modify: `src/pm_traitbench/tables/schema.py` (new models `Signal`, `Stance`, `Skeleton`)
- Modify: `src/pm_traitbench/tables/specs.py` (`SIGNALS`, `SKELETONS`, `PLAN_TABLES`, `HIDDEN_COLUMNS["skeletons"]`, `_check_hidden_columns` covers plan tables)
- Modify: `src/pm_traitbench/errors.py` (`PlanError`)
- Create: `tests/tables/test_plan_specs.py`
- Test: `tests/test_errors.py` (extend)

**Interfaces (binding):**

```python
# enums.py
class SignalMode(StrEnum):
    STATED = "stated"
    REVEALED = "revealed"
    CONTRADICTION = "contradiction"

class Valence(StrEnum):
    CONFIRM = "confirm"
    RETRACTED = "retracted"

class Ownership(StrEnum):
    SELF = "self"
    COLLEAGUE = "colleague"
    CLIENT = "client"

class SessionKind(StrEnum):
    DECISION = "decision"
    CHECK_IN = "check_in"
    SILENCE = "silence"

class StanceEntry(StrEnum):
    REVEALED = "revealed"
    STATED = "stated"
    CLAIM = "claim"
    RETRACT = "retract"
    THIRD_PARTY = "third_party"
    DRIFT_UPDATE = "drift_update"
    DRIFT_DORMANT = "drift_dormant"
    DRIFT_REVIVE = "drift_revive"
    REVEALED_REACTION = "revealed_reaction"
    VIOLATION = "violation"

class CarrierSource(StrEnum):
    LEDGER = "ledger"
    POSITION_DAY = "position_day"
    RULE_EVENT = "rule_event"
    IDEA = "idea"
```

```python
# schema.py
_SIGNAL_ID_PATTERN = r"^sg_\d{3,}$"
_SESSION_ID_PATTERN = r"^s_pm\d{3,}_\d{4}-\d{2}-\d{2}_[a-z]$"

class Signal(BaseModel):          # frozen, extra="forbid", every field with a description
    signal_id: str                # _SIGNAL_ID_PATTERN
    pm_id: str                    # _PM_ID_PATTERN
    session_id: str               # _SESSION_ID_PATTERN
    date: datetime.date
    trait_id: str                 # _TRAIT_ID_PATTERN
    mode: SignalMode
    trade_idea_id: str | None     # _IDEA_ID_PATTERN when set
    valence: Valence
    ownership: Ownership
    third_party_value: str | None
    claim_session_id: str | None  # _SESSION_ID_PATTERN when set

class Stance(BaseModel):          # frozen, extra="forbid"
    signal_id: str                # _SIGNAL_ID_PATTERN
    trait_id: str                 # _TRAIT_ID_PATTERN
    mode: SignalMode
    entry: StanceEntry
    stance: str                   # min_length=1

class Skeleton(BaseModel):        # frozen, extra="forbid"
    session_id: str               # _SESSION_ID_PATTERN
    pm_id: str
    date: datetime.date
    kind: SessionKind
    trade_idea_ids: tuple[str, ...]   # each _IDEA_ID_PATTERN, sorted, unique
    stances: tuple[Stance, ...]
    advisor_violation: str | None
    forbidden_trait_ids: tuple[str, ...]
    forbidden_pref_params: tuple[str, ...]
```

`Signal` validators (each a `ValueError` naming the rule):
- `session_id` starts with `f"s_{pm_id.replace('_', '')}_{date.isoformat()}_"` (a signal sits on a session of its own PM and date).
- `claim_session_id` is set exactly when `mode == CONTRADICTION`; when set it belongs to the same PM and its date part is earlier than `date`.
- `mode == CONTRADICTION` requires `trade_idea_id` set (it points at a flagged decision). `REVEALED` does not (a revealed preference reaction has no idea).
- `ownership != SELF` requires `mode == STATED` and `valence == CONFIRM`.
- `third_party_value` set requires `ownership != SELF`.

`Skeleton` validators:
- `session_id` prefix matches `pm_id` and `date` (same rule as `Signal`).
- `kind == SILENCE` requires `stances == ()`, `trade_idea_ids == ()` and `advisor_violation is None`.
- stance `trait_id`s are unique within the session.
- `advisor_violation` is set exactly when some stance has `entry == REVEALED_REACTION`.
- `trade_idea_ids` sorted and unique.

Specs:
- `SIGNALS = TableSpec("signals", Signal, ("pm_id", "signal_id"))`
- `SKELETONS = TableSpec("skeletons", Skeleton, ("pm_id", "session_id"))`
- `PLAN_TABLES: tuple[TableSpec, ...] = (SIGNALS, SKELETONS)`
- `HIDDEN_COLUMNS["skeletons"]` = every `Skeleton` field not in `SKELETONS.key` (hidden in full, the `position_days` pattern). `signals` has no hidden columns.
- `_check_hidden_columns` builds `models_by_table` from `(*ENGINE_TABLES, *PLAN_TABLES)`.

`errors.py`: `class PlanError(PmTraitbenchError)`, docstring "Raised when the signal plan's inputs are inconsistent or its stance bank cannot serve a request.", `exit_code = 1`.

- [ ] **Step 1: Write failing tests** in `tests/tables/test_plan_specs.py`:
  - a valid `Signal` and a valid `Skeleton` build; round-trip through `DataStore.write`/`read` in both `jsonl` and `parquet` (copy the pattern in `tests/tables/test_formats.py`), so the nested `stances` tuple survives parquet.
  - one test per validator above, each asserting `ValidationError` on the violating input: wrong session prefix; contradiction without claim; claim on a non-contradiction; claim dated on or after the signal; contradiction without idea; third-party with `mode=revealed`; third-party retracted; `third_party_value` with `ownership=self`; silence with a stance; duplicate stance trait; violation without a `revealed_reaction` stance and vice versa; unsorted `trade_idea_ids`.
  - `HIDDEN_COLUMNS["skeletons"]` equals the non-key fields; `PLAN_TABLES` names and keys.
  - `tests/test_errors.py`: `PlanError` subclasses `PmTraitbenchError`, `exit_code == 1`.
  Red: `ImportError` for the new names.
- [ ] **Step 2: Run** `uv run pytest tests/tables/test_plan_specs.py tests/test_errors.py -n auto -q`. Expected FAIL on import.
- [ ] **Step 3: Implement** the enums, models, specs and error exactly as above.
- [ ] **Step 4: Run** `uv run pytest tests/tables tests/test_errors.py -n auto -q`; ruff check and format on touched files. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: add signal and skeleton tables for the plan stage"`

---

### Task 3: `plan` config section

**Files:**
- Modify: `src/pm_traitbench/config.py` (new `PlanConfig`, field `plan` on `Config`)
- Test: `tests/test_config.py` (extend)

**Interfaces (binding):** `class PlanConfig(BaseModel)`, `model_config = ConfigDict(frozen=True, extra="forbid")`, every field carries `json_schema_extra={"basis": ..., "note": ...}` in the style of `Gate1Config`. Notes state the reason in domain terms and never mention a document.

| Field | Type | Default | Constraint | basis | note (reason) |
|---|---|---|---|---|---|
| `bias_signals_min` | int | 8 | ge 1 | guess | fewest planted carriers per active bias that a full-context reader can still pick up |
| `bias_signals_max` | int | 10 | ge 1 | guess | upper end of the per-bias carrier range |
| `pref_signals` | int | 3 | ge 1 | guess | a preference is stated or shown about three times a year |
| `bias_revealed_weight` | float | 0.65 | gt 0 | design | most bias evidence is visible only in decisions |
| `bias_stated_weight` | float | 0.175 | gt 0 | design | a minority of bias evidence is the PM describing it |
| `bias_contradiction_weight` | float | 0.10 | gt 0 | design | about one in ten bias signals sets a stated view against a later act |
| `pref_stated_weight` | float | 0.65 | gt 0 | guess | preferences are mostly said outright |
| `pref_revealed_weight` | float | 0.35 | gt 0 | guess | the rest show as a reaction or an instrument choice |
| `retracted_share` | float | 0.075 | ge 0, lt 1 | design | a few statements are taken back in the same session and must not count as evidence |
| `third_party_share` | float | 0.10 | ge 0, lt 1 | design | about one signal in ten belongs to a colleague or client, as an ownership distractor |
| `claim_lead_days` | tuple[int, int] | (10, 40) | 1 le first le second | guess | the earlier claim is its own session yet inside a quarter of the act it contradicts |
| `ledger_session_percentile` | float | 75.0 | gt 0, le 100 | guess | orders at or above the PM's own upper quartile of risk are large enough that the PM mentions them |
| `signal_session_cap` | float | 0.40 | gt 0, le 1 | design | most sessions carry no planted signal, so "links nothing" stays a common correct outcome |
| `filler_silence_share` | float | 0.5 | ge 0, le 1 | guess | half of filler is a pure market question |
| `drift_min_per_side` | int | 6 | ge 0 | design | enough evidence on each side of a drift event to tell the old value from the new |
| `max_signals_per_session` | int | 2 | ge 1 | design | keeps a PM's year at about 55-80 sessions without crowding one short exchange |

Validators: `bias_signals_min <= bias_signals_max`; `claim_lead_days[0] <= claim_lead_days[1]` and `claim_lead_days[0] >= 1`.
`Config.plan: PlanConfig = Field(default_factory=PlanConfig)`, placed after `gate1`.

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`: defaults equal the table; a YAML override of `plan.max_signals_per_session` keeps siblings (copy `test_yaml_override_of_nested_leaf_keeps_siblings`); `bias_signals_min > bias_signals_max` raises; `claim_lead_days: [40, 10]` raises; `claim_lead_days: [0, 5]` raises; every `PlanConfig` field appears in `Config().dump_with_basis()` with a non-empty basis. Red: `AttributeError: plan`.
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_config.py -n auto -q`; ruff on touched files. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: add plan config section"`

---

### Task 4: Stance bank catalogue

**Files:**
- Create: `src/pm_traitbench/catalogues/stances.yaml`
- Modify: `src/pm_traitbench/catalogues/models.py` (`BiasStances`, `PreferenceStances`, `Stances`, `Catalogue.stances`)
- Modify: `src/pm_traitbench/catalogues/loader.py` (load `stances.yaml`; public `check_stances`; call it from `check_catalogue`; `render_stance`)
- Test: `tests/catalogues/test_stances.py` (create), `tests/catalogues/test_shipped.py` (packaged file list)

**Interfaces (binding):**

```python
# models.py
StanceLines = dict[str, tuple[str, ...]]   # key: "all" or an AssetClass value

class BiasStances(BaseModel):              # frozen, extra="forbid"
    revealed: StanceLines
    stated: StanceLines
    claim: StanceLines
    retract: StanceLines
    third_party: StanceLines
    drift_update: StanceLines
    drift_dormant: StanceLines
    drift_revive: StanceLines

class PreferenceStances(BaseModel):        # frozen, extra="forbid"
    stated: StanceLines
    revealed_reaction: StanceLines
    violation: StanceLines
    retract: StanceLines
    third_party: StanceLines
    drift_update: StanceLines
    revealed: StanceLines = {}             # required (non-empty) for the expression group only

class Stances(BaseModel):                  # frozen, extra="forbid"
    biases: dict[str, BiasStances]
    preferences: dict[PreferenceGroup, PreferenceStances]

    def lines(self, key: str, entry: StanceEntry, asset_class: AssetClass) -> tuple[str, ...]:
        """Asset-class lines if present, else the "all" lines.

        `key` is a bias param name or a PreferenceGroup value. Raises PlanError
        naming key and entry when the bank has no such entry.
        """

# Catalogue gains: stances: Stances
```

```python
# loader.py
STANCE_SLOTS: dict[tuple[Kind, StanceEntry], frozenset[str]] = {
    (Kind.BIAS, StanceEntry.REVEALED): frozenset({"instrument", "entry", "target", "stop"}),
    (Kind.BIAS, StanceEntry.STATED): frozenset(),
    (Kind.BIAS, StanceEntry.CLAIM): frozenset(),
    (Kind.BIAS, StanceEntry.RETRACT): frozenset(),
    (Kind.BIAS, StanceEntry.THIRD_PARTY): frozenset({"who"}),
    (Kind.BIAS, StanceEntry.DRIFT_UPDATE): frozenset(),
    (Kind.BIAS, StanceEntry.DRIFT_DORMANT): frozenset(),
    (Kind.BIAS, StanceEntry.DRIFT_REVIVE): frozenset(),
    (Kind.PREFERENCE, StanceEntry.STATED): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.REVEALED): frozenset({"value", "instrument"}),
    (Kind.PREFERENCE, StanceEntry.REVEALED_REACTION): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.VIOLATION): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.RETRACT): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.THIRD_PARTY): frozenset({"value", "who"}),
    (Kind.PREFERENCE, StanceEntry.DRIFT_UPDATE): frozenset({"value", "old_value"}),
}
BANNED_STANCE_WORDS: tuple[str, ...] = (
    "loss aversion", "loss averse", "disposition", "anchor", "extrapolat",
    "herd", "overconfiden", "miscalibrat", "exit deficiency", "bias",
)

def check_stances(catalogue: Catalogue) -> None: ...
def render_stance(line: str, slots: Mapping[str, str]) -> str: ...   # str.format; CatalogueError on a missing slot
```

`check_stances` raises `CatalogueError` with the offending key, entry and line when:
- `biases` keys differ from `BIAS_PARAMS`, or `preferences` keys differ from the full `PreferenceGroup` set;
- any entry lacks an `"all"` key, a key is neither `"all"` nor an `AssetClass` value, or any key holds fewer than 2 lines (except `PreferenceStances.revealed` on non-expression groups, which must be empty);
- the expression group's `revealed` is empty;
- a line uses a slot outside `STANCE_SLOTS[(kind, entry)]` (reuse the loader's `_template_fields` helper) or uses no slot where the entry's slot set contains `value` (a preference line must name the preference);
- a line contains any `BANNED_STANCE_WORDS` entry, case-insensitive (a stance naming the bias leaks the label the dataset hides; "conviction" stays allowed because it is ordinary desk vocabulary and a public ledger column).

`stances.yaml` content requirements: top-level keys `biases` and `preferences`. For each of the eight bias params and each entry, 2-4 `all` lines written as second-person instructions to the PM narrator ("take part of {instrument} off ..."), lowercase-first, no trailing period, describing behaviour, never naming the bias. `revealed` lines must describe the engine's action for that bias (disposition: selling a winner early or holding a loser; loss aversion: adding to or holding a loser; exit deficiency: letting a fired stop or signpost pass, adding after it, or rolling late; anchoring: exiting right at a round or entry level; herding: entering with the street's view against the PM's own read; overconfidence: sizing big with a tight range; conviction: sizing out of line with stated conviction; extrapolation: buying after a run or selling after a drop because of the move). `claim` lines assert the opposite habit. `retract` lines state the habit and take it back in one line ("say ..., then correct yourself: ..."). `drift_*` lines say the PM has been working on it / has stopped doing it / has slipped back. For each preference group, entries per the model, lines using `{value}`; `violation` lines instruct the advisor ("answer ... even though the PM prefers {value}"); `third_party` lines attribute `{value}` to `{who}`.

- [ ] **Step 1: Write failing tests** in `tests/catalogues/test_stances.py`: the packaged bank loads and passes `check_stances`; `Stances.lines` returns asset-class lines when present and `all` otherwise, and raises `PlanError` for a missing entry; each `check_stances` failure above is triggered by a minimal hand-built `Stances` (one test per rule); `render_stance` fills slots and raises `CatalogueError` on a missing slot. Extend `tests/catalogues/test_shipped.py::test_packaged_yaml_files_exist` with `stances.yaml`. Red: import errors.
- [ ] **Step 2: Run** `uv run pytest tests/catalogues -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement** models, loader changes and the YAML.
- [ ] **Step 4: Run** `uv run pytest tests/catalogues tests/sampling -n auto -q` (sampling calls `check_catalogue`); ruff. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: add stance bank catalogue for the signal plan"`

---

### Task 5: Plan inputs and carrier pools

**Files:**
- Create: `src/pm_traitbench/signals/__init__.py` (docstring only)
- Create: `src/pm_traitbench/signals/inputs.py`
- Create: `src/pm_traitbench/signals/carriers.py`
- Create: `tests/signals/__init__.py` if the test tree uses them (check `tests/gates/`; match it)
- Create: `tests/signals/conftest.py` (shared builders, consumed by Tasks 6-9)
- Test: `tests/signals/test_inputs.py`, `tests/signals/test_carriers.py`

**Interfaces (binding):**

```python
# inputs.py
@dataclass(frozen=True)
class PlanInputs:
    persona: Persona
    traits: tuple[Trait, ...]              # this PM's, sorted by trait_id
    drift_events: tuple[DriftEvent, ...]   # this PM's, sorted by (date, trait_id, event)
    ideas: Mapping[str, Idea]              # by trade_idea_id
    ledger: tuple[LedgerRow, ...]          # sorted by (date, trade_idea_id)
    rule_events: tuple[RuleEvent, ...]
    position_days: tuple[PositionDay, ...] # sorted by (date, trade_idea_id)
    trading_days: tuple[date, ...]         # ascending

    def trait(self, trait_id: str) -> Trait: ...                 # PlanError if unknown
    def dormant_windows(self, trait_id: str) -> tuple[tuple[date, date], ...]: ...
    def is_dormant(self, trait_id: str, day: date) -> bool: ...
    def value_at(self, trait_id: str, day: date) -> float | str: ...

def trading_days(config: Config) -> tuple[date, ...]: ...
def build_inputs(personas, traits, drift_events, ideas, ledger, rule_events, position_days,
                 days: tuple[date, ...], skipped: Collection[str]) -> list[PlanInputs]: ...
```

- `trading_days(config)`: `axis = build_axis(config.timeline(), config.market.burn_in_days)`; return `tuple(axis.dates[axis.horizon])` (the engine's published horizon, so every engine date is a trading day).
- `dormant_windows`: half-open `[dormant_date, revive_date)` per `dormant` event of that trait, paired with the first later `revive`; with no later revive the window ends at `date.max`.
- `value_at(trait_id, day)`: start from `trait.value`, apply every `update` event of that trait with `date <= day` in date order (value becomes `to_value`).
- `build_inputs`: one `PlanInputs` per persona not in `skipped`, sorted by `pm_id`. Raises `PlanError` naming PM and id when a drift event's `trait_id` is not a trait of that PM, or a ledger, rule-event or position-day row names a `trade_idea_id` missing from that PM's ideas.

```python
# carriers.py
@dataclass(frozen=True)
class Carrier:
    trait_id: str
    trade_idea_id: str
    date: date
    source: CarrierSource

FLAG_PREFIX_PARAM: dict[str, str] = {
    "disposition": "disposition_ratio",
    "loss_aversion": "loss_aversion_lambda",
    "exit_deficiency": "exit_deficiency",
    "anchoring": "anchoring_rho",
    "herding": "herding_weight",
    "overconfidence": "overconfidence_coverage",
    "conviction": "conviction_size_miscalibration",
}
HOLD_FLAGS: frozenset[str] = frozenset({"loss_aversion:hold", "disposition:hold_loser"})
BREACH_RESPONSES: frozenset[RuleResponse] = frozenset(
    {RuleResponse.ACKED_NO_ACTION, RuleResponse.ADDED}
)

def carrier_pools(inputs: PlanInputs) -> dict[str, tuple[Carrier, ...]]: ...
```

`carrier_pools` rules:
- Keys: the `trait_id` of every active bias (possibly an empty tuple), plus every preference trait whose value in force on some idea's `entry_date` maps to a non-None form in `engine.adapters.FORM_FOR_PREFERENCE`.
- Every `<prefix>:<pattern>` flag in a `ledger` or `position_days` `bias_flag` (split on `;`) is a carrier for the bias `FLAG_PREFIX_PARAM[prefix]`, if that bias is active for the PM; an unknown prefix raises `PlanError` (guards against the engine adding a flag this stage does not understand). `HOLD_FLAGS` count only on their first `position_days` date per `(trade_idea_id, flag)`, because they mark every held day.
- `rule_events` with `response in BREACH_RESPONSES` are carriers for `exit_deficiency` (if active), dated `response_date`.
- Ideas with `chased_trend` true are carriers for `extrapolation_theta` if active, dated `entry_date`, source `IDEA`.
- An expression preference's carriers are ideas whose `expression` equals the mapped form of the value in force on the idea's `entry_date`, source `IDEA`.
- Drop carriers whose date is in a dormant window of that trait. Deduplicate on `(trait_id, trade_idea_id, date)`, keeping the first by source order `LEDGER, POSITION_DAY, RULE_EVENT, IDEA`. Sort each pool by `(date, trade_idea_id, source)`.

`tests/signals/conftest.py` (shared; later tasks import from it): re-export `idea_row`, `ledger_row`, `rule_event`, `position_day` from `tests.gates.conftest`; add `bias_trait(param, *, active=True, value=None, trait_id=...)`, `pref_trait(param, value, *, trait_id=...)` using the shipped catalogue's params, `drift_event(trait_id, day, event, from_value=None, to_value=None)`, `persona(asset_class=AssetClass.EQUITIES)`, `plan_inputs(**overrides) -> PlanInputs` (one equities PM, eight neutral inactive biases, no preferences, trading days = 260 weekdays from 2026-01-05), and a `plan_config` fixture returning `Config().plan`.

- [ ] **Step 1: Write failing tests.**
  - `test_inputs.py`: `trading_days(Config())` equals the engine's horizon dates (build via `build_axis` in the test); `dormant_windows` with and without a revive; `is_dormant` at the boundaries (dormant date inside, revive date outside); `value_at` before, on and after an update; `build_inputs` skips listed PMs, sorts by `pm_id`, raises `PlanError` on an unknown drift trait and on a ledger row with an unknown idea.
  - `test_carriers.py`: each flag in `FLAG_PREFIX_PARAM` maps to its bias; a `;`-joined flag yields one carrier per flag; a flag on an inactive bias yields nothing; hold flags yield only the first day per idea while `loss_aversion:add` on `position_days` yields every day; `acked_no_action` and `added` rule events are carriers dated `response_date`, `acted` and `overridden` are not; `chased_trend` ideas count only when extrapolation is active; a same-day ledger and position-day flag dedupe to one `LEDGER` carrier; dormant-window carriers are dropped; an unknown prefix raises `PlanError`; an expression preference mapped to `Expression.CURVE` collects curve ideas only, and after a drift update to a value mapped to `OUTRIGHT` collects outright ideas after the update date; every active bias key exists even with an empty pool; `FLAG_PREFIX_PARAM` values are all in `BIAS_PARAMS` and cover every bias param except `extrapolation_theta`.
  Red: `ModuleNotFoundError: pm_traitbench.signals`.
- [ ] **Step 2: Run** `uv run pytest tests/signals -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/signals -n auto -q`; ruff. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: build plan inputs and engine-flagged carrier pools"`

---

### Task 6: Quotas

**Files:**
- Create: `src/pm_traitbench/signals/quotas.py`
- Test: `tests/signals/test_quotas.py`

**Interfaces (binding):**

```python
@dataclass(frozen=True)
class DateWindow:
    first: date   # inclusive
    last: date    # inclusive

@dataclass(frozen=True)
class PlannedSignal:
    trait_id: str
    mode: SignalMode
    valence: Valence
    ownership: Ownership
    entry: StanceEntry
    window: DateWindow
    needs_carrier: bool
    third_party_value: str | None = None
    drift_date: date | None = None     # set only on drift notes

def largest_remainder(total: int, weights: Sequence[tuple[K, float]]) -> dict[K, int]: ...
def round_half_up(x: float) -> int: ...          # math.floor(x + 0.5); Python's round() is banker's
def plan_quotas(inputs: PlanInputs, pools: Mapping[str, Sequence[Carrier]],
                catalogue: Catalogue, knobs: PlanConfig,
                rng: np.random.Generator) -> list[PlannedSignal]: ...
```

`largest_remainder` (binding): floors of `total * w / sum(w)`, then the leftover units go one each to the largest fractional parts; ties go to the earlier item in `weights` order. Result sums to `total`; zero-weight items get 0.

`plan_quotas` rules, in this order:
1. **Segments.** For each trait, its segments are the maximal runs of trading days split at every `update` date and with every dormant window removed. A trait with no drift events has one segment (all trading days). A segment is dropped if empty.
2. **Drift notes.** Each drift event on the PM yields one `PlannedSignal(mode=STATED, valence=CONFIRM, ownership=SELF, entry=DRIFT_UPDATE or DRIFT_DORMANT or DRIFT_REVIVE, window=DateWindow(event_date, last trading day), needs_carrier=False, drift_date=event_date)`. An `update` or `revive` note counts toward the segment that starts on or after its date.
3. **Confirm count per trait.** Active bias: `n = rng.integers(bias_signals_min, bias_signals_max + 1)`. Preference: `n = pref_signals`. If the trait has more than one segment, each segment needs `drift_min_per_side` signals minus the notes counted in it (floor 0), and `n = max(n, sum of those needs)`. Inactive biases get no confirm signals.
4. **Allocate to segments.** Give each segment its need, then spread the remaining `n - sum(needs)` over segments with `rng.multinomial` in proportion to segment length in trading days. Each signal's `window` is its segment's first and last day.
5. **Modes.** Bias: `largest_remainder(n, [(REVEALED, bias_revealed_weight), (STATED, bias_stated_weight), (CONTRADICTION, bias_contradiction_weight)])`; preference: `[(STATED, pref_stated_weight), (REVEALED, pref_revealed_weight)]`. Assign the mode multiset to the trait's `n` signals in the order given by `rng.permutation`, so segments get a mix.
6. **Entry and carrier need.** Bias `REVEALED` and `CONTRADICTION`: `entry=REVEALED`, `needs_carrier=True`. Bias `STATED`: `entry=STATED`. Preference `STATED`: `entry=STATED`. Preference `REVEALED`: `entry=REVEALED, needs_carrier=True` when `pools` has a non-empty pool for the trait, else `entry=REVEALED_REACTION, needs_carrier=False`. All confirm signals: `valence=CONFIRM`, `ownership=SELF`.
7. **Retracted.** `r = round_half_up(retracted_share * total_confirm)` where `total_confirm` counts step 3-6 signals over all traits (notes excluded). Targets: the PM's active biases and preferences in `rng.permutation` order, cycled. Each: `mode=STATED, valence=RETRACTED, ownership=SELF, entry=RETRACT, window=` all trading days, `needs_carrier=False`.
8. **Third-party.** `k = round_half_up(third_party_share * (total_confirm + r))`. Preference targets: the PM's preferences whose catalogue entry has at least one value the PM never holds (not `trait.value` and not any drift `from`/`to`). If the PM has inactive biases: `k_bias = ceil(k / 2)` if preference targets exist else `k`; else `k_bias = 0`. `k_pref = k - k_bias` if preference targets exist, else those are dropped (no target exists). Bias targets: inactive biases in `rng.permutation` order, cycled. Preference targets likewise, each with `third_party_value = rng.choice` over its never-held values (sorted before choosing). Each: `mode=STATED, valence=CONFIRM, ownership=COLLEAGUE or CLIENT` with probability 0.5 each, `entry=THIRD_PARTY`, window all trading days, `needs_carrier=False`.
9. Return confirm signals in trait order, then notes, then retracted, then third-party.

- [ ] **Step 1: Write failing tests** (using `tests/signals/conftest.py` builders; assert properties, not hand-computed seeded values):
  - `largest_remainder`: sums to total; `(10, [(a, .65), (b, .175), (c, .10)])` gives 7, 2, 1; ties go to the earlier item; zero weight gives zero.
  - `round_half_up(0.5) == 1`, `round_half_up(2.5) == 3`.
  - an active bias gets between 8 and 10 confirm signals with modes split by `largest_remainder`; an inactive bias gets none; a preference gets 3.
  - a preference with no expression carriers plans `REVEALED_REACTION`; one with a non-empty pool plans `REVEALED` with `needs_carrier=True`.
  - retracted and third-party rows are additional (confirm counts unchanged when shares go to 0 vs default) and have the counts from the formulas.
  - third-party preference rows carry a `third_party_value` the PM never holds; bias third-party rows target only inactive biases; a PM with no inactive bias gets all third-party rows on preferences.
  - a bias with an update event: every segment gets at least `drift_min_per_side` confirm signals counting the update note; a dormant/revive pair gives two segments with the dormant window excluded from both windows; one note per drift event with the right entry and `drift_date`.
  - same `rng` seed twice gives equal lists.
  Red: `ModuleNotFoundError`.
- [ ] **Step 2: Run** `uv run pytest tests/signals/test_quotas.py -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/signals -n auto -q`; ruff. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: plan per-trait signal quotas"`

---

### Task 7: Session assembly

**Files:**
- Create: `src/pm_traitbench/signals/assemble.py`
- Test: `tests/signals/test_assemble.py`

**Interfaces (binding):**

```python
@dataclass(frozen=True)
class PlacedSignal:
    signal_id: str
    planned: PlannedSignal
    date: date
    trade_idea_id: str | None
    claim_date: date | None            # set only for contradiction

@dataclass(frozen=True)
class PlannedSession:
    session_id: str
    date: date
    kind: SessionKind
    trade_idea_ids: tuple[str, ...]    # sorted, unique
    signals: tuple[PlacedSignal, ...]  # signals whose row sits on this session
    claims: tuple[PlacedSignal, ...]   # contradiction signals whose claim stance sits here

@dataclass(frozen=True)
class Assembly:
    sessions: tuple[PlannedSession, ...]   # sorted by session_id
    signals: tuple[Signal, ...]            # table rows, sorted by signal_id
    warnings: tuple[str, ...]
    counts: dict[str, Any]

def session_id(pm_id: str, day: date, index: int) -> str: ...   # index 0 -> "a"; PlanError past "z"
def assemble(inputs: PlanInputs, planned: Sequence[PlannedSignal],
             pools: Mapping[str, Sequence[Carrier]], knobs: PlanConfig,
             rng: np.random.Generator) -> Assembly: ...
```

`session_id(pm_id, day, index)` = `f"s_{pm_id.replace('_', '')}_{day.isoformat()}_{letter}"`, letter `chr(ord("a") + index)`.

Session capacity (binding, every step): a draft session "has room for trait t" when it is not silence, holds fewer than `max_signals_per_session` stances (signals plus claims), holds no stance of trait `t`, and, if the incoming stance is a `REVEALED_REACTION`, holds no other `REVEALED_REACTION`.

`assemble` steps, in order:
1. **Carrier signals** (`needs_carrier`), grouped by trait in `trait_id` order, each trait's signals in plan order. For each: eligible carriers are unused carriers of that trait whose date is in the signal's window; for `CONTRADICTION`, also at trading-day index `>= claim_lead_days[0]`. None eligible: drop the signal and warn `f"{pm_id} {trait_id}: {mode} signal dropped, no carrier in {first}..{last}"`. Otherwise draw one uniformly with `rng`, mark it used; place the signal on the first draft session that date with room, else open a new draft session that date.
2. **Claims.** For each placed contradiction, draw a claim day uniformly from trading days at index `[i - claim_lead_days[1], i - claim_lead_days[0]]` (clipped at 0) of the carrier day `i`, excluding the trait's dormant days; place the claim on the first draft session that date with room, else a new one. If no day qualifies, drop the whole contradiction signal (carrier returned unused) and warn.
3. **Packable signals**: every remaining non-note signal, in `rng.permutation` order. Allowed dates: trading days in its window and not dormant for its trait. Candidates: draft sessions dated on an allowed date with room. If any, join one drawn uniformly; else open a new draft session on an allowed date drawn uniformly; no allowed date: drop and warn.
4. **Drift notes**, in `drift_date` order: join the earliest draft session (by date, then creation order) dated on or after `drift_date` with room; else open a new draft session on the first trading day on or after `drift_date`.
5. **Ledger sessions.** `threshold = numpy.percentile([row.risk_amount for row in inputs.ledger], ledger_session_percentile)` (numpy's default linear method); skip if the PM has no ledger rows. For each date with a row at or above threshold, in date order: add those rows' idea ids to the first non-silence draft session that date if any, else open a new draft session with those ideas and no stances.
6. **Idea ids and kind.** Each draft's ideas = carrier ideas of its signals plus ledger-added ideas, sorted, unique. Kind: `DECISION` when the PM has a ledger row that date for any of the session's ideas, else `CHECK_IN`.
7. **Filler.** `S` = sessions with at least one stance, `T` = all sessions. Needed filler `f = max(0, ceil(S / signal_session_cap) - T)` (compute with a `1e-9` guard against float error). Free days are trading days with no session. Draw `min(f, len(free))` distinct free days uniformly; each is `SILENCE` with probability `filler_silence_share`, else `CHECK_IN` with no ideas. If `f > len(free)`, warn that the cap cannot be met.
8. **Ids.** Order sessions by `(date, creation order)`; assign letters per date with `session_id`. Order stance-bearing signals by `(session date, session letter, placement order)` and number them `sg_001`, `sg_002`, ... per PM. Build `Signal` rows (claim signals' `claim_session_id` = the claim session's id).
9. **Counts** (JSON-safe, sorted keys): `signals_by_trait_mode` (`{trait_id: {mode: n}}` over placed rows), `revealed_planned` and `revealed_placed` per trait (carrier signals planned vs placed), `segments` per drifted trait (`[n placed per segment]`), `sessions_by_kind`, `signal_session_share` (float, 4 decimals).

- [ ] **Step 1: Write failing tests** (builders from `tests/signals/conftest.py`; properties over outputs, plus small exact cases where no randomness is involved):
  - every placed revealed or contradiction signal's `date` equals one carrier of its trait; no carrier is used twice.
  - a trait with 2 carriers and 6 planned revealed signals places 2 and warns once per drop.
  - contradiction: claim session date precedes the carrier by 10-40 trading days; `claim_session_id` points at a session holding a `claim` for that signal.
  - capacity: no session holds more than `max_signals_per_session` stances, never two of one trait, never two `REVEALED_REACTION`s; with `max_signals_per_session=1` every signal session holds one stance.
  - packable signals avoid dormant windows and stay in their window.
  - drift note lands on the first session on or after its date (construct an existing session after the date and check it is chosen).
  - ledger: every date with an order at or above the percentile has a session containing that order's idea; a session with such an order is `DECISION`.
  - filler: `signal sessions / total <= signal_session_cap` whenever free days suffice; with a 5-day horizon and many signals, a cap warning is emitted.
  - ids: session ids match `_SESSION_ID_PATTERN`, letters start at `a` per date; signal ids are contiguous from `sg_001`; `session_id(..., 26)` raises `PlanError`.
  - rows validate as `Signal`; running twice with equal seeds gives equal `Assembly`.
  Red: `ModuleNotFoundError`.
- [ ] **Step 2: Run** `uv run pytest tests/signals/test_assemble.py -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement.** Keep draft sessions a private mutable dataclass inside the module; the public outputs are the frozen types above.
- [ ] **Step 4: Run** `uv run pytest tests/signals -n auto -q`; ruff. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: assemble planned signals into dated sessions"`

---

### Task 8: Skeleton rendering

**Files:**
- Create: `src/pm_traitbench/signals/skeleton.py`
- Test: `tests/signals/test_skeleton.py`

**Interfaces (binding):**

```python
def forbidden_sets(inputs: PlanInputs, catalogue: Catalogue) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(inactive bias trait_ids sorted, catalogue preference params for the PM's
    asset class that the PM does not hold, sorted)."""

def format_level(x: float) -> str: ...       # f"{x:.4g}"

def render_skeletons(inputs: PlanInputs, assembly: Assembly, catalogue: Catalogue,
                     rng: np.random.Generator) -> list[Skeleton]: ...
```

Rules:
- One `Skeleton` per `PlannedSession`, in session order. `forbidden_trait_ids` and `forbidden_pref_params` from `forbidden_sets` on every skeleton (a third-party stance in the session does not lift a trait from the forbidden set: the PM must still not show it).
- Stances: claims first (entry `CLAIM`, mode `CONTRADICTION`), then signals in placement order with their planned `entry`. Lines come from `catalogue.stances.lines(key, entry, asset_class)` with `key` = the bias param or the preference's `PreferenceGroup` value (look up the group via `catalogue.preferences`), one line chosen by `rng.choice` over the tuple, then `render_stance(line, slots)`.
- Slots: `instrument`, `entry`, `target`, `stop` from the carrier idea (`instrument_id`, `format_level` of `entry_level`, `target_level`, `stop_level`); `value` = `inputs.value_at(trait_id, session date)` for self preference signals, `third_party_value` for third-party preference signals; `old_value` = the drift event's `from_value` on a preference `DRIFT_UPDATE` note; `who` = `"a colleague"` or `"a client"` from `ownership`. Only the slots the entry's `STANCE_SLOTS` set names are passed.
- `advisor_violation`: when the session has a `REVEALED_REACTION` stance, render one line of that preference group's `VIOLATION` entry with the same `value`; else `None`.
- A missing stance entry propagates `PlanError` from `Stances.lines`.

- [ ] **Step 1: Write failing tests:**
  - `forbidden_sets`: inactive biases only; preference params exclude held ones and other asset classes' params.
  - a revealed bias stance has the carrier's instrument and formatted levels in its text and no `{` left.
  - a claim stance appears on the claim session with `entry=CLAIM` and the contradiction's `signal_id`; the carrier session has the `REVEALED` stance for the same id.
  - a preference stance after a drift update uses the new value; a `DRIFT_UPDATE` note names both values.
  - third-party preference stance text contains `third_party_value` and "a colleague" or "a client".
  - `advisor_violation` set exactly when a `REVEALED_REACTION` stance is present and contains the preference value.
  - every rendered skeleton validates as `Skeleton`; silence skeletons have no stances; equal seeds give equal output.
  Red: `ModuleNotFoundError`.
- [ ] **Step 2: Run** `uv run pytest tests/signals/test_skeleton.py -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/signals -n auto -q`; ruff. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: render session skeletons from the stance bank"`

---

### Task 9: `plan` stage, CLI wiring, README and end-to-end

**Files:**
- Create: `src/pm_traitbench/signals/stage.py`
- Modify: `src/pm_traitbench/pipeline.py` (append `PLAN_STAGE` to `STAGES`)
- Modify: `README.md` (usage block, new "Signal plan" section after "Gate 1")
- Test: `tests/signals/test_stage.py`, `tests/test_end_to_end.py`

**Interfaces (binding):**

```python
def run(config: Config, store: DataStore) -> dict[str, Any]: ...

PLAN_STAGE = Stage(
    number=5,
    name="plan",
    help="plant trait signals on dated sessions and write one narrator skeleton per session",
    run=run,
    reads=(PERSONAS, TRAITS, DRIFT_EVENTS, *ENGINE_TABLES),
    writes=PLAN_TABLES,
)
```

`run`:
1. `meta = store.read_run_metadata("engine")`; `None` raises `PlanError("engine run metadata is missing; run the engine stage first")`.
2. `catalogue = load_catalogue()`; `check_stances(catalogue)`.
3. `inputs = build_inputs(..., days=trading_days(config), skipped=meta["skipped"])`.
4. Per PM, in `pm_id` order: `pools = carrier_pools(pm)`; `planned = plan_quotas(pm, pools, catalogue, config.plan, stream(root, "plan", pm_id, "quotas"))`; `assembly = assemble(pm, planned, pools, config.plan, stream(root, "plan", pm_id, "assemble"))`; `skeletons = render_skeletons(pm, assembly, catalogue, stream(root, "plan", pm_id, "skeleton"))`.
5. `store.write(SIGNALS, all signal rows)`, `store.write(SKELETONS, all skeletons)`.
6. Return `{"skipped": sorted(meta["skipped"]), "pms": {pm_id: assembly.counts}, "warnings": [every warning, in PM order]}`. No verdict hook: shortfalls never block.

README:
- Usage block: add `uv run pm-traitbench plan --config configs/demo.yaml --data-dir data` after `gate1`.
- New `## Signal plan` section after `## Gate 1` (before `## Development`): what `plan` reads and writes; `signals` columns and what `third_party_value` and `claim_session_id` mean; `skeletons` is hidden in full and is the narrator's whole input; carriers are engine-flagged evidence only (list the per-bias sources in one short table, noting disposition and anchoring flags come from `position_days`); signals without a carrier pack into existing sessions up to `plan.max_signals_per_session`; shortfalls (revealed below quota, drift side below minimum, session cap not met) are warnings in `run_metadata/plan.json`, never failures; multi-asset PMs are skipped.

- [ ] **Step 1: Write failing tests.**
  - `tests/signals/test_stage.py`: build inputs with `write_stage_inputs(store, fixture_market, neutral_pm)` and `stage_config()` from `tests/engine/conftest.py` (copy the setup of `tests/engine/test_stage.py::test_engine_stage_writes_four_tables_and_rules_gains_idea_rows`). Before running the engine, rewrite `traits` so the first PM has `loss_aversion_lambda` and `overconfidence_coverage` active (values at their active medians, 2.0 and 0.40) and one preference from the shipped catalogue per group for its asset class (trait ids after the biases), so the plan has carriers and preferences to place. Then `run_stage(ENGINE_STAGE, ...)` and `run_stage(PLAN_STAGE, ...)`; assert both tables exist and validate; every `Signal.session_id` exists in `skeletons`; every skeleton stance's `signal_id` exists in `signals`; the stage raises `PlanError` without engine metadata; a second run into another directory writes byte-identical `signals` and `skeletons`; multi-asset PMs appear in metadata `skipped` and in no row.
  - `tests/test_end_to_end.py::test_sample_market_engine_then_plan_on_synthetic_seeds`: with `_SYNTHETIC_CONFIG`, run `sample`, `market`, `engine`, `plan` through `main`; `plan` returns 0; `run_metadata/plan.json` exists; every direct-asset PM has at least one skeleton.
  - `test_help_output_lists_the_plan_subcommand`.
  Red: `ImportError` / unknown subcommand.
- [ ] **Step 2: Run** `uv run pytest tests/signals/test_stage.py tests/test_end_to_end.py -n auto -q`. Expected FAIL.
- [ ] **Step 3: Implement** stage, pipeline registration and README.
- [ ] **Step 4: Run** `uv run pytest tests/signals tests/test_end_to_end.py tests/test_stages.py tests/test_cli_stages.py -n auto -q`; ruff on `src tests`. Then one full suite run: `uv run pytest -n auto -q`. Green.
- [ ] **Step 5: Commit** `git commit -m "feat: add the plan stage"`
