# Engine Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pipeline stage 3, `engine`: a deterministic daily loop per PM over the PM's market seed that turns sampled traits and rules into `ideas`, idea-scope `rules`, `ledger`, `rule_events` and the hidden `position_days` table, for equities, rates_credit and commodities PMs.

**Architecture:** One pure function `step(state, t, params, adapter, view, config, rng_for) -> (state, DayOutput)` with no I/O. Adapters (one per asset class) turn instruments into level series, rule fields, risk units and anchors. Bias rules live in a registry keyed by parameter name. `MarketView` holds one seed in memory. Only `engine/stage.py` touches the `DataStore`. Multi-asset PMs are skipped.

**Tech Stack:** Python 3.13, numpy, scipy (normal quantile), pydantic v2, pyyaml, pytest, ruff, uv.

## Global Constraints

- Branch `feat/engine-stage` off `main`, in the working directory `/Users/ianpoey/code/github/pm-traitbench`. Run every command with `uv run`.
- No file in the repo (code, docstrings, comments, tests, YAML, README, commit messages) may mention `docs/`, a spec or a plan. Docstrings state the rule itself. Citing a published paper or a named public series as the basis of a value is allowed.
- Never commit anything under `docs/` except as instructed by subagent-driven-development for the spec and plan.
- Commit messages: one Conventional Commits subject line, no body, then a blank line and the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. No other trailer.
- Plain dash `-`, never an em dash. No filler words ("genuinely", "actually", "truly", "really"). Docstrings concise; inline comments 1-2 lines.
- Match surrounding style: frozen pydantic row models with `extra="forbid"` and `Field(description=...)` on every field; frozen dataclasses for internal state; `json_schema_extra={"basis": ..., "note": ...}` on every config leaf with a non-empty note; `Config.dump_with_basis()` keeps working.
- Randomness only through `pm_traitbench.rng.stream(root_seed, *keys)`. Engine streams are `stream(config.seed.root, "engine", pm_id, purpose, *extra)`; every draw type has its own purpose string and includes the day index `t` so adding a PM or a day never perturbs another's draws.
- Dependency direction: `engine/` imports `config`, `rng`, `enums`, `errors`, `tables`, `catalogues`, `market.axis`, `market.regimes`, `market.levels`, `market.constants`. Nothing in `market/` or `sampling/` imports `engine/`. Only `engine/stage.py` reads or writes the store.
- Purity: `step` and every bias rule never mutate their inputs; state is frozen dataclasses replaced with `dataclasses.replace`. A test asserts the input state is unchanged after `step`.
- Invariants enforced with `EngineError`: no position size below 0; `size_pct_book` never above the mandate cap after sizing; an idea is closed once; a curve or calendar-spread leg always has a tenor; a rule `field` unknown to the adapter fails before any day runs.
- Hidden provenance columns (`HIDDEN_COLUMNS`) are ground truth; nothing in this stage strips them, but the registry must list every one.
- Lint and tests green after every task: `uv run ruff check && uv run ruff format --check && uv run pytest -q`.
- A statistical test failing at default config is a design problem, not a tolerance to loosen: stop and report the numbers (status BLOCKED) instead of changing a threshold.

## Decisions made while planning (binding, refine the design)

12. **Adapters carry the horizon and take the held set.** Every adapter is constructed as `Adapter(sub_style: str, horizon_days: int)`, `adapter_for(asset_class, sub_style, horizon_days)`, and `build_legs(form, instrument_id, view, t, universe, held: frozenset[str], rng)` excludes held instruments explicitly; `relative_move` uses the adapter's horizon. Decided at the Task 10 review; Tasks 11, 12, 15 and 17 follow this signature wherever they say `adapter_for(asset_class, sub_style)` or `build_legs(..., rng)`.

1. **Series convention.** A `Series` is `legs: tuple[LegRef, ...]`, `bullish_sign: int` (+1 or -1) and `unit: str` (`"pct"` or `"bp"`), where `LegRef = (instrument_id, tenor | None, coeff: float)`. Raw leg levels are quoted, not flipped: `100 * ln(price)` for equities and commodities (so a difference is a percent move), `100 * yield_pct` (bp) for a sovereign tenor, `spread_bp` for a credit issuer. `level(series, t) = sum(coeff * raw_level(leg, t))`. `bullish_sign` is +1 when a rising series is good for a long (prices, a steepener series `10Y - 2Y`, a calendar spread `100 * (ln M1 - ln Mk)`), -1 when a falling series is good (an outright yield, a spread). Reason: rule levels and thesis text stay in quoted terms ("out through 175") while one P&L formula covers every family.
2. **P&L and leg sides.** `side_sign = +1 for buy, -1 for sell`. `pnl_unit = bullish_sign * side_sign * (level_now - entry_level)`. Ledger side of a leg is `buy` when `side_sign * bullish_sign * coeff * leg_bullish > 0`, where `leg_bullish` is -1 for a yield or spread leg and +1 for a price leg. Worked checks (binding): credit outright buy -> leg buy (long bond); rates outright buy -> leg buy (long bond); steepener buy (series `10Y - 2Y`, coeffs 10Y +1, 2Y -1) -> 10Y leg sell, 2Y leg buy; pair buy -> A buy, B sell; calendar spread buy -> M1 buy, Mk sell.
3. **Adverse direction.** `adverse_dir = -bullish_sign * side_sign`. Stop, level-signpost and target levels are `entry_level + adverse_dir * distance` (stop, signpost) and `entry_level - adverse_dir * rr * stop_distance` (target). Idea-scope rows use `field = "level"`, `op = ">="` when `adverse_dir > 0` else `"<="` for stop and signposts, the reverse for the target, and `unit` = the series unit.
4. **The PM-scope stop is applied through the idea's stop row only.** `stop_loss` is the source of every idea's `stop` row and is not evaluated daily itself, so a stop firing produces one event, not two. `trim_at_target`, `min_holding_period` and `roll_before_expiry` are evaluated daily on every open idea; `max_risk_pct`, `max_positions`, `no_add_before_trigger` and exclusions are constraints and never fire.
5. **Target action.** An idea `target` row firing alone exits the whole position (take profit). When `trim_at_target` also fires the same day the trim wins (one event each, both `acted`), half the size comes off and the target row is consumed; the remainder exits later by stop, signpost, discretionary sell or close-out.
6. **arrival_rate default 0.6.** Scratch simulation (daily vol 1.5%, stop -10%, target 2x, hazard 0.03, threshold 1.0): 0.4 gives about 33 ideas per PM-year (10th percentile 25), 0.6 gives about 48 (10th percentile 39), 0.8 gives about 65; about 4 concurrent positions, so `max_positions` never binds. 0.6 sits mid-range of the 30-80 target.
7. **Statistical test tolerances** (from the probe): own-signal correlation within 0.02 of `skill` and interval coverage within 0.02 of `c` over 20,000 draws (standard errors about 0.007 and 0.003); neutral breach rate within 0.03 of `e` when pooled over at least 500 trigger firings (standard error about 0.011); each bias direction test compares two values on the same fixed state over 2,000 repeated draws of the same stream family.
8. **Consensus lookup returns `None`** for an instrument with no consensus rows (curves and commodities have them; a fixture may omit some), and herding treats `None` as neutral.
9. **Position days are written for every open position on every day**, including the day it closes (with that day's action), and not before the entry day.
10. **`EffectiveParams` treats `overconfidence_coverage` on `1 - c`** for the logit-scale regime multiplier because it is lower-is-stronger.
11. **Real-seed commodity ids share the synthetic table**, so contract multipliers keyed by commodity code cover both universes.

---

### Task 1: Engine enums, error, row models, table specs, hidden-column registry

**Files:**
- Modify: `src/pm_traitbench/enums.py`
- Modify: `src/pm_traitbench/errors.py`
- Modify: `src/pm_traitbench/tables/schema.py`
- Modify: `src/pm_traitbench/tables/specs.py`
- Test: `tests/tables/test_schema.py` (append), `tests/tables/test_specs.py` (create), `tests/test_errors.py` (append)

**Interfaces:**
- Produces enums in `enums.py` (`StrEnum`, value = snake_case name): `Side` (`BUY="buy"`, `SELL="sell"`); `Expression` (`OUTRIGHT`, `PAIR`, `CURVE`, `CALENDAR_SPREAD`); `RuleResponse` (`ACTED`, `ACKED_NO_ACTION`, `ADDED`, `OVERRIDDEN`); `PositionAction` (`NONE`, `HOLD`, `ADD`, `CUT`, `TRIM`, `EXIT`, `ROLL`); `PnlState` (`GAIN`, `LOSS`, `FLAT`). Module constant `MULTI_LEG_FORMS: frozenset[Expression] = {PAIR, CURVE, CALENDAR_SPREAD}`.
- Produces `EngineError(PmTraitbenchError)`, `exit_code = 1`, docstring "Raised when the behaviour engine breaks one of its own invariants."
- Produces row models (frozen, `extra="forbid"`, every field with `description`), added to `__all__`:
  - `Leg`: `instrument_id: str`, `tenor: Tenor | None`, `side: Side`, `weight: float` (`gt=0`).
  - `Idea` (id pattern `_IDEA_ID_PATTERN = r"^ti_\d{3,}$"`): `pm_id`, `trade_idea_id`, `instrument_id: str`, `expression: Expression`, `side: Side`, `legs: tuple[Leg, ...]` (`min_length=1`), `entry_date: date`, `exit_date: date | None`, `entry_level: float`, `target_level: float`, `stop_level: float`, `thesis: str`, `outcome: str | None`, `own_signal: float`, `forecast: float`, `interval_lo: float`, `interval_hi: float`, `street_view_at_entry: StreetView`, `conflict: bool`, `followed_street: bool | None`, `conviction: int` (`ge=1, le=5`), `size_rank: int` (`ge=1, le=5`). Validators: `interval_lo < interval_hi`; `followed_street` is null exactly when `conflict` is false; `outcome` is null exactly when `exit_date` is null; `expression in MULTI_LEG_FORMS` exactly when `len(legs) == 2`; every leg of a `curve` or `calendar_spread` idea has a tenor, every leg of an `outright` or `pair` idea has none; `exit_date >= entry_date` when set.
  - `LedgerRow`: `pm_id`, `date`, `trade_idea_id` (idea pattern), `instrument_id: str`, `tenor: Tenor | None`, `instrument_type: InstrumentKind`, `side: Side`, `size: float` (`gt=0`), `risk_amount: float` (`gt=0`), `price_or_yield: float`, `stated_conviction: int` (`ge=1, le=5`), `bias_flag: str | None`, `rule_id: str | None` (rule pattern when set). Validator: `bias_flag`, when set, matches `^[a-z_]+:[a-z_]+(;[a-z_]+:[a-z_]+)*$`.
  - `RuleEvent`: `pm_id`, `rule_id` (rule pattern), `trade_idea_id` (idea pattern), `date_fired: date`, `response: RuleResponse`, `response_date: date`. Validator: `response_date >= date_fired`.
  - `PositionDay`: `pm_id`, `date`, `trade_idea_id`, `pnl_unit: float`, `pnl_z: float`, `pnl_state: PnlState`, `sessions_held: int` (`ge=0`), `triggers_fired: int` (`ge=0`), `trigger_pending: bool`, `action: PositionAction`, `bias_flag: str | None` (same pattern as the ledger), `anchor_level: float | None`, `effective_exit_level: float | None`. Validator: `anchor_level` and `effective_exit_level` are both null or both set; `pnl_state` is `flat` exactly when `abs(pnl_z) < 1e-9`.
- Produces specs: `IDEAS = TableSpec("ideas", Idea, ("pm_id", "trade_idea_id"))`, `LEDGER = TableSpec("ledger", LedgerRow, ("pm_id", "date", "trade_idea_id", "instrument_id", "tenor", "side"))`, `RULE_EVENTS = TableSpec("rule_events", RuleEvent, ("pm_id", "rule_id", "trade_idea_id", "date_fired"))`, `POSITION_DAYS = TableSpec("position_days", PositionDay, ("pm_id", "date", "trade_idea_id"))`, `ENGINE_TABLES = (IDEAS, LEDGER, RULE_EVENTS, POSITION_DAYS)`, and `HIDDEN_COLUMNS: dict[str, tuple[str, ...]] = {"ledger": ("bias_flag", "rule_id"), "ideas": ("own_signal", "forecast", "interval_lo", "interval_hi", "street_view_at_entry", "conflict", "followed_street", "conviction", "size_rank"), "position_days": every PositionDay column except the key}`. A module-level check at import time asserts every listed column exists on the model (so a rename cannot silently unhide a column).

- [ ] **Step 1: Write failing tests**
  - `test_schema.py`: one valid row per new model; `Idea` rejects `interval_lo >= interval_hi`, `followed_street` set without `conflict`, `outcome` without `exit_date`, a `pair` with one leg, a `curve` leg without tenor, an `outright` leg with a tenor, `exit_date` before `entry_date`; `LedgerRow` rejects `size=0`, a bad `bias_flag` like `"herding"` and accepts `"herding:followed_street;overconfidence:oversized"`; `RuleEvent` rejects `response_date` before `date_fired`; `PositionDay` rejects `anchor_level` set alone and `pnl_state=gain` with `pnl_z=0`; each model survives `columns()` introspection and a `to_record` round trip (null `tenor` becomes `None`).
  - `test_specs.py`: `ENGINE_TABLES` names and keys as listed; every `HIDDEN_COLUMNS` entry names a real column; the `position_days` hidden tuple equals its non-key columns.
  - `test_errors.py`: `EngineError` is a `PmTraitbenchError` with exit code 1.
- [ ] **Step 2: Run** `uv run pytest tests/tables tests/test_errors.py -q`. Expected: FAIL with ImportError on the new names.
- [ ] **Step 3: Implement** per the Interfaces block.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add engine enums, row models and table specs`

---

### Task 2: Stage `appends` contract

**Files:**
- Modify: `src/pm_traitbench/stages.py`
- Test: `tests/test_stages.py` (append)

**Interfaces:**
- Consumes: `Stage`, `run_stage`, `DataStore`.
- Produces: `Stage.appends: tuple[TableSpec, ...] = ()`. In `run_stage`: every `appends` table must exist before the run (missing -> `StageIOError` naming it, same message shape as `reads`); it is not subject to the `--force` existence check; before the run its rows are read and kept as records; after the run the table must have been written by this store (`was_written`) and every pre-existing record must appear unchanged among the written records (compare `to_record` dicts), else `StageIOError("stage '<name>' altered pre-existing rows of table(s): <names>")`. Reason: stage 3 adds idea-scope rows to stage 1's `rules` table and must not touch PM-scope rows.

- [ ] **Step 1: Write failing tests** in `tests/test_stages.py`: a fake stage with `appends=(RULES,)` whose `run` reads `RULES` and rewrites it with one extra idea-scope row passes and the table has both rows; the same stage when `RULES` is missing raises `StageIOError` matching `rules` and never calls `run`; a stage that rewrites the table dropping the original row raises matching `altered`; a stage that rewrites the original row with a changed `text` raises; an `appends` table is not blocked by the no-force existence check; a run that does not write the appends table raises the existing "did not write" error.
- [ ] **Step 2: Run** `uv run pytest tests/test_stages.py -q`. Expected: FAIL with `TypeError: unexpected keyword argument 'appends'`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: let a stage append rows to an existing table`

---

### Task 3: Engine config, constants, shared round-level helpers

**Files:**
- Create: `src/pm_traitbench/engine/__init__.py` (docstring: `"""Stage 3: the deterministic behaviour engine that writes each PM's ledger."""`)
- Create: `src/pm_traitbench/engine/constants.py`
- Create: `src/pm_traitbench/market/levels.py`
- Modify: `src/pm_traitbench/market/synthetic/processes/common.py` (import the three helpers from `market/levels.py` and re-export; no behaviour change)
- Modify: `src/pm_traitbench/market/constants.py` (`CommoditySpec` gains `multiplier: float`)
- Modify: `src/pm_traitbench/config.py`
- Test: `tests/test_config.py` (append), `tests/engine/test_constants.py` (create), `tests/market/test_levels.py` (create)

**Interfaces:**
- Produces `market/levels.py`: `log_grid_step(price)`, `nearest_level(value, step)`, `round_log_gap(price, step)` moved verbatim from `processes/common.py`, plus `YIELD_STEP_PCT = 0.25`, `SPREAD_STEP_BP = 10.0` (the grid steps the rates and credit processes already use as literals; the processes keep their literals, this module just names them for the engine).
- Produces `CommoditySpec.multiplier` (currency per one contract per one unit of quoted price; design values from exchange contract specs): crude 1000, brent 1000, gasoil 100, gasoline 42000, heating_oil 42000, coal 1000, copper 25, aluminium 25, nickel 6, zinc 25, gold 100, silver 5000, platinum 50, wheat 50, corn 50, soybeans 50, sugar 1120, coffee 375, cotton 500, cocoa 10. Add `CONTRACT_MULTIPLIER: dict[str, float]` keyed by code (`"CRD"` ...) built from the table.
- Produces `engine/constants.py`, each with a one-line comment giving its basis:
  - `ENTRY_THRESHOLD = 1.0`, `CONVICTION_CUTS = (1.0, 1.14, 1.31, 1.53, 1.86)` (quintiles of a standard normal's absolute value given it exceeds 1), `SIGNPOSTS_PER_IDEA = (2, 3)`, `ADD_FRACTION = 0.5`, `RISK_STEPS = (0.2, 0.4, 0.6, 0.8, 1.0)`, `NO_ENTRY_LAST_SESSIONS = 5`, `TRAILING_SD_DAYS = 60`, `MIN_SD_DAYS = 5`, `SD_FLOOR = 1e-6`, `LEVEL_WINDOW_CHOICES = (3, 4, 5)`, `TRAILING_HIGH_DAYS = 60`.
  - `DV01_PER_MILLION: dict[Tenor, float] = {Y2: 190, Y5: 450, Y10: 800, Y30: 1700}`.
  - `OUTRIGHT_TENOR = Tenor.Y10`, `CURVE_PAIRS = ((Tenor.Y2, Tenor.Y10), (Tenor.Y5, Tenor.Y30))`, `CALENDAR_BACK_TENORS = (Tenor.M4, Tenor.M5, Tenor.M6)`.
  - `BIAS_FLAG_ORDER = ("herding", "overconfidence", "conviction")` for joining entry flags.
- Produces `EngineConfig(BaseModel)` (frozen, `extra="forbid"`), attached as `Config.engine`:

  | Field | Default | basis | note |
  |---|---|---|---|
  | `horizon_days: int` (`ge=5`) | 20 | guess | forward window of the own signal, trailing window for vol and extrapolation |
  | `skill: float` (`ge=0, lt=1`) | 0.15 | guess | correlation of the own signal with the realised move; a small positive edge |
  | `arrival_rate: float` (`gt=0`) | 0.6 | guess | candidate attempts per day; sets ideas per PM-year (about 48 at default) |
  | `rr_range: tuple[float, float]` | (1.5, 3.0) | guess | target distance as a multiple of stop distance |
  | `signpost_k: float` (`gt=0`) | 1.0 | guess | level and relative signposts sit at k times the horizon vol from entry |
  | `preferred_form_weight: float` (`gt=0, le=1`) | 0.7 | guess | share of ideas in the preferred expression when a mapped preference is held |
  | `softmax_tau: float` (`gt=0`) | 1.0 | guess | temperature of the loss-side action draw in vol units |
  | `base_hazard: float` (`gt=0, le=1`) | 0.03 | guess | daily discretionary sell hazard at zero progress, about one in 33 days |

  Validators: `rr_range` ascending and `rr_range[0] >= 1`. `Config` cross-check: `engine.horizon_days * 4 <= calendar.n_weeks * 5`, message `engine.horizon_days must be at most a quarter of the horizon in trading days`.

- [ ] **Step 1: Write failing tests**
  - `test_config.py`: `Config().engine.arrival_rate == 0.6`; `rr_range=(3, 1.5)` rejected; `rr_range=(0.5, 2)` rejected; `horizon_days=70` with `n_weeks=52` rejected with the message; every `engine` leaf appears in `dump_with_basis()` with a basis and non-empty note.
  - `test_constants.py`: `CONTRACT_MULTIPLIER` has one entry per commodity code in `COMMODITIES`, all positive; `DV01_PER_MILLION` covers `SOVEREIGN_TENORS`; `CONVICTION_CUTS` ascending with first equal to `ENTRY_THRESHOLD`; `RISK_STEPS` ascending in (0, 1].
  - `test_levels.py`: `log_grid_step(72) == 5`, `nearest_level(4.13, 0.25) == 4.25`, `round_log_gap` is 0 on a grid level; the synthetic `processes/common.py` still exposes the three names (import test).
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py tests/engine/test_constants.py tests/market/test_levels.py -q`. Expected: FAIL (no `engine` attribute, ImportError).
- [ ] **Step 3: Implement.** `tests/engine/__init__.py` is not needed (importlib mode); create `tests/engine/` directory with the test file.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add engine config, constants and shared round-level helpers`

---

### Task 4: Signpost and thesis catalogues

**Files:**
- Create: `src/pm_traitbench/catalogues/signposts.yaml`
- Create: `src/pm_traitbench/catalogues/theses.yaml`
- Modify: `src/pm_traitbench/catalogues/models.py`
- Modify: `src/pm_traitbench/catalogues/loader.py`
- Test: `tests/catalogues/test_loader.py` (append), `tests/catalogues/test_shipped.py` (append)

**Interfaces:**
- Produces YAML shapes:
  - `signposts.yaml`: top key `signposts`, then `{asset_class: {event: [templates], level: [templates], relative: [templates]}}` for `equities`, `rates_credit`, `commodities`, at least 2 templates per cell. Slots: `event` templates use `{event}` (event type with underscores as spaces); `level` templates use `{level}`, `{unit}`, `{window}`; `relative` templates use `{level}`, `{unit}`, `{peer}` (`sector`, `curve`, `rating band`, `commodity group`).
  - `theses.yaml`: top keys `theses` and `outcomes`. `theses: {asset_class: {expression: [templates]}}` for every (asset class, expression) pair the adapters expose: equities `outright, pair`; rates_credit `outright, curve`; commodities `outright, calendar_spread`; at least 2 templates each. Slots: `{side}` (`long`/`short` for outright and pair, `steepener`/`flattener` for curve, `long the front`/`short the front` for calendar spread; templates carry no other direction word), `{name}` (instrument name; for a two-leg idea `{name}` is "A versus B" or "2Y versus 10Y" or "M1 versus M5", built by the renderer), `{entry}`, `{target}`, `{move}` (signed expected move, `format(x, "+.1f")`), `{unit}`, `{horizon}` (days). `outcomes: {win: [templates], loss: [templates], open: [templates]}` with slots `{pnl}` (signed, `"+.1f"`), `{unit}`, `{closer}` (a phrase from `CLOSER_PHRASES` in the loader keyed by `stop`, `target`, `signpost`, `trim_at_target`, `roll`, `discretionary`, `horizon_end`; templates assert no cause of their own).
- Produces models: `SignpostTemplates(BaseModel)` with `event: tuple[str, ...]`, `level: tuple[str, ...]`, `relative: tuple[str, ...]` (each `min_length=2`); `ThesisTemplates(BaseModel)` with `theses: dict[AssetClass, dict[Expression, tuple[str, ...]]]` and `outcomes: dict[str, tuple[str, ...]]` (keys exactly `win`, `loss`, `open`, validator). `Catalogue` gains `signposts: dict[AssetClass, SignpostTemplates]` and `theses: ThesisTemplates`.
- Produces loader changes: `_FILE_NAMES` gains the two files; `check_catalogue` gains `_check_engine_templates(catalogue)`: every direct asset class (equities, rates_credit, commodities) has signposts and theses for every expression in `ADAPTER_FORMS` (a constant `dict[AssetClass, tuple[Expression, ...]]` defined in `catalogues/models.py` so the loader does not import the engine: `{EQUITIES: (OUTRIGHT, PAIR), RATES_CREDIT: (OUTRIGHT, CURVE), COMMODITIES: (OUTRIGHT, CALENDAR_SPREAD)}`); every template's slots are a subset of the allowed slots for its cell (use `string.Formatter().parse`); no template is blank.
- Produces `render_signpost(template, *, level=None, unit=None, window=None, event=None, peer=None) -> str` and `render_thesis(template, **slots) -> str` in `loader.py` (levels formatted with `format(x, "g")`, event underscores to spaces).

- [ ] **Step 1: Write failing tests**
  - `test_loader.py`: a temp directory catalogue with a signpost cell of one template fails `min_length`; a thesis template with an unknown slot `{foo}` fails `check_catalogue` naming the cell; an `outcomes` key `draw` is rejected; `render_signpost` fills every slot and turns `rating_downgrade` into `rating downgrade`.
  - `test_shipped.py`: the packaged catalogue passes `check_catalogue`; every direct asset class has all three signpost kinds; every `ADAPTER_FORMS` cell has a thesis template.
- [ ] **Step 2: Run** `uv run pytest tests/catalogues -q`. Expected: FAIL (missing files, ImportError).
- [ ] **Step 3: Implement.** Write the YAML in the PM's voice, short, one clause each, in the same register as `rules.yaml`. Examples of the level: signpost level equities "if it holds under {level} for {window} sessions the setup is broken"; thesis rates curve "{name} at {entry}{unit}, looking for {move}{unit} over {horizon} sessions, target {target}"; outcome loss "loss: closed {pnl}{unit}, stopped by {closer}".
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add signpost and thesis template catalogues`

---

### Task 5: Market view and engine test fixtures

**Files:**
- Create: `src/pm_traitbench/engine/series.py`
- Create: `src/pm_traitbench/engine/market_view.py`
- Create: `tests/engine/conftest.py`
- Test: `tests/engine/test_market_view.py`

**Interfaces:**
- Produces `engine/series.py`: `LegRef(instrument_id: str, tenor: Tenor | None, coeff: float)` and `Series(legs: tuple[LegRef, ...], bullish_sign: int, unit: str)`, both frozen dataclasses; `Series.__post_init__` rejects `bullish_sign not in (-1, 1)` and `unit not in ("pct", "bp")`.
- Produces `MarketView` (class, built once per seed):
  - `MarketView.build(seed: str, dates: Sequence[date], instruments: Sequence[Instrument], prices: Sequence[Price], curves: Sequence[CurvePoint], consensus: Sequence[ConsensusRow], calendar: Sequence[CalendarEvent], regimes: Sequence[RegimeSpan]) -> MarketView`; `dates` is the horizon axis in order (from `build_axis(...).dates[axis.horizon]`); rows for other seeds are ignored; a horizon date with no price for an instrument that has any price raises `EngineError` naming the instrument and date.
  - Attributes: `seed`, `dates: tuple[date, ...]`, `n_days`, `instruments: dict[str, Instrument]`.
  - `raw_level(instrument_id, tenor, t) -> float`: equity or commodity with `tenor None` -> `100 * ln(price)`; commodity with a futures tenor -> `100 * ln(curve level)`; sovereign curve tenor -> `100 * curve level` (bp); credit -> `spread_bp`. Unknown combination raises `EngineError`.
  - `level(series, t) -> float = sum(coeff * raw_level(...))`.
  - `daily_moves(series, t) -> np.ndarray`: differences of `level` over days `max(0, t - TRAILING_SD_DAYS) .. t`.
  - `sd_h(series, t, H) -> float`: sample sd of `daily_moves` when at least `MIN_SD_DAYS` moves exist, else the sd over whatever exists (at least 2), times `sqrt(H)`, floored at `SD_FLOOR`; on `t < 2` use the sd over days `0..min(TRAILING_SD_DAYS, n_days - 1)` (look ahead only for vol, a design choice documented in the docstring so day 0 has a scale).
  - `trailing_move(series, t, H) = level(t) - level(max(t - H, 0))`; `forward_move(series, t, H) = level(min(t + H, n_days - 1)) - level(t)`; `forward_days(t, H) = min(t + H, n_days - 1) - t`.
  - `trailing_high(series, t, days) -> float`: max of `level` over `max(0, t - days) .. t` (used as the "prior high" anchor; for a bearish series the engine takes the low, so also `trailing_low`).
  - `street_view(instrument_id, t) -> StreetView | None`, `positioning(instrument_id, t) -> Positioning | None`.
  - `events_on(instrument_id, t) -> frozenset[EventType]` (instrument rows plus market-wide rows), `event_types(instrument_id) -> frozenset[EventType]` (instrument rows only, over the whole seed, excluding `contract_expiry`, `positioning_report`, `consensus_flip`).
  - `regime(t) -> Regime` from the seed's spans; a date outside every span raises `EngineError`.
  - `days_to_expiry(instrument_id, t) -> int | None`: sessions until the next `contract_expiry` row for that instrument at or after `t` (0 on the expiry day); `None` when there is no later expiry row or the instrument is not a commodity.
  - `is_expiry_day(instrument_id, t) -> bool`.
  - `peer_move(instrument_ids, series_for, t, H) -> float`: mean of `trailing_move` over the given ids' outright series (`series_for(instrument_id) -> Series` supplied by the adapter).
- Produces `tests/engine/conftest.py` fixtures (module scope):
  - `fixture_market() -> dict` with row lists for seed `"T"`: 60 horizon weekdays from 2026-01-05; instruments `EQ-0001..EQ-0003` (sector `sector_01`, beta 1.0), `EQ-0004` (sector `sector_02`), `RT-USD`, `CR-IG-001` (AA, sector_01, duration 6.0), `CR-IG-002` (BBB), `CM-CRD` (energy), `CM-GLD` (precious); prices as deterministic random walks from `np.random.default_rng(7)` with daily vol 1.5% (equities), 1% (commodities), yields 4 tenors around 4% with 5bp daily moves, spreads 60 and 110 with 2bp moves; curves for `RT-USD` tenors and `CM-*` `M1..M12` (`M_k = M1 * (1 + 0.02 * (k - 1) / 12)`); consensus rows daily for every instrument except `RT-USD` (street score from a fixed pattern that flips sign on day 30 with a `consensus_flip` calendar row); calendar: `earnings` for `EQ-0001` on days 10 and 40, `cb_meeting` for `RT-USD` on day 20, `inventory_report` for `CM-CRD` on day 15, `contract_expiry` for both commodities on the third Friday of each month, one `macro_print` on day 25; regimes: `range` days 0-29, `risk_off` days 30-59.
  - `fixture_view(fixture_market) -> MarketView`.
  - `fixture_instruments(fixture_market) -> list[Instrument]`.
  - `neutral_pm(asset_class, sub_style) -> tuple[Persona, list[Trait], list[Rule]]` factory: a persona on seed `"T"` with `book_size 1e8`, eight bias traits at their neutral medians (from `Config().biases.params[p].neutral.median_value()`) and `active False`, multipliers 1.0, no preferences; rules: `max_risk_pct` cap 10, `stop_loss` (equities -10 pct on `pnl_from_entry`; sovereign_rates 20 bp on `adverse_yield_move_bp`; long_short_credit 30 bp on `adverse_spread_move_bp`; commodities -10 pct), `trim_at_target`, `no_add_before_trigger`, `min_holding_period` 5 sessions, `max_positions` 20, and for commodities `roll_before_expiry` 5. Rule rows use the same fields, ops and units as `rules.yaml`.
  - `engine_config() -> Config`: `Config()` with defaults; engine functions take the seed name and the view directly, so the fixture seed `"T"` never has to exist in the market config.

- [ ] **Step 1: Write failing tests** in `test_market_view.py`: `raw_level` per kind matches `100 * ln(price)`, `100 * yield`, `spread_bp`; `level` of a steepener series equals `100 * (y10 - y2)`; `sd_h` at `t=0` is positive and finite, at `t=59` uses 60 trailing moves; `trailing_move` at `t=3, H=20` uses day 0; `forward_move` at `t=55, H=20` uses day 59 and `forward_days == 4`; `street_view` for `RT-USD` is `None`; `events_on` day 25 includes `macro_print` for every instrument; `event_types("EQ-0001") == {earnings}`; `event_types("CM-CRD")` excludes `contract_expiry`; `days_to_expiry` counts down to 0 on the expiry day and is `None` after the last expiry; `regime(29) == range`, `regime(30) == risk_off`; a missing price for a listed instrument raises `EngineError`; `peer_move` over the three sector_01 equities equals the mean of their trailing moves.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_market_view.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.** Arrays indexed `[instrument][t]`; store `dict[(instrument_id, tenor | None), np.ndarray]` of raw levels built once in `build`.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add engine market view and fixtures`

---

### Task 6: Engine state and effective parameters

**Files:**
- Create: `src/pm_traitbench/engine/state.py`
- Create: `src/pm_traitbench/engine/params.py`
- Test: `tests/engine/test_state.py`, `tests/engine/test_params.py`

**Interfaces:**
- Produces `state.py` (all frozen dataclasses):
  - `Position`: `trade_idea_id: str`, `expression: Expression`, `instrument_id: str`, `legs: tuple[Leg, ...]`, `series: Series`, `side: Side`, `entry_t: int`, `entry_level: float`, `target_level: float`, `stop_level: float`, `sd_h_at_entry: float`, `forecast: float`, `size_pct_book: float`, `original_size_pct_book: float`, `conviction: int`, `size_rank: int`, `triggers_fired: int`, `consumed_rule_ids: frozenset[str]` (idea-scope rules that fired or were breached, plus PM-scope `trim_at_target` once fired), `run_counters: tuple[tuple[str, int], ...]` (rule_id, consecutive days condition held), `rolled_offset: float` (cumulative roll shifts folded into `entry_level`; 0 for non-commodities), `roll_breached: bool` (a roll rule fired and was not honoured), `size_changed_t: int` (day the size last changed). Helpers: `side_sign`, `adverse_dir` (per planning decision 3), `counter(rule_id) -> int`, `with_counter(rule_id, n) -> Position`.
  - `PmState`: `pm_id`, `positions: tuple[Position, ...]` (sorted by `trade_idea_id`), `next_idea: int` (1-based), `next_rule: int` (continues after the PM's highest `r_NN`), `n_positions` property. Helpers: `position(trade_idea_id)`, `replace_position(pos)`, `remove_position(trade_idea_id)`, `add_position(pos)`; `held_instruments -> frozenset[str]`.
  - `idea_id(n) -> f"ti_{n:03d}"`, `rule_id(n) -> f"r_{n:02d}"`.
- Produces `params.py`:
  - `EffectiveParams`: frozen dataclass with `values: dict[str, float]` (every `BIAS_PARAMS` name), `active: dict[str, bool]`, and `value(param)`, `is_active(param)`.
  - `ParamSchedule.build(traits: Sequence[Trait], drift_events: Sequence[DriftEvent], config: Config) -> ParamSchedule`: keeps base values, activity, the per-regime multipliers per param and the drift events sorted by date; `for_day(day: date, regime: Regime) -> EffectiveParams`. Rules (binding): apply drift events with `date <= day` in date order: `update` sets the value to `to`; `dormant` sets it to `config.biases.params[param].neutral.median_value()` and remembers the pre-dormant value; `revive` restores it. Then multiply by `mult_<regime>`: params in `LOGIT_PARAMS = {anchoring_rho, extrapolation_theta, herding_weight, conviction_size_miscalibration, exit_deficiency}` on the logit scale (`x' = sigmoid(logit(x) * m)`? no: `x' = sigmoid(logit(x) + ln(m))`, so `m=1` is identity and the result stays in (0, 1)); `overconfidence_coverage` the same on `1 - c` (lower is stronger); `loss_aversion_lambda` and `disposition_ratio` by plain multiplication. Values are clipped to `[1e-6, 1 - 1e-6]` for [0, 1] params.
- Consumes: `Trait`, `DriftEvent` (`from_value`/`to_value` attributes), `Regime`, `BIAS_PARAMS`.

- [ ] **Step 1: Write failing tests**
  - `test_state.py`: `adverse_dir` for the four sign combinations; `with_counter` returns a new object and leaves the old unchanged; `PmState.add_position` keeps order by id; `remove_position` of an unknown id raises `EngineError`; `idea_id(7) == "ti_007"`.
  - `test_params.py`: no events -> base values; an `update` applies from its date, not before; `dormant` sets the neutral median and `revive` restores the pre-dormant value; a multiplier of 1.0 is identity on every param; a multiplier 1.3 on `herding_weight` 0.5 gives `sigmoid(ln 1.3)` about 0.565 and stays below 1 for 0.99; multiplier 1.3 on `overconfidence_coverage` 0.4 lowers it; multiplier on `loss_aversion_lambda` 2.0 gives 2.6; events on a param that is inactive still apply to the value (activity unchanged).
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_state.py tests/engine/test_params.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add engine state and effective bias parameters`

---

### Task 7: Own signal, forecast, interval, conviction

**Files:**
- Create: `src/pm_traitbench/engine/own_signal.py`
- Create: `src/pm_traitbench/engine/biases/__init__.py` (empty docstring module for now; the registry is filled in Tasks 13 and 14)
- Create: `src/pm_traitbench/engine/biases/extrapolation.py`
- Test: `tests/engine/test_own_signal.py`

**Interfaces:**
- Produces:
  - `SignalDraw` frozen dataclass: `own_signal: float`, `sd_h: float`, `thesis_move: float`, `forecast: float`, `interval_lo: float`, `interval_hi: float`, `conviction: int`.
  - `draw_signal(view: MarketView, series: Series, t: int, params: EffectiveParams, config: Config, rng: np.random.Generator) -> SignalDraw`. Binding formulas: `H = config.engine.horizon_days`; `sd = view.sd_h(series, t, H)`; `fd = view.forward_days(t, H)`; `sd_fwd = sd * sqrt(fd / H)` (truncated forward window); `z = series.bullish_sign * view.forward_move(series, t, H) / sd_fwd` (0 when `fd == 0`; `forward_move` is in series units, `z` in bullish units); `own_signal = skill * z + sqrt(1 - skill^2) * n` with `n` one standard normal from `rng`; `thesis_move = bullish_sign * own_signal * sd` (the signal is in bullish units; the thesis move is in series units, so a bearish series flips); `trailing = view.trailing_move(series, t, H)`; `forecast = extrapolation.blend(thesis_move, trailing, params)` where `biases/extrapolation.py` defines `blend(thesis_move, trailing_move, params) -> float = (1 - theta) * thesis_move + theta * trailing_move` with `theta = params.value("extrapolation_theta")` (the bias lives in its own module so the registry can point at it); `c = params.value("overconfidence_coverage")`; `z_c = scipy.stats.norm.ppf((1 + c) / 2)`; `centre = series.bullish_sign * skill * own_signal * sd_fwd` (the honest conditional mean of the forward move given the signal, in series units); half-width `w = z_c * sd_fwd * sqrt(1 - skill^2)`; `interval_lo, interval_hi = centre - w, centre + w`, so realised coverage equals `c` by construction for any `fd`; `conviction = 1 + sum(1 for cut in CONVICTION_CUTS[1:] if abs(own_signal) >= cut)` (five cuts give buckets 1-5, bucket 1 covers `[1.0, 1.14)`).
  - `z_for_coverage(c: float) -> float` and `Z_80 = z_for_coverage(0.8)` (about 1.2816).
  - `signal_sign(draw) -> Side` (`buy` when `own_signal > 0`).
- Consumes: `MarketView`, `Series`, `EffectiveParams`.

- [ ] **Step 1: Write failing tests**: over 20,000 draws on a fixed series and `t` (fresh `stream(1, "test", i)` each) the correlation of `own_signal` with the series' `z` (recompute `z` in the test) is within 0.02 of `skill` for `skill` in {0.0, 0.15, 0.5}; with `c = 0.8` (any `theta`) the realised forward move falls inside `[interval_lo, interval_hi]` in 0.80 +- 0.02 of 20,000 draws at `fd == H` and again at `fd == 5` where the forward move is simulated as `z * sd_fwd` with `z` from the same joint draw (construct the joint draw in the test: draw `z ~ N(0,1)`, build a synthetic `MarketView`-free path by monkeypatching `forward_move`, or use a stub view object with the needed methods); `theta = 1` makes `forecast == trailing`; a bearish series flips `thesis_move` sign; conviction buckets: 1.0 -> 1, 1.13 -> 1, 1.14 -> 2, 1.86 -> 5, 5.0 -> 5.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_own_signal.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.** A stub view for the tests is fine: any object with `sd_h`, `forward_days`, `forward_move`, `trailing_move`.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the engine's own signal and forecast`

---

### Task 8: Condition grammar evaluator

**Files:**
- Create: `src/pm_traitbench/engine/rules_eval.py`
- Test: `tests/engine/test_rules_eval.py`

**Interfaces:**
- Produces:
  - `FieldValues = Mapping[str, float | int | str | frozenset[str]]` (a field's value; `event` maps to the set of event type values on the day).
  - `condition_holds(rule: Rule, fields: FieldValues) -> bool`: numeric ops compare `float(fields[rule.field])` with `float(rule.level)`; string levels use `==` / `!=` on strings; when the field value is a `frozenset`, `==` means membership and `!=` means absence. A field missing from `fields` raises `EngineError` naming the rule id and field.
  - `advance(position: Position, rule: Rule, holds: bool) -> tuple[Position, bool]`: updates the run counter (`n + 1` when holds, else 0) and returns `(new_position, fired)` where `fired = holds and counter_new >= rule.window`. Idea-scope rules and `trim_at_target` never fire again once in `consumed_rule_ids` (returns `fired=False` without touching the counter).
  - `evaluate_day(position: Position, rules: Sequence[Rule], fields: FieldValues) -> tuple[Position, list[Rule]]`: applies `advance` for every rule whose `trade_idea_id` is `None` (PM scope, evaluated daily: those with `param in DAILY_PM_PARAMS = {"trim_at_target", "min_holding_period", "roll_before_expiry"}`) or equals the position's id (idea scope), in `rule_id` order; returns the fired rules. Rules with other params (`stop_loss`, `max_risk_pct`, `no_add_before_trigger`, `max_positions`, `exclusion`) are skipped (constraints or sources, per planning decision 4).
  - `required_fields(rules) -> frozenset[str]`: every `field` of a rule that `evaluate_day` would evaluate, for the adapter check in the stage.

- [ ] **Step 1: Write failing tests**: each `Op` on numbers; `sector != energy` on a string; `event == earnings` against a set with and without it; a `window 3` rule fires on the third consecutive holding day and resets after a break; an idea-scope rule in `consumed_rule_ids` never fires; `stop_loss` and `max_risk_pct` PM rules are skipped; a missing field raises `EngineError` naming rule and field; `required_fields` excludes skipped params.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_rules_eval.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the rule condition evaluator`

---

### Task 9: Action precedence

**Files:**
- Create: `src/pm_traitbench/engine/precedence.py`
- Test: `tests/engine/test_precedence.py`

**Interfaces:**
- Produces:
  - `RANK: dict[Action, int] = {EXCLUDE: 0, CAP: 0, EXIT: 1, SIGNPOST: 1, ROLL: 2, TRIM_HALF: 3, TARGET: 3, HOLD: 4, NO_ADD: 5}` (lower wins).
  - `Resolution` frozen dataclass: `winner: Rule | None`, `also_acted: tuple[Rule, ...]`, `overridden: tuple[Rule, ...]`.
  - `resolve(fired: Sequence[Rule]) -> Resolution`, binding table:
    - No rules -> `winner None`.
    - Among rules with the minimum rank: if several have rank 1 (stop and signposts, or two signposts), the winner is the rule whose `param == "stop"` if present else the lowest `rule_id`; the others are `also_acted` (one event each, one exit).
    - Rank 2 (`roll`) with rank 3 (`target` or `trim_half`) both present: winner is the roll, the rank-3 rules go to `also_acted` (the trim executes on the rolled position the same day).
    - `target` and `trim_half` together: winner is `trim_half`, `target` in `also_acted`.
    - Every rule with a rank higher than the winner's (and not in `also_acted`) is `overridden`; this includes `hold` whenever anything else fires.
    - `hold` alone: `winner` is the hold rule (it produces no action; the caller treats a `hold` winner as "no trigger" and writes no event, see Task 14).
  - `is_exit(rule) -> bool` (`action in (EXIT, SIGNPOST)`).

- [ ] **Step 1: Write failing tests** (build minimal `Rule` rows): every row of the table above as a parametrised case; ordering of `also_acted` and `overridden` by `rule_id`; a `no_add` rule is always overridden (it should never be passed, but if it is, it loses).
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_precedence.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add rule action precedence`

---

### Task 10: Adapter protocol and equities adapter

**Files:**
- Create: `src/pm_traitbench/engine/adapters/__init__.py`
- Create: `src/pm_traitbench/engine/adapters/base.py`
- Create: `src/pm_traitbench/engine/adapters/equities.py`
- Test: `tests/engine/adapters/test_equities.py`, `tests/engine/adapters/test_registry.py`

**Interfaces:**
- Produces `base.py`:
  - `Adapter(Protocol)`:
    - `asset_class: AssetClass`
    - `universe(self, instruments: Mapping[str, Instrument], rules: Sequence[Rule]) -> tuple[str, ...]`: candidate ids in sorted order after applying every PM-scope `exclusion` rule (`field` in `sector | rating_band | commodity_group`, `op !=`, `level` the excluded value).
    - `forms(self, sub_style: str) -> tuple[Expression, ...]`.
    - `build_legs(self, form: Expression, instrument_id: str, view: MarketView, t: int, universe: Sequence[str], rng: np.random.Generator) -> tuple[LegRef, ...] | None` (coefficients only; `None` when a two-leg form has no partner).
    - `series(self, form: Expression, legs: tuple[LegRef, ...]) -> Series`.
    - `outright_series(self, instrument_id: str) -> Series` (for peers and relative signposts).
    - `stop_distance(self, rule: Rule, series: Series) -> float`: the PM stop rule turned into a distance in series units (equities: `abs(level)` pct; rates_credit: `level` bp; commodities: `abs(level)` pct). Raises `EngineError` when the stop rule's field is not one the adapter serves.
    - `position_fields(self, pos: Position, view: MarketView, t: int, state: PmState, pnl_unit: float) -> dict[str, float | int | str | frozenset[str]]`: every field name in the adapter's `FIELDS` constant.
    - `FIELDS: frozenset[str]` (class attribute) = the rule fields this adapter can evaluate; the stage checks every PM rule against it.
    - `size_and_risk(self, size_pct_book: float, legs: tuple[LegRef, ...], view: MarketView, t: int, book_size: float) -> tuple[float, float]` returning `(size in risk unit, risk_amount notional)`.
    - `leg_price(self, leg: LegRef, view: MarketView, t: int) -> float` (`price_or_yield` on the ledger: price for equities and commodities, yield in percent for a tenor, spread in bp for credit).
    - `instrument_type(self, instrument_id: str, view: MarketView) -> InstrumentKind`.
    - `anchors(self, pos: Position, view: MarketView, t: int) -> tuple[float, ...]`: entry level; nearest round level when `view.regime(t) == range`; the trailing 60-day high of the series on the target side (`trailing_high` when `adverse_dir < 0`, else `trailing_low`).
    - `peer_ids(self, instrument_id: str, instruments: Mapping[str, Instrument]) -> tuple[str, ...]` and `peer_label(self, instrument_id) -> str` (`sector`, `curve`, `rating band`, `commodity group`).
    - `round_step(self, series: Series, level: float) -> float`.
    - `leg_bullish(self, leg: LegRef) -> int` (+1 price leg, -1 yield or spread leg).
  - Shared helpers in `base.py` (binding): `pnl_unit(pos, level_now) = pos.series.bullish_sign * pos.side_sign * (level_now - pos.entry_level)`; `leg_side(pos_side_sign, bullish_sign, coeff, leg_bullish) -> Side`; `common_fields(pos, state, t, pnl_unit, target_reached: bool) -> dict` giving `sessions_held = t - pos.entry_t`, `triggers_fired`, `n_positions = state.n_positions`, `size_pct_book`, `target_hit = 1 if target_reached else 0`, `level`; `target_reached(pos, level_now) = (level_now - pos.target_level) * (-pos.adverse_dir) >= 0`; `relative_move(view, adapter, pos, t, H) = bullish_sign * side_sign * (trailing_move(pos.series) - peer_move)` (positive = in the PM's favour).
- Produces `adapters/__init__.py`: `ADAPTERS: dict[AssetClass, Callable[[str], Adapter]]` (factories taking the sub-style) filled by Tasks 10-12 (`equities` now), `adapter_for(asset_class: AssetClass, sub_style: str) -> Adapter` raising `EngineError` for `multi_asset` or an unregistered class; `FORM_FOR_PREFERENCE: dict[tuple[str, str], Expression | None]` with the mapping from the design (param, value) -> form or `None`, covering every value of `duration_expression`, `pair_vs_outright`, `curve_trade_expression`, `hedge_instrument`, `futures_vs_etf`, `fx_hedge_expression`, `credit_index_vs_single_name` exactly as spelled in `preferences.yaml`; `preferred_form(traits: Sequence[Trait]) -> Expression | None` (the first mapped preference in trait order); a test asserts every expression-group value in the shipped catalogue has a mapping entry.
- Produces `equities.py` `EquitiesAdapter`: `FIELDS = {pnl_from_entry, sector, n_positions, sessions_held, size_pct_book, triggers_fired, target_hit, level, event, relative_move}`; forms `(OUTRIGHT, PAIR)` for both sub-styles; `build_legs` outright -> `((id, None, 1.0),)`; pair -> partner = the id in `universe` with the same sector, not the candidate, not held, chosen by the most negative own-signal proxy `-trailing_move` (no second signal draw: the partner is the same-sector name that has moved least in the candidate's favour over H days; ties by id), legs `((cand, None, 1.0), (partner, None, -1.0))`, `None` when no partner exists; series unit `pct`, `bullish_sign +1`; `stop_distance` reads `pnl_from_entry` rules; `pnl_from_entry = pnl_unit`; `sector` = the candidate's sector; `size_and_risk`: `size = size_pct_book`, `risk_amount = book_size * size_pct_book / 100`; `leg_price` = price; `round_step` = `log_grid_step(price)` converted to series units (`100 * ln(nearest grid level)`: implement `anchors` by rounding the price, then converting); `peer_ids` = same sector; `leg_bullish` +1.

- [ ] **Step 1: Write failing tests**
  - `test_registry.py`: `adapter_for(multi_asset, "global_macro")` raises `EngineError`; `EquitiesAdapter(sub_style)` accepts any sub-style; every expression preference value in the shipped catalogue is a key of `FORM_FOR_PREFERENCE`; `preferred_form` on a trait list with `duration_expression = steepeners over outright duration` is `curve`, with only unmapped preferences is `None`.
  - `test_equities.py` on the fixture view and neutral equities PM: `universe` drops `sector_02` when an exclusion rule names it; pair legs pick a same-sector partner and never the candidate; `pnl_unit` for a buy at entry 100 pct-log units and level 105 is 5, for a sell is -5; `leg_side` for a pair buy gives buy, sell; `position_fields` has every `FIELDS` name and `target_hit` flips when the level crosses the target; `size_and_risk(5, ..., book 1e8)` gives `(5, 5e6)`; `anchors` in range includes the round level and out of range does not; `relative_move` sign flips with side.
- [ ] **Step 2: Run** `uv run pytest tests/engine/adapters -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the adapter protocol and equities adapter`

---

### Task 11: Rates and credit adapter

**Files:**
- Create: `src/pm_traitbench/engine/adapters/rates_credit.py`
- Modify: `src/pm_traitbench/engine/adapters/__init__.py` (register)
- Test: `tests/engine/adapters/test_rates_credit.py`

**Interfaces:**
- Produces `RatesCreditAdapter` with `FIELDS = {adverse_yield_move_bp, adverse_spread_move_bp, pnl_from_entry, rating_band, n_positions, sessions_held, size_pct_book, triggers_fired, target_hit, level, event, relative_move, spread_bp}`.
  - `universe`: sub-style `sovereign_rates` -> ids of kind `sovereign_curve`; `long_short_credit` -> kind `credit_issuer` minus `rating_band` exclusions. Constructor `RatesCreditAdapter(sub_style: str)`, registered as the factory for `rates_credit`.
  - Forms: `sovereign_rates` -> `(OUTRIGHT, CURVE)`; `long_short_credit` -> `(OUTRIGHT,)`.
  - Legs: outright rates -> `((curve_id, Y10, 1.0),)`, series `bullish_sign -1`, unit `bp`; curve -> pick a pair from `CURVE_PAIRS` uniformly with `rng`, legs `((curve_id, long_tenor, 1.0), (curve_id, short_tenor, -1.0))`, series `bullish_sign +1` (a steepener is the bullish side), unit `bp`; credit outright -> `((issuer, None, 1.0),)`, `bullish_sign -1`, unit `bp`.
  - `stop_distance`: reads `adverse_yield_move_bp` (sovereign) or `adverse_spread_move_bp` (credit) rules; level in bp.
  - Fields: `adverse_yield_move_bp = max(-pnl_unit, 0)` (sovereign; 0 for credit), `adverse_spread_move_bp = max(-pnl_unit, 0)` (credit; 0 for sovereign), `pnl_from_entry = pnl_unit * dv01_per_million / 10000` (percent of notional, DV01 of the outright leg or the long-tenor leg; credit uses `duration_years * 100`), `rating_band` (credit; `""` for curves), `spread_bp` (credit level; for a curve the series level).
  - `size_and_risk`: `risk_amount = book_size * size_pct_book / 100`; `size = risk_amount / 1e6 * dv01_per_million(tenor or credit)`; for a curve the DV01 is that of the long-tenor leg and the short-tenor leg's notional is implied (not stored; the ledger row per leg carries the same `size` and `risk_amount` scaled by the DV01 ratio: `risk_amount_leg = size / dv01_per_million(leg) * 1e6`).
  - `leg_price`: yield in percent (`curve level`) for a tenor, spread bp for credit. `leg_bullish` -1.
  - `anchors`: entry; round level in range (`nearest_level(yield_pct, 0.25) * 100` bp for a tenor series; `nearest_level(spread, 10)` for credit; for a curve series the round level of the slope in 25bp steps); trailing 60-day extreme on the target side.
  - `peer_ids`: curve -> the same curve (peer move is the 10Y outright, so `relative_move` for a curve trade is the slope move minus the 10Y move, the "relative to the curve" signpost); credit -> issuers of the same rating band. `peer_label`: `curve` / `rating band`.

- [ ] **Step 1: Write failing tests** on the fixture: sovereign universe is `("RT-USD",)`, credit universe drops `BBB` when excluded; a steepener series level is `100 * (y10 - y2)` with `bullish_sign +1`; `leg_side` for a steepener buy gives 10Y sell, 2Y buy; for a credit buy gives buy; `pnl_unit` for an outright rates buy is positive when yields fall; `adverse_yield_move_bp` is the positive part of `-pnl_unit`; `pnl_from_entry` for 10bp adverse on 10Y is -0.8; `size_and_risk(5, 10Y, book 1e8)` gives `(4000, 5e6)`; the curve short leg's `risk_amount` follows the DV01 ratio; anchors in range include a 25bp-rounded slope for a curve and a 10bp-rounded spread for credit.
- [ ] **Step 2: Run** `uv run pytest tests/engine/adapters -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** and register the factory.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the rates and credit adapter`

---

### Task 12: Commodities adapter

**Files:**
- Create: `src/pm_traitbench/engine/adapters/commodities.py`
- Modify: `src/pm_traitbench/engine/adapters/__init__.py` (register)
- Test: `tests/engine/adapters/test_commodities.py`

**Interfaces:**
- Produces `CommoditiesAdapter(sub_style)` with `FIELDS = {pnl_from_entry, commodity_group, days_to_expiry, price, n_positions, sessions_held, size_pct_book, triggers_fired, target_hit, level, event, relative_move}`.
  - `universe`: kind `commodity` minus `commodity_group` exclusions.
  - Forms: `commodity_futures_directional` -> `(OUTRIGHT,)`; `curve_and_spread` -> `(OUTRIGHT, CALENDAR_SPREAD)`.
  - Legs: outright -> `((id, M1, 1.0),)` unit `pct`, `bullish_sign +1`; calendar spread -> back tenor from `CALENDAR_BACK_TENORS` uniformly, legs `((id, M1, 1.0), (id, Mk, -1.0))`, series `100 * (ln M1 - ln Mk)`, `bullish_sign +1`, unit `pct`.
  - `stop_distance` reads `pnl_from_entry` rules; `pnl_from_entry = pnl_unit`; `days_to_expiry = view.days_to_expiry(id, t)` (an `int`; when `None`, 10_000 so a roll rule never fires); `price` = M1 price; `commodity_group` from the instrument.
  - `size_and_risk`: `risk_amount = book_size * size_pct_book / 100`; `size = max(1, floor(risk_amount / (price_M1 * CONTRACT_MULTIPLIER[code])))` where `code` is the id suffix after `CM-`.
  - `leg_price`: the tenor's curve level. `leg_bullish` +1.
  - `roll_legs(pos, view, t) -> tuple[tuple[LegRef, float], tuple[LegRef, float]]`: `((M1 leg, level_M1), (M2 leg, level_M2))` and `roll_shift(pos, view, t) -> float` = `100 * (ln M2 - ln M1)` for an outright, and for a calendar spread the shift of the spread series when both legs move one month out (`M1 -> M2`, `Mk -> Mk+1`); used by Task 14.
  - `anchors`: entry; round level in range (`log_grid_step` of the M1 price); trailing 60-day extreme.
  - `peer_ids`: same commodity group. `peer_label`: `commodity group`.

- [ ] **Step 1: Write failing tests** on the fixture: universe drops energy when excluded; calendar spread series equals `100 * (ln M1 - ln M5)` for a chosen back tenor; `leg_side` for a spread buy gives M1 buy, Mk sell; `size_and_risk(5, crude at 72, book 1e8)` gives `(69, 5e6)` (`5e6 / 72000 = 69.4`); `days_to_expiry` reaches 0 on an expiry day; `roll_shift` for an outright equals `100 * (ln M2 - ln M1)`; anchors in range include the rounded M1 price in series units.
- [ ] **Step 2: Run** `uv run pytest tests/engine/adapters -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the commodities adapter`

---

### Task 13: Entry-side bias rules and the registry

**Files:**
- Modify: `src/pm_traitbench/engine/biases/__init__.py`
- Modify: `src/pm_traitbench/engine/biases/extrapolation.py`
- Create: `src/pm_traitbench/engine/biases/herding.py`
- Create: `src/pm_traitbench/engine/biases/overconfidence.py`
- Create: `src/pm_traitbench/engine/biases/conviction.py`
- Test: `tests/engine/biases/test_entry_side.py`

**Interfaces:**
- Produces `biases/__init__.py`: `BIAS_RULES: dict[str, ModuleType]` mapping a bias param name to its module; this task registers the four entry-side modules, Task 14 adds the other four and the import-time check that the keys equal `set(BIAS_PARAMS)`.
- Produces `herding.py`: `HerdingDecision(conflict: bool, followed_street: bool | None, side: Side, flag: str | None)`; `decide(own_side: Side, street: StreetView | None, params: EffectiveParams, rng) -> HerdingDecision`. Binding: `street None or neutral` -> no conflict; street `overweight` means bullish; conflict when the street's side differs from `own_side`; on conflict draw `u`; `u < w` -> side becomes the street's, `followed_street True`, `flag = "herding:followed_street"` when `params.is_active("herding_weight")` else `None`; otherwise `followed_street False`.
- Produces `overconfidence.py`: `size_factor(params) -> tuple[float, str | None]` = `(Z_80 / z_for_coverage(c), "overconfidence:oversized" if active and factor > 1 else None)`.
- Produces `conviction.py`: `size_rank(conviction: int, params, rng) -> tuple[int, str | None]`: `u ~ U(1, 5)`, `rank = int(round((1 - m) * conviction + m * u))` clipped to 1-5, flag `"conviction:mis_sized"` when active and `rank != conviction`.
- Produces in `extrapolation.py` (which already holds `blend`): `entered_after_run(trailing_move, sd_h, side_sign, bullish_sign) -> bool` = `bullish_sign * side_sign * trailing_move > sd_h` (the opportunity counter for Gate 1).
- Produces `join_flags(flags: Sequence[str | None]) -> str | None` in `biases/__init__.py`: drops `None`, orders by `BIAS_FLAG_ORDER` prefix, joins with `;`.

- [ ] **Step 1: Write failing tests**: herding with `w=0` never follows, `w=1` always follows on conflict, no conflict when street neutral or `None`, the flag only when active; over 2,000 draws with `w=0.58` the follow share is within 0.05 of 0.58; overconfidence factor is 1 at `c=0.8`, greater than 1 at `c=0.4`, flag only when active; conviction with `m=0` returns the conviction, with `m=1` the rank correlation with conviction over 2,000 draws of random convictions is below 0.15, flag only when active and different; `blend` at `theta` 0 and 1; `entered_after_run` sign cases; `join_flags` ordering and `None`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/biases/test_entry_side.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add entry-side bias rules`

---

### Task 14: Position-side bias rules

**Files:**
- Create: `src/pm_traitbench/engine/biases/exit_deficiency.py`
- Create: `src/pm_traitbench/engine/biases/loss_aversion.py`
- Create: `src/pm_traitbench/engine/biases/disposition.py`
- Create: `src/pm_traitbench/engine/biases/anchoring.py`
- Modify: `src/pm_traitbench/engine/biases/__init__.py` (complete registry and the import-time key check)
- Test: `tests/engine/biases/test_position_side.py`

**Interfaces:**
- Produces `exit_deficiency.py`: `Response(response: RuleResponse, flag: str | None)`; `respond(params, at_loss: bool, rng) -> Response`. Binding: `u < e` -> breach: `added` with flag `"exit_deficiency:added"` when `params.is_active("loss_aversion_lambda") and at_loss`, else `acked_no_action` with flag `None`; otherwise `acted`. `late_roll_flag = "exit_deficiency:late_roll"`.
- Produces `loss_aversion.py`: `LossSideChoice(action: PositionAction, flag: str | None)` with action in `CUT | HOLD | ADD`; `choose(pnl_z: float, forecast_remaining_z: float, params, config, rng, add_allowed: bool) -> LossSideChoice`. Binding: `lam = params.value("loss_aversion_lambda")`, `L = abs(pnl_z)`; `V_cut = -lam * L`; `V_hold = forecast_remaining_z`; `V_add = V_hold * (1 + ADD_FRACTION) - lam * L * ADD_FRACTION`; softmax over the allowed actions (`add` excluded when `add_allowed` is false) at temperature `config.engine.softmax_tau`; draw with `rng.uniform()`. Flags when `loss_aversion_lambda` is active: `add -> "loss_aversion:add"`, `hold -> "loss_aversion:hold"`, `cut -> None`.
- Produces `disposition.py`: `sell_hazard(progress: float, pnl_state: PnlState, params, config) -> float` = `h = base_hazard * (1 + max(progress, 0))`, times `sqrt(D)` at a gain, divided by `sqrt(D)` at a loss, unchanged when flat, clipped to [0, 1]; `draw_sell(h, rng) -> bool`; `flag(pnl_state, progress, sold: bool, params) -> str | None`: active `D` and gain and sold and `progress < 1` -> `"disposition:realise_gain_early"`; active and loss and not sold -> `"disposition:hold_loser"`; else `None`.
- Produces `anchoring.py`: `AnchoredExit(anchor_level: float, effective_exit_level: float, reached: bool)`; `evaluate(pos: Position, level_now: float, anchors: Sequence[float], params) -> AnchoredExit`. Binding: with `fav = -pos.adverse_dir`, candidates are anchors `a` strictly between entry and target, `(a - pos.entry_level) * fav > 0` and `(a - pos.target_level) * fav < 0`; if none, `anchor = pos.target_level`; else the candidate nearest the target (largest `(a - entry) * fav`); `effective = (1 - rho) * pos.target_level + rho * anchor`; `reached = (level_now - effective) * (-pos.adverse_dir) >= 0`. `flag(params) = "anchoring:exit_at_anchor" if active else None`.
- Produces the completed `BIAS_RULES` with the import-time check `set(BIAS_RULES) == set(BIAS_PARAMS)`.

- [ ] **Step 1: Write failing tests** (fixed one-position states, 2,000 draws on `stream(1, "test", i)`): `respond` with `e=0` always `acted`, `e=1` never, `added` only when lambda active and at a loss; with `e=0.06` the breach share over 2,000 draws is within 0.02 of 0.06; `choose` with `lam=1` versus `lam=4` at `pnl_z=-1`, `forecast_remaining_z=0.5`: the add-or-hold share rises and the cut share falls; `add_allowed=False` never returns `add`; `sell_hazard` at `D=1` equals the base at zero progress, at `D=3` gain hazard is `sqrt(3)` times loss hazard; clip at 1; flags per case; `evaluate` with `rho=0` gives `effective == target`, `rho=1` gives `effective == anchor`, `reached` respects direction for buy and sell; no candidate on the target side falls back to the target.
- [ ] **Step 2: Run** `uv run pytest tests/engine/biases -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add position-side bias rules`

---

### Task 15: Idea generation and templates

**Files:**
- Create: `src/pm_traitbench/engine/templates.py`
- Create: `src/pm_traitbench/engine/ideas.py`
- Test: `tests/engine/test_templates.py`, `tests/engine/test_ideas.py`

**Interfaces:**
- Produces `templates.py`: `render_thesis(catalogue, asset_class, expression, *, side: str, name: str, entry: float, target: float, move: float, unit: str, horizon: int, rng) -> str`. `side` per expression: outright and pair `long`/`short`, curve `steepener`/`flattener`, calendar spread `long the front`/`short the front`. For an equities or commodities `outright`, `entry` and `target` are passed as quoted prices (`exp(level / 100)`), since those cells carry no unit; every other cell gets the series level in its unit. (template chosen with `pick_index`-style uniform draw on `rng`; `name` built by `idea_name(view, legs, expression)`: outright -> instrument name; pair -> "A versus B"; curve -> "2Y versus 10Y" using tenor values; calendar spread -> "M1 versus M5"); `render_outcome(catalogue, *, kind: str, pnl: float, unit: str, closer: str, rng) -> str` (`closer` is a `CLOSER_PHRASES` key: `stop`, `target`, `signpost`, `trim_at_target`, `roll`, `discretionary`, `horizon_end`); `render_signpost_text(catalogue, asset_class, kind, ..., rng)`.
- Produces `ideas.py`:
  - `NewIdea` frozen dataclass: `position: Position`, `idea: Idea`, `rules: tuple[Rule, ...]`, `ledger_rows: tuple[LedgerRow, ...]`, `conflict: bool`, `entered_after_run: bool`.
  - `attempt_entry(state: PmState, t: int, view: MarketView, adapter: Adapter, params: EffectiveParams, persona: Persona, pm_rules: Sequence[Rule], traits: Sequence[Trait], universe: Sequence[str], config: Config, catalogue: Catalogue, rng_for: Callable[..., np.random.Generator], attempt: int) -> tuple[PmState, NewIdea | None]`. Binding sequence:
    1. Return `None` when `t >= view.n_days - NO_ENTRY_LAST_SESSIONS`, or a `max_positions` rule exists and `state.n_positions >= level`, or every universe id is held.
    2. Candidate: uniform over `universe` minus `state.held_instruments`, on `rng_for("candidate", t, attempt)`.
    3. Form: `forms = adapter.forms(sub_style)`; `preferred = preferred_form(traits)`; weights `preferred_form_weight` on the preferred form when it is in `forms`, remainder split evenly over the rest; else uniform; draw on `rng_for("form", t, attempt)`. `build_legs` with `rng_for("legs", t, attempt)`; `None` -> fall back to `outright` legs.
    4. `draw_signal` on `rng_for("signal", t, attempt)`; return `None` when `abs(own_signal) < ENTRY_THRESHOLD`.
    5. `own_side = signal_sign`; herding on `rng_for("herding", t, attempt)` with `view.street_view(candidate, t)`; `side` = decision side.
    6. Conviction from the draw; `size_rank` on `rng_for("size_rank", t, attempt)`; `factor` from overconfidence; `cap_level` = the `max_risk_pct` rule level; `size_pct_book = min(cap_level, cap_level * RISK_STEPS[size_rank - 1] * factor)`.
    7. `stop_distance = adapter.stop_distance(stop_rule, series)`; `rr ~ U(rr_range)` on `rng_for("rr", t, attempt)`; `entry_level = view.level(series, t)`; `stop_level`, `target_level` per planning decision 3.
    8. Idea-scope rules with ids from `state.next_rule` onward: `stop` (`param "stop"`, `action EXIT`), `target` (`param "target"`, `action TARGET`), then `n ~ U{2, 3}` signposts on `rng_for("signposts", t, attempt)` drawn without replacement from the kinds available (`event` only when `view.event_types(candidate)` is non-empty; when only two kinds are available and `n == 3`, use two): event -> `field "event"`, `op EQ`, `level` = a uniform event type, `unit None`, `window 1`; level -> `field "level"`, adverse op, `level = entry + adverse_dir * signpost_k * sd`, `unit` = series unit, `window ~ U{3, 4, 5}`; relative -> `field "relative_move"`, `op LE`, `level = -signpost_k * sd`, `window 1`. All `action SIGNPOST`, `param "signpost"`, `source SELF`, `scope IDEA`, `text` from the catalogue. Sd is `draw.sd_h`.
    9. `size, risk_amount = adapter.size_and_risk(...)`; one `LedgerRow` per leg with `side = leg_side(...)`, `tenor`, `instrument_type`, `price_or_yield = adapter.leg_price`, `stated_conviction`, `bias_flag = join_flags([herding, overconfidence, conviction])`, `rule_id None`. For a two-leg rates trade the second leg's `risk_amount` follows the DV01 ratio (Task 11); other two-leg forms carry equal `size` and `risk_amount` per leg.
    10. `Idea` row with `expression`, `legs` (as `Leg` rows: `weight` = `abs(coeff)`), `entry_date = view.dates[t]`, hidden columns from the draw and decisions, `thesis` rendered with `move = draw.forecast`, `horizon = horizon_days`.
    11. New `Position` (`triggers_fired 0`, `consumed_rule_ids` empty, `size_changed_t = t`) and the state with `next_idea`, `next_rule` advanced.
  - `entries_for_day(state, t, ..., rng_for) -> tuple[PmState, list[NewIdea]]`: `n_attempts ~ Poisson(arrival_rate)` on `rng_for("arrival", t)`, loop `attempt_entry` with `attempt = 0..n-1`, stop at the first structural `None` from step 1.

- [ ] **Step 1: Write failing tests**
  - `test_templates.py`: `idea_name` for each expression; rendered thesis contains the entry and target; outcome for `win` contains the signed pnl; a signpost text contains the window.
  - `test_ideas.py` on the fixture (equities neutral PM, forced draws by seeding): no entry in the last 5 sessions; no entry below the threshold (monkeypatch `draw_signal` to return 0.5); with a signal of 2.0 an idea is created with `stop_level` below entry for a buy and `target_level` above, `target - entry == rr * (entry - stop)` within 1e-9; the cap is never exceeded for `factor` 3.0 (patch `c` to 0.3); signposts count is 2 or 3, ids continue from `next_rule`, an instrument without event types gets no `event` signpost; the ledger rows for a pair have opposite sides; `preferred_form(pair)` with weight 1.0 always draws `pair`; the `max_positions` rule stops entries; the returned state has the new position and the input state is unchanged.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_templates.py tests/engine/test_ideas.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add idea generation and thesis templates`

---

### Task 16: The daily step

**Files:**
- Create: `src/pm_traitbench/engine/step.py`
- Test: `tests/engine/test_step.py`

**Interfaces:**
- Produces:
  - `DayOutput` frozen dataclass: `ledger_rows: tuple[LedgerRow, ...]`, `rule_events: tuple[RuleEvent, ...]`, `position_days: tuple[PositionDay, ...]`, `new_ideas: tuple[Idea, ...]`, `idea_rules: tuple[Rule, ...]`, `closed: tuple[tuple[str, date, str], ...]` (idea id, exit date, outcome text), `opportunities: Counter[str]` with keys `loss_side_untriggered_days`, `triggers_fired`, `conflict_entries`, `exits`, `sell_day_position_days`, `ideas`, `entries_after_run`.
  - `PmContext` frozen dataclass bundling what does not change per day: `persona`, `traits`, `pm_rules` (PM scope), `adapter`, `universe`, `catalogue`, `config`, `schedule: ParamSchedule`, `rng_for: Callable[..., Generator]` (`rng_for(purpose, *keys)` = `stream(root, "engine", pm_id, purpose, *keys)`).
  - `step(state: PmState, t: int, view: MarketView, ctx: PmContext, idea_rules: Mapping[str, tuple[Rule, ...]]) -> tuple[PmState, DayOutput]`; `idea_rules` maps a trade idea id to its idea-scope rows (kept by the loop). Binding order per open position (ids ascending), then entries, then close-out on the last day:
    1. `params = ctx.schedule.for_day(view.dates[t], view.regime(t))`.
    2. Mark: `level_now`, `pnl_unit`, `pnl_z = pnl_unit / pos.sd_h_at_entry`, `pnl_state`, `progress = pnl_unit / abs(target - entry)`, `fields = adapter.position_fields(...)`.
    3. `evaluate_day` over `ctx.pm_rules + idea_rules[id]`; `resolve` the fired list. If the winner is a `hold` rule alone: no event, `trigger_pending False`. Otherwise write one `RuleEvent` per fired rule: winner and `also_acted` get the exit-deficiency `respond` draw on `rng_for("response", t, id)` (one draw shared by the winner and `also_acted`, so a breached stop also breaches its signposts); `overridden` rules get `overridden`. Consumed ids: every fired idea-scope rule and a fired `trim_at_target`. `triggers_fired += len(fired non-hold rules)`.
    4. Execute the winner when `acted`: `exit` or `signpost` -> exit the whole position (ledger rows per leg, opposite sides, `rule_id None`, `bias_flag None`), outcome `closer = winner.param`; `roll` -> commodity roll rows per Task 12 (`sell` M1 leg, `buy` M2 leg at their prices; `entry_level += roll_shift`; a `trim_half` or `target` in `also_acted` then executes on the rolled position); `trim_half` or `target` -> when the trim rule is present sell half (`size_pct_book /= 2`, `size_changed_t = t`) else exit whole. When the response is `added`: buy `ADD_FRACTION * original_size_pct_book` more (cap-checked: the add is reduced so `size_pct_book <= cap`; skipped entirely when already at cap, response stays `added` but no row), ledger `rule_id = winner.rule_id`, `bias_flag = "exit_deficiency:added"`. When `acked_no_action`: nothing. Late roll: a commodity position still holding its M1 leg on `view.is_expiry_day` is force-rolled with the same two rows; the rows carry `bias_flag = late_roll_flag` when `pos.roll_breached` is true (set when a roll rule's response was `acked_no_action`), else no flag; no event row either way.
    5. Discretionary block only when no non-hold trigger fired today and the position is still open:
       a. Anchoring: `evaluate` with `adapter.anchors`; if `reached` and `sessions_held >= min_holding` (when the PM has that rule) -> exit, `bias_flag = anchoring flag`, closer `discretionary`; record `anchor_level` and `effective_exit_level` on the position day either way.
       b. Loss side when `pnl_state == loss`: `forecast_remaining_z = (pos.forecast - (level_now - entry) * bullish_sign) * bullish_sign * side_sign / sd_h_at_entry` (the forecast in bullish P&L units minus what has happened); `add_allowed` = not (`no_add_before_trigger` rule held and `triggers_fired == 0`) or a breach draw on `rng_for("no_add_breach", t, id)` passes at `e`; `choose` on `rng_for("loss_side", t, id)`; `cut` -> exit (`bias_flag None`, closer `discretionary`) if `sessions_held >= min_holding`, else hold; `add` -> buy `ADD_FRACTION * original` (cap-checked as above), ledger `rule_id` = the no-add rule id and `bias_flag "loss_aversion:add_before_trigger"` when the add went through a breach, else `rule_id None` and the choice's flag; `hold` -> position-day flag from the choice.
       c. Sell hazard (skipped when b already acted, i.e. cut or add): `h = sell_hazard(...)`, draw on `rng_for("hazard", t, id)`; sell -> exit whole if `sessions_held >= min_holding` (else no sale), `bias_flag = disposition flag`, closer `discretionary`; no sale -> flag `disposition:hold_loser` when applicable.
    6. Position day row per open-at-start position with the action taken (`none` when nothing happened), `trigger_pending`, flags joined (`join_flags` of the day's flags), anchor fields.
    7. Entries: `entries_for_day`; append rows and rules; `opportunities["ideas"] += n`, `conflict_entries`, `entries_after_run`.
    8. Last day (`t == view.n_days - 1`): exit every open position at the close (`bias_flag None`), outcome kind `open`, closer `horizon end`.
    9. Opportunities: `loss_side_untriggered_days` += positions at a loss with no trigger today; `triggers_fired` += fired non-hold rules; `exits` += positions closed today; `sell_day_position_days` += (number of open positions at start of day) when any position was sold today (Odean's position-days on selling days).
  - Outcome kind: `win` when the realised `pnl_unit` at exit `> 0`, `loss` when `<= 0`, `open` on close-out.
- Consumes everything from Tasks 5-15.

- [ ] **Step 1: Write failing tests** on the fixture equities PM with hand-built states (one position each):
  - Determinism: two calls with equal inputs give equal outputs; the input state is unchanged (`state == state_before`).
  - A position whose level has crossed its stop row: one `RuleEvent` `acted`, one ledger sell row, position removed, `closed` has the id with `closer == "stop"`, `opportunities["exits"] == 1`.
  - Stop and a level signpost firing the same day: two events, one exit.
  - `e = 1`: the stop event is `acked_no_action`, the position stays, a later day does not re-fire the consumed stop.
  - `e = 1` with lambda active at a loss: `added`, a buy row with `rule_id` = the stop id and flag `exit_deficiency:added`.
  - Trim: target reached with `trim_at_target`: two events (`target`, `trim_at_target`) `acted`, size halved, one sell row.
  - Commodity roll on `days_to_expiry <= 5`: two ledger rows (sell M1, buy M2), `entry_level` shifted by `roll_shift`, `pnl_unit` continuous across the roll.
  - Hold rule with a stop firing: the hold event is `overridden`, the stop `acted`.
  - Discretionary: with `rho = 1` and the level at the entry-side anchor, exit with `anchoring:exit_at_anchor`; with `D` huge and a gain, a sell with `disposition:realise_gain_early` (force the hazard draw by patching `draw_sell`).
  - Position day rows: one per open position, `trigger_pending` true on the stop day, `action exit`.
  - Last day close-out: every open position exits with outcome kind `open`.
  - Min holding: a discretionary cut on day 2 with `min_holding_period 5` does not exit.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_step.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement.** Keep `step.py` under about 400 lines by moving the trigger block into `engine/triggers.py` (`handle_triggers(...)`) and the discretionary block into `engine/discretionary.py` (`handle_discretionary(...)`), both pure, both tested through `test_step.py`.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the engine daily step`

---

### Task 17: Loop, stage, pipeline, README, end to end

**Files:**
- Create: `src/pm_traitbench/engine/loop.py`
- Create: `src/pm_traitbench/engine/stage.py`
- Modify: `src/pm_traitbench/pipeline.py`
- Modify: `README.md`
- Test: `tests/engine/test_loop.py`, `tests/engine/test_stage.py`, `tests/test_end_to_end.py` (append), `tests/test_cli_stages.py` (append if it lists subcommands)

**Interfaces:**
- Produces `loop.py`:
  - `PmResult` frozen dataclass: `ideas: list[Idea]`, `idea_rules: list[Rule]`, `ledger: list[LedgerRow]`, `rule_events: list[RuleEvent]`, `position_days: list[PositionDay]`, `opportunities: dict[str, int]`.
  - `run_pm(persona, traits, pm_rules, drift_events, view, config, catalogue) -> PmResult`: builds `adapter_for(asset_class, sub_style)`, checks `required_fields(pm_rules) <= adapter.FIELDS` (else `EngineError` naming the rule id and field, before any day), builds `ParamSchedule`, `PmContext`, the initial `PmState` (`next_rule` = highest existing `r_NN` + 1), then iterates `t = 0 .. n_days - 1` calling `step`, collecting rows, keeping `idea_rules` per idea, and finalising `Idea` rows with `exit_date` and `outcome` from `closed` (an idea closed twice raises `EngineError`).
- Produces `stage.py`: `run(config, store) -> dict`: read `PERSONAS, TRAITS, RULES, DRIFT_EVENTS` and the six market tables; group personas by `market_seed`; build `MarketView` per seed with `dates = build_axis(config.timeline(), config.market.burn_in_days).dates[axis.horizon]`; for every persona in `pm_id` order: `multi_asset` -> skipped list; else `run_pm`; write `IDEAS`, `LEDGER`, `RULE_EVENTS`, `POSITION_DAYS` and rewrite `RULES` as the original rows plus every idea rule; return `{"ideas_per_pm": {...}, "opportunities": {pm_id: {...}}, "skipped": [...]}`. `ENGINE_STAGE = Stage(number=3, name="engine", help="run the behaviour engine: ideas, ledger, rule events per PM", run=run, reads=(PERSONAS, TRAITS, RULES, DRIFT_EVENTS, *MARKET_TABLES), writes=ENGINE_TABLES, appends=(RULES,))`.
- Produces `pipeline.STAGES = (SAMPLE_STAGE, MARKET_STAGE, ENGINE_STAGE)`.
- README: usage line `uv run pm-traitbench engine --config configs/demo.yaml --data-dir data`; a paragraph on what the engine writes (five tables, `position_days` and the hidden columns are generator provenance never shown to a system under test); an "Engine model" section in the style of "Market model" listing the design points (deterministic daily loop, own signal as a noisy peek at the forward move with fixed skill, unscaled thesis move, one bias rule per parameter with its formula in one line each, precedence order, expression forms per asset class and which preferences map to them, no transaction costs, multi-asset PMs skipped for now); the limitations from the design's README list.

- [ ] **Step 1: Write failing tests**
  - `test_loop.py`: the neutral fixture PMs (equities, sovereign_rates, long_short_credit, commodities directional, commodities curve_and_spread) over the 60-day fixture: every ledger row's idea exists in `ideas`; every idea-scope rule's idea exists; every closed idea has `exit_date` and `outcome`, every open one at the end was closed out (no idea without `exit_date`); no position day before its idea's entry date; `rule_events` responses are only `acted` or `overridden` when `e = 0` (set the fixture `exit_deficiency` value to 0 for this test); with `exit_deficiency = 0.06` pooled over the five PMs times 20 root seeds (rerun with different `config.seed.root`) and at least 500 non-overridden events, the breach share is within 0.03 of 0.06; `opportunities["ideas"]` equals `len(ideas)`; an unknown rule field raises `EngineError` before any output; a `multi_asset` persona is not run by `run_pm` (the stage skips it; `run_pm` raises `EngineError`).
  - `test_stage.py`: through a `DataStore` on `tmp_path` with the fixture rows written as stage 1 and 2 outputs (write `PERSONAS`, `TRAITS`, `RULES`, `DRIFT_EVENTS` and the six market tables from the fixture): all four engine tables exist, `RULES` keeps every original row and gains idea rows, run metadata has the three extras and lists the multi-asset PM under `skipped`; a second run without `--force` fails; a rule with an unknown field fails before any engine table is written.
  - `test_end_to_end.py`: `sample`, `market` (with the fake cache fixture, as the existing market test does), then `engine` on the demo config returns 0 and writes the four tables; `--help` lists `engine`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_loop.py tests/engine/test_stage.py tests/test_end_to_end.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement**, including README.
- [ ] **Step 4: Run tests; full suite + lint green.**
- [ ] **Step 5: Commit** - `feat: add the engine stage`

---

### Task 18: Default run and calibration record

**Files:**
- No repo changes unless a defect is found. Output goes to the final report, not the repo (`data/` is gitignored).

- [ ] **Step 1:** `uv run pm-traitbench sample --data-dir data --force`, `uv run pm-traitbench market --data-dir data --force` (the raw cache in `data/raw/market` already exists; if `fetch-market` complains about the window, run it), `uv run pm-traitbench engine --data-dir data --force`. Record wall time of the engine stage.
- [ ] **Step 2:** From `data/run_metadata/engine.json` and the tables, report: ideas per PM per asset class (min, median, max); per (seed, asset class) the 10th-percentile PM's opportunity counts against the plan's minimums (`triggers_fired >= 7`, `loss_side_untriggered_days >= 16`, `conflict_entries >= 14`, `exits >= 28`); the share of ideas in each expression per asset class; the breach share on neutral PMs.
- [ ] **Step 3:** Any cell below a minimum is reported as an input to the Gate 1 sub-project, not fixed here, unless ideas per PM fall outside 30-80 at default config, in which case adjust `engine.arrival_rate` in one commit `fix: tune engine arrival rate` with the numbers in the final report.
