# Evaluation Harness Part 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the validated corpus into an in-process copilot memory system (the system under test) chronologically, ask it every probe at its checkpoint, store its responses, and score the responses that have a deterministic answer key.

**Architecture:** A new package `pm_traitbench/harness/`, outside the pipeline's `STAGES`, exposed as `pm-traitbench eval run` and `pm-traitbench eval score`. The runner reads the corpus only through an allow-list view layer, builds one adapter per PM from a factory, feeds it sessions in date order and asks each checkpoint's probes once every session at or before that date has been observed. Two built-in baselines (no memory, full context) implement the same protocol over the Gate 2 model; the scorer grades option letters and routine-question format against the hidden probe columns.

**Tech Stack:** Python 3.13, pydantic 2, PyYAML, the existing `DataStore`, `CachedClient`/`AnthropicClient`, argparse, pytest with pytest-xdist, ruff.

## Global Constraints

- No file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself. Citing a published paper or a named public data series is allowed.
- Docstrings concise; inline comments 1-2 lines. No em dashes anywhere; use a plain dash.
- Commit messages: one Conventional Commits subject line, no body, ending with the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer only.
- Every closed set of values is a `StrEnum` in `src/pm_traitbench/enums.py`.
- Every new config leaf carries `json_schema_extra={"basis": "sourced" or "design" or "guess", "note": "<non-empty reason>"}`; `tests/test_config.py::test_dump_with_basis_covers_every_leaf` enforces it.
- **Leak invariant:** nothing a system under test receives is built from a hidden column or a withheld table. Only `harness/views.py` turns corpus rows into objects a system under test sees, and it copies named fields from an allow-list. `harness/score.py` reads hidden columns and passes nothing to a system under test.
- **Dependency direction:** nothing outside `pm_traitbench/harness/` imports from it, except `pm_traitbench/cli.py` importing `pm_traitbench.harness.cli`. The harness is not registered in `pipeline.STAGES` or `scripts/generate.sh`.
- **Replay invariant:** a system under test never observes a session dated after a checkpoint before that checkpoint's last probe is answered, and every session dated at or before a checkpoint is observed before that checkpoint's first probe.
- Harness errors raise `HarnessError` (exit code 1).
- Lint and format: `uv run ruff check` and `uv run ruff format --check` on touched files. Tests: `uv run pytest <path> -q` per task; full suite `uv run pytest -n auto -q` once before a task's final commit.

## File Structure

| File | Status | Responsibility | Task |
|---|---|---|---|
| `src/pm_traitbench/enums.py` | modify | `Scorer`, `FormatOutcome` enums | 1 |
| `src/pm_traitbench/errors.py` | modify | `HarnessError` | 1 |
| `src/pm_traitbench/config.py` | modify | `HarnessConfig`, `Config.harness` | 1 |
| `src/pm_traitbench/tables/schema.py` | modify | `ResponseRow`, `ScoreRow` | 1 |
| `src/pm_traitbench/tables/specs.py` | modify | `RESPONSES`, `SCORES`, `parts_spec`, `personas.typicality` hidden, `PERSONAS` in the hidden-column check | 1 |
| `src/pm_traitbench/gates/gate2/transcript.py` | modify | type `render_session`/`render_pm` against a structural protocol | 2 |
| `src/pm_traitbench/harness/__init__.py` | create | package marker, docstring only | 2 |
| `src/pm_traitbench/harness/protocol.py` | create | `SystemUnderTest`, `SutFactory`, `PublicProfile`, `PublicSession`, `PublicProbe` | 2 |
| `src/pm_traitbench/harness/views.py` | create | allow-list builders and per-PM replay inputs | 2 |
| `src/pm_traitbench/harness/checks.py` | create | check-map loader and format checks | 3 |
| `src/pm_traitbench/harness/checks.yaml` | create | value-to-check map | 3 |
| `src/pm_traitbench/harness/runner.py` | create | factory loading, freshness, replay, parts, merge, run metadata | 4 |
| `src/pm_traitbench/harness/baselines.py` | create | `NoMemory`, `FullContext`, `baseline_factory` | 5 |
| `src/pm_traitbench/harness/score.py` | create | option-letter and format scorers, summary | 6 |
| `src/pm_traitbench/harness/cli.py` | create | `eval` argparse group and dispatch | 7 |
| `src/pm_traitbench/cli.py` | modify | register the `eval` group and dispatch to it | 7 |
| `README.md` | modify | hidden-column sentence (task 1), "Evaluation" section and usage lines (task 7) | 1, 7 |
| `docs/pm-dataset-plan.md` | modify | typicality marked hidden (task 1); scope sentence and section 9 harness paragraph (task 7) | 1, 7 |
| `tests/harness/__init__.py` | create | empty | 2 |
| `tests/harness/fixtures.py` | create | profile, session, probe builders; `RecordingSut`; `validated_corpus_with_probes` | 2 (builders), 4 (`RecordingSut`, corpus) |
| `tests/harness/test_*.py` | create | per task below | 2-7 |

**Shared fixture ownership.** `tests/harness/fixtures.py` is created in Task 2 with the row builders `persona_row`, `rule_row`, `session_row`, `probe_row`. Task 4 adds `RecordingSut`, `recording_factory` and `validated_corpus_with_probes`. Tasks 5, 6 and 7 consume them; they must not redefine them.

---

### Task 1: Tables, enums, errors and config

**Files:**
- Modify: `src/pm_traitbench/enums.py`, `src/pm_traitbench/errors.py`, `src/pm_traitbench/config.py`, `src/pm_traitbench/tables/schema.py`, `src/pm_traitbench/tables/specs.py`, `README.md`, `docs/pm-dataset-plan.md`
- Test: `tests/tables/test_harness_schema.py` (create), `tests/tables/test_specs.py` (modify), `tests/test_config.py` (modify), `tests/test_errors.py` (modify)

**Interfaces:**
- Produces:
  - `enums.Scorer(StrEnum)`: `OPTION_LETTER = "option_letter"`, `FORMAT = "format"`.
  - `enums.FormatOutcome(StrEnum)`: `PASS = "pass"`, `FAIL = "fail"`, `NOT_APPLICABLE = "not_applicable"`.
  - `errors.HarnessError(PmTraitbenchError)`, `exit_code = 1`, docstring "Raised when the evaluation harness cannot run a system under test or score its responses."
  - `schema.ResponseRow(BaseModel)`, frozen, `extra="forbid"`: `probe_id: str` (the probe id pattern `ProbeRow` uses), `pm_id: str` (PM id pattern), `response: str` (may be empty: a system may return nothing, which scores as wrong), `latency_ms: int` (`ge=0`). Validator: `probe_id` starts with `p_<pm_id without underscores>_`, as `ProbeRow._check_probe_id_prefix` does.
  - `schema.ScoreRow(BaseModel)`, frozen, `extra="forbid"`: `probe_id: str`, `pm_id: str`, `scorer: Scorer`, `correct: bool`, `detail: str | None` (`min_length=1` when set). Same prefix validator.
  - `specs.RESPONSES = TableSpec("responses", ResponseRow, ("pm_id", "probe_id"))`, `specs.SCORES = TableSpec("scores", ScoreRow, ("pm_id", "probe_id"))`, `specs.HARNESS_TABLES = (RESPONSES, SCORES)`.
  - `specs.parts_spec(pm_id: str) -> TableSpec` returning `TableSpec(f"parts/{pm_id}", ResponseRow, ("pm_id", "probe_id"))`.
  - `specs.HIDDEN_COLUMNS["personas"] = ("typicality",)`; `_check_hidden_columns` adds `PERSONAS` to the tables it resolves so the entry is validated.
  - `config.HarnessConfig(BaseModel)`, frozen, `extra="forbid"`, registered as `Config.harness: HarnessConfig = Field(default_factory=HarnessConfig)`. Leaves (defaults, basis, note meaning):

    | Leaf | Type and default | Basis | Note must say |
    |---|---|---|---|
    | `model` | `str = DEFAULT_MODEL` | design | the Gate 2 model, so the full-context baseline is the Gate 2 ceiling |
    | `effort` | `Effort = Effort.HIGH` | design | Gate 2's recovery effort, for the same reason |
    | `max_answer_tokens` | `int = 8000`, `ge=256` | design | high-effort thinking plus a reply up to a short page; a max-tokens stop is an unparsable reply |
    | `short_page_words` | `int = 400`, `ge=1` | design | about half a printed page, the ceiling for "up to a short page" |
    | `pm_token_budget` | `int or None = None`, `ge=1` when set | guess | soft stop on fresh tokens per PM for a baseline, since each PM's adapter owns its client |

- [ ] **Step 1: Write failing tests**
  - `test_harness_schema.py::test_response_row_round_trips` - a valid `ResponseRow` writes and reads back equal through `DataStore` for every format in `FORMATS` (copy the parametrisation of `tests/tables/test_plan_specs.py::test_round_trip_signal`). Red: `ImportError` on `ResponseRow`.
  - `test_response_row_rejects_foreign_probe_prefix` - `probe_id="p_pm002_0001"` with `pm_id="pm_001"` raises `ValidationError`.
  - `test_response_row_allows_empty_response` - `response=""` builds.
  - `test_score_row_rejects_blank_detail` - `detail=""` raises.
  - `test_parts_spec_names_pm` - `parts_spec("pm_001").name == "parts/pm_001"` and its key equals `RESPONSES.key`.
  - `tests/tables/test_specs.py::test_personas_typicality_is_hidden` - `"typicality" in HIDDEN_COLUMNS["personas"]`.
  - `tests/test_config.py::test_harness_defaults` - `Config().harness.model == DEFAULT_MODEL`, `effort == Effort.HIGH`, `max_answer_tokens == 8000`, `short_page_words == 400`, `pm_token_budget is None`. The existing basis test must still pass with the new section.
  - `tests/test_errors.py`: `HarnessError` is a `PmTraitbenchError` with `exit_code == 1` (follow the file's existing pattern).
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/tables/test_harness_schema.py tests/tables/test_specs.py tests/test_config.py tests/test_errors.py -q`
  Expected: FAIL with import errors for `ResponseRow`, `parts_spec`, `HarnessError`, and a `KeyError` on `HIDDEN_COLUMNS["personas"]`.
- [ ] **Step 3: Implement** the interfaces above. Field descriptions follow the file's style (one sentence each).
- [ ] **Step 4: Living docs.**
  - `README.md`, the paragraph that lists hidden tables and columns (the one starting "`position_days` and `skeletons` are hidden tables in full"): add a sentence "`personas.typicality` is hidden too: it says whether the self-description contradicts the PM's strongest biases."
  - `docs/pm-dataset-plan.md` section 8 layout comment for `personas.jsonl` (line reading "one row per PM: pm_id, market_seed, split (pilot | full), mandate, stated_profile, typicality"): append " (typicality hidden)". Section 8.2 diagram line `string typicality "typical, anti_typical"`: change the comment to `"hidden: typical, anti_typical"`.
- [ ] **Step 5: Run this task's tests and lint on touched files; all green**
- [ ] **Step 6: Commit** - `git commit -m "feat(harness): add response and score tables, harness config and hidden typicality"`

---

### Task 2: Protocol and public views

**Files:**
- Create: `src/pm_traitbench/harness/__init__.py`, `src/pm_traitbench/harness/protocol.py`, `src/pm_traitbench/harness/views.py`, `tests/harness/__init__.py`, `tests/harness/fixtures.py`
- Modify: `src/pm_traitbench/gates/gate2/transcript.py`
- Test: `tests/harness/test_views.py`

**Interfaces:**
- Consumes: `Persona`, `Rule`, `Session`, `Turn`, `ProbeRow`, `Mandate` from `tables/schema.py`; `RuleScope`, `ProbeForm` from `enums.py`.
- Produces (binding contract; other tasks and external adapters compile against it):

```python
# harness/protocol.py
class PublicProfile(BaseModel):          # frozen, extra="forbid"
    pm_id: str
    mandate: Mandate
    self_description: str
    rules: tuple[Rule, ...]              # PM-scope rules, sorted by rule_id

class PublicSession(BaseModel):          # frozen, extra="forbid"
    session_id: str
    date: datetime.date
    turns: tuple[Turn, ...]
    idea_rules: tuple[Rule, ...]         # idea-scope rules of the ideas discussed, sorted by rule_id

class PublicProbe(BaseModel):            # frozen, extra="forbid"
    probe_id: str
    form: ProbeForm
    question: str
    options: tuple[str, ...]             # non-null options in A-D order; empty for an open probe

@runtime_checkable
class SystemUnderTest(Protocol):
    def observe(self, session: PublicSession) -> None: ...
    def answer(self, as_of: datetime.date, probe: PublicProbe) -> str: ...

SutFactory = Callable[[PublicProfile], SystemUnderTest]
```

  The `SystemUnderTest` docstring must state: one instance serves one PM; `observe` is called in date order; `answer` must not write memory, since a probe's premise may be deliberately stale; an optional `close()` method is called once after the PM finishes or fails.

```python
# harness/views.py
def public_profile(persona: Persona, rules: Sequence[Rule]) -> PublicProfile
def public_session(session: Session, rules: Sequence[Rule]) -> PublicSession
def public_probe(row: ProbeRow) -> PublicProbe

@dataclass(frozen=True)
class Checkpoint:
    day: datetime.date
    probes: tuple[ProbeRow, ...]                 # sorted by probe_id

@dataclass(frozen=True)
class PmReplay:
    profile: PublicProfile
    sessions: tuple[PublicSession, ...]          # sorted by (date, session_id)
    checkpoints: tuple[Checkpoint, ...]          # sorted by day

def pm_replays(
    personas: Sequence[Persona],
    rules: Sequence[Rule],
    sessions: Sequence[Session],
    probes: Sequence[ProbeRow],
) -> dict[str, PmReplay]                         # one entry per pm_id present in probes, key-sorted
```

  - `public_profile` keeps only rules with `pm_id == persona.pm_id` and `scope == RuleScope.PM`. `self_description` is `persona.stated_profile.self_description`. `typicality`, `split` and `market_seed` are never copied.
  - `public_session` keeps rules with the session's `pm_id`, `scope == RuleScope.IDEA` and `trade_idea_id` in `session.trade_idea_ids`. `pm_id` and `trade_idea_ids` are not copied: ids carry no memory content, and the rules already name their ideas.
  - `public_probe` copies `probe_id`, `form`, `question`, and the non-null options among `option_a..option_d` in order. It never touches `answer`, `source_*`, `supporting_signal_ids`, `trait_id`, `probe_type`, `checkpoint_label` or `context_chars`: the last five would tell the system which trait or slice is being tested.
  - `pm_replays` raises `HarnessError` when a PM in `probes` has no persona row.
- Modifies `gates/gate2/transcript.py`: add

```python
class TranscriptSession(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def date(self) -> datetime.date: ...
    @property
    def turns(self) -> Sequence[Turn]: ...
```

  and change the parameter types of `_date_then_id`, `render_session`, `render_pm` and `pm_turn_text` from `Session` to `TranscriptSession`. Behaviour is unchanged; the existing tests in `tests/gates/gate2/test_transcript.py` must pass untouched.
- `tests/harness/fixtures.py` (created here): `persona_row(pm_id="pm_001", **overrides) -> Persona`, `rule_row(pm_id, rule_id, scope, trade_idea_id=None, text="...") -> Rule`, `session_row(pm_id, session_id, day, trade_idea_ids=(), turns=None) -> Session`, `probe_row(pm_id, n, day, form=ProbeForm.MCQ, probe_type=ProbeType.TRAIT_MCQ, options=("x","y"), answer="A", trait_id="t_01") -> ProbeRow`. Build valid rows by reusing `tests/gates/gate2/fixtures.py` helpers (`session_of`, `trait`) where they fit; read `ProbeRow`'s validators to set consistent `source_*` (non-null exactly where the option is non-null) and a probe id `p_<pm id without underscores>_<nnnn>`.

- [ ] **Step 1: Write failing tests** in `tests/harness/test_views.py`
  - `test_public_model_field_sets_are_exact` - `set(PublicProfile.model_fields) == {"pm_id", "mandate", "self_description", "rules"}`, and likewise for `PublicSession` (`session_id, date, turns, idea_rules`) and `PublicProbe` (`probe_id, form, question, options`). Adding a field later must be a deliberate test change.
  - `test_no_hidden_name_reaches_public_objects` - dump every public object built from fixture rows with `model_dump_json()`; assert none of the strings `typicality`, `anti_typical`, `market_seed`, `bias_flag`, the probe's `answer` value when it is not an option letter, `source_`, `supporting_signal`, `trait_id`, `checkpoint_label` appears as a key. Use a probe whose `trait_id` is `t_07` and assert `t_07` is absent from the dump.
  - `test_profile_keeps_pm_scope_rules_of_its_pm_only` - rules for two PMs, both scopes; the profile holds only its PM's PM-scope rules, sorted by id.
  - `test_session_carries_rules_of_discussed_ideas_only` - three idea rules on ideas `ti_1`, `ti_2`, `ti_3`; a session discussing `ti_1` and `ti_3` gets exactly those rules.
  - `test_public_probe_drops_null_options` - a 3-option MCQ gives a 3-tuple; an open probe gives `()`.
  - `test_pm_replays_groups_probes_by_checkpoint_and_sorts` - probes on two dates given out of order come back as two checkpoints in date order, each with probes in id order; sessions sorted by `(date, session_id)`.
  - `test_pm_replays_raises_on_missing_persona` - `HarnessError`.
  - `test_render_pm_accepts_public_sessions` - `render_pm` over `PublicSession` objects equals `render_pm` over the source `Session` rows.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_views.py -q`
  Expected: FAIL with `ModuleNotFoundError: pm_traitbench.harness`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run this task's tests, `tests/gates/gate2/test_transcript.py`, and lint on touched files; all green**
- [ ] **Step 5: Commit** - `git commit -m "feat(harness): add system-under-test protocol and allow-list public views"`

---

### Task 3: Format checks and the check map

**Files:**
- Create: `src/pm_traitbench/harness/checks.py`, `src/pm_traitbench/harness/checks.yaml`
- Modify: `src/pm_traitbench/enums.py` (add `CheckKind`)
- Test: `tests/harness/test_checks.py`

**Interfaces:**
- Consumes: `Catalogue` and `load_catalogue` from `catalogues/loader.py` (`catalogue.preferences` is a tuple of `PreferenceEntry` with `param` and `values`); `FormatOutcome`; `HarnessError`.
- Produces:

```python
class CheckKind(StrEnum):   # in enums.py
    BULLETS = "bullets"; PROSE_PARAGRAPH = "prose_paragraph"; TABLE = "table"; HEADERS = "headers"
    UNITS_BP = "units_bp"; UNITS_PERCENT = "units_percent"; UNITS_BOTH = "units_both"
    ONE_SENTENCE = "one_sentence"; TWO_TO_THREE_SENTENCES = "two_to_three_sentences"
    SHORT_PAGE = "short_page"; NO_HEDGES = "no_hedges"; CONFIDENCE_LEVEL = "confidence_level"
    JUDGE = "judge"

CHECKED_PARAMS: tuple[str, ...] = (
    "response_format", "number_language", "length_on_routine_questions", "hedging_language",
)

def load_check_map(catalogue: Catalogue, path: Path | None = None) -> dict[tuple[str, str], CheckKind]
def run_check(kind: CheckKind, reply: str, short_page_words: int) -> FormatOutcome
def parse_routine_answer(answer: str) -> tuple[tuple[str, str], ...]   # (param, value) pairs
```

  - `checks.yaml` shape: `{param: {value: check_kind}}` for exactly the four `CHECKED_PARAMS`, with the mapping of the table below. `load_check_map` reads the packaged file when `path` is None (use `importlib.resources`, as `read_advisor_prompt` does), and raises `HarnessError` when a param is missing, when a value of a `CHECKED_PARAMS` entry in the catalogue is absent from the map (message names the param and value), when the map names a value the catalogue lacks, or when a check kind is unknown. Rationale for the coverage rule, for the docstring: a new catalogue value must never go silently unscored.
  - `run_check(CheckKind.JUDGE, ...)` raises `ValueError`: callers filter judge values first.
  - `parse_routine_answer`: the input is `format: <p>=<v>; <p>=<v>; intrusion: none` or `format: none; intrusion: none`. Strip the `format: ` prefix and the `; intrusion: none` suffix, return `()` for `none`, else split on `"; "` and split each item on the first `=`. Raise `HarnessError` on any other shape. Values contain commas but never `;` or `=`.

  **Check map and rules (binding):**

  | Param | Catalogue value | CheckKind |
  |---|---|---|
  | response_format | short bullets | `bullets` |
  | response_format | one prose paragraph | `prose_paragraph` |
  | response_format | a table with columns | `table` |
  | response_format | a memo with headers | `headers` |
  | number_language | quote moves in basis points | `units_bp` |
  | number_language | quote moves in percent | `units_percent` |
  | number_language | quote moves in both basis points and percent | `units_both` |
  | length_on_routine_questions | one sentence | `one_sentence` |
  | length_on_routine_questions | two to three sentences | `two_to_three_sentences` |
  | length_on_routine_questions | up to a short page | `short_page` |
  | hedging_language | state views plainly, no qualifiers | `no_hedges` |
  | hedging_language | flag uncertainty once, then commit to a view | `judge` |
  | hedging_language | give an explicit confidence level on every call | `confidence_level` |

  Check semantics, on `reply.strip()`; "lines" are its non-empty lines after `rstrip`:
  - List marker: a line matching `^\s*([-*•]|\d+[.)])\s+`.
  - Table row: a line with at least two `|` characters. Separator row: a line matching `^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$`.
  - Header: a line matching `^#{1,6}\s+\S`.
  - `bullets`: at least 2 lines and every line is a list marker line.
  - `prose_paragraph`: no blank line between the first and last text line, and no line is a list marker, table row or header.
  - `table`: some table row is immediately followed by a separator row.
  - `headers`: at least 2 header lines.
  - Units: `bp` present when `\bbps?\b` or `basis points?` matches (case-insensitive); percent present when `%` or `\bper ?cent\b` matches. `units_bp`: bp and not percent; `units_percent`: percent and not bp; `units_both`: both. Each returns `NOT_APPLICABLE` when neither is present, since a reply that quotes no move cannot break a rule about how moves are quoted.
  - Sentences: split on `(?<=[.!?])\s+`, count non-empty pieces; a decimal like `4.25` never splits because no whitespace follows the dot. `one_sentence`: count == 1; `two_to_three_sentences`: 2 <= count <= 3.
  - `short_page`: word count (`len(reply.split())`) <= `short_page_words`.
  - `no_hedges`: none of `might`, `could`, `perhaps`, `possibly`, `likely`, `unlikely`, `uncertain` as whole words case-insensitive, nor `may` as a whole lowercase word (capitalised "May" is usually the month).
  - `confidence_level`: a confidence or conviction word adjacent to a level, case-insensitive: `\b(high|medium|low|\d+(\.\d+)?\s*%)\s+(confidence|conviction)\b` or `\b(confidence|conviction)(\s+level)?\s*(of|at|is|:)?\s*(high|medium|low|\d+(\.\d+)?\s*%)(?!\w)`. A bare percentage is a quoted move, not a confidence level.

- [ ] **Step 1: Write failing tests** in `tests/harness/test_checks.py`
  - `test_packaged_map_covers_catalogue` - `load_check_map(catalogue)` (the `catalogue` fixture in `tests/conftest.py`) returns an entry for every value of the four params, and `("hedging_language", "flag uncertainty once, then commit to a view")` maps to `JUDGE`.
  - `test_map_missing_catalogue_value_raises` - write a copy of the map without "one sentence" to `tmp_path`; `HarnessError` names `length_on_routine_questions` and `one sentence`.
  - `test_map_with_unknown_value_raises`, `test_map_with_unknown_kind_raises`.
  - Parametrised `test_check_passes` and `test_check_fails`: one passing and one failing reply per non-judge kind (13 kinds minus judge). Include: bullets with one prose line mixed in fails; prose with a blank line fails; a pipe-table without separator fails; a single header fails; `"up 12bp"` passes `units_bp` and `"up 12bp, 0.3%"` fails it; `"Yields rose 4.25 today."` is one sentence; a 401-word reply fails `short_page` at 400; `"This could rally."` fails `no_hedges`; `"High confidence: buy."` passes `confidence_level`.
  - `test_units_not_applicable_without_units` - `"Rates were flat."` gives `NOT_APPLICABLE` for all three unit kinds.
  - `test_judge_kind_raises`.
  - `test_parse_routine_answer` - the two-pair example `format: response_format=short bullets; hedging_language=flag uncertainty once, then commit to a view; intrusion: none` returns both pairs with the comma intact; `format: none; intrusion: none` returns `()`; `"garbage"` raises `HarnessError`.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_checks.py -q`
  Expected: FAIL with `ModuleNotFoundError: pm_traitbench.harness.checks`.
- [ ] **Step 3: Implement.** `checks.yaml` must be included in the wheel: check `pyproject.toml`'s build backend includes non-Python package files (the catalogues' YAML already ships this way; follow it).
- [ ] **Step 4: Run this task's tests and lint on touched files; all green**
- [ ] **Step 5: Commit** - `git commit -m "feat(harness): add routine-question format checks and check map"`

---

### Task 4: Runner

**Files:**
- Create: `src/pm_traitbench/harness/runner.py`
- Modify: `tests/harness/fixtures.py` (add `RecordingSut`, `recording_factory`, `validated_corpus_with_probes`)
- Test: `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: `pm_replays`, `PmReplay`, `public_probe` (Task 2); `SystemUnderTest`, `SutFactory` (Task 2); `ResponseRow`, `RESPONSES`, `parts_spec` (Task 1); `PERSONAS`, `RULES`, `SESSIONS`, `PROBES` specs; `DataStore`; `Config`; `HarnessError`.
- Produces:

```python
RUN_NAME_PATTERN = r"^[a-z0-9_-]+$"
RUN_METADATA = "eval-run"

def default_run_name(sut: str) -> str             # sut.replace(".", "_").replace(":", "_").lower()
def check_run_name(name: str) -> str              # returns name; HarnessError if not RUN_NAME_PATTERN
def run_dir(data_dir: Path, run_name: str) -> Path   # data_dir / "eval" / run_name
def load_factory(path: str) -> SutFactory         # "package.module:attr"; HarnessError on bad form,
                                                  # import failure, missing attr, or non-callable
def probes_sha256(store: DataStore) -> str        # sha256 hex of the probes table file bytes
def check_probes_fresh(store: DataStore) -> None

@dataclass(frozen=True)
class RunResult:
    run_store: DataStore
    completed: tuple[str, ...]                    # PMs with a part file after this run, sorted
    skipped: tuple[str, ...]                      # PMs already done before this run, sorted
    failed: dict[str, str]                        # pm_id -> formatted traceback

def run_sut(
    config: Config,
    store: DataStore,
    factory: SutFactory,
    *,
    sut_name: str,
    run_name: str,
    workers: int = 1,
    force: bool = False,
) -> RunResult
```

  - `check_probes_fresh`: `store.read_run_metadata("probes")` missing raises `HarnessError("probes run metadata is missing; run the probes stage first")`; a `sessions_sha256` differing from the sha256 of `store.path(SESSIONS)` bytes raises `HarnessError` saying sessions changed after the probes stage ran. Same guard the probe stage records the hash for.
  - `run_sut` steps:
    1. `check_run_name(run_name)`; `check_probes_fresh(store)`.
    2. `rd = run_dir(store.data_dir, run_name)`; if `force`, `shutil.rmtree(rd, ignore_errors=True)`. `run_store = DataStore(rd, config.output)`.
    3. If `run_store.read_run_metadata(RUN_METADATA)` exists and its `probes_sha256` differs from the current one, raise `HarnessError` telling the user to rerun with `--force` (parts from other probes must not merge with new ones).
    4. Read personas, rules, sessions, probes; `replays = pm_replays(...)`.
    5. PMs whose `parts_spec(pm_id)` already exists in `run_store` are `skipped`. The rest run through `concurrent.futures.ThreadPoolExecutor(max_workers=workers)`, one task per PM running `_replay_pm` below.
    6. `_replay_pm(replay, factory) -> list[ResponseRow]`: build `sut = factory(replay.profile)` and replay with the binding loop below; write `parts_spec(pm_id)` with its rows only when the loop finishes. In `finally`, call `sut.close()` when the attribute exists and is callable. An exception from the factory, `observe`, `answer`, a non-`str` answer (`HarnessError` naming the probe id), or `close` fails the PM: record `traceback.format_exc()` under its id; other PMs continue.
    7. Merge every existing part file into `RESPONSES` in `run_store` (`DataStore.write` sorts by key).
    8. `run_store.write_run_metadata(RUN_METADATA, config, extra)` with `extra = {"sut": sut_name, "run_name": run_name, "probes_sha256": ..., "pms_completed": [...], "pms_skipped": [...], "pms_failed": {pm_id: traceback}, "pm_seconds": {pm_id: float}}`, covering PMs run in this invocation for `pm_seconds`.
  - Replay loop (binding, the replay invariant depends on it):

```python
i = 0
rows: list[ResponseRow] = []
for checkpoint in replay.checkpoints:
    while i < len(replay.sessions) and replay.sessions[i].date <= checkpoint.day:
        sut.observe(replay.sessions[i])
        i += 1
    for row in checkpoint.probes:
        start = time.perf_counter()
        reply = sut.answer(checkpoint.day, public_probe(row))
        latency_ms = int((time.perf_counter() - start) * 1000)
        if not isinstance(reply, str):
            raise HarnessError(f"probe {row.probe_id}: answer returned {type(reply).__name__}")
        rows.append(ResponseRow(probe_id=row.probe_id, pm_id=row.pm_id,
                                response=reply, latency_ms=latency_ms))
```

- Fixtures added to `tests/harness/fixtures.py`:
  - `RecordingSut`: records an ordered event log of `("observe", session)` and `("answer", as_of, probe)`; answers with a scripted callable `(as_of, probe) -> str` (default `"A"`); `closed: bool`; optional `fail_on_probe_id: str | None` that raises `RuntimeError` in `answer`.
  - `recording_factory(**kwargs) -> tuple[SutFactory, dict[str, RecordingSut]]` - the factory stores each built instance by `profile.pm_id` so tests can inspect it.
  - `ECHO_FACTORY: SutFactory` - module-level, builds a `RecordingSut` answering `"A"`; `FAILING_FACTORY: SutFactory` - module-level, builds one whose `answer` raises `RuntimeError`. Both are loadable by import path for the CLI tests.
  - `validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch) -> tuple[Config, DataStore]` - runs `_run_validated_corpus` from `tests/probes/test_stage.py` then `run_stage(PROBES_STAGE, config, store)`. Import the helper rather than copying it.

- [ ] **Step 1: Write failing tests** in `tests/harness/test_runner.py` (use the corpus fixture unless a test says otherwise)
  - `test_every_probe_gets_one_response` - after `run_sut` with `recording_factory()`, `responses` holds exactly one row per probe id in `probes`, and `RunResult.failed == {}`.
  - `test_replay_invariant` - for every recorded `answer(as_of, probe)` event, every session with `date <= as_of` for that PM was observed earlier in the log and no session with `date > as_of` was. Also assert observe order is `(date, session_id)` ascending.
  - `test_received_objects_carry_no_hidden_fields` - every recorded object is a `PublicSession` or `PublicProbe` instance (not a corpus row), and no recorded probe's `model_dump_json()` contains the probe's hidden `answer` text when that text is an open reference (form open).
  - `test_close_called_per_pm` - every built `RecordingSut` has `closed is True`, including one configured to fail.
  - `test_failing_pm_does_not_stop_others` - fail one PM on its first probe id; `failed` holds only that PM with a traceback containing `RuntimeError`; other PMs have parts; run metadata lists it under `pms_failed`.
  - `test_resume_skips_completed_pms` - run once with a PM failing, rerun with a non-failing factory: `skipped` holds the previously completed PMs, the factory is called only for the failed PM, and `responses` is complete.
  - `test_force_reruns_everything` - rerun with `force=True` calls the factory for every PM.
  - `test_stale_probes_metadata_raises` - rewrite `run_metadata/probes.json` with a wrong `sessions_sha256`; `HarnessError`; nothing written under `data/eval`.
  - `test_changed_probes_without_force_raises` - after a run, edit the recorded `probes_sha256` in `eval-run.json`; rerun without force raises `HarnessError`.
  - `test_workers_give_same_responses` - `workers=1` and `workers=4` with a deterministic answer callable give identical `responses` rows ignoring `latency_ms`.
  - `test_non_str_answer_fails_pm` - an answer callable returning `None` fails the PM with a traceback naming the probe id.
  - Unit tests without the corpus: `test_default_run_name` (`"mycopilot.bench:make_adapter"` gives `"mycopilot_bench_make_adapter"`), `test_check_run_name_rejects_slash_and_space`, `test_load_factory_errors` (no colon, missing module, missing attribute, non-callable each raise `HarnessError`), `test_load_factory_loads_callable` (`load_factory("tests.harness.fixtures:ECHO_FACTORY")` returns that object).
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_runner.py -q`
  Expected: FAIL with `ModuleNotFoundError: pm_traitbench.harness.runner`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run this task's tests and lint on touched files; all green**
- [ ] **Step 5: Commit** - `git commit -m "feat(harness): add chronological replay runner with per-PM resume"`

---

### Task 5: Baselines

**Files:**
- Create: `src/pm_traitbench/harness/baselines.py`
- Test: `tests/harness/test_baselines.py`

**Interfaces:**
- Consumes: `PublicProfile`, `PublicSession`, `PublicProbe`, `SutFactory` (Task 2); `render_pm` (`gates/gate2/transcript.py`, accepts `PublicSession` after Task 2); `mandate_line`, `pm_rules_section` (`gates/gate2/prompt_parts.py`); `base_request`, `read_advisor_prompt` (`dialogue/prompts.py`); `CachedClient`, `AnthropicClient`, `LlmClient`, `send_parsed`, `last_text_json` (`dialogue/client.py`); `Config`; `HarnessError`; `ProbeForm`.
- Produces:

```python
BASELINES: tuple[str, ...] = ("no-memory", "full-context")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
MCQ_INSTRUCTION = "Reply with the letter of one option only."
OPEN_INSTRUCTION = "Reply to the PM as you would in the session."

def render_probe(probe: PublicProbe) -> str
def profile_text(profile: PublicProfile) -> str
def parse_answer(response: Mapping[str, Any]) -> str | None

class NoMemory:      # implements SystemUnderTest plus close()
class FullContext:   # implements SystemUnderTest plus close()

def baseline_factory(
    name: str,
    config: Config,
    run_dir: Path,
    run_name: str,
    client_factory: Callable[[Config], LlmClient] | None = None,
) -> SutFactory
```

  - `render_probe`: `question`, then for an MCQ one line per option `"<letter>. <text>"` with letters A-D by position, then a blank line and `MCQ_INSTRUCTION` or `OPEN_INSTRUCTION` by form.
  - `profile_text`: `mandate_line(profile.mandate)`, `"Self-description: <text>"`, `pm_rules_section(profile.rules)`, joined by blank lines.
  - `parse_answer`: `last_text_json(response)` must be a dict whose `answer` is a `str`; else `None`.
  - Request: `{**base_request(config.harness.model, config.harness.max_answer_tokens, config.harness.effort, system, [{"role": "user", "content": user}], ANSWER_SCHEMA), "cache_control": {"type": "ephemeral"}}`. `system` = `read_advisor_prompt(config.dialogue.advisor_prompt_path)` + blank line + `profile_text(profile)`. The top-level `cache_control` lets the long shared prefix be cached across a PM's probes, as the narrator request does.
  - `user` for `FullContext`: `render_pm(observed_sessions)`, a blank line, an "Idea rules discussed:" section listing each distinct observed idea rule's `text` sorted by `rule_id` (omitted when none), a blank line, `f"Today is {as_of.isoformat()}."`, a blank line, `render_probe(probe)`. For `NoMemory`: the last three parts only. `NoMemory.observe` does nothing.
  - Each instance owns `asyncio.Runner()` and a `CachedClient(lambda: factory(config), run_dir / "cache", config.harness.pm_token_budget)` where `factory` is `client_factory` or `lambda c: AnthropicClient(1, c.dialogue.api_max_retries)`; one concurrent call per PM because answers are sequential. `answer` runs `send_parsed(client, request, parse_answer, scope=f"eval:{run_name}:{probe.probe_id}", max_retries=config.dialogue.max_retries, error_type=HarnessError, label="probe", reason="reply is not an answer object")` on the runner and returns the parsed string. `close` awaits `client.aclose()` on the runner, then closes the runner. A per-instance runner is required because `AnthropicClient`'s semaphore and HTTP client bind to the loop they first run on, and PMs run on different threads.
  - `baseline_factory` raises `HarnessError` for a name not in `BASELINES`.

- [ ] **Step 1: Write failing tests** in `tests/harness/test_baselines.py`, using `FakeClient` and `fake_message` from `tests/dialogue/fixtures.py` with a responder returning `fake_message([{"type": "text", "text": json.dumps({"answer": "B"})}])`, and `tmp_path` as `run_dir`.
  - `test_render_probe_mcq_and_open` - exact text for a 3-option MCQ and an open probe.
  - `test_full_context_sends_observed_transcript` - observe two sessions, answer; the one request's user content starts with `render_pm` of both, contains the idea-rules section, `Today is <as_of>`, and ends with the rendered probe; the system contains the advisor prompt, the mandate line and the PM rules; the reply is `"B"`.
  - `test_full_context_includes_only_observed_sessions` - observe one of two sessions; the other's session id is absent from the request.
  - `test_no_memory_ignores_sessions` - after observing, the request has no `Session ` header and does contain the profile.
  - `test_request_uses_harness_config` - `model`, `max_tokens`, `output_config.effort` match `config.harness`; `cache_control` present.
  - `test_unparsable_reply_retries_then_raises` - a responder returning plain text fails `max_retries + 1` times and raises `HarnessError`.
  - `test_second_answer_hits_cache` - a new `FullContext` over the same `run_dir`, same observed sessions and probe, sends no request to the fake client.
  - `test_close_closes_client` - `FakeClient.closed is True` after `close()`.
  - `test_unknown_baseline_raises`.
  - `test_baseline_runs_through_runner` - `run_sut` with `baseline_factory("full-context", ..., client_factory=lambda c: FakeClient(responder))` on the corpus fixture completes with no failures and one response per probe; run in a thread pool with `workers=2` to prove per-instance loops work.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_baselines.py -q`
  Expected: FAIL with `ModuleNotFoundError: pm_traitbench.harness.baselines`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run this task's tests and lint on touched files; all green**
- [ ] **Step 5: Commit** - `git commit -m "feat(harness): add no-memory and full-context baselines"`

---

### Task 6: Scoring and summary

**Files:**
- Create: `src/pm_traitbench/harness/score.py`
- Test: `tests/harness/test_score.py`

**Interfaces:**
- Consumes: `RESPONSES`, `SCORES`, `ScoreRow`, `Scorer`, `FormatOutcome` (Task 1); `load_check_map`, `run_check`, `parse_routine_answer`, `CheckKind` (Task 3); `RUN_METADATA`, `probes_sha256`, `run_dir` (Task 4); `PROBES`, `TRAITS`, `SIGNALS` specs; `ProbeType`, `ProbeForm`, `SignalMode`, `Kind`; `load_catalogue`.
- Produces:

```python
SCORE_METADATA = "eval-score"
LETTER_PATTERN = r"^\(?([A-D])\)?(?=$|[\s.):])"

def parse_letter(response: str, n_options: int) -> str | None
def score_option_letter(row: ProbeRow, response: str) -> ScoreRow
def score_format(row: ProbeRow, response: str,
                 check_map: dict[tuple[str, str], CheckKind],
                 short_page_words: int) -> ScoreRow | None
def evidence_type(signal_ids: Sequence[str], modes: Mapping[str, SignalMode]) -> str
def summarise(probes: Sequence[ProbeRow], scores: Sequence[ScoreRow],
              kinds: Mapping[tuple[str, str], Kind],
              modes: Mapping[str, SignalMode]) -> dict[str, Any]
def score_run(config: Config, store: DataStore, run_name: str) -> dict[str, Any]   # returns summary
```

  - `parse_letter`: match `LETTER_PATTERN` on `response.strip()`; accept only letters among the first `n_options` of A-D; else `None`. Rationale for the docstring: strict parsing, because a lenient parser rewards a reply that lists every letter.
  - `score_option_letter`: applies to `probe_type` in `trait_presence`, `trait_mcq` with `form == mcq`. `n_options` is the count of non-null options. `correct = parsed == row.answer`; `detail = "parse_error"` when unparsable (and `correct=False`), else `None`.
  - `score_format`: applies to `routine_question`. Held pairs from `parse_routine_answer(row.answer)`; drop pairs whose param is not in `CHECKED_PARAMS` or whose map entry is `JUDGE`. No pair left returns `None` (no score row). Else run each check; `correct` is True when no outcome is `FAIL` (`NOT_APPLICABLE` ignored); `detail` is the failed pairs as `param=value` joined by `"; "`, or `None`.
  - `evidence_type`: `"none"` for no ids; `"explicit"` if all modes are `stated`; `"implicit"` if all are `revealed` or `contradiction`; `"mixed"` otherwise. This is the plan's evidence-type slice, derived at scoring time.
  - `summarise` returns (binding shape):

```python
{
  "by_type": [   # one entry per (probe_type, form, scorer) with at least one score, sorted
    {"probe_type": str, "form": str, "scorer": str, "n": int, "correct": int,
     "accuracy": float, "chance": float | None, "parse_errors": int}
  ],
  "awaiting_judge": {   # counts of probes with no deterministic score yet
    "trait_mcq/open": int, "in_situ/open": int, "governance/open": int,
    "routine_question/intrusion": int,          # every routine probe: intrusion needs a judge
    "routine_question/format_judge_only": int,  # routine probes with no score row
  },
  "slices": {   # per scorer, then per dimension, then per value
    "<scorer>": {
      "kind": {"bias" | "preference" | "none": {"n": int, "accuracy": float}},
      "checkpoint_label": {"<label>": {"n": int, "accuracy": float}},
      "evidence": {"explicit" | "implicit" | "mixed" | "none": {"n": int, "accuracy": float}},
    }
  },
  "presence": {"yes": {"n": int, "accuracy": float | None},
               "no": {"n": int, "accuracy": float | None},
               "balanced_accuracy": float | None},
}
```

    `chance` is the mean of `1 / n_options` over the entry's rows for `option_letter`, `None` for `format`. `kind` joins `traits` on `(pm_id, trait_id)`; a null `trait_id` is `"none"`. Presence splits by whether the answer letter's option text is `"yes"`; `balanced_accuracy` is the mean of the two accuracies when both n are positive, else `None`. Rationale for the docstring: most presence rows are "no", so an always-no reply would otherwise look strong. Keys with zero n are omitted from `slices`.
  - `score_run`:
    1. `run_store = DataStore(run_dir(store.data_dir, run_name), config.output)`; its `RUN_METADATA` missing raises `HarnessError`; non-empty `pms_failed` raises `HarnessError` naming the PMs (a partial run's accuracy would be over a biased subset); `probes_sha256` differing from the current probes table raises `HarnessError`.
    2. Read responses (run store), probes, traits, signals (corpus store). A probe without a response raises `HarnessError`.
    3. Score each probe; write `SCORES`; write `summary.json` (indent 2, sorted keys) under the run dir; write `SCORE_METADATA` run metadata with `{"run_name", "probes_sha256", "n_scored"}`; return the summary.

- [ ] **Step 1: Write failing tests** in `tests/harness/test_score.py` (probe rows from `tests/harness/fixtures.py`; corpus fixture only for the last test)
  - Parametrised `test_parse_letter_accepts`: `"A"`, `"A."`, `"(B)"`, `" C) because"`, `"D: yes"`, `"B\n"` with 4 options.
  - Parametrised `test_parse_letter_rejects`: `"Absolutely"`, `"a"`, `"The answer is A"`, `"C"` with 2 options, `""`.
  - `test_parse_letter_first_token_decides` - `"A or B"` parses to `"A"`; the docstring of `parse_letter` states that the leading letter decides.
  - `test_option_letter_score_and_parse_error`.
  - `test_format_score_all_pass_fail_and_na` - bullets reply for `response_format=short bullets; number_language=quote moves in basis points` with no units: correct, detail None; a prose reply: incorrect, detail `response_format=short bullets`.
  - `test_format_score_none_when_only_judge_values` - answer lists only `register=...` and `hedging_language=flag uncertainty once, then commit to a view`: returns `None`.
  - `test_evidence_type` - all four outcomes.
  - `test_summary_chance_and_presence_balance` - presence rows: 3 "no" answered correctly, 1 "yes" answered wrongly gives yes accuracy 0.0, no accuracy 1.0, balanced 0.5, chance 0.5; a 4-option MCQ and a 3-option MCQ give chance `(0.25 + 1/3) / 2`.
  - `test_awaiting_judge_counts` - open twin, in-situ, governance and routine probes counted as specified.
  - `test_score_run_refuses_failed_run`, `test_score_run_refuses_changed_probes`, `test_score_run_refuses_missing_response`.
  - `test_score_run_end_to_end` - corpus fixture, `run_sut` with a `RecordingSut` whose answer callable returns the hidden answer for MCQ probes (look it up by probe id from the probes table in the test) and a two-line bullet reply for open probes; `score_run` writes `scores` and `summary.json`; every `option_letter` entry has accuracy 1.0.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_score.py -q`
  Expected: FAIL with `ModuleNotFoundError: pm_traitbench.harness.score`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run this task's tests and lint on touched files; all green**
- [ ] **Step 5: Commit** - `git commit -m "feat(harness): add option-letter and format scoring with summary"`

---

### Task 7: CLI and living docs

**Files:**
- Create: `src/pm_traitbench/harness/cli.py`
- Modify: `src/pm_traitbench/cli.py`, `README.md`, `docs/pm-dataset-plan.md`
- Test: `tests/harness/test_cli.py`

**Interfaces:**
- Consumes: `run_sut`, `load_factory`, `default_run_name`, `check_run_name`, `run_dir`, `RunResult` (Task 4); `baseline_factory`, `BASELINES` (Task 5); `score_run` (Task 6); `load_config`; `DataStore`.
- Produces:

```python
def add_eval_parser(subparsers: argparse._SubParsersAction) -> None
def run_eval(args: argparse.Namespace) -> int
```

  - `add_eval_parser` adds `eval` with its own subparsers `run` and `score`. Both take `--config PATH` (default None) and `--data-dir PATH` (default `data`), matching the stage subcommands. `run` adds `--sut NAME` (required; a name in `BASELINES` or `package.module:factory`), `--run-name NAME` (default `default_run_name(--sut)`), `--workers N` (int, default 1, `>= 1`, argparse error otherwise), `--force`. `score` adds `--run-name NAME` (required).
  - `run_eval`: load config; build the corpus `DataStore`; for `run`, resolve the factory (`baseline_factory(name, config, run_dir(...), run_name)` for a baseline, else `load_factory`), call `run_sut`, print one line `completed N, skipped N, failed N` and each failed PM id with the last line of its traceback to stderr, return 1 when any PM failed else 0. For `score`, call `score_run` and print the `by_type` entries as a fixed-width table (`probe_type`, `form`, `scorer`, `n`, `accuracy`, `chance`, `parse_errors`) plus the `awaiting_judge` counts; return 0.
  - `pm_traitbench/cli.py`: `build_parser` calls `add_eval_parser(subparsers)` after `fetch-market`; `main` dispatches `args.command == "eval"` to `run_eval(args)` inside the existing `PmTraitbenchError` handler (the handler prints `error: <message>` and returns the exit code). `eval` must be rejected as a stage name so the two namespaces cannot collide: `build_parser` raises `ValueError` if a stage is named `eval`.

- [ ] **Step 1: Write failing tests** in `tests/harness/test_cli.py` (corpus fixture; `ECHO_FACTORY` and `FAILING_FACTORY` from `tests/harness/fixtures.py`)
  - `test_eval_run_then_score` - `main(["eval", "run", "--sut", "tests.harness.fixtures:ECHO_FACTORY", "--run-name", "echo", "--data-dir", str(dir)])` returns 0; then `main(["eval", "score", "--run-name", "echo", "--data-dir", ...])` returns 0 and `data/eval/echo/summary.json` exists; stdout of score contains `trait_presence`.
  - `test_eval_run_exit_1_on_failed_pm` - `--sut tests.harness.fixtures:FAILING_FACTORY`; exit 1; stderr names a PM.
  - `test_eval_run_bad_sut_exit_1` - `--sut nosuch.module:x` returns 1 with `error:` on stderr.
  - `test_eval_run_default_run_name` - omitting `--run-name` writes under `data/eval/tests_harness_fixtures_echo_factory/`.
  - `test_eval_run_rejects_zero_workers` - `SystemExit` from argparse.
  - `test_baseline_name_resolves` - monkeypatch `pm_traitbench.harness.baselines.AnthropicClient` to a `FakeClient`-building callable; `--sut no-memory` returns 0.
  - `test_stage_named_eval_rejected` - `build_parser` with a stub stage named `eval` raises `ValueError`.
  - `test_harness_not_a_pipeline_stage` - no stage in `pipeline.STAGES` is named `eval` and `"eval"` is absent from `scripts/generate.sh`'s `stages=(...)` line.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_cli.py -q`
  Expected: FAIL with `argparse` error "invalid choice: 'eval'" surfacing as `SystemExit`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Living docs.**
  - `README.md` usage block: after the `probes` line add
    ```
    uv run pm-traitbench eval run --sut full-context --data-dir data
    uv run pm-traitbench eval score --run-name full-context --data-dir data
    ```
    and in the stage paragraph after the `probes` sentence: "The `eval` commands are not a stage: they replay the corpus into a system under test and score its answers; see Evaluation."
  - `README.md`: new section `## Evaluation` between `## Probes` and `## Development`, covering, in prose with the one code block and one table:
    - the protocol (the `SystemUnderTest` and `SutFactory` code block from Task 2) and the optional `close()`;
    - what a system observes (profile: mandate, self-description, PM-scope rules; per session: date, turns, idea-scope rules of the ideas discussed) and what it never sees (typicality, split, market seed, session kind, ledger, ideas, rule events, market tables, signals, traits, drift events, hidden probe columns); why: this matches what Gate 2 saw, so Gate 2's recovery is the ceiling;
    - replay order and the read-only `answer` contract with its reason (governance premises are stale on purpose);
    - commands for a baseline and for a custom adapter, `--run-name`, `--workers`, `--force`, per-PM resume, and the output layout under `data/eval/<run_name>/`;
    - the two baselines and what each bounds (no memory: a profile-only floor; full context: the Gate 2 ceiling as a probe score);
    - scoring: strict option-letter parsing, the check-map table from Task 3 (without the `judge` value row, plus a sentence that `register`, `pushback_style`, `answer_ordering` and "flag uncertainty once" wait for judges), the summary fields and slices, and that open probes are counted as awaiting a judge;
    - the `harness` config knobs with their bases.
  - `docs/pm-dataset-plan.md` header status line: replace "Scope is data generation only: a frozen synthetic corpus ... for evaluating the copilot's behavioural memory." so it says the scope is a frozen synthetic corpus of PMs, ledgers, advisory conversations and probes with ground truth, plus an evaluation harness that replays the corpus into a copilot memory system and scores its answers to the probes. Keep the rest of the paragraph.
  - `docs/pm-dataset-plan.md` section 9: after the Stage 10 paragraph, add a paragraph "**Evaluation harness (not a stage).**" stating: an in-process protocol (`observe`, `answer`, one instance per PM from a factory); the system sees the profile and the sessions only, replayed in date order with every session at or before a checkpoint observed before its probes; `answer` must not write memory; two baselines on the Gate 2 model, profile-only and full-context; deterministic scores (strict option letter for presence and MCQ, a published check map for the checkable routine-format values); LLM judges for the open twin, in-situ, governance, intrusion and the judge-only format values, and a full report, are later work.
- [ ] **Step 5: Run this task's tests and lint on touched files; all green. Then the full suite once: `uv run pytest -n auto -q`**
- [ ] **Step 6: Commit** - `git commit -m "feat(harness): add eval run and eval score commands"`
