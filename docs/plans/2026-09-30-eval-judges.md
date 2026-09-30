# Evaluation Harness Part 2: LLM Judges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use minipowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the open probes (open twin, in-situ, governance, routine intrusion, judge-only format values) with one LLM judge call each, and report judge-human agreement on an exported sample.

**Architecture:** A new `eval judge` command reads a finished run's `responses` plus the hidden probe columns, traits, drift events and the probe bank, sends one structured-output request per judged item through the run's `CachedClient`, and writes a run-scoped `judgements` table. `eval score` converts judgements to `ScoreRow`s with pure functions and folds them into the summary. `eval sample` exports a blind, stratified CSV for human raters; `eval score` reads the filled CSV back and reports agreement and Cohen's kappa per judge.

**Tech Stack:** Python 3.13, pydantic, the existing `pm_traitbench.dialogue.client` (`CachedClient`, `send_parsed`, `AnthropicClient`), `asyncio`, numpy `stream` RNG, `csv`, pytest with `tests.dialogue.fixtures.FakeClient`.

## Global Constraints

- No file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator; docstrings state the rule itself. Every non-obvious value carries its reason in a 1-2 line comment.
- Only `harness/` imports `harness/`; nothing in `pipeline.STAGES` or `scripts/generate.sh` references the harness. The judges never touch the corpus tables: everything they write lands under `data/eval/<run_name>/`.
- A system under test never sees hidden columns; the judge does, and nothing built for a judge is ever passed to a `SystemUnderTest`.
- The deterministic scorers (`option_letter`, `format`) are unchanged: no judge verdict alters a deterministic row.
- One judge call per (probe, judge); a reply still unparsable after `dialogue.max_retries` raises `HarnessError` (a broken judge is a harness fault). Empty responses are never sent to a judge.
- All judge JSON schemas put `rationale` first and set `additionalProperties: false`.
- Every closed set of values is an enum in `enums.py`; `Judge` and the new `Scorer` members share string values.
- Markdown tables in README: no literal `|` in a cell.
- Lint: `uv run ruff check <files>` and `uv run ruff format --check <files>` on touched files; line length 100.
- Test command shape: `uv run pytest tests/<path> -q`. Never a full-suite run inside a task step.
- Commit messages: a single Conventional Commits subject line, no body, plus the harness `Co-Authored-By` trailer only.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/pm_traitbench/enums.py` | modify | `Judge` enum; five `Scorer` members |
| `src/pm_traitbench/tables/schema.py` | modify | `JudgementRow` |
| `src/pm_traitbench/tables/specs.py` | modify | `JUDGEMENTS` spec; `SCORES` key gains `scorer` |
| `src/pm_traitbench/config.py` | modify | `JudgeConfig`, registered as `Config.judge` |
| `src/pm_traitbench/harness/judge.py` | create | items, briefs, prompts, schemas, requests, verdict parsing, `judge_run` |
| `src/pm_traitbench/harness/sample.py` | create | human sample allocation and export; ratings read-back; agreement and kappa |
| `src/pm_traitbench/harness/score.py` | modify | judgement to score conversion; judge slices; new `awaiting_judge`; `agreement` |
| `src/pm_traitbench/harness/cli.py` | modify | `judge` and `sample` subcommands; printed agreement |
| `tests/harness/judge_fixtures.py` | create (Task 2) | hand-built open probe rows, traits, drift events and canned judge replies shared by Tasks 2-6 |
| `tests/harness/test_judge_briefs.py` | create | Task 2 |
| `tests/harness/test_judge_run.py` | create | Task 3 |
| `tests/harness/test_score.py` | modify | Task 4 |
| `tests/harness/test_sample.py` | create | Task 5 |
| `tests/harness/test_cli.py` | modify | Task 6 |
| `tests/tables/test_harness_schema.py`, `tests/test_config.py` | modify | Task 1 |
| `README.md`, `docs/pm-dataset-plan.md` | modify | Tasks 1, 4, 5, 6 |

---

### Task 1: Enums, rows, specs and config

**Files:**
- Modify: `src/pm_traitbench/enums.py` (after `Scorer`, line ~446)
- Modify: `src/pm_traitbench/tables/schema.py` (after `ScoreRow`, line ~1657)
- Modify: `src/pm_traitbench/tables/specs.py` (lines 108-109)
- Modify: `src/pm_traitbench/config.py` (after `HarnessConfig`; registration at line ~2396)
- Modify: `README.md` Evaluation "Config." paragraph (line ~929)
- Test: `tests/tables/test_harness_schema.py`, `tests/test_config.py`

**Interfaces:**
- Produces, `enums.py`:
  ```python
  class Judge(StrEnum):
      OPEN = "judge_open"
      IN_SITU = "judge_in_situ"
      GOVERNANCE = "judge_governance"
      INTRUSION = "judge_intrusion"
      FORMAT = "judge_format"
  ```
  `Scorer` gains `JUDGE_OPEN = "judge_open"`, `JUDGE_IN_SITU = "judge_in_situ"`, `JUDGE_GOVERNANCE = "judge_governance"`, `JUDGE_INTRUSION = "judge_intrusion"`, `JUDGE_FORMAT = "judge_format"` (binding: same strings as `Judge`, so `Scorer(judge.value)` always resolves).
- Produces, `schema.py`: `JudgementRow(BaseModel)` frozen, `extra="forbid"`, fields `probe_id: str` (pattern `_PROBE_ID_PATTERN`), `pm_id: str` (pattern `_PM_ID_PATTERN`), `judge: Judge`, `correct: bool`, `detail: str` (min_length 1; every verdict field as `name=value` joined by `; ` in schema order, or `empty_reply`), `rationale: str` (may be empty: an empty reply has no rationale). Docstring: "One judge's verdict on one reply." Flat columns because the arrow format has no nested column.
- Produces, `specs.py`: `SCORES = TableSpec("scores", ScoreRow, ("pm_id", "probe_id", "scorer"))`; `JUDGEMENTS = TableSpec("judgements", JudgementRow, ("pm_id", "probe_id", "judge"))`.
- Produces, `config.py`: `JudgeConfig(BaseModel)` frozen, `extra="forbid"`, registered as `judge: JudgeConfig = Field(default_factory=JudgeConfig)` next to `harness`. Leaves, each with `json_schema_extra={"basis": ..., "note": ...}`:
  - `model: str = DEFAULT_MODEL`, design, "one judge model; the human-rated sample bounds its noise"
  - `effort: Effort = Effort.HIGH`, design, "grading against a rubric benefits from thinking, at Gate 2's effort"
  - `max_tokens: int = 4000` (ge 256), design, "thinking plus a short JSON verdict, the dialogue stage's output cap"
  - `max_concurrency: int = 8` (ge 1), guess, "the dialogue stage's value, well inside default API rate limits"
  - `sample_size: int = 100` (ge 5), guess, "20 items per judge, enough to notice an agreement rate below 0.8"
  - `token_budget: int | None = None` (ge 1), guess, "cap on fresh tokens for one judge pass; spending it stops the pass and finished items stay cached"

- [ ] **Step 1: Write failing tests**
  - `tests/tables/test_harness_schema.py`: `test_scores_key_includes_scorer` (`SCORES.key == ("pm_id", "probe_id", "scorer")`); `test_judgements_spec` (`JUDGEMENTS.name == "judgements"`, model `JudgementRow`, key `("pm_id", "probe_id", "judge")`); `test_judgement_row_round_trip` (a row with `judge=Judge.OPEN` survives `model_dump` / `model_validate`; `detail=""` raises `ValidationError`; `rationale=""` is accepted); `test_judge_and_scorer_share_values` (`{j.value for j in Judge} <= {s.value for s in Scorer}`). Red: `ImportError` on `Judge`, `JudgementRow`, `JUDGEMENTS`; wrong key tuple.
  - `tests/test_config.py`: copy the existing test that checks another section's defaults carry a basis and note (grep `basis` in that file) for `judge`; assert `Config().judge.model == DEFAULT_MODEL`, `effort == Effort.HIGH`, `max_tokens == 4000`, `max_concurrency == 8`, `sample_size == 100`, `token_budget is None`; a YAML override `judge: {sample_size: 4}` raises `ValidationError`. Red: `AttributeError: judge`.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/tables/test_harness_schema.py tests/test_config.py -q`
- [ ] **Step 3: Implement** the interfaces above. Check that the `scores` key change breaks nothing that reads `SCORES.key` (grep `SCORES` in `src/` and `tests/`); `store.write` deduplication or ordering by key, if any, now includes `scorer`. README "Config." paragraph of the Evaluation section gains one sentence per `judge` knob with its default and basis, in the style of the `harness` sentences.
- [ ] **Step 4: Run this task's tests and lint on touched files; all green**
- [ ] **Step 5: Commit** - `feat(harness): add judge enum, judgement row and judge config`

---

### Task 2: Judge items, briefs, prompts and requests

**Files:**
- Create: `src/pm_traitbench/harness/judge.py` (first half)
- Create: `tests/harness/judge_fixtures.py`
- Test: `tests/harness/test_judge_briefs.py`

**Interfaces:**
- Consumes: `ProbeRow`, `Trait`, `DriftEvent`, `ProbeBank` (`catalogue.probes`, `bank.biases[param].behaviour`), `bias_active_at(trait, drift_events, day)` from `pm_traitbench.traits_truth`, `parse_routine_answer(answer) -> tuple[tuple[str, str], ...]` and `CHECKED_PARAMS` from `harness/checks.py`, `load_check_map`, `CheckKind.JUDGE`, `base_request(model, max_tokens, effort, system, messages, schema)` from `dialogue/prompts.py`, `JudgeConfig`, `Judge`.
- Produces (all in `judge.py`):
  ```python
  class InSituCase(StrEnum):  # in enums.py
      COMPLY = "comply"
      COUNTERACT = "counteract"
      DECLINE = "decline"

  class GovernanceKind(StrEnum):  # in enums.py
      PREFERENCE = "preference"
      UPDATE = "update"
      DORMANT = "dormant"

  @dataclass(frozen=True)
  class JudgeItem:
      probe: ProbeRow
      judge: Judge
      response: str
      brief: str            # what the judge is told beyond question and reply
      case: str             # InSituCase, GovernanceKind, or "" for other judges
      letters: tuple[str, ...]        # judge_open: the sibling's offered letters; else ()
      answer_letter: str | None       # judge_open: the sibling's answer; else None
      fields: tuple[str, ...]         # in_situ/governance: verdict boolean field names; judge_format: the "param=value" names; else ()

  @dataclass(frozen=True)
  class JudgeInputs:
      probes: Sequence[ProbeRow]              # with hidden columns
      responses: Mapping[str, str]            # probe_id -> response
      traits: Sequence[Trait]
      drift_events: Sequence[DriftEvent]
      bank: ProbeBank
      check_map: Mapping[tuple[str, str], CheckKind]

  @dataclass(frozen=True)
  class ItemSelection:
      items: tuple[JudgeItem, ...]
      skipped: dict[str, int]   # keys: "no_active_bias", "no_judge_only_values"

  def in_situ_case(answer: str) -> InSituCase          # HarnessError on an unknown prefix
  def governance_kind(answer: str) -> GovernanceKind   # HarnessError on an unknown form
  def judge_only_values(row: ProbeRow, check_map) -> tuple[tuple[str, str], ...]
  def sibling_mcq(row: ProbeRow, probes: Sequence[ProbeRow]) -> ProbeRow   # HarnessError if 0 or >1
  def active_bias_phrases(pm_id: str, day: date, traits, drift_events, bank) -> tuple[str, ...]
  def select_items(inputs: JudgeInputs) -> ItemSelection
  def build_request(item: JudgeItem, config: JudgeConfig) -> dict[str, Any]
  def prompts_sha256() -> str      # sha256 over SYSTEM_PROMPT and every judge instruction, in Judge order
  SYSTEM_PROMPT: str
  INSTRUCTIONS: dict[Judge, str]
  SCHEMAS: dict[Judge, dict[str, Any]]   # judge_format and judge_open schemas are built per item, see below
  def schema_for(item: JudgeItem) -> dict[str, Any]
  ```
- Binding rules:
  - `in_situ_case`: prefix before the first `:` must be one of `comply`, `counteract`, `decline`. `governance_kind`: `premise rejected: dormant since <date>` is `DORMANT`; `premise rejected: changed on <date>; current: much less than before` is `UPDATE`; `premise rejected: changed on <date>; current: <anything else>` is `PREFERENCE`; anything else raises `HarnessError` naming the probe id.
  - `judge_only_values`: from `parse_routine_answer(row.answer)`, keep `(param, value)` where `param not in CHECKED_PARAMS` or `check_map[(param, value)] == CheckKind.JUDGE`; an unmapped value of a checked param raises `HarnessError` (same rule as `checkable_values`).
  - `sibling_mcq`: the `trait_mcq` row with `form == ProbeForm.MCQ` and equal `pm_id`, `checkpoint_date`, `trait_id`, `question`; the probes stage emits both from one draft, so exactly one exists.
  - `active_bias_phrases`: for each trait of the PM with `kind == Kind.BIAS` and `bias_active_at(trait, drift_events, day)` true, `bank.biases[trait.param].behaviour`, in `trait_id` order; `KeyError` on a param missing from the bank becomes `HarnessError`.
  - `select_items`: for every probe in `probe_id` order: `trait_mcq`/`open` gives one `Judge.OPEN` item (`letters` = the sibling's non-null option letters, `answer_letter` = sibling `answer`, brief = the sibling's options as `A. text` lines); `in_situ` gives `Judge.IN_SITU` (`case` from `in_situ_case`, `fields` = `("honours",)`, `("accounts", "names")` or `("refuses", "gives_reason")`, brief = the rubric text `row.answer`); `governance` gives `Judge.GOVERNANCE` (`case` from `governance_kind`, `fields` = `("rejects_premise", "corrects")`, brief = `row.answer`); `routine_question` gives `Judge.INTRUSION` when `active_bias_phrases` is non-empty (brief = the phrases as `- ` lines; else `skipped["no_active_bias"] += 1`) and `Judge.FORMAT` when `judge_only_values` is non-empty (`fields` = `f"{param}={value}"` names, brief = those lines; else `skipped["no_judge_only_values"] += 1`). Other probe types and forms give nothing. `responses` missing a probe id raises `HarnessError`. Empty responses still produce items (Task 3 short-circuits them).
  - `SYSTEM_PROMPT` (constant, wording free but must contain these points): the grader is scoring a copilot's reply to a portfolio manager; grade only against the brief; reply as JSON matching the schema; be strict: a reply that hedges between answers gets the stricter verdict; give the rationale first.
  - `INSTRUCTIONS[judge]` per judge, must state: OPEN "which lettered option does the reply's predicted action correspond to; `none` if it names none, several, or an action outside the list"; IN_SITU the meaning of each field: `honours` (the advice follows the value the rubric names), `accounts` (the advice takes the tendency into account), `names` (it states the tendency to the PM), `refuses` (it does not carry out or endorse the request), `gives_reason` (it cites the rule or cap the rubric quotes); GOVERNANCE `rejects_premise` (the reply does not go along with the stale premise), `corrects` (it states the current state given in the brief); INTRUSION "the reply may follow the PM's communication formats; `intrudes` is true only when content is shaped by or mentions one of the listed tendencies; `evidence` quotes the offending text or is empty"; FORMAT "for each listed value, `pass`, `fail`, or `not_applicable` when the reply gives the value nothing to act on (a pushback style when nothing is disputed)".
  - `schema_for(item)`: `{"type": "object", "properties": {"rationale": {"type": "string"}, ...}, "required": [all], "additionalProperties": False}` with: OPEN `choice: {"type": "string", "enum": [*letters, "none"]}`; IN_SITU and GOVERNANCE one `{"type": "boolean"}` per `fields` entry; INTRUSION `intrudes: boolean`, `evidence: string`; FORMAT `values: {"type": "object", "properties": {name: {"type": "string", "enum": ["pass", "fail", "not_applicable"]} for name in fields}, "required": fields, "additionalProperties": False}`.
  - `build_request`: user message = `Question put to the copilot:\n<question>\n\nCopilot reply:\n<response>\n\nBrief:\n<brief>`; `base_request(config.model, config.max_tokens, config.effort, SYSTEM_PROMPT + "\n\n" + INSTRUCTIONS[judge], [user], schema_for(item))`. No top-level `cache_control` (no long shared prefix).
- Fixtures (`tests/harness/judge_fixtures.py`): builders returning validated rows via `tests.harness.fixtures.probe_row` and `_variant`-style copies: `open_pair(pm_id, n, day, options=("sell now", "hold", "add"), answer="B")` returning `(mcq_row, twin_row)` with `question` identical and twin `answer` = option text; `in_situ_row(n, case)` with `answer` = `"comply: honour short bullets"`, `"counteract: the advice accounts for the PM's tendency to add to losing positions rather than cut them and names it"` or `'decline: the request breaches r_01 "max risk 5% of book"; refuse and give the reason'`; `governance_row(n, kind)` with the three answer forms; `routine_row(n, held)` with `answer = "format: " + "; ".join(f"{p}={v}") + "; intrusion: none"`; `traits_for(pm_id)` returning two bias `Trait`s (params `loss_aversion_lambda` active, `disposition_ratio` active) and one preference; `dormant_event(pm_id, trait_id, day)`; a `bank()` accessor returning `load_catalogue().probes`; `verdict_reply(judge, **fields)` returning `fake_message([{"type": "text", "text": json.dumps({"rationale": "r", **fields})}])`. Read `tests/probes/fixtures.py` and `tests/harness/fixtures.py` first and reuse their row builders; use real bias param names from `BIAS_PARAMS`.

- [ ] **Step 1: Write failing tests** (`tests/harness/test_judge_briefs.py`)
  - `test_in_situ_case_and_governance_kind`: the three prefixes and three governance forms map; `"bogus: x"` and `"premise rejected: whatever"` raise `HarnessError`.
  - `test_judge_only_values`: `register=blunt trading-desk tone` and `hedging_language=flag uncertainty once, then commit to a view` are judge-only; `response_format=short bullets` is not; an unmapped checked value raises.
  - `test_sibling_mcq_found_and_errors`: found for a pair; a twin with no MCQ raises; two MCQs with the same key raise.
  - `test_active_bias_phrases_respects_dormancy`: two active biases give two phrases from the bank; a dormant event on or before the day drops that one; a revive after the dormant restores it.
  - `test_select_items_per_type`: a corpus of one open pair, one in-situ, one governance, one routine with a judge-only value, one routine with only checked values, one presence probe: item judges are exactly `{OPEN, IN_SITU, GOVERNANCE, INTRUSION x2, FORMAT}`; the open item's `letters == ("A","B","C")` and `answer_letter == "B"`; the presence probe yields nothing; `skipped == {"no_active_bias": 0, "no_judge_only_values": 1}`.
  - `test_select_items_skips_intrusion_without_active_bias`: all biases inactive gives no INTRUSION item and `skipped["no_active_bias"] == 1`.
  - `test_select_items_missing_response_raises`.
  - `test_build_request_content`: for each judge, the user message contains the question, the reply and the brief; the schema has `rationale` first in `properties`, `additionalProperties` false; OPEN enum is letters plus `none`; FORMAT `values.required` equals the field names; the system prompt contains the judge's instruction; `model`, `max_tokens` and effort come from a `JudgeConfig(model="m", max_tokens=512)`.
  - `test_prompts_sha256_stable_and_sensitive`: equal across two calls; differs when `INSTRUCTIONS` is monkeypatched.
  Red: `ImportError` on `pm_traitbench.harness.judge`.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_judge_briefs.py -q`
- [ ] **Step 3: Implement** per Interfaces. `InSituCase` and `GovernanceKind` go in `enums.py`.
- [ ] **Step 4: Run this task's tests and lint on touched files; all green**
- [ ] **Step 5: Commit** - `feat(harness): build judge items, briefs and requests`

---

### Task 3: Verdict parsing and the judge pass

**Files:**
- Modify: `src/pm_traitbench/harness/judge.py` (second half)
- Test: `tests/harness/test_judge_run.py`

**Interfaces:**
- Consumes: Task 2 symbols; `CachedClient(inner_factory, cache_dir, token_budget)`, `send_parsed(client, request, parse, *, scope, max_retries, error_type, label, reason)`, `last_text_json`, `AnthropicClient(max_concurrency, max_retries)`, `DialogueBudgetError`; `run_dir`, `probes_sha256`, `RUN_METADATA` from `harness/runner.py`; the guard `_check_run(store, run_store, run_name)` in `harness/score.py` (rename it to public `check_scorable(store, run_store, run_name)` and reuse); `DataStore`, `JUDGEMENTS`, `RESPONSES`, `PROBES`, `TRAITS`, `DRIFT_EVENTS`; `load_catalogue`, `load_check_map`.
- Produces:
  ```python
  JUDGE_METADATA = "eval-judge"

  def parse_verdict(item: JudgeItem, response: Mapping[str, Any]) -> dict[str, Any] | None
  def judgement_from_verdict(item: JudgeItem, verdict: Mapping[str, Any]) -> JudgementRow
  def empty_judgement(item: JudgeItem) -> JudgementRow
  def judge_scope(run_name: str, item: JudgeItem) -> str      # f"judge:{run_name}:{item.probe.probe_id}:{item.judge}"

  @dataclass(frozen=True)
  class JudgeResult:
      judgements: tuple[JudgementRow, ...]
      skipped: dict[str, int]
      run_store: DataStore

  def judge_run(
      config: Config,
      store: DataStore,
      run_name: str,
      *,
      force: bool = False,
      client_factory: Callable[[Config], LlmClient] | None = None,
  ) -> JudgeResult
  ```
- Binding rules:
  - `parse_verdict`: `last_text_json(response)` must be a dict with `rationale` a string and every schema field present with the right type (OPEN `choice` in `letters + ("none",)`; booleans for boolean fields; INTRUSION `intrudes` bool and `evidence` str; FORMAT `values` a dict whose keys equal `item.fields` and values in `pass`, `fail`, `not_applicable`); otherwise `None`.
  - `judgement_from_verdict` `correct`: OPEN `choice == item.answer_letter`; IN_SITU all fields true; GOVERNANCE `rejects_premise or corrects`; INTRUSION `not intrudes`; FORMAT no value `fail`. `detail`: fields as `name=value` joined by `; ` in schema order, booleans as `true`/`false`, FORMAT as `param=value: outcome` lines joined by `; `, INTRUSION `evidence` included verbatim as `evidence=<text>`. `rationale` = the verdict's rationale.
  - `empty_judgement`: `correct=False`, `detail="empty_reply"`, `rationale=""`. Used when `item.response.strip() == ""`; no request is sent.
  - `judge_run`: `check_run_name`; `run_store = DataStore(run_dir(store.data_dir, run_name), config.output)`; `check_scorable`; if `force`, delete the `judgements` table file (`run_store.path(JUDGEMENTS)`) and the `eval-judge` metadata file if present; else if `eval-judge` metadata exists and its `config["judge"]` differs from `config.judge.model_dump(mode="json")` or its `prompts_sha256` differs from `prompts_sha256()`, raise `HarnessError("... rerun with --force to replace it")`. Build `JudgeInputs` from the run's `RESPONSES` and the corpus `PROBES`, `TRAITS`, `DRIFT_EVENTS`, `load_catalogue().probes`, `load_check_map(catalogue)`. Client: `CachedClient(lambda: factory(config), run_dir / "cache", config.judge.token_budget)` with `factory = client_factory or (lambda c: AnthropicClient(c.judge.max_concurrency, c.dialogue.api_max_retries))`. Run all items with `asyncio.Runner` and `asyncio.gather` (the client's semaphore bounds concurrency); per item `send_parsed(..., parse=partial(parse_verdict, item), scope=judge_scope(run_name, item), max_retries=config.dialogue.max_retries, error_type=HarnessError, label="judge", reason="reply is not a verdict object")`. A `DialogueBudgetError` becomes `HarnessError("judge token budget spent; finished items are cached, rerun to continue")`. Close the client in `finally`. Write `JUDGEMENTS` sorted by key, then `write_run_metadata(JUDGE_METADATA, config, {"run_name", "probes_sha256", "prompts_sha256", "counts": {judge: n}, "skipped", "usage": client.totals.as_dict()})` (check the `UsageTotals` method name in `client.py` and use it).
- Fixture additions in `tests/harness/judge_fixtures.py`: `write_run(tmp_path, config, store, probes, responses)` that writes a finished `eval-run` metadata (`status: finished`, `probes_sha256`, `pms_failed: []`, `sut: "fake"`) and a `RESPONSES` table into `run_dir(store.data_dir, "r1")`; a `responder_for(items_by_scope)` that returns canned verdict replies keyed on the request's user text (match by probe question plus brief substring) and a default all-correct verdict otherwise. Read `tests/harness/test_score.py::test_score_run_*` for how a run directory is faked today and reuse that.

- [ ] **Step 1: Write failing tests** (`tests/harness/test_judge_run.py`)
  - `test_parse_verdict_accepts_and_rejects` per judge: a valid reply parses; missing `rationale`, wrong type, unknown letter, FORMAT with an extra key, non-JSON text each give `None`.
  - `test_judgement_correct_rules`: OPEN right/wrong/`none`; IN_SITU counteract with `names=False` is wrong and `detail == "accounts=true; names=false"`; GOVERNANCE `rejects_premise=False, corrects=True` is correct; INTRUSION `intrudes=True` wrong with `evidence=...` in detail; FORMAT one `fail` wrong, all `not_applicable` correct (the score-row rule lives in Task 4).
  - `test_empty_response_not_sent`: an item with response `""` yields `empty_judgement` and the fake client received no request.
  - `test_judge_run_writes_every_item`: on the corpus of Task 2's `test_select_items_per_type` written through `write_run`, `judge_run` with `FakeClient` writes one row per item, metadata counts per judge and `skipped`, and `run_store.read(JUDGEMENTS)` keys are unique.
  - `test_judge_run_unparsable_raises`: a responder returning `"nonsense"` raises `HarnessError` mentioning `judge:` and the probe id after `dialogue.max_retries` retries (assert request count is `1 + max_retries`).
  - `test_judge_run_replays_cache`: a second `judge_run` sends zero requests (`FakeClient.requests` empty on the second client) and writes equal rows.
  - `test_judge_run_refuses_changed_config_without_force`: a second run with `config.judge.model` changed raises `HarnessError` matching `--force`; with `force=True` it succeeds and the old table is replaced.
  - `test_judge_run_refuses_unscorable_run`: a run with `pms_failed` raises `HarnessError` (reuses the existing failed-run fixture pattern).
  - `test_budget_spent_raises_harness_error`: `token_budget=1` with a responder reporting `output_tokens=50` on the first call raises `HarnessError` matching `budget`.
  Red: `ImportError` on `judge_run` and friends.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_judge_run.py -q`
- [ ] **Step 3: Implement** per Interfaces; rename `_check_run` to `check_scorable` in `score.py` and update its callers and tests.
- [ ] **Step 4: Run this task's tests, `tests/harness/test_score.py`, and lint on touched files; all green**
- [ ] **Step 5: Commit** - `feat(harness): add the judge pass writing a judgements table`

---

### Task 4: Fold judgements into scores and the summary

**Files:**
- Modify: `src/pm_traitbench/harness/score.py`
- Modify: `README.md` Evaluation section: the `summary.json` paragraph (line ~918-927) and the "Scoring" sentence "no score row and is counted under `awaiting_judge`" (line ~879)
- Test: `tests/harness/test_score.py`

**Interfaces:**
- Consumes: `JudgementRow`, `JUDGEMENTS`, `Judge`, `Scorer`, `InSituCase`, `GovernanceKind`, `in_situ_case`, `governance_kind`, `judge_only_values`, `active_bias_phrases` (for the awaiting count), Task 1-3 symbols.
- Produces:
  ```python
  def judgement_score(row: JudgementRow) -> ScoreRow | None
  def awaiting_counts(probes, judgements, check_map, traits, drift_events, bank) -> dict[str, int]
  # summarise gains keyword-only `judgements: Sequence[JudgementRow] = ()`, `traits`, `drift_events`, `bank`
  ```
- Binding rules:
  - `judgement_score`: `scorer = Scorer(row.judge.value)`. `detail`: `None` when correct; `"empty_reply"` when `row.detail == "empty_reply"`; OPEN `chose=<letter or none>` (parse `choice=` from `row.detail`); IN_SITU and GOVERNANCE the names of the false fields joined by `; `; INTRUSION `evidence=<text>`; FORMAT the `param=value` names whose outcome is `fail` joined by `; `. Returns `None` for a FORMAT judgement whose outcomes are all `not_applicable` (same rule as the deterministic format scorer: nothing was checkable). Parsing `row.detail` back is the one place the `name=value` format is read; keep a shared `split_detail(detail) -> dict[str, str]` helper in `judge.py` used by both directions, with `; ` as the pair separator and the first `=` as the key split (an `evidence` quote may contain `=` or `;`, so the intrusion detail is always `intrudes=<bool>; evidence=<text>` and `split_detail` takes everything after the first `evidence=` as the quote; state that in its docstring).
  - `score_run`: after the deterministic rows, if `run_store.path(JUDGEMENTS)` exists, read it and append `judgement_score` rows that are not `None`. Pass judgements and the corpus inputs to `summarise`.
  - `awaiting_counts` keys: `trait_mcq/open`, `in_situ/open`, `governance/open`, `routine_question/intrusion`, `routine_question/format_judge`: each the number of probes of that type that would produce that judge's item (per `select_items` rules, so a routine probe with no active bias is not awaiting intrusion, and one with no judge-only values is not awaiting format) minus those with a matching judgement row. The old keys `routine_question/format_not_applicable` and `routine_question/format_judge_only` are removed.
  - `summarise` additions: `by_type` includes judge scorers with `chance` null and `parse_errors` 0; `slices[scorer]` gains `case` for `judge_in_situ` (keys `comply`, `counteract`, `decline` from `in_situ_case(row.answer)`) and `answer_kind` for `judge_governance` (keys `preference`, `update`, `dormant` from `governance_kind`); kind, checkpoint and evidence slices apply to every scorer as today. `awaiting_judge` = `awaiting_counts(...)`.
- README: replace "Open probes are not scored here: ... are counted under `awaiting_judge`." with: judge scorers appear in `by_type` with `chance` null; `slices` gains `case` for `judge_in_situ` and `answer_kind` for `judge_governance`; `awaiting_judge` counts open items with no judgement yet, and after a full `eval judge` only `routine_question/intrusion` can be non-zero, equal to the routine questions whose PM had no active bias at the checkpoint. Keep the change to the "Scoring" sentence minimal: "gets no score row" (drop "and is counted under `awaiting_judge`").

- [ ] **Step 1: Write failing tests** (`tests/harness/test_score.py`)
  - `test_judgement_score_details` per judge: correct gives `detail None`; the wrong cases give the details above; FORMAT all-not-applicable gives `None`; `empty_reply` passes through.
  - `test_awaiting_counts_before_and_after_judging`: on the Task 2 corpus with no judgements the counts are `{"trait_mcq/open": 1, "in_situ/open": 1, "governance/open": 1, "routine_question/intrusion": 2, "routine_question/format_judge": 1}`; with a full set of judgement rows every count is 0; with all biases inactive `routine_question/intrusion` is 0 even with no judgements. Rewrite `test_awaiting_judge_counts` to this.
  - `test_summary_judge_rows_and_slices`: `by_type` has a `judge_in_situ` entry with `chance None`; `slices["judge_in_situ"]["case"]` has the three cases with the right `n`; `slices["judge_governance"]["answer_kind"]` likewise.
  - `test_score_run_includes_judgements`: extend `test_score_run_end_to_end` (or add a sibling) by writing a `JUDGEMENTS` table into the run dir and asserting the `scores` table has judge scorers and the summary's `awaiting_judge` values are 0 for judged types.
  Red: `ImportError` on `judgement_score`, `awaiting_counts`; `TypeError` on the new `summarise` keywords.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_score.py -q`
- [ ] **Step 3: Implement** per Interfaces; update README as specified.
- [ ] **Step 4: Run `tests/harness -q` and lint on touched files; all green**
- [ ] **Step 5: Commit** - `feat(harness): score judgements and report them in the summary`

---

### Task 5: Human sample export and agreement

**Files:**
- Create: `src/pm_traitbench/harness/sample.py`
- Modify: `src/pm_traitbench/harness/score.py` (read ratings; `agreement` in the summary)
- Modify: `README.md` Evaluation section (new "Human sample" paragraph after the summary paragraph)
- Test: `tests/harness/test_sample.py`, `tests/harness/test_score.py`

**Interfaces:**
- Consumes: `JudgementRow`, `JUDGEMENTS`, `Judge`, `InSituCase`, `select_items` / `JudgeItem` (for `brief` and `case`), `stream(root_seed, *keys)` from `pm_traitbench.rng`, `csv`.
- Produces:
  ```python
  SAMPLE_FILE = "human_sample.csv"
  SAMPLE_COLUMNS = ("sample_id", "judge", "probe_id", "pm_id", "case", "question",
                    "response", "brief", "human_correct", "human_note")
  IN_SITU_WEIGHTS = {InSituCase.COUNTERACT: 0.4, InSituCase.DECLINE: 0.4, InSituCase.COMPLY: 0.2}

  def allocate(counts: Mapping[Judge, int], size: int) -> dict[Judge, int]
  def allocate_in_situ(counts: Mapping[InSituCase, int], size: int) -> dict[InSituCase, int]
  def draw_sample(items: Sequence[JudgeItem], judgements: Sequence[JudgementRow], size: int, rng: np.random.Generator) -> list[dict[str, str]]
  def write_sample(config, store, run_name) -> Path
  @dataclass(frozen=True)
  class Rating: sample_id: str; pm_id: str; probe_id: str; judge: Judge; human_correct: bool
  def read_ratings(path: Path) -> list[Rating]
  def agreement(ratings: Sequence[Rating], judgements: Sequence[JudgementRow]) -> dict[str, dict[str, Any]]
  def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None
  ```
- Binding rules:
  - `allocate`: `size // 5` per judge in `Judge` order, the remainder (`size % 5`) going one each to the first judges in order; a judge whose count is below its share is capped at its count and the shortfall is handed to the next judges in order that still have room, looping until nothing moves. `allocate_in_situ`: shares `round(size * weight)` in the order counteract, decline, comply, with the last share taking `size` minus the others so the total is exact; the same cap-and-redistribute rule.
  - `draw_sample`: only judgements with `detail != "empty_reply"` are eligible (an empty reply needs no rater); within a judge (and within an in-situ case) pick `rng.choice(n, share, replace=False)` over the eligible items sorted by `(pm_id, probe_id)`; output rows in draw order with `sample_id = f"s_{i:04d}"` from 1; `human_correct` and `human_note` empty; `case` = `item.case`; `brief` = `item.brief`. No verdict or rationale column (raters are blind).
  - `write_sample`: `HarnessError` when the run has no `judgements` table ("run eval judge first"); rebuilds items with `select_items` from the same inputs `judge_run` used; `rng = stream(config.seed.root, "eval_sample", run_name)`; writes `run_dir / SAMPLE_FILE` with `csv.writer` (`newline=""`, utf-8); size = `config.judge.sample_size` unless overridden by the caller; returns the path.
  - `read_ratings`: `csv.DictReader`; a missing column raises `HarnessError`; rows with blank `human_correct` (after strip) are skipped; `yes`/`no` case-insensitive map to `True`/`False`; any other value raises `HarnessError` naming the `sample_id`.
  - `agreement`: join each rating to its judgement by `(pm_id, probe_id, judge)`; a rating with no judgement raises `HarnessError`. Per judge present: `{"n": k, "agreement": matches / k, "kappa": cohen_kappa(pairs)}` where `pairs = [(human, judge_correct)]`.
  - `cohen_kappa` (binding, Cohen 1960): with `po` = fraction of pairs that agree, `pa` = fraction of human `True` times fraction of judge `True`, `pb` = the same for `False`, `pe = pa + pb`, return `None` when `pe == 1` (a rater used one label only, so chance agreement is total), else `(po - pe) / (1 - pe)`.
  - `score_run`: if `run_dir / SAMPLE_FILE` exists, `summary["agreement"] = agreement(read_ratings(path), judgements)`; else `summary["agreement"] = {}`.
- README "Human sample" paragraph: the command, the columns, the rating values `yes`/`no` (blank = unrated), the allocation (even over judges, in-situ 40/40/20 counteract/decline/comply, seeded), that verdicts are withheld, and that `eval score` reports `agreement` per judge with `n`, agreement rate and Cohen's kappa (kappa because intrusion and governance verdicts are mostly one-sided).

- [ ] **Step 1: Write failing tests** (`tests/harness/test_sample.py`)
  - `test_allocate_even_with_remainder`: size 12 over five judges with ample counts gives `[3, 3, 2, 2, 2]` in `Judge` order.
  - `test_allocate_caps_short_judge`: `counts[Judge.GOVERNANCE] == 0` gives it 0 and its share moves to others; total equals `min(size, sum(counts))`.
  - `test_allocate_in_situ_weights`: size 10 gives counteract 4, decline 4, comply 2; size 7 sums to 7.
  - `test_draw_sample_blind_and_deterministic`: rows have exactly `SAMPLE_COLUMNS`; no `correct`/`rationale`; empty-reply judgements excluded; the same seed gives the same rows; `sample_id`s are `s_0001...`.
  - `test_write_sample_requires_judgements`: `HarnessError` matching `eval judge`.
  - `test_read_ratings_values_and_errors`: `Yes`, ` no ` parse; blank skipped; `maybe` raises naming the sample id; a missing column raises.
  - `test_agreement_and_kappa`: a hand-built table: 8 pairs with 6 agreeing and both labels used gives `agreement 0.75` and kappa equal to the hand computation (state the numbers in the test); all-`True` judge gives `kappa None`; an unmatched rating raises.
  - In `test_score.py`: `test_score_run_reports_agreement` writes a judgements table and a filled `human_sample.csv` into the run dir and asserts `summary["agreement"][judge]["n"]`.
  Red: `ImportError` on `pm_traitbench.harness.sample`.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_sample.py tests/harness/test_score.py -q`
- [ ] **Step 3: Implement** per Interfaces; README paragraph.
- [ ] **Step 4: Run `tests/harness -q` and lint on touched files; all green**
- [ ] **Step 5: Commit** - `feat(harness): export a blind human sample and report judge agreement`

---

### Task 6: CLI commands, end-to-end test and living docs

**Files:**
- Modify: `src/pm_traitbench/harness/cli.py`
- Modify: `README.md` Evaluation section (commands block, "Judges" paragraph after "Scoring", run directory listing at line ~856)
- Modify: `docs/pm-dataset-plan.md` section 9 "Evaluation harness (not a stage)" paragraph (line ~779)
- Test: `tests/harness/test_cli.py`

**Interfaces:**
- Consumes: `judge_run`, `JudgeResult`, `write_sample`, `score_run`, `check_run_name`, `BASELINES`.
- Produces: subcommands `eval judge --run-name NAME [--force]` and `eval sample --run-name NAME [--size N]` (`--size` positive int via the existing `_workers`-style validator, renamed `_positive_int`; default `config.judge.sample_size`). `run_eval` returns 0 on success; `HarnessError` propagates to `main` as today (exit 1). `eval judge` prints `judged <n> items (<judge>: n, ...), skipped <k>`. `eval sample` prints the path and row count. `eval score` prints, after the by-type table and the awaiting line, one line per judge in `agreement`: `agreement <judge>: n=<k> rate=<r:.3f> kappa=<k or ->`. The by-type table includes judge rows unchanged in shape.
- The baselines' `client_factory` pattern is mirrored: `eval judge` calls `judge_run(config, store, run_name, force=args.force)`; tests inject `FakeClient` by monkeypatching `pm_traitbench.harness.judge.AnthropicClient` (copy `test_baseline_name_resolves`).

- [ ] **Step 1: Write failing tests** (`tests/harness/test_cli.py`)
  - `test_eval_judge_sample_score_end_to_end`: `run` with `ECHO`; monkeypatch `AnthropicClient` in `harness.judge` to a `FakeClient` whose responder returns a valid all-correct verdict for whichever judge the request's schema names (inspect the request's structured-output schema properties: `choice` for OPEN, `intrudes` for INTRUSION, `values` for FORMAT, `rejects_premise` for GOVERNANCE, else IN_SITU); `judge` exits 0 and prints `judged`; `judgements` table exists with one row per open probe present in the fixture corpus (compute the expected count with `select_items` on the corpus; do not assume any given type is present); `sample --size 5` exits 0 and the CSV has the columns and at most 5 rows; fill every `human_correct` with `yes` and `score` exits 0 printing `agreement` lines; `summary.json` has `agreement` keys for the sampled judges and `awaiting_judge` values of 0 for every key except `routine_question/intrusion`.
  - `test_eval_sample_before_judge_exit_1`: `sample` on a run without judgements exits 1 and prints `eval judge` to stderr.
  - `test_eval_judge_bad_run_name_exit_1`: `--run-name "Bad Name"` exits 1.
  Red: `argparse` error `invalid choice: 'judge'`.
- [ ] **Step 2: Run them; they fail for the right reason**
  Run: `uv run pytest tests/harness/test_cli.py -q`
- [ ] **Step 3: Implement** the subcommands and printing. README: add `uv run pm-traitbench eval judge --run-name mine --data-dir data` and `eval sample --run-name mine --size 100` to the commands block; add a "Judges" paragraph after "Scoring" containing the table below (no literal `|` in cells), the intrusion brief rule (active biases' behaviour phrases at the checkpoint, communication formats never count, biases only because a preference is how the PM wants to be advised), the empty-reply rule, the unparsable-judge rule, the `judgements` table columns, the cache and `--force` behaviour, `judge.model` defaulting to the Gate 2 model with the self-preference caveat; extend the run directory sentence with `judgements` and `human_sample.csv`.

    | Judge | Probes | Told | Verdict | Correct |
    |---|---|---|---|---|
    | `judge_open` | open twin of a `trait_mcq` | the sibling's lettered options | `choice` | equals the sibling's answer letter |
    | `judge_in_situ` | `in_situ` | the rubric | `honours`, or `accounts` and `names`, or `refuses` and `gives_reason` | all true |
    | `judge_governance` | `governance` | the event date and current state | `rejects_premise`, `corrects` | either true |
    | `judge_intrusion` | `routine_question` | the PM's active bias tendencies | `intrudes`, `evidence` | not intrudes |
    | `judge_format` | `routine_question` with judge-only held values | those values | `pass`, `fail` or `not_applicable` per value | no fail |

  `docs/pm-dataset-plan.md` section 9 paragraph: replace "LLM judges for the open twin, in-situ, governance, intrusion and the judge-only format values, and a full report, are later work." with a sentence stating that one LLM judge per open case scores those probes into a `judgements` table, a blind stratified sample is exported for human rating and judge-human agreement with Cohen's kappa is reported per judge, and that the full report remains later work.
- [ ] **Step 4: Run `tests/harness -q` and lint on touched files; all green**
- [ ] **Step 5: Commit** - `feat(harness): add eval judge and eval sample commands`
