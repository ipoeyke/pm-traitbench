**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** `docs/specs/2026-09-29-probes-design.md` (its Scope's "Out of scope: the scoring harness" - this spec builds the first part of it); changes `docs/pm-dataset-plan.md` header scope sentence ("Scope is data generation only") and section 9

# Design: evaluation harness, part 1 - runner and deterministic scoring

Date: 2026-09-29. Source of requirements: `docs/pm-dataset-plan.md` sections 5 (probe types and scoring column, evidence-type slice, "the harness passes the system under test nothing but the question, the options, and the context") and 8 (hidden columns are never shown to a system under test). Builds on `docs/specs/2026-09-29-probes-design.md` (the `probes` table, its hidden columns, the context cut at a checkpoint, `sessions_sha256` in probes run metadata) and `docs/specs/2026-09-28-gate2-design.md` (`render_pm`, the Gate 2 model).

## Scope

The harness evaluates a copilot memory system, the **system under test**, on the probes. It is decomposed into three sub-projects, each with its own spec:

1. **This spec: runner and deterministic scoring.** The system-under-test protocol, the public-view loader, chronological replay, the `responses` table, two baselines (no memory, full context), the option-letter scorer (presence, trait MCQ), the routine-question format scorer for checkable values, and a summary.
2. **Later: LLM judges.** The open twin, in-situ, governance, routine intrusion, the judge-only format values, and the human-agreement sample of plan section 5.
3. **Later: full report.** Every remaining slice (drift, typicality, asset class, position in context, `source_*`), confidence intervals, cross-run comparison.

Out of scope here: judges, the full report, freeze (stage 10; the harness checks probes freshness instead, decision 9), ledger or idea visibility to the system under test (decision 4), a network protocol for remote systems (decision 2).

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself.

## Decisions

1. **The harness lives in the same package, as `pm_traitbench/harness/`, outside the pipeline.** It shares the row models, enums, hidden-column registry, `DataStore`, `render_pm` and the cached model client, so row shapes have one source of truth. It is not a stage: it consumes the dataset rather than producing it, so it is not in `pipeline.STAGES` or `scripts/generate.sh`. It is exposed as a CLI group `pm-traitbench eval` with subcommands `run` and `score`. Rejected: a separate repo reading the frozen dataset (duplicated row models that drift from the generator's); a uv workspace with a second package (workspace plumbing for no boundary the public-view loader does not already give).

2. **The system under test is an in-process Python object with a stateful, chronological protocol.** A memory system has a write path and a read path, so the harness replays history rather than handing over a context string.

   ```python
   class SystemUnderTest(Protocol):
       def observe(self, session: PublicSession) -> None: ...
       def answer(self, as_of: date, probe: PublicProbe) -> str: ...

   SutFactory = Callable[[PublicProfile], SystemUnderTest]
   ```

   The factory is called once per PM, so every PM starts from an empty memory; if the returned object has a `close()` method the runner calls it after the PM finishes or fails, so an adapter can release its resources and PMs can run concurrently without the adapter being thread-safe. It is loaded by import path, `--sut package.module:factory`; `no-memory` and `full-context` are built-in names for the baselines. The harness never imports the copilot's code except through that path, so no copilot dependency enters this repo. Rejected: stateless per-probe calls with a rendered context (tests long-context reading only; a memory system cannot consolidate incrementally); a `reset(pm_id)` method on one shared instance (forces thread-safety on the adapter and is redundant with per-PM construction); an HTTP protocol (the target copilot runs in-process; a service can be wrapped by a thin adapter class).

3. **`answer` must not write memory.** A governance probe asserts a stale value on purpose; a system that stores probe text would carry that premise into later checkpoints and every later score would measure the harness's own contamination. The protocol docstring states the contract. It cannot be enforced in-process (limitation).

4. **The system under test sees dialogue plus the static profile, nothing else.**
   - `PublicProfile` (factory argument): `pm_id`, `mandate`, `stated_profile.self_description`, the PM-scope rules (all `Rule` fields).
   - `PublicSession` (per `observe`): `session_id`, `date`, `turns`, and the idea-scope rules whose `trade_idea_id` is in the session's `trade_idea_ids`. The same idea's rules repeat on every session that discusses it, so a system never needs to join across sessions to read a rule.
   - `PublicProbe` (per `answer`): an opaque `probe_id` (`q_` plus the first 16 hex digits of the sha256 of the root seed and the real probe id), `form`, `question`, and the non-null options as a tuple. The real id is contiguous per checkpoint in emission order, so it would reveal position.

   Withheld: `typicality`, `split`, `market_seed`, the session `kind` (planning metadata Gate 2's transcript also omits), ledger, ideas, rule events, market tables, signals, traits, drift events, and every hidden probe column. This matches what Gate 2 saw (transcript, mandate, rules), so Gate 2's recovery rate is the ceiling for a probe score, and matches the probe stage's context cut and supporting-signal rule. Rejected: adding the public ledger, ideas and rule events interleaved by date (closer to a deployed copilot, but bias probes then become answerable by Gate 1-style trade statistics that no ceiling covers, mixing ledger analytics into a dialogue-memory score).

5. **Public objects are built from an allow-list.** `harness/views.py` is the runner's only reader of corpus tables, and nothing the scorer reads is ever passed to a system under test; `views.py` constructs each public object by copying named fields, never by deleting hidden ones, so a column added later is invisible until someone allows it. `personas.typicality` joins `HIDDEN_COLUMNS` (it says whether the self-description contradicts the PM's biases, which is ground truth); `_check_hidden_columns` gains the sampling tables so the entry is validated.

6. **Replay order.** Per PM, sessions sorted by `(date, session_id)` and checkpoints sorted by date are merged: every session dated at or before a checkpoint date is observed before that checkpoint's probes are asked, the same cut the probe stage used for `context_chars` and supporting signals. Probes at one checkpoint are asked in a seeded shuffle keyed on the root seed, PM and checkpoint date: the probe stage emits a checkpoint's current-value (yes) presence rows before its no rows, so asking in emission order would leak answers by position. Sessions after the last checkpoint are not observed.

7. **Resume and parallelism are per PM.** A stateful system cannot resume mid-history, so a PM's responses are written to `eval/<run_name>/parts/<pm_id>` only once the PM finishes; a rerun skips PMs with a part file and `--force` deletes the run directory first. A rerun without `--force` refuses when the recorded probes digest, SUT name or harness config differs, so one `responses` table never mixes systems. After the last PM, parts are merged into the `responses` table. `--workers N` (default 1) runs N PMs in threads. An exception from the adapter fails that PM only: its traceback goes to run metadata, the other PMs continue, and the command exits 1 if any PM failed.

8. **Output layout.** Everything goes under `data/eval/<run_name>/`, written through a `DataStore` rooted there, so run metadata lands at `data/eval/<run_name>/run_metadata/eval-run.json` and `eval-score.json`. `--run-name` defaults to the SUT name with `.` and `:` replaced by `_`; it must match `[a-z0-9_-]+` (lower-cased), else a `HarnessError`. Tables: `responses` (`probe_id, pm_id, response, latency_ms`, keyed by `pm_id, probe_id`), `scores` (decision 12), and `summary.json`. Run metadata for `eval-run`: SUT name, probes table sha256, per-PM status and wall time, failures. `latency_ms` is the one non-deterministic column.

9. **Freshness check.** `eval run` fails with `HarnessError` when probes run metadata is missing or its `sessions_sha256` differs from the sha256 of the current `sessions` file, the same guard Gate 2 and the probe stage use against stale upstream output. `eval score` fails when the probes table's sha256 differs from the one recorded by the run, or when the run recorded a failed PM (a partial run's accuracy would be over a biased subset).

10. **Baselines use the Gate 2 model through the cached client.** Both answer with `DEFAULT_MODEL` via `CachedClient` (cache under `data/eval/<run_name>/cache`, scope `eval:<run_name>:<probe_id>`), with the copilot system prompt the simulated advisor uses (`catalogues/advisor_prompt.md`), the profile rendered as text, and one fixed answer instruction per form: MCQ "Reply with the letter of one option only."; open "Reply to the PM as you would in the session.".
    - `NoMemory` ignores `observe` and answers from the profile and the question. It is the floor: what the self-description alone gives, which misleads on anti-typical PMs.
    - `FullContext` buffers observed sessions and answers from `render_pm` over them plus the profile. It is the Gate 2 ceiling expressed as a probe score.
    - A reply still unparsable after the retries is recorded as an empty response, which scores wrong; spending `pm_token_budget` or an API error fails the PM.

11. **Option-letter scorer (trait presence and trait MCQ `mcq` form).** The response is stripped and must start with a letter among the probe's offered options, optionally in parentheses, followed by end of text, whitespace, `.`, `)` or `:`. Anything else is incorrect with `detail: parse_error`, reported separately. Rejected: lenient parsing (rewards a system that lists every letter).

12. **Format scorer (routine questions).** The probe's hidden `answer` (`format: p=v; ...; intrusion: none`) lists the held communication values. A check map `harness/checks.yaml` maps each catalogue value of four params to a check or to `judge`:

    | Param | Value | Check |
    |---|---|---|
    | `response_format` | short bullets | at least 2 non-empty lines, every one starting with `-`, `*`, a bullet character or `N.` |
    | `response_format` | one prose paragraph | no blank line between text lines, no list marker, no table row, no header |
    | `response_format` | a table with columns | a markdown table: a row with at least 2 cells followed by a separator row of dashes |
    | `response_format` | a memo with headers | at least 2 markdown header lines |
    | `number_language` | quote moves in basis points | contains `bp`, `bps` or "basis point"; no `%` or "percent"; not applicable when neither appears |
    | `number_language` | quote moves in percent | the reverse; not applicable when neither appears |
    | `number_language` | quote moves in both basis points and percent | both present; not applicable when neither appears |
    | `length_on_routine_questions` | one sentence | exactly 1 sentence |
    | `length_on_routine_questions` | two to three sentences | 2 or 3 sentences |
    | `length_on_routine_questions` | up to a short page | at most `harness.short_page_words` (400) words |
    | `hedging_language` | state views plainly, no qualifiers | none of a fixed hedge-word list (`might`, `may`, `could`, `perhaps`, `possibly`, `likely`, `unlikely`, `uncertain`) as whole words; `may` only in lowercase, since capitalised May is usually the month |
    | `hedging_language` | give an explicit confidence level on every call | "confidence" or "conviction" next to a level (high, medium, low or a percentage); a bare percentage is a quoted move, not a confidence level |
    | `hedging_language` | flag uncertainty once, then commit to a view | `judge` |

    `register`, `pushback_style` and `answer_ordering` are wholly `judge` and are not listed. Sentences are split on `.`, `!` or `?` followed by whitespace or end of text, so decimals never split. A routine probe passes when every checkable held value passes (not-applicable checks are ignored); an empty reply is incorrect; one with no checkable held value, or whose checkable values are all not applicable, gets no score row. The loader fails with `HarnessError` if any catalogue value of the four params is missing from the map, so a new catalogue value cannot go silently unscored. The map lives in the harness, not in `preferences.yaml`, because the probes spec made value-to-check mapping scoring logic that the dataset does not ship. Rejected: leaving all routine scoring to the judges sub-project (the four checkable params are the plan's deterministic scoring and need no model).

13. **`scores` table.** One row per scored probe: `probe_id, pm_id, scorer, correct, detail`, keyed by `pm_id, probe_id`. `scorer` is `option_letter` or `format` (enum in `enums.py`); `detail` is null, `parse_error`, or the failed values joined by `; `. The open twin, in-situ, governance and the routine intrusion half get no row; the summary counts them under `awaiting_judge`, with routine probes lacking a format row split into judge-only and all-not-applicable counts.

14. **Summary (`summary.json`, also printed as a table).** Per `(probe_type, form)`: n, accuracy, chance (1 over the number of offered options for `option_letter`, null for `format`), parse errors. Then accuracy with n by each of:
    - kind (bias, preference, or `none` when `trait_id` is null), joined from `traits`;
    - `checkpoint_label`;
    - evidence type, derived as the plan says from the `mode` of the probe's supporting signals: `explicit` if all are `stated`, `implicit` if all are `revealed` or `contradiction`, `mixed` otherwise, `none` when there are none;
    - trait presence split by answer (yes, no) with balanced accuracy, since most presence rows are "no" and an always-no system would otherwise score high.

15. **Config.** `HarnessConfig` registered as `Config.harness`, every leaf with a basis and note: `model` = `DEFAULT_MODEL` design (the Gate 2 model, so the full-context baseline is the Gate 2 ceiling), `effort` = `high` design (Gate 2's recovery effort, for the same reason), `max_answer_tokens 8000` design (high-effort thinking plus a reply up to a short page; Gate 2's classify call uses the same cap for the same shape), `short_page_words 400` design (about half a printed page), `pm_token_budget` null guess (soft stop on fresh tokens per PM for a baseline, since each PM's adapter owns its own client).

16. **Errors.** New `HarnessError(PmTraitbenchError)` in `errors.py`, exit code 1, for a bad run name, missing or stale probes metadata, an incomplete check map, an unloadable `--sut`, or scoring a run whose probes changed.

## Package layout

```
src/pm_traitbench/harness/
  __init__.py
  protocol.py   # SystemUnderTest, SutFactory, PublicProfile, PublicSession, PublicProbe
  views.py      # allow-list builders from corpus rows to public objects
  runner.py     # load factory, replay per PM, parts, merge to responses, run metadata
  baselines.py  # NoMemory, FullContext, profile and probe rendering
  checks.py     # check map loader and the format checks
  checks.yaml
  score.py      # option-letter and format scorers, summary
  cli.py        # the `eval` group: `run`, `score`
```

Modified: `tables/schema.py` (`ResponseRow`, `ScoreRow`), `tables/specs.py` (`HIDDEN_COLUMNS["personas"]`, `_check_hidden_columns` over sampling tables, a function building run-scoped `TableSpec`s), `enums.py` (`Scorer`), `errors.py`, `config.py`, `cli.py` (registers the `eval` group).

### Data flow

1. `eval run`: check freshness (decision 9); load factory; read `personas`, `rules`, `sessions`, `probes` public view; per PM not yet done: build profile, construct adapter, replay (decision 6), write part; merge parts to `responses`; write run metadata.
2. `eval score`: check the probes sha256; read `responses`, `probes` with hidden columns, `traits`, `signals`; score; write `scores` and `summary.json`.

## Limitations

- **Read-only `answer` is a contract, not enforced.** A system that writes during `answer` contaminates later checkpoints. Accepted: in-process isolation would need a snapshot and restore API the target copilot does not have; the contract is stated in the protocol.
- **Most open probes are unscored in this sub-project.** The open twin, in-situ, governance and intrusion wait for judges. Accepted: the deterministic scores are usable alone, and responses are stored so judges score them later without rerunning the system.
- **Format checks cover four of seven communication params, and not every value of those.** Accepted: the rest has no defensible deterministic check and goes to the judges.
- **Format checks are surface heuristics.** A hedge-word list and a sentence splitter can misfire on unusual phrasing. Accepted: the checks are fixed and published, so every system is scored by the same rule; the judges sub-project can report agreement with them.
- **No ledger visibility.** A deployed copilot sees the book. Accepted: decision 4; a ledger-visible mode would need its own ceiling.
- **The no-memory baseline still sees the profile.** It is a profile-only floor, not a zero-information one. Accepted: a copilot always knows the mandate, and the gap between it and the full-context baseline is what dialogue adds.
- **Baseline answers are not reproducible across cache loss.** Accepted: the same artefact-level reproducibility the plan accepts for dialogue.
- **Resume granularity is one PM.** A crash mid-PM repeats that PM. Accepted: the system is stateful.

## Living docs impact

- `README.md`: new section "Evaluation" after "Probes": the system-under-test protocol with the code block of decision 2, what the system observes and what is withheld, replay order, the read-only contract, the commands (`uv run pm-traitbench eval run --sut package.module:factory --data-dir data` and `eval score --run-name <name>`), the two baselines and what each bounds, the scorers with the check-map table, the summary slices, and the config knobs with their bases. The hidden-columns paragraph adds `personas.typicality`.
- `docs/pm-dataset-plan.md`: the header's "Scope is data generation only" becomes data generation plus an evaluation harness that replays the corpus into a system under test and scores the probes. Section 9 gains a paragraph after the stage paragraphs, "Evaluation harness (not a stage)", summarising decisions 2, 4, 6, 10, 11 and 12 and naming the judge and report sub-projects as later work. Section 8's hidden-column comment list gains `typicality`.
- No `ARCHITECTURE.md` exists; the README and the plan remain the living docs.

## Implementation notes

Heavy tier: the plan pins interfaces per task. Key points the plan must carry:

- `PublicProfile`, `PublicSession`, `PublicProbe` are frozen pydantic models with only the allowed fields; a test asserts their field sets exactly, so adding a field is a deliberate test change.
- A recording fake adapter in `tests/harness/fixtures.py` records every call; tests assert no hidden field name appears in any object it received, and that no session dated after a checkpoint is observed before that checkpoint's first probe.
- Baselines are tested with a stub `LlmClient`, as `tests/gates/gate2/test_stage.py` stubs the model.
- `tests/harness/test_checks.py`: one passing and one failing reply per checkable value; the loader rejects a map missing a catalogue value.
- `tests/harness/test_score.py`: parse rules of decision 11 (accepted and rejected forms), chance per option count, evidence-type derivation, balanced accuracy on presence.
- `tests/harness/test_cli.py` runs `eval run` then `eval score` end to end on the probe stage's fixture corpus (`tests/probes/fixtures.py`) with the fake adapter; copies the structure of `tests/probes/test_stage.py::test_stage_writes_probes_for_every_narrated_pm`; covers per-PM resume, an adapter exception failing one PM with exit 1, and stale probes metadata raising `HarnessError`.
- `tests/tables`: `personas.typicality` in the hidden set.
- Full suite: `uv run pytest -q`.
