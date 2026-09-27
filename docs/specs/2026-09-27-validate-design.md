**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** none (changes `docs/pm-dataset-plan.md` section 4 "Validator", the section 9 stage 7 row and paragraph, and the section 9.1 "Judges judging judges" risk; no earlier spec's decisions change)

# Design: stage 7, validate

Date: 2026-09-27. Source of requirements: `docs/pm-dataset-plan.md`, sections 4 (validator, three layers), 8 (`signals.jsonl` join rule), 9 (stage 7 row and paragraph), 9.1 (prose parsing, judges judging judges, narrator reverts to stereotype). Builds on `docs/specs/2026-09-26-dialogue-design.md`, whose `sessions`, `dialogue_logs`, `Mention` schema and `feedback` hook are this stage's contract.

## Scope

Seventh sub-project. Delivers **stage 7, validate**: check every session stage 6 wrote against the ledger, the leakage rule and the PM's forbidden set; regenerate failing sessions through stage 6's session driver with the failure reason as narrator feedback; drop sessions that still fail after a fixed attempt cap; write a hidden `validation` table and rewrite `sessions` and `dialogue_logs` in place.

Out of scope: Gate 2 (recovery, cross-PM n-gram overlap); probes; re-planning a dropped session's signals onto a new carrier (see decision 2); a second judge model (decision 3); changes to stage 5 or stage 6 beyond what regeneration needs.

## Decisions

1. **Results live in place, plus a `validation` table.** Stage 7 appends to `sessions` and `dialogue_logs` (`Append` with `owned = record["pm_id"] in validated PMs`): passing sessions are kept, regenerated ones replaced, dropped ones removed. A new hidden `validation` table records one row per session per attempt. Downstream stages keep reading `sessions` as the corpus. Rejected: separate `sessions_validated` tables (every later stage would have to know which table is the corpus, and stage 6's raw output would sit next to the real one).
2. **A dropped session voids its signals by join, not by a column.** The session's rows are removed; its signals stay in `signals.jsonl` untouched. Plan section 8 already fixes the join direction (`signals.session_id` -> `sessions`), so a signal whose session no longer exists is void by construction and Gate 2 and probes filter by that join. Void signal ids and the per-PM shortfall are reported in run metadata. Rejected: a `status` column on `signals` (stage 5 would have to write it too, `Signal` is `extra="forbid"`); re-planning inside stage 7 (stage 7 would own stage 5's carrier allocation and the skeleton table; a re-plan is a rerun of stages 5-7 for that PM); failing the run (a handful of stubborn sessions would block the whole corpus). Cost: the signal mix per PM can fall short of plan section 4's targets; accepted because the shortfall is reported and Gate 2 measures the result directly.
3. **One judge model, no agreement rate.** `judge_model` defaults to `claude-opus-5-5`. Plan section 9 asked for two models with a logged agreement rate; changed here to one judge to halve judge cost and keep the loop simple. Cost: the plan 9.1 "judges judging judges" bound (agreement below 0.8 means tighten the layer) cannot be reported; accepted because the deterministic layers carry the hard rules (ledger, grep) and the judge layers are tuned on pilot output either way. The plan text is updated (Living docs impact).
4. **Judge scope: leakage on revealed carriers, forbidden on every session.** Only a session with a `revealed` or `contradiction` stance has a planted bias whose label could leak; an unplanted trait or preference can surface anywhere, so the forbidden judge runs on all sessions, silence included. About 1.3 judge calls per session. Rejected: both judges everywhere (a false-positive baseline for the leakage judge at twice the cost); judges on carriers only (forbidden traits in filler unchecked).
5. **The leakage judge tests labels, not behaviour.** Zero-context, no option list. A judge offered behaviour descriptions (the `avoid.yaml` lines) would name the bias on every well-planted revealed signal, since revealed signals are behaviour by definition, and the corpus would never pass. The judge is asked whether the speaker explicitly names or self-labels a tendency (a term for it, or a general habit stated about themselves), as opposed to describing a one-off decision; a session fails only when the label maps to the param of a revealed stance in that session. Stated signals in the same session ("I tend to take profits early") are legitimate and map to a different trait (stance trait ids are unique per session), so they do not fail the check.
6. **Level mentions are report-only.** Advisor `level` mentions are compared with that turn's logged tool results, PM `level` mentions with the market on or before the session date; mismatches are counted in `level_warnings`, never fail. The dialogue spec's limitation stands: the advisor may derive figures (a spread change, a percentage move) that no tool returned verbatim. Rejected: failing on level mismatch (regenerates whole sessions for arithmetic the transcript is allowed to contain).
7. **Every same-day trade must be mentioned.** Stage 5 applied `plan.ledger_session_percentile` when it chose which ideas a session discusses, and stage 6 puts every same-day ledger row of those ideas on the turn-0 directive, so "above the size threshold" is already "listed on the directive". Stage 7 does not recompute the threshold. Trade mentions may also point at earlier rows of the session's ideas (a PM recalls last week's entry), so a mention matches any row dated at or before the session date.
8. **Regeneration re-narrates the whole session.** `feedback` changes every narrator request, so the PM texts change and every advisor request misses the cache too. The feedback line carries the attempt number, so a repeated reason still yields a new cache key. Rejected: regenerating single turns (the driver has no per-turn entry point and later turns depend on the changed one).
9. **No verdict raise.** Dropped sessions are an accepted outcome; the stage exits 0 unless the run itself fails (budget, API error, missing input), in which case nothing is written, as in stage 6. Regeneration rate above 30% in a typicality cell is a warning in run metadata (plan 9.1, "narrator reverts to stereotype").
10. **Validates every PM present in `sessions`.** Stage 6's `pm_filter` already chose the PMs; a second filter would let the two tables disagree about which PMs are validated.

## Components

```
src/pm_traitbench/dialogue/validate/
  __init__.py
  stage.py      # VALIDATE_STAGE: read, partition per PM, run loop, write tables and metadata
  ledger.py     # trade-mention matching, coverage, level warnings (pure functions)
  grep.py       # param-name and banned-word regex over PM turns (pure)
  judge.py      # prompt builders, JSON schemas, result parsing, label-to-param mapping
  loop.py       # per-session validate -> regenerate -> revalidate, attempt cap
src/pm_traitbench/catalogues/bias_labels.yaml   # phrases per bias param a judge label may use
```

`dialogue/session.py`, `dialogue/client.py`, `dialogue/context.py` and `dialogue/prompts.py` are reused unchanged. `feedback` is the only stage 6 surface stage 7 drives.

### Data flow

1. Read: every table stage 6 reads (`personas`, `rules`, `traits`, `drift_events`, `ideas`, `ledger`, `position_days`, `skeletons`, five `market/` tables), plus `signals`, `sessions`, `dialogue_logs`. Require `run_metadata/dialogue.json` (for the advisor prompt path and model ids the regeneration must reuse).
2. Per PM: rebuild `SessionContext`s exactly as stage 6 does (`draw_voice`, `MarketLookup`, `build_contexts`), index ledger rows by `(trade_idea_id, instrument_id, tenor, side)`, index stances by session.
3. Per session, `loop.validate_session`: run the deterministic layers, then the judge layers for whichever the session qualifies for; collect `reasons`. Pass -> row `status=pass`. Fail -> row `status=regenerate`, build feedback, call `narrate_session(ctx, client, config.dialogue, advisor_prompt, feedback=...)`, validate the new result; repeat until pass or `attempt == max_attempts`, then `status=dropped`.
4. Write: `validation` (all attempts); `sessions` and `dialogue_logs` with the final rows of validated PMs replacing their earlier rows and dropped sessions removed; run metadata.

Sessions run concurrently under `Semaphore(max_concurrency)` with `gather(return_exceptions=True)`; the same all-or-nothing failure grouping as stage 6 (`_raise_on_failure`, extracted to a shared helper if the two stages would otherwise duplicate it).

### Layer 1: ledger consistency (`ledger.py`)

Input: the session's PM turns' `mentions` (advisor turns cannot carry trade mentions), the PM's ledger rows, the skeleton's `date` and `trade_idea_ids`.

- A `trade` mention with `trade_idea_id` null fails: `invented trade`.
- A `trade` mention matches a ledger row when `(trade_idea_id, instrument_id, tenor, side)` are equal, the row's date is at or before the session date, and `abs(size - row.size) <= size_tolerance * row.size`. A mention with no match fails: `trade not in ledger`, naming the mention.
- Coverage: every ledger row with `date == session date` and `trade_idea_id in skeleton.trade_idea_ids` must have at least one matching mention. A missing row fails: `trade not mentioned`, naming the row.
- Level warnings (report-only): an advisor `level` mention is confirmed when any numeric value in the same turn's `tool_calls[].result_json` (recursively, including `levels` maps and `points` lists) is within `level_tolerance` relative of `value`; a PM `level` mention is confirmed against `MarketLookup` for its `instrument_id`, `tenor`, `field` on or before the date. Unconfirmed mentions increment `level_warnings`.

### Layer 2: grep (`grep.py`)

Over every PM turn's `text`, lower-cased: a whole-word or whole-phrase match on any of `BIAS_PARAMS`, the catalogue's preference params (each in its raw and underscore-to-space form, the `_voice_line_matched_param` rule, which moves to a shared helper), or `BANNED_STANCE_WORDS` (substring match, as the loader uses it). A match fails: `names a parameter: <word>`. Advisor turns are not grepped: the advisor never sees the PM's traits, and the narrator is the only side regeneration changes.

### Layer 3: judges (`judge.py`, `bias_labels.yaml`)

Both judges go through the stage's `CachedClient` (same cache directory `cache/llm`, scope = session id) with `output_config.format` JSON schemas and `output_config.effort = config.validation.effort`, `max_tokens = config.validation.max_output_tokens`. Requests carry only the transcript text (all turns, both roles, in order, as one user message) and the judge instruction; never a trait id, param, stance line, mode or the PM's row. A refusal, truncation or unparsable reply is retried up to `config.dialogue.max_retries` fresh attempts, then the session fails the run (as a stage 6 rejected reply does).

Leakage judge (sessions with a `revealed` or `contradiction` stance):
- Instruction: read the PM side; decide whether the PM explicitly names or self-labels a psychological or trading tendency of their own, either by a term for it or by stating it as a general habit ("I always", "I tend to", "my weakness is"); describing one decision on its merits is not explicit. Return `{"explicit": bool, "label": string or null, "quote": string}`.
- `catalogues/bias_labels.yaml`: an authored catalogue, `param -> list of lower-case phrases`, one key per bias param (for example `disposition_ratio`: "disposition", "sell winners", "selling winners", "hold losers", "holding losers"; `loss_aversion_lambda`: "loss aversion", "loss averse", "averaging down", "average down", "get back to even"). Loaded and checked by `catalogues/loader.py` like `avoid.yaml`: keys equal `BIAS_PARAMS`, every list non-empty, phrases lower-case and unique across params, no phrase equal to a param string. `judge.map_label(label, labels) -> param or None` returns the first param whose phrase is a substring of the lower-cased label. Same split as the rest of the repo: authored data in YAML, code in the module.
- Fail when `explicit` and `map_label(label)` equals the `trait.param` of a revealed or contradiction stance in the session: `leaks <param>: "<quote>"`.

Forbidden judge (every session):
- Instruction: here is a transcript and a numbered list of things the PM must not do; list every item the PM does, with a short quote. Return `{"violations": [{"index": int, "quote": string}]}`. The list is the session's `avoid_lines` (already rendered from `avoid.yaml` for the skeleton's forbidden sets by `context.py`), numbered from 1.
- Fail when `violations` is non-empty and every index is in range: `forbidden: <avoid line>: "<quote>"` per item. Out-of-range indices are dropped with a warning; if nothing remains the layer passes.

### Regeneration (`loop.py`)

Feedback text, one block appended to the narrator system prompt by stage 6's existing hook:

```
Attempt {n}. A validator rejected the previous version of this session:
- <reason 1>
- <reason 2>
Fix these and keep everything else as instructed.
```

Reasons use the layer wording above. Each attempt produces one `validation` row; the final attempt's `SessionResult` replaces the session's rows, or the session is dropped at `max_attempts`. Warnings and `rejected_replies` from regeneration accumulate into run metadata like stage 6's.

## Tables

`validation` (hidden in full; every non-key column in `HIDDEN_COLUMNS`), key `(pm_id, session_id, attempt)`:

| Column | Type | Meaning |
|---|---|---|
| `pm_id`, `session_id` | str | as `sessions` |
| `attempt` | int, ge 1 | 1 is stage 6's version |
| `status` | `pass`, `regenerate`, `dropped` | `regenerate` means a later attempt exists |
| `ledger_ok`, `grep_ok`, `forbidden_ok` | bool | layer results |
| `leak_judged` | bool | whether the leakage judge ran |
| `leak_ok` | bool | true when not judged |
| `level_warnings` | int, ge 0 | report-only count |
| `reasons` | tuple[str] | empty when `status` is `pass` |
| `judge_model` | str | recorded per row so a later rerun with a different judge is visible |

Validators: `status == pass` iff all `*_ok` are true; `reasons` non-empty iff `status != pass`.

`sessions` and `dialogue_logs`: unchanged schemas; stage 7 appends with `owned = pm_id in validated PMs`.

## Config

`ValidateConfig`, registered as `Config.validation` (not `validate`, which would shadow `BaseModel.validate` and warn on import), every leaf with `basis` and `note`:

| Field | Default | Basis |
|---|---|---|
| `judge_model` | `claude-opus-5-5` | design: strongest current model; one judge per decision 3 |
| `effort` | `low` | design: a yes/no reading task |
| `max_output_tokens` | 1000 | design: two short JSON objects |
| `size_tolerance` | 0.05 | guess: the narrator sees raw floats and may round to a desk-sized figure |
| `level_tolerance` | 0.01 | guess: quoted levels are rounded to display precision |
| `max_attempts` | 3 | guess: one original plus two regenerations; plan 9.1 expects about 20% regeneration, so a third failure is a prompt problem |
| `max_concurrency` | 8 | design: same as dialogue |
| `token_budget` | None | design: same semantics as dialogue |

## Run metadata extras

`judge_model`, `pms`, `sessions_checked`, `fails_by_layer` (ledger, grep, leak, forbidden), `regenerated`, `dropped`, `dropped_session_ids`, `void_signal_ids`, `void_signals_by_pm`, `regeneration_rate_by_typicality`, `warnings` (includes the 30% cell warning and out-of-range judge indices), `calls`, `cache_hits`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `rejected_replies`.

## Error handling

- Missing inputs or `run_metadata/dialogue.json`: `StageIOError` / `ValidateError` before any call.
- Budget exhaustion: `DialogueBudgetError` wins over other failures; nothing written.
- Judge or narrator API failure past retries: `ValidateError` naming the session ids, grouped by reason; nothing written.
- `validation` exists and no `--force`: refused by `run_stage`. Under `--force` the stage re-validates the current `sessions` rows; attempt numbering restarts at 1.

## Testing

- `ledger.py`, `grep.py`, `judge.py` parsing and `map_label`: pure unit tests on hand-built `Mention`, `LedgerRow`, `TurnLog` objects and fake judge replies.
- `loop.py`: fake client responder that fails a session on a chosen layer for the first k attempts, asserting feedback text, attempt rows and the drop at the cap.
- `stage.py`: chain engine, plan, dialogue (fake) and validate (fake) on the fixture market; assert 1:1 `validation` rows on the pass path, replaced and removed rows on regenerate and drop, `Append` ownership preserved for an unvalidated PM, byte-identical rerun from cache, budget error writes nothing, metadata keys.
- `bias_labels.yaml`: loader checks tested in `tests/catalogues/test_loader.py` (bad keys, empty list, duplicate phrase, param string as phrase) and the shipped file in `tests/catalogues/test_shipped.py`. Validate unit tests live under `tests/dialogue/validate/`.
- `ValidateError` is added to `errors.py` with exit code 1, following the one-class-per-stage pattern.

## Living docs impact

`docs/pm-dataset-plan.md`:
- Section 4 "Validator": layer 1 (leakage) judge tests explicit labels, not behaviour; layer 2 (ledger consistency) states the coverage rule as "every same-day trade of the session's ideas"; adds that level mentions are checked report-only.
- Section 9 stage 7 row: writes `validation`, rewrites `sessions` and `dialogue_logs`. Stage 7 paragraph: one judge model, no agreement rate; drop voids signals by join; regeneration re-narrates the whole session with the attempt number in the feedback.
- Section 9.1 "Judges judging judges": replaced with the single-judge statement and the pilot human check as the noise bound.

No `ARCHITECTURE.md` exists; the dataset plan is this repository's living doc.

## Limitations

- **One judge.** No agreement rate, so judge noise is unbounded until the section 7 human check; accepted per decision 3.
- **Leakage judge measures labels only.** A revealed signal narrated as blatant, repeated behaviour without a label passes; that is Gate 2's job to measure as recoverability, not leakage.
- **Label catalogue is hand-written.** A judge label outside the table maps to no param and the session passes; the unmapped labels are logged in warnings so the table can grow on pilot output.
- **Rerun restarts attempt counts.** Under `--force` the previous `validation` table is overwritten and attempt 1 is whatever `sessions` currently holds; accepted because judge and narrator replies are cached, so the rerun is cheap and the earlier table is in git history of the data directory if kept.
- **Level checks are report-only** (decision 6).
- **Dropped signals are not re-planned** (decision 2).
- **Forbidden judge sees behaviour descriptions**, so a neutral PM's ordinary decision (taking a profit at target) can be flagged as a forbidden bias by an over-eager judge; the regeneration rate per typicality cell is the monitor, and the avoid lines are the tuning surface.
