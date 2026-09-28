**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** none (changes `docs/pm-dataset-plan.md` section 4 "Gate 2" paragraph, the section 9 stage 8 row and paragraph, and the section 9.1 "Sameness" risk; no earlier spec's decisions change)

# Design: stage 8, Gate 2

Date: 2026-09-28. Source of requirements: `docs/pm-dataset-plan.md`, sections 4 (Gate 2 paragraph: full-context recovery with a strong model, report per trait, kind and mode, classify each stated signal as bias or preference), 6 (mix guesses "to be checked at Gate 2"), 8 (one-way `signals` to `sessions` join), 9 (stage 8 row and paragraph, pilot order: narrate static PMs, run Gate 2, then narrate drift PMs), 9.1 (Sameness: cross-PM n-gram overlap reported by Gate 2). Builds on `docs/specs/2026-09-26-dialogue-design.md` (decision 4 defers the overlap row to Gate 2; `sessions` and `dialogue_logs` schemas) and `docs/specs/2026-09-27-validate-design.md` (decision 2: a dropped session voids its signals by join; `run_metadata/validate.json`).

## Scope

Eighth sub-project. Delivers **stage 8, Gate 2**: for every PM with validated sessions, ask one strong model, given the whole transcript, the mandate and the rules, which traits the PM shows; score the answer against ground truth; pool per trait, kind, mode and reporting slices; block the pipeline when a trait's recovery is not above chance. Two report-only companions: per stated signal, whether a model given the session, the mandate, the rules and the relevant ledger rows classifies the statement as a bias or a preference; and the cross-PM n-gram overlap of PM turns.

Out of scope: probes (stage 9), freeze (stage 10), multi-asset PMs (no sessions exist), drift detection (whether the model notices a value changed mid-year; that is the pre-drift, post-drift and governance probes' job), a per-checkpoint ceiling (recovery on a context cut at week 4, 13 and so on, which the probes' contexts use), any change to stages 5-7. Acting on a failed row is a config edit to the plan stage's mix weights and a rerun of stages 5-8 (decision 11).

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself. Citing a published paper or a named public series as the basis of a value is allowed.

## Decisions

1. **Scope is the three Gate 2 parts together: recovery, stated-signal classification, n-gram overlap.** Recovery is the gate; the other two are small once the client and transcript plumbing exist and were promised to Gate 2 by the plan and the dialogue spec. Probes and freeze stay separate.
2. **Recovery is one full-context call per PM with closed-form answers and cited sessions.** The model sees every session of the PM in date order, the mandate and the PM-scope rules, and answers in one JSON reply: for each of the 8 bias params (named, one authored definition line each), whether the PM shows it; for each catalogue preference param of the PM's asset class, which catalogue value the PM holds or that the PM does not hold it. Every positive answer cites the session ids that show it. Scoring is deterministic against `traits` and `drift_events`, and chance is known (0.5 per bias; `1/k` for the value of a held preference param with `k` catalogue values). Per-mode recovery falls out of the citations: a signal is recovered when its trait is correct and its session is cited. Rejected: per-session calls asking which candidate traits a session shows (about 80 calls per PM, and not a ceiling, since a revealed signal is unrecognisable in isolation by design); a free-form profile scored by a second judge (judge on judge, no deterministic chance level). Cost accepted: naming the trait vocabulary frames the task. The ceiling being measured is "given the vocabulary, is the trait findable"; the probes also supply options.
3. **Blocking rows use exact tests at one `alpha`, pooled over every narrated direct-asset PM, all seeds.** Nine blocking rows: one per bias param and one pooled preference row.
   - Bias param: the 2x2 table of PMs (active or inactive by truth, predicted present or not), judged by a one-sided Fisher exact test that the present rate is higher among active PMs. Pass when `p < alpha`; `insufficient` when either class has fewer than `min_class` PMs (2). With 3 active and 9 inactive PMs, perfect recovery gives `p = 1/220`, so the pilot's 12 static PMs can pass.
   - Preferences: every held (PM, param) pair pooled (about 70 at pilot), a hit when the predicted value equals the truth, judged by an exact Poisson-binomial upper tail against per-pair chance `1/k_param`. Pass when `p < alpha`; `insufficient` below `min_class` pairs. Per-param preference rows get the same test, report-only, since 2-3 holders per param cannot carry a verdict.
   - One test family and one knob, `alpha` (0.05, guess), instead of a z rule with a normal approximation that is poor at n of 4-5. Pooling real seeds with synthetic ones differs from Gate 1: narration does not depend on regime order the way ledger statistics do, and the first Gate 2 run has only about 12 PMs. At 9 blocking rows the chance of at least one false fail when every trait is only marginally recoverable is bounded by `1 - (1 - alpha)^9`, about 37%; accepted because a fail is a prompt to read the report-only rows and rerun, not an automatic re-plan. Rejected: a report-only gate (the plan makes Gate 2 a gate); per-param preference verdicts (no per-param sample); `min_pms` 5 as in Gate 1 (every pilot row would be `insufficient`, and drift PMs are only narrated after Gate 2 passes, so there would be nothing to widen the filter to).
4. **Stated-signal classification is a per-session call without the trait vocabulary, scored per signal on its own turn.** Unit: every surviving session carrying at least one own, confirm, `stated` signal (drift announcements included, since they are stated). Retracted and third-party signals are distractors, not evidence, and are skipped. Inputs: mandate, PM-scope rule texts, the idea-scope stop and target rule texts of the session's ideas, the ledger rows of the session's ideas dated at or before the session, the session transcript, and the number of statements to find (the count of such signals in the session, 1 or 2). The model quotes each statement and classifies it `bias` (honouring or breaching it would move P&L or risk) or `preference` (it would not). A signal's PM turn is known: its stance line sits on exactly one `dialogue_logs` turn as `directive`, so a signal is scored on a quote inside that turn, and a quote from some other statement in the session (a contradiction's claim, a retraction, a colleague's view) cannot score for it. Rejected: folding kind into the recovery call (the trait names give the kind away); multiset matching of kinds per session (a quote of the wrong statement with the right kind would score). The ledger is limited to the session's ideas because the whole ledger is large and mostly irrelevant to one statement.
5. **Overlap is over PM turns, word 5-grams, containment against the nearest other PM.** The advisor is one model under one prompt everywhere, and the PM side is what carries traits. Lower-cased, split on non-letters. Per PM: the maximum over other PMs of the share of this PM's distinct 5-grams that the other PM's PM turns also contain. Containment against one PM does not grow with the PM count, so the threshold carries from 12 PMs to 150; a share against the union of all others would not. Reported as a per-PM column and as median, p90 and max in run metadata, with a warning above `overlap_warning`. Never blocks; a second narrator model is a later decision (dialogue spec decision 2). Rejected: excluding n-grams that come from rule texts, stance lines or catalogue values (template phrasing is the sameness risk the plan names, so it counts).
6. **Ground truth is the value in force at the PM's last session date.** A bias with an `update` event stays active (attenuated); `dormant` and `revive` windows always close within the year, so a bias is inactive at the end only if a `dormant` event has no `revive` on or before the last session date. A preference `update` makes the `to` value the truth; the `from` value sits among the catalogue options as a natural distractor. Preference signals dated before an `update` of their trait express the old value and are excluded from the mode rows; they still appear in `gate2_signals` with `cited`. Recovery is reported split static versus drift.
7. **The gate runs only on a validated corpus.** `run_metadata/validate.json` must exist, its `created_at` must be at or after the dialogue metadata's `created_at` (the dialogue stage rewrites `sessions` whole under `--force` and leaves an older validate metadata behind), and every PM in `sessions` must be in validate's `pms`; else `Gate2Error` before any call. The PM set is every PM with at least one row in `sessions`; a validated PM whose sessions were all dropped is listed in metadata as `pms_without_sessions` and not scored. The sha256 of the `sessions` file is recorded in run metadata so freeze can check Gate 2 scored the frozen corpus.
8. **One model, high effort.** `gate2.model` defaults to `claude-opus-5-5` and `gate2.effort` to `high`: this is the ceiling, so the strong model runs at strength, unlike the narrator and judges at `low`. One recovery call per PM at about 100k input tokens costs about $0.50, so effort is not the cost driver. `recovery_max_output_tokens` is 32000 because high-effort thinking over a long context can be long, and a `max_tokens` stop is an unparsable reply that identical retries repeat.
9. **Preference params the PM does not hold are scored too, report-only.** For each candidate param the truth is the held value or "not held". Held-versus-not is reported as balanced accuracy per param (chance 0.5), and value accuracy on holders is the blocking quantity (decision 3). The `kind` row for preferences pools holders only; non-holder nulls (about 20 per PM) are easy and would swamp it.
10. **Cited sessions are verified, not trusted.** A cited id that is not one of the PM's surviving sessions is dropped with a warning. A cited session that carries a retracted or third-party signal of the same trait is a false attribution, counted on the trait row. On biases a third-party signal only ever targets an inactive bias, so a bias false attribution is always also a false positive; on preferences it is a wrong value taken from a colleague or client.
11. **The lever a failed row pulls is the plan stage's global mix weights.** `plan.bias_mode_weights` and `plan.pref_mode_weights` are per kind, not per param; the per-param mode rows say which mode of which trait is not carrying, and the user edits the weights and reruns stages 5-8. A per-param override in the plan config is follow-up work if a single trait keeps failing.

## Package layout

```
src/pm_traitbench/gates/gate2/
  __init__.py
  stage.py        # GATE2_STAGE: read, check inputs, build units, run calls, score, write, verdict
  transcript.py   # render a PM's sessions in date order as one text; render one session
  recover.py      # per-PM recovery request builder, JSON schema, reply parser, scorer
  classify.py     # per-session classification request, schema, parser, per-signal scorer
  overlap.py      # cross-PM 5-gram containment (pure)
  aggregate.py    # pool trait and signal rows into cells; exact tests; verdicts
src/pm_traitbench/catalogues/bias_definitions.yaml   # one neutral definition line per bias param
```

Reused: `dialogue/client.py` (`CachedClient`, `send_until_accepted`, `last_text_json`), `dialogue/prompts.py::base_request`, `dialogue/stage.py` (`stage_client`, `run_bounded`, `collect_results`), `catalogues/loader.py` (preference catalogue, YAML loading pattern). Generalised: `dialogue/stage.py::raise_on_failure` takes unit ids (`tuple[str, ...]`) instead of session contexts, and `send_until_accepted` takes a `label` ("session" or "pm") for its error prefix, so a PM-scoped failure is not reported as a session. New: `Gate2Error` in `errors.py` (exit code 1), `Gate2Config` registered as `Config.gate2`, stage registered in `pipeline.py` as number 8 through `make_stage(client_factory)` as validate does, CLI subcommand `gate2`.

### Data flow

1. Read `personas`, `traits`, `drift_events`, `rules`, `ledger`, `signals`, `skeletons`, `sessions`, `dialogue_logs`. Check validate and dialogue metadata (decision 7).
2. Per PM with sessions: sort sessions by `(date, session_id)`; compute truth per candidate param (decision 6); build the recovery unit; build one classification unit per session with own confirm stated signals, locating each signal's PM turn (skeleton stance text equals the turn's `directive`; no match or two matches is a `Gate2Error`, since stage 6 puts each stance on exactly one turn); compute the PM's 5-gram set.
3. Run every recovery and classification unit through one `CachedClient` under `run_bounded(max_concurrency)`, all or nothing: any unit failing after retries raises `Gate2Error` listing them, and a `DialogueBudgetError` wins over other failures; nothing is written.
4. Score: trait rows, signal rows, PM rows; pool into cells; judge.
5. Write `gate2_traits`, `gate2_signals`, `gate2_pm`, `gate2_cells` in key order (a fully cached rerun writes byte-identical tables); write run metadata; verdict hook raises `Gate2Error` on failed or insufficient blocking rows.

### Transcript (`transcript.py`)

`render_pm(sessions) -> str`: sessions in date order, each as

```
Session <session_id>, <date>
PM: ...
ADVISOR: ...
```

separated by blank lines. The session `kind` is planning metadata a deployed copilot never sees and hints where signals sit, so it is not shown. `render_session(session) -> str` is one such block.

### Recovery (`recover.py`)

Request, `scope = pm_id`, via `base_request(config.gate2.model, config.gate2.recovery_max_output_tokens, config.gate2.effort, system, messages, schema)`:

- `system`: role ("You are reviewing a year of conversations between a portfolio manager and their advisor. Decide what this PM does and wants, using only the transcript."); mandate facts (asset class, sub-style, book size, risk unit, benchmark); the PM-scope rule texts; the 8 bias params each as `<param>: <definition line>` from `bias_definitions.yaml`; the candidate preference params for the PM's asset class, each with its catalogue values listed; the instruction that a `present` bias or a held preference must cite the session ids that show it, that most PMs show a minority of the biases, and that a preference is "not held" when the transcript gives no evidence.
- `messages`: one `user` message holding `render_pm(sessions)`.
- `output_config.format` schema:

```json
{"type": "object", "properties": {
  "biases": {"type": "array", "items": {"type": "object", "properties": {
    "param": {"type": "string"}, "present": {"type": "boolean"},
    "session_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["param", "present", "session_ids"], "additionalProperties": false}},
  "preferences": {"type": "array", "items": {"type": "object", "properties": {
    "param": {"type": "string"}, "value": {"type": ["string", "null"]},
    "session_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["param", "value", "session_ids"], "additionalProperties": false}}},
 "required": ["biases", "preferences"], "additionalProperties": false}
```

Parser: exactly one entry per candidate param on each list, no unknown params, a preference `value` either null or one of that param's catalogue values (compared after strip and lower-case); anything else is unparsable and retried up to `config.dialogue.max_retries` fresh attempts through `send_until_accepted`, then the PM fails the run. Cited ids not in the PM's sessions are dropped with a warning (decision 10).

`bias_definitions.yaml`: `param -> one line`, neutral wording that describes the behaviour without the loaded term (for example `disposition_ratio: "sells positions at a gain sooner than positions at a loss, holding losers longer than winners"`). Loader check: keys equal `BIAS_PARAMS`, every line non-empty. The bias param name itself is shown to the model alongside the line, so the vocabulary is given by design (decision 2).

Scorer, per candidate param:
- Bias: `truth_active` from decision 6; `predicted_active = present`; `correct = (predicted_active == truth_active)`.
- Preference: `truth_value` is the held value in force or null; `predicted_value` from the reply; `correct = (predicted_value == truth_value)`.
- `cited_session_ids`: the verified citations. `false_attribution_ids`: cited sessions carrying a `retracted` or non-`self` signal of this param.
- Signal rows: for every signal of the PM whose `session_id` is in `sessions`: `cited = session_id in cited_session_ids of its param's row`, `recovered = cited and that row is correct`.

### Classification (`classify.py`)

Request, `scope = session_id`, `max_tokens = config.gate2.classify_max_output_tokens`:

- `system`: role ("In this conversation the PM makes N statements about how they trade or about what they want from their advisor. Find each one."); the definition of `bias` and `preference` in materiality terms (plan section 1: a bias is something whose honouring or breaching would move expected P&L or risk; a preference would not); mandate facts; PM-scope rule texts; the idea-scope stop and target rule texts of the session's ideas; the instruction to quote each statement verbatim from a PM turn.
- `messages`: one `user` message: the ledger rows of the session's `trade_idea_ids` dated at or before the session (`date, trade_idea_id, instrument_id, side, size, price_or_yield`), then `render_session(session)`, then `Find N statements.` with N the count of own confirm stated signals in the session.
- Schema: `{"statements": [{"quote": string, "kind": "bias" | "preference"}]}`, `kind` an enum.
- Parser: 1 to N statements; each quote must appear in a PM turn after whitespace and case normalisation, else dropped with a warning. Zero surviving statements is a scored miss, not a failure.
- Scorer, per signal in the unit: `kind_predicted` is the kind of the first surviving quote that lies inside the signal's own PM turn (decision 4), or null when none does; `kind_ok = (kind_predicted == kind)`, false when null. `classified = True` for every signal in a unit.

### Overlap (`overlap.py`)

`ngrams(text, n) -> set[tuple[str, ...]]`: lower-case, split on runs of non-letters, consecutive `n`-tuples. `containment_by_pm(pm_turn_texts_by_pm, n) -> dict[pm_id, float]`: for each PM, `max` over other PMs of `|own ∩ other| / |own|`; 0.0 for a PM with no n-grams or no other PM. Summary: median, p90 (nearest-rank), max.

### Aggregation (`aggregate.py`)

Cells from trait rows and signal rows. "Own confirm" below means `ownership = self` and `valence = confirm`.

| `slice` | `slice_value` | `param` | Population | `rate` | `chance` | Test |
|---|---|---|---|---|---|---|
| `all` | `all` | bias param | every scored PM | balanced accuracy | 0.5 | Fisher exact, one-sided, on the 2x2; blocking |
| `all` | `all` | null (pooled preferences) | held (PM, param) pairs | value accuracy | mean of `1/k` | Poisson-binomial upper tail; blocking |
| `all` | `all` | preference param | PMs holding the param | value accuracy | `1/k` | Poisson-binomial (here binomial) upper tail; report-only |
| `held` | `all` | preference param | every scored PM of an asset class the param covers | held-versus-not balanced accuracy | 0.5 | Fisher exact; report-only |
| `kind` | `bias` or `preference` | null | bias rows, or preference rows of holders | share correct | null | none |
| `mode` | `stated`, `revealed`, `contradiction` | bias or preference param, or null for pooled | surviving own confirm signals of that mode (and param), excluding preference signals dated before an `update` of their trait | share recovered | null | none |
| `asset_class`, `typicality`, `drift` | the cell value | bias param or null (pooled preferences) | scored PMs in the cell | as the `all` row | as the `all` row | as the `all` row, report-only |

Exact tests: Fisher's one-sided p is the hypergeometric upper tail of the active-and-present count given the margins. The Poisson-binomial upper tail is `P(X >= hits)` for `X` a sum of independent Bernoullis with success `1/k_i`, computed by the standard dynamic programme over pairs (about 70 terms, exact in floating point). `p` is stored on the row; `z` is not. Verdict: `insufficient` when a 2x2 class has fewer than `min_class` PMs or a pooled row has fewer than `min_class` pairs; `pass` when `p < alpha`; `fail` otherwise. Rows without a test get verdict null. `blocking` is true only for the nine `all` rows with a bias param or the null pooled-preference param.

## Tables

`gate2_traits`, key `(pm_id, param)`: `trait_id` (str or None: null when the PM does not hold the preference param), `kind`, `truth_active` (bool or None), `truth_value` (str or None), `predicted_active` (bool or None), `predicted_value` (str or None), `correct` (bool), `cited_session_ids` (tuple[str]), `false_attribution_ids` (tuple[str]). Validators: a bias row has `truth_active` and `predicted_active` non-null and both value fields null; a preference row has both active fields null (its value fields may each be null, meaning not held or predicted not held).

`gate2_signals`, key `(pm_id, signal_id)`: `session_id`, `trait_id`, `param`, `kind`, `mode`, `valence`, `ownership`, `pre_update` (bool: a preference signal dated before an `update` of its trait), `cited` (bool), `recovered` (bool), `classified` (bool), `kind_predicted` (Kind or None), `kind_ok` (bool or None). Validators: `recovered` implies `cited`; `kind_ok` set exactly when `classified`; `kind_predicted` null implies `kind_ok` false or null.

`gate2_pm`, key `(pm_id,)`: `asset_class`, `typicality`, `drift` (`static` or `drift`), `seed`, `sessions` (int), `context_chars` (int), `ngram_containment` (float in [0, 1]), `biases_correct` (int), `preferences_held` (int), `preferences_correct` (int), `stated_signals` (int), `stated_kind_ok` (int).

`gate2_cells`, key `(slice, slice_value, param)`: `slice` (enum above), `slice_value` (str), `param` (str or None), `n` (int), `n_positive` (int: correct, recovered or hit count), `rate` (float or None), `chance` (float or None), `p` (float or None), `verdict` (Gate2Verdict or None), `blocking` (bool). Null `param` sorts first within a slice, as `gate1_cells` does with a null `asset_class`.

All four public, like `gate1_pm` and `gate1_cells`: they are dataset checks over ground truth, not narrator provenance.

## Config: `gate2`

| Field | Default | Basis |
|---|---|---|
| `model` | `claude-opus-5-5` | design: strongest current model, one model |
| `effort` | `high` | design: the ceiling wants the strong model at strength (decision 8) |
| `recovery_max_output_tokens` | 32000 | design: high-effort thinking over about 100k tokens of transcript plus one JSON object; a `max_tokens` stop is an unparsable reply that retries repeat |
| `classify_max_output_tokens` | 2000 | design: thinking plus one or two quotes |
| `alpha` | 0.05 | guess: one-sided exact test level per blocking row; nine rows, so the family-wise false-fail bound is about 0.37 |
| `min_class` | 2 | design: the smallest class an exact test can say anything about |
| `ngram_n` | 5 | guess: long enough to be a phrase, short enough to recur |
| `overlap_warning` | 0.15 | guess: to be re-centred on pilot output |
| `max_concurrency` | 4 | design: long-context calls, half the dialogue stage's |
| `token_budget` | None | design: same semantics as dialogue |

Every leaf carries `basis` and `note`.

## Run metadata extras

`model`, `pms`, `pms_without_sessions`, `sessions_sha256`, `failed` (blocking rows not passing, `all/<param>` or `all/preferences`), `insufficient` (blocking rows short of `min_class`), `overlap` (`median`, `p90`, `max`), `warnings` (unknown cited ids, dropped quotes, containment above threshold, in PM order), `calls`, `cache_hits`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `rejected_replies`, `thresholds` (the `gate2` config dump).

## Error handling

- Missing tables: caught by `run_stage`. Missing or stale `run_metadata/validate.json`, a `sessions` PM absent from validate's `pms`: `Gate2Error` before any call.
- A signal whose stance text matches no turn or more than one turn of its session: `Gate2Error` naming the signal, before any call.
- Budget exhaustion: `DialogueBudgetError` wins; nothing written.
- A PM or session whose reply stays unparsable after `config.dialogue.max_retries`: collected; `Gate2Error` listing them after every unit has run; nothing written.
- Verdict: `raise_on_failures` reads `extra["failed"]` and raises `Gate2Error("gate 2 failed for: all/<param>, ...")` after tables and metadata are on disk, as Gate 1 does.
- Output exists and no `--force`: refused by `run_stage`.

## Testing

`tests/gates/gate2/` (Gate 1's tests stay where they are).

- `transcript.py`: date order, header has id and date only, both roles; one session render equals its block.
- `recover.py`: request contains mandate, rules, every candidate param and every catalogue value, no trait id, trait value or stance line; parser rejects a missing, duplicate or unknown param and an off-catalogue value; unknown cited ids dropped with a warning; scorer on hand-built traits and drift events (update keeps a bias active, dormant without revive makes it inactive, preference update moves the truth); false attribution on a cited third-party session.
- `classify.py`: N derived from the session's own confirm stated signals; the signal's turn found from the stance text; a quote outside the signal's turn does not score for it; quote-not-in-PM-turn dropped; 2-signal session with one right and one wrong.
- `overlap.py`: identical PMs give 1.0, disjoint PMs 0.0, a PM with fewer than `n` words gives 0.0, containment is against the nearest PM not the union, summary statistics.
- `aggregate.py`: Fisher p on a hand-computed 2x2 (3 active, 9 inactive, perfect recovery gives 1/220); Poisson-binomial tail against a brute-force enumeration on 6 pairs; `insufficient` below `min_class`; pass and fail at `alpha`; blocking only on the nine `all` rows; `held`, `kind` and `mode` rows present; pre-update preference signals excluded from mode rows.
- `stage.py`: chain sample, market, engine, plan, dialogue (fake), validate (fake) and gate2 (fake client scripted by system prompt: recovery replies built from the fixture's ground truth so the pass path is exact, plus a wrong-answer script for the fail path); asserts one `gate2_traits` row per PM per candidate param, one `gate2_signals` row per surviving signal, verdict raises after tables exist, byte-identical rerun from cache with zero inner calls, stale validate metadata raises, budget error writes nothing. Copy `tests/dialogue/validate/test_stage.py`'s chained fixture and fake-client pattern.
- `bias_definitions.yaml`: loader checks in `tests/catalogues/test_loader.py`, shipped file in `tests/catalogues/test_shipped.py`.
- `raise_on_failure` and `send_until_accepted` generalisation: existing dialogue and validate tests unchanged in behaviour; one test that a PM-scoped failure is labelled `pm`.
- `tests/test_end_to_end.py`: extend the CLI chain through `gate2` with the fake client.

## Living docs impact

- `README.md`: usage block gains `uv run pm-traitbench gate2 --config configs/demo.yaml --data-dir data`; new "Gate 2" section after "Validate": what it reads, the four tables, the nine blocking rows and the exact tests, the report-only rows, the containment overlap, that it needs a validated corpus, that a failed row is acted on by editing the plan stage's mix weights and rerunning stages 5-8, credentials only on cache misses.
- `docs/pm-dataset-plan.md`: section 4 Gate 2 paragraph gains the closed-form recovery with citations and the per-mode rule (recovered when the trait is correct and the session is cited); section 9 row 8 reads stage 7 output and writes `gate2_traits`, `gate2_signals`, `gate2_pm`, `gate2_cells`; stage 8 paragraph states the blocking rule (per bias param a one-sided Fisher exact test, preferences one pooled Poisson-binomial row, `alpha` 0.05, pooled over all narrated PMs and seeds) and the classification unit; section 9.1 "Sameness" names the 5-gram containment definition and its warning threshold.
- No `ARCHITECTURE.md`; the README and the dataset plan are the living docs, as in every earlier spec.

## Limitations

- **Vocabulary is given.** The recovery call names the bias params and lists catalogue values, so it measures findability given the frame, not open discovery. Accepted: the probes give options too, and the copilot's memory has its own vocabulary.
- **Citations are self-reported.** A correct trait with missing or wrong citations undercounts per-mode recovery; per-mode rows are report-only for this reason, and only the nine `all` rows block.
- **Classification ledger is the session's ideas only.** A statement about a trade outside the session's ideas cannot be checked against its rows. Accepted: stage 6 lists only the session's ideas on the narrator directive, so the statement almost always concerns them.
- **Overlap counts template phrasing** from stance lines, rule texts and catalogue values, ignores advisor turns, and uses one n and one threshold, both guesses. Accepted: template sameness is the risk being measured; report-only, re-centred on pilot output.
- **Pilot sample is small.** About 12 direct-asset static PMs on the first run and 3-5 active per bias, so a pass needs near-perfect recovery on those PMs, and a single miss on a 3-active bias fails it. That is what the pilot is for; a fail is read with the report-only rows before any re-plan.
- **Nine blocking rows at `alpha` 0.05** give a family-wise false-fail bound near 0.37 when recovery is marginal. Accepted per decision 3.
- **No per-checkpoint ceiling.** Recovery is on the full year against end-of-year truth; the probes' contexts are cut at earlier checkpoints, where the ceiling may be lower. Stage 9 can tag probes on traits Gate 2 did not recover from `gate2_traits`.
- **Drift detection is not measured.** Truth is the end-of-year value; whether the model notices the change is left to the probes.
- **Per-trait mix has no per-trait lever** (decision 11).
- **One model.** No second recoverer or agreement rate; the plan's section 7 human check bounds judge noise, as for stage 7.
- **Pooling real seeds.** A real-seed PM's recovery counts like a synthetic one, unlike Gate 1. Accepted per decision 3.
