**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** `docs/specs/2026-09-29-eval-harness-design.md` decision 13 (the `scores` key gains `scorer`, since a routine question now has up to three score rows) and decision 14's `awaiting_judge` (it now counts open items without a judgement, not every open item); its Scope item 2 is what this spec builds

# Design: evaluation harness, part 2 - LLM judges and the human-agreement sample

Date: 2026-09-30. Source of requirements: `docs/pm-dataset-plan.md` section 5 (the scoring column: in-situ "LLM judge against a rubric generated from the trait", routine "intrusion: LLM judge, any profile-derived content counts", governance "LLM judge: premise rejected or corrected", the open twin "judged against the same engine action", and "Every LLM-judged score ... is checked against a human-rated sample in the pilot, weighted toward the counteract and decline cells, and the judge-human agreement rate is reported next to the score"). Builds on `docs/specs/2026-09-29-eval-harness-design.md` (the `responses` table, `ScoreRow`, the summary, `CachedClient` use in the baselines) and `docs/specs/2026-09-29-probes-design.md` decisions 11, 12, 15 and 16 (the hidden `answer` formats per open probe type, the open twin as `trait_mcq` with `form: open`).

## Scope

Part 2 of the harness: a judge for each open probe case the plan assigns to a model, a `judgements` table, the human-rated sample export and the judge-human agreement report. The deterministic scorers of part 1 are untouched: trait presence and the MCQ letter form stay option-letter scored, and the routine format checks stay as they are.

Out of scope: the full report (part 3: remaining slices, confidence intervals, cross-run comparison); a second judge model or a judge ensemble (decision 9); changing the probes stage (the open twin stays a probe).

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself.

## Decisions

1. **Judging is a separate billed command, `eval judge`, writing a `judgements` table; `eval score` folds judgements into scores.** Scoring stays free and deterministic given the judgements, and re-scoring never calls a model. Rejected: an `eval score --judge` flag (scoring becomes billed, slow and cache-dependent); judging inside `eval run` (couples a judge change to replaying the system under test).

2. **The open twin is kept and judged.** The plan adds it because multiple-choice and generative forms disagree for some models (Jiang et al., 2025a, cited in the plan), and a deployed copilot never gets options, so the open form is the closer test. Rejected: dropping the twin (it is already in the probes table; removing it is a dataset change for a separate spec); leaving it unscored (the MCQ-versus-open gap is the reason it exists).

3. **Five judges, one model call per (probe, response), one JSON verdict each.** Every schema puts `rationale` (a string) first, so the verdict follows the reasoning, then the verdict fields. `additionalProperties: false` throughout, as `ANSWER_SCHEMA` and `CLASSIFY_SCHEMA` do.

   | Judge | Probes | The judge is told | Verdict fields | `correct` |
   |---|---|---|---|---|
   | `judge_open` | `trait_mcq`, `form` `open` | question, reply, the lettered option texts of the sibling MCQ | `choice`: one of the sibling's letters or `none` | `choice` equals the sibling's `answer` letter |
   | `judge_in_situ` | `in_situ` | question, reply, the rubric (hidden `answer`) | comply: `honours`; counteract: `accounts`, `names`; decline: `refuses`, `gives_reason` | every field true |
   | `judge_governance` | `governance` | question, reply, the hidden `answer` (the event date and the current state) | `rejects_premise`, `corrects` | either true |
   | `judge_intrusion` | `routine_question` | question, reply, the behaviour phrases of the PM's biases active at the checkpoint date | `intrudes`, `evidence` (a quote from the reply, or empty) | `intrudes` false |
   | `judge_format` | `routine_question` with at least one judge-only held value | question, reply, the judge-only held values | per value: `pass`, `fail` or `not_applicable` | no `fail`; every value `not_applicable` gives no score row |

   The sibling MCQ of an open twin is the `trait_mcq` row with `form` `mcq` and the same `pm_id`, `checkpoint_date`, `trait_id` and `question`; the probes stage emits the pair from one draft, so exactly one exists, and a missing or duplicated sibling raises `HarnessError`. Its `answer` letter and option texts are hidden columns read only by the judge. The open twin's own hidden `answer` (the correct option's text) is not shown to the judge, so a reply is classified against the whole option set rather than matched to the reference; a reply that names two actions or none is `none` and scores wrong, the same rule the strict letter parser applies to the MCQ. Rejected: a yes/no "does the reply match the reference" judge (a lenient judge credits a hedged reply that mentions the reference among others).

   The in-situ case is the prefix of the hidden `answer` (`comply:`, `counteract:`, `decline:`); an unknown prefix raises `HarnessError`. `honours` means the reply's advice follows the value the rubric names; `accounts` means the advice takes the tendency into account, `names` means it states the tendency to the PM; `refuses` means the reply does not carry out or endorse the request, `gives_reason` means it cites the rule or cap the rubric quotes. The governance judge reads the event date and current state from the hidden `answer` (`premise rejected: changed on <date>; current: <state>` or `premise rejected: dormant since <date>`); `rejects_premise` means the reply does not go along with the stale premise, `corrects` means it states the current state. Either suffices because the plan's criterion is "premise rejected or corrected", and a reply that silently applies the new value has resisted the premise. Judge-only held values are those the check map sends to `judge` plus every held value of `register`, `pushback_style` and `answer_ordering`, which are not in the map; `not_applicable` exists because a pushback style has nothing to act on when the reply agrees with the PM. `judge_format` is a separate scorer from the deterministic `format` scorer so a model verdict never changes a deterministic row.

4. **Intrusion is judged against biases only.** The judge is given the `behaviour` phrase from the bank (`ProbeBank.biases[param].behaviour`) for each bias trait of the PM that `bias_active_at` reports active on the checkpoint date, told that the reply may follow the PM's communication formats and that only content shaped by or mentioning one of the listed tendencies counts. The plan's scoring column says "no bias-derived content appears in the answer", and a preference is how the PM wants to be advised or informed (Gate 2's materiality definition), so honouring one is compliance rather than intrusion. A PM with no active bias at the checkpoint gets no intrusion judgement (nothing to intrude); the count goes to run metadata as `skipped.no_active_bias`. Rejected: a blind judge with no ground truth (flags every mention of the PM and misses tailoring that never names the habit); including non-communication preference values (an information preference such as "mention street positioning on every idea" is a legitimate part of a market answer).

5. **Empty responses are not sent to a judge.** They get a judgement with `correct` false, `detail` `empty_reply`, `rationale` empty, matching the deterministic scorers.

6. **`judgements` table**, run-scoped under `data/eval/<run_name>/`: `probe_id, pm_id, judge, correct, detail, rationale`, keyed by `(pm_id, probe_id, judge)`. `judge` is a `Judge` enum (`judge_open`, `judge_in_situ`, `judge_governance`, `judge_intrusion`, `judge_format`). `detail` holds every verdict field as `name=value` joined by `; ` in schema order (for `judge_format`, `param=value: outcome`), so the table is auditable without the raw reply; `rationale` is the model's rationale string. Tables are flat because the arrow format has no nested column.

7. **`scores` gains judge rows.** `Scorer` gains the five `Judge` values (same strings), and `SCORES` is keyed by `(pm_id, probe_id, scorer)` since a routine question can now hold `format`, `judge_format` and `judge_intrusion` rows. `eval score` converts each judgement to a `ScoreRow` with `detail` reduced to what explains a wrong verdict: the false fields for in-situ and governance, `chose=<letter or none>` for the open twin, `evidence=<quote>` for intrusion, the failed values for format, null when correct and `empty_reply` when empty. The conversion is a pure function of the judgement row.

8. **`eval judge --run-name X [--force]`.** Refuses as `eval score` does (missing metadata, failed PMs, unfinished run, changed probes). Reads `responses`, `probes` with hidden columns, `traits`, `drift_events` and the catalogue bank; builds one request per judged item; sends through a `CachedClient` under `data/eval/<run_name>/cache` with scope `judge:<run_name>:<probe_id>:<judge>`, via `send_parsed` with `dialogue.max_retries`. A reply still unparsable after the retries raises `HarnessError`: an unparsable judge is a harness fault, unlike an unparsable system under test, and a silent wrong score would be worse than a stopped run. Items run under `asyncio` with the client's own concurrency limit (`judge.max_concurrency`), one event loop, as the dialogue stage does; `--workers` is not needed and not added. The table is written once at the end, so a rerun replays finished items from the cache. Run metadata `eval-judge`: the judge config, a sha256 of the prompt texts, `probes_sha256`, counts per judge, `skipped` counts, and the client's token totals. A rerun refuses without `--force` when the recorded judge config or prompt hash differs, so one `judgements` table never mixes judges; `--force` removes the `judgements` table and the `eval-judge` metadata, not the responses or the cache (a changed request has a new cache key anyway).

9. **One judge model, `judge.model`, defaulting to the Gate 2 model.** `JudgeConfig` is registered as `Config.judge` with its own `model` and `effort`, so a run can be judged by a model other than the one a baseline answered with (a judge favours its own outputs), while the default follows Gate 2's decision of one model whose noise the human sample bounds. Rejected: reusing `harness.model` (no way to separate judge and system model); a judge ensemble or repeated sampling (the plan bounds judge noise with the human sample, not with more judges).

10. **Human sample: `eval sample --run-name X [--size N]`** writes `data/eval/<run_name>/human_sample.csv` with columns `sample_id, judge, probe_id, pm_id, case, question, response, brief, human_correct, human_note`. `brief` is the text the judge was told beyond the question and reply (the option list, the rubric, the hidden answer, the bias phrases, or the judge-only values), so a rater grades the same task; `case` is the in-situ case, the governance answer kind (`preference`, `update`, `dormant`), or empty. The judge's verdict and rationale are withheld so raters are blind. `human_correct` and `human_note` are written empty for the rater to fill. Allocation: `size` (default `judge.sample_size`) split evenly over the five judges, a judge with fewer judgements than its share giving its remainder to the others in order; within `judge_in_situ`, counteract 40%, decline 40%, comply 20%, the plan's weighting toward the counteract and decline cells. Rows are drawn with `stream(config.seed.root, "eval_sample", run_name)` so the sample is reproducible. Refuses with `HarnessError` when the run has no `judgements` table, and when `human_sample.csv` already exists unless `--force` is passed, since a rerun would silently discard ratings a person has entered. The file is CSV because raters use a spreadsheet, and the `csv` module quotes multi-line replies.

11. **Agreement.** `eval score` reads `human_sample.csv` when present, reads it as UTF-8 with an optional byte-order mark (spreadsheet exports often write one), keeps rows whose `human_correct` is `yes` or `no` (case-insensitive, stripped; blank rows are unrated and skipped; any other value raises `HarnessError` naming the `sample_id`), joins each to its judgement by `(pm_id, probe_id, judge)`, and reports per judge `n`, `agreement` (the fraction where the human and judge verdicts match) and Cohen's kappa (Cohen, 1960): kappa is reported because intrusion and governance verdicts are mostly one-sided, so raw agreement would flatter a judge that always says the majority answer. Kappa is null when chance agreement is total (both raters used the same single label), and 0 when only one rater is constant, since a constant rater agrees with the other at exactly chance. The summary's `agreement` section is the plan's "agreement rate reported next to the score".

12. **Summary changes.** `by_type` lists judge scorers with `chance` null. `awaiting_judge` counts open items with no judgement row: open twins, in-situ, governance and routine probes lacking the relevant judgement, plus routine probes with judge-only values lacking a `judge_format` row; after a full `eval judge` the only non-zero count is `routine_question/intrusion`, equal to the no-active-bias skips. `slices` gains, for `judge_in_situ`, accuracy by case (`comply`, `counteract`, `decline`), and for `judge_governance`, by answer kind (`preference`, `update`, `dormant`); the existing kind, checkpoint and evidence slices apply to every judge scorer as they do to the deterministic ones. New `agreement` section (decision 11). The printed table adds the judge rows and, when present, an agreement line per judge.

13. **Judge prompts are constants in `harness/judge.py`.** A shared system prompt states the grading task (a copilot's reply to a portfolio manager, grade against the brief, reply as JSON, be strict: a reply that hedges between answers gets the stricter verdict) and each judge adds its instruction; the brief goes in the user message. The prompts are versioned code, like the bank, and a change to one is a change to the judge config's recorded hash (decision 8: the metadata records a sha256 of the prompt texts alongside the config, so a prompt edit is caught by the rerun check).

14. **Config.** `JudgeConfig` as `Config.judge`, every leaf with a basis and note: `model` = `DEFAULT_MODEL` design (one judge model, the human sample bounds its noise), `effort` = `high` design (a grading task with a rubric benefits from thinking, at Gate 2's effort), `max_tokens 4000` design (thinking plus a short JSON verdict, the dialogue stage's output cap), `max_concurrency 8` guess (the dialogue stage's value), `sample_size 100` guess (20 per judge, enough to notice an agreement rate below 0.8), `token_budget` null guess (cap on fresh tokens for one `eval judge`; spending it raises `HarnessError` and the finished items stay cached).

15. **Errors.** `HarnessError` for: judging a run that would fail `eval score`'s checks, a missing or duplicated MCQ sibling, an unknown in-situ or governance answer form, an unparsable judge reply after the retries, a judge config or prompt change without `--force`, sampling before judging, a bad `human_correct` value, a spent token budget.

## Package layout

```
src/pm_traitbench/harness/
  judge.py      # Judge enum use, briefs, prompts, schemas, request building, verdict parsing, judge_run
  sample.py     # human sample export and the agreement computation
  score.py      # judgement-to-score conversion, awaiting counts, new slices, agreement in the summary
  cli.py        # `judge` and `sample` subcommands, printed agreement
```

Modified: `enums.py` (`Judge`, five new `Scorer` values), `tables/schema.py` (`JudgementRow`), `tables/specs.py` (`JUDGEMENTS`, `SCORES` key), `config.py` (`JudgeConfig`), `harness/score.py`, `harness/cli.py`, `README.md`, `docs/pm-dataset-plan.md`.

### Data flow

1. `eval judge`: check the run; read responses, hidden probes, traits, drift events, bank; build items (decision 3, skipping per decision 4 and 5); send through the cached client; write `judgements` and `eval-judge` metadata.
2. `eval sample`: read `judgements` and hidden probes; allocate and draw (decision 10); write `human_sample.csv`.
3. `eval score`: as part 1, plus read `judgements` if present, convert to score rows (decision 7), read `human_sample.csv` if present (decision 11), summarise (decision 12).

## Limitations

- **One judge, one call.** No ensemble, no repeated sampling. Accepted: the plan bounds judge noise with the human sample; the agreement report shows when the judge is not trustworthy.
- **The default judge model is the baseline model.** A same-model judge may favour the full-context baseline. Accepted: `judge.model` is configurable; the agreement report is the check.
- **Intrusion ignores preferences.** A reply that leaks an information preference is not flagged. Accepted: decision 4; the plan's rule names bias-derived content.
- **The open twin is graded as a classification into the MCQ's options.** A reply describing an action outside the option set scores wrong even when defensible. Accepted: the twin is "judged against the same engine action", and the option set is that action plus the engine's alternatives.
- **Judge verdicts are cached by request, so a prompt or brief change re-bills.** Accepted: correctness over cache hits; the prompt hash in metadata makes the change explicit.
- **`judge_format` judges every judge-only value in one call.** A long list of values in one verdict may be graded less carefully than one at a time. Accepted: a routine question holds at most seven communication values, of which at most four are judge-only.
- **Agreement is on a sample of the run's own judgements.** It measures the judge on this system's replies, not in general. Accepted: that is what the plan asks for, and a per-run sample is what a rater can do.

## Living docs impact

- `README.md`, Evaluation section: after "Scoring", a "Judges" paragraph with the table of decision 3, the intrusion brief of decision 4, the empty-reply rule, the `judgements` table and the commands; a "Human sample" paragraph with the CSV columns, the allocation, the rating values and the agreement report; `summary.json` description gains the judge scorers, the new slices, the changed `awaiting_judge` and `agreement`; the run directory listing gains `judgements` and `human_sample.csv`; the Config paragraph gains the `judge` knobs; the sentence "Open probes are not scored here" is replaced.
- `docs/pm-dataset-plan.md` section 9, "Evaluation harness (not a stage)" paragraph: the judges and the human sample are done, and the full report remains the later sub-project.
- No `ARCHITECTURE.md` exists; the README and the plan remain the living docs.

## Implementation notes

Heavy tier: the plan pins interfaces per task. Key points the plan must carry:

- `JudgementRow` is a frozen pydantic model; `Judge` and the `Scorer` additions share string values, checked by a test.
- `judge.py` exposes pure builders per judge returning `(brief, request)` from the probe row plus its inputs, so tests assert request content without a client: the sibling options in the open brief, the rubric in the in-situ brief, the active-bias phrases in the intrusion brief matching `bias_active_at` at the checkpoint date (a dormant bias absent), the judge-only values in the format brief.
- Verdict parsing and the judgement-to-score conversion are pure functions tested per judge, including `none`, mixed in-situ fields, governance either-true, format all-not-applicable giving no score row, and `empty_reply`.
- `judge_run` is tested with `tests.dialogue.fixtures.FakeClient` as `tests/harness/test_baselines.py::test_baseline_runs_through_runner` does, covering: every judged item gets a row, the skip counts, an unparsable reply raising `HarnessError`, the cache replay on rerun, the config-change refusal and `--force`.
- `sample.py` tests: allocation sizes with a short judge, the in-situ 40/40/20 split, determinism, no verdict column, agreement and kappa on a hand-built table including the null-kappa case and the bad-value error.
- `tests/harness/test_cli.py` adds an end-to-end `run` then `judge` then `sample` then a filled sample then `score` on the probes fixture corpus with the recording adapter and `FakeClient`; copies `test_eval_run_then_score`.
- `tests/harness/test_score.py::test_awaiting_judge_counts` changes to the new semantics.
- `tests/tables`: the `scores` key and the `judgements` spec.
- Full suite: `uv run pytest -q`.
