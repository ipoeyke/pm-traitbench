**Tier:** heavy (provisional until triage)
**Escalation threshold:** n/a (heavy)
**Supersedes:** none (adds the `chased_trend` column to `ideas`, which `docs/specs/2026-09-24-engine-design.md` did not persist; the engine's counter and predicate are unchanged)

# Design: stage 5, signal plan and session skeletons

Date: 2026-09-25. Source of requirements: `docs/pm-dataset-plan.md`, sections 4 (session sampling, signal plan, target mix, silence set), 6 (per-PM session budget, drift carrier minimum), 8 (`signals.jsonl`, `sessions.jsonl`), 9 (stage 5 row and paragraph). Builds on `docs/specs/2026-09-24-engine-design.md`, `docs/specs/2026-09-25-engine-bias-fixes-design.md` and `docs/specs/2026-09-25-gate1-design.md`.

## Scope

Fifth sub-project. Delivers **stage 5, plan**: a deterministic pass over stage 1 and stage 3 output that decides, per PM, which trait evidence is planted, in which mode, on which date and against which trade idea; groups that evidence into dated sessions; adds ledger-event and filler sessions; and writes one skeleton per session, the whole contract between the deterministic world and the stage 6 narrator. Pure Python, no model calls.

In scope: `signals/` package (carriers, quotas, assemble, skeleton, stage), two output tables (`signals`, hidden `skeletons`), a `stances` catalogue, a `plan` config section, a `PlanError`, one new hidden `ideas` column (`chased_trend`) written by the engine, README section, tests.

Out of scope: stage 6 (narrator and advisor loop), stage 7 (validator), Gate 2, probes. Multi-asset PMs (no engine output yet; skipped and counted). Tuning the mix after Gate 2 (a config edit made after reading the Gate 2 report).

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself. Citing a published paper or a named public series as the basis of a value is allowed.

## Decisions taken

1. **Scope is stage 5 only.** Stage 5 is deterministic and testable without a model, like stages 1-4; the skeleton becomes a pinned interface and stage 6 gets its own spec against it. Rejected: stages 5 and 6 together (mixes deterministic and LLM work; large spec) and stages 5-7 (would need decomposition anyway).
2. **Signal-first, three passes.** (1) plan signals per trait (quotas, then carriers), (2) assemble sessions around them, (3) render skeletons. Each pass is a pure function. Rejected: session-first (revealed carriers are pinned to ledger dates, so the calendar would be rebuilt around them anyway) and a single day-by-day pass (per-trait quotas and the session cap need lookahead).
3. **Revealed and contradiction bias carriers come from engine-flagged evidence only.** A carrier is a row where the engine recorded that the bias drove the decision (table below). The narrated justification therefore always matches what the engine did, and the plan's rule that dialogue never adds what the ledger lacks holds by construction. Rejected: topping up with unflagged but consistent rows (some carriers would narrate a bias the engine did not apply on that trade) and any trade on a matching day (dialogue could contradict the ledger).
4. **Shortfall caps revealed and logs a warning.** When an active bias has fewer carriers than its revealed quota, every available carrier is used, the other modes keep their counts, and the per-trait shortfall goes to run metadata. Gate 2 decides whether the trait is still recoverable. Rejected: backfilling with stated signals (skews the mode mix exactly where the ledger is thin) and failing the stage (the pilot's real seed R1 gives rates_credit PMs 10-26 ideas and would block every run).
5. **Third-party distractors anchor on an inactive bias or a same-param preference value.** A bias distractor is a colleague or client showing one of the PM's inactive biases, which already has a `trait_id`. A preference distractor is a colleague or client stating a different catalogue value for a param the PM holds; the value goes in `third_party_value` and `trait_id` points at the PM's own trait for that param. Feeds the trait-presence negatives and the MCQ third-party option later. Rejected: inactive biases only (preferences get no ownership distractors) and third-party rows in `traits.jsonl` (changes the stage 1 schema and puts non-PM rows into ground truth). The column name says whose value it is, so it cannot be misread as the PM's.
6. **Skeletons are a new hidden table.** Stage 5 writes `skeletons.jsonl`, hidden in full; stage 6 reads it and writes `sessions.jsonl` complete with turns. Matches the hidden-table pattern of `position_days`, keeps `sessions.jsonl` valid whenever it exists, and the narrator contract never ships. Rejected: pre-filling `sessions.jsonl` with `turns: null` and hidden plan columns (half-valid table between stages, hidden columns in a public table).
7. **Extrapolation carriers come from a new hidden `ideas` column, `chased_trend`.** Extrapolation is a forecast blend and leaves no `bias_flag`. The engine already computes, per new idea, whether the entry is on the side of a trailing move already past one standard deviation of a horizon move (`extrapolation.entered_after_run`), counts it into the run metadata counter `entries_after_run`, and discards the per-idea value. The engine now persists it. Gate 1's own recomputation from `ideas` and the market stays as the independent cross-check. Name: `chased_trend`, a past-tense "the PM did X" bool like its neighbours `conflict` and `followed_street`; the repo uses no `is_` prefix on any boolean, and the user chose consistency with that over an `is_` prefix. "Run" was rejected in the column name because it collides with pipeline runs. The existing function, field and counter names are left unchanged to keep the diff focused. Type: non-null bool on every idea. Rejected: a float `trend_z_at_entry` (richer, but moves the one-standard-deviation cutoff into every reader) and recomputing in stage 5 (a third copy of the same predicate).
8. **`chased_trend` is an opportunity, not a bias flag.** Neutral PMs chase trends too. Stage 5 uses it as a carrier only when `extrapolation_theta` is active for that PM, the same way flags are used only for the bias they name.
9. **Contradiction is one signal with two dated stances.** The signal row sits on the later revealed carrier session and `claim_session_id` points at an earlier `check_in` where the PM claimed the opposite (a `claim` stance). The pair is traceable from one row.
10. **Stances come from a template bank.** `catalogues/stances.yaml`, loaded and validated by the existing catalogue loader, holds short instruction lines keyed by (param, mode, asset class) with an `all` fallback. Stage 5 picks a line with the PM's seeded stream and fills slots from the carrier's rows; stage 6 paraphrases it in the PM's voice. The narrator never sees the param, the value, the mode or which lines are signals. Keeps signal content deterministic and human-reviewable, and puts leakage control in one file.
11. **Drift events are announced in dialogue.** Each drift event plants one stated signal on the first session on or after its date, from the bank's `drift` entries (`update`, `dormant`, `revive`), so the change the probes later test is visible in the transcript. It counts toward that trait's signals and toward the before-and-after minimum on the after side.
12. **README stays the living doc.** Follows Gate 1's decision 6: a "Signal plan" section beside "Gate 1". No `ARCHITECTURE.md` is created. (An earlier draft of this design proposed creating one; dropped for consistency with that decision.)

## Package layout

```
src/pm_traitbench/signals/__init__.py
src/pm_traitbench/signals/inputs.py      # PlanInputs per PM: persona, traits, drift events, rules, ideas, ledger, rule_events, position_days, trading days
src/pm_traitbench/signals/carriers.py    # carrier pool per active bias and per expression preference
src/pm_traitbench/signals/quotas.py      # per-trait counts by mode, valence, ownership
src/pm_traitbench/signals/assemble.py    # dates, sessions, ledger-event sessions, filler
src/pm_traitbench/signals/skeleton.py    # stance rendering, advisor violation, forbidden sets
src/pm_traitbench/signals/stage.py       # CLI stage `plan`
src/pm_traitbench/catalogues/stances.yaml
```

## Carriers (`carriers.py`)

A carrier is one dated piece of engine evidence for one trait. Only the PM's own rows count.

| Trait | Carrier source | Carrier date |
|---|---|---|
| disposition_ratio | `ledger` rows flagged `disposition:realise_gain_early`; the first `position_days` row per idea flagged `disposition:hold_loser` | row date |
| loss_aversion_lambda | `ledger` rows flagged `loss_aversion:add` or `loss_aversion:add_before_trigger`; the first `position_days` row per idea flagged `loss_aversion:hold` | row date |
| exit_deficiency | `rule_events` with response `acked_no_action` or `added`; `ledger` rows flagged `exit_deficiency:late_roll` or `exit_deficiency:added` | `response_date` for rule events, row date otherwise |
| anchoring_rho | `ledger` rows flagged `anchoring:exit_at_anchor` | row date |
| herding_weight | `ledger` rows flagged `herding:followed_street` | row date |
| overconfidence_coverage | `ledger` rows flagged `overconfidence:oversized` | row date |
| conviction_size_miscalibration | `ledger` rows flagged `conviction:mis_sized` | row date |
| extrapolation_theta | `ideas` with `chased_trend` true | `entry_date` |
| expression preference whose (param, value) maps to a form in the engine's `FORM_FOR_PREFERENCE` | `ideas` whose `expression` equals that form | `entry_date` |

`bias_flag` may join several flags with `;`; a row is a carrier for every flag it carries. One carrier per (trait, idea, date). A carrier that falls in a dormant window (on or after a `dormant` date and before the matching `revive`) is dropped.

Expression preferences with no form mapping (`hedge_instrument`, `futures_vs_etf`, `fx_hedge_expression` and any other unmapped value) have no ledger fingerprint; their revealed signals use an advisor violation, like the other preference groups.

## Quotas (`quotas.py`)

Per PM, from `traits` and `drift_events`. All counts use largest-remainder rounding so they sum exactly.

- **Active bias, confirm signals.** `n ~ U{bias_signals_min..bias_signals_max}` (8-10, plan section 6). Split over `revealed`, `stated`, `contradiction` by weights 0.65, 0.175, 0.10 (the plan's 60-70%, 15-20%, 10% midpoints, renormalised over the three modes). A trait with no carriers places only the carriers it has; the missing revealed and contradiction signals are not reassigned to stated (decision 4).
- **Preference, confirm signals.** `n = pref_signals` (3, plan section 6). Split `stated` 0.65 and `revealed` 0.35 (plan section 4, guess).
- **Retracted.** Per PM, `round(retracted_share x total confirm signals)` extra rows with `mode = stated`, `valence = retracted`, spread over the PM's active biases and preferences in a seeded order. Share 0.075 (midpoint of the plan's 5-10%). A retracted row is additional: it never replaces a confirm signal, since a memory system must not count it.
- **Third-party.** Per PM, `round(third_party_share x total own signals)` extra rows (share 0.10, plan section 4), `mode = stated`, `valence = confirm`, `ownership` colleague or client with equal probability. Half go to inactive biases and half to preferences with a different catalogue value (odd count: the extra one to biases); if the PM has no inactive bias, all go to preferences.
- **Drift.** For each drift event on a trait, the trait's confirm signals are raised if needed so at least `drift_min_per_side` (6, plan section 4) fall before the event and at least 6 after (after the `revive` for a dormant event), counting the drift-announcement signal (decision 11) on the after side. For a bias the extra signals keep the trait's mode split; revealed ones still need carriers on the right side of the date, and a side left short is a warning.

## Assembly (`assemble.py`)

Trading days are the PM's market-seed calendar. Session ids follow the plan: `s_<pm>_<date>_<letter>`, letter `a`, `b`, ... per extra session that day.

1. **Date each signal.**
   - Revealed or contradiction bias signal, revealed expression preference: a carrier drawn without replacement from the trait's pool. For a drifted trait, carriers are drawn to fill each side of the event first.
   - Contradiction claim: a `check_in` on a trading day drawn uniformly from `claim_lead_days` (10 to 40 trading days, guess) before the carrier; clipped to the first trading day.
   - Stated or retracted bias signal, all non-expression preference signals, third-party signals: a trading day drawn uniformly, outside any dormant window of that trait.
   - Drift announcement: the first session on or after the event date; if none exists, a new `check_in` on the event date.
2. **Group into sessions.** Signals on the same PM and date merge into one session when their traits differ; a second signal of the same trait on that date opens session `b`. Kind: `decision` if the PM has a ledger row that day for an idea in the session, `check_in` otherwise. `trade_idea_ids` holds every carrier's idea.
3. **Ledger-event sessions.** Every ledger row with `risk_amount` at or above the PM's own `ledger_session_percentile` (75, guess; per PM so it scales with book size) that is not already in a session that day gets a `decision` session with no signals, or joins an existing session that day.
4. **Filler.** Sessions on random trading days with no session, kind `check_in` or `silence` with probability `filler_silence_share` (0.5, guess), added until signal-carrying sessions are at most `signal_session_cap` (0.40, plan section 4) of the total. If the calendar runs out of free days first, a warning.

## Skeleton (`skeleton.py`)

For each session: every signal's stance is picked from `stances.yaml` at `[param][entry][asset_class]` falling back to `[all]`, with the PM's stream, and its slots filled from the carrier's rows (`{name}` from the idea, `{price}` from the ledger row, `{target}` and `{stop}` from the idea-scope rules, `{value}` from the preference or `third_party_value`, `{who}` = colleague or client). Entries: `revealed`, `stated`, `claim` (contradiction's earlier session), `retract` (a state-then-correct pair), `third_party`, `drift_update`, `drift_dormant`, `drift_revive`, plus per preference group `stated`, `revealed_reaction` and `violation`.

A revealed preference signal without an expression carrier sets `advisor_violation` from the group's `violation` template (the advisor breaks the preference; the PM's `revealed_reaction` stance reacts). At most one violation per session.

Forbidden sets: `forbidden_trait_ids` are the PM's inactive biases; `forbidden_pref_params` are catalogue preference params for the PM's asset class that the PM does not hold. Third-party signals in the same session do not remove a trait from the forbidden set: the PM still must not exhibit it.

## Tables

`signals` (key `pm_id, signal_id`):
`signal_id, pm_id, session_id, date, trait_id, mode (stated, revealed, contradiction), trade_idea_id (str or None), valence (confirm, retracted), ownership (self, colleague, client), third_party_value (str or None), claim_session_id (str or None)`.
Invariants checked by the row model: `third_party_value` is set exactly when `ownership` is not `self` and the trait is a preference; `claim_session_id` is set exactly when `mode` is `contradiction`.

`skeletons` (key `pm_id, session_id`, hidden in full):
`session_id, pm_id, date, kind (decision, check_in, silence), trade_idea_ids, stances (list of {signal_id, trait_id, mode, stance}), advisor_violation (str or None), forbidden_trait_ids, forbidden_pref_params`.

`ideas` gains hidden column `chased_trend: bool`.

Run metadata per PM: signal counts by trait and mode, planned versus placed revealed counts, drift side counts, sessions by kind, signal-session share, and every shortfall as a warning. Skipped multi-asset PMs are counted.

## Config: `plan`

| Key | Default | Basis |
|---|---|---|
| `bias_signals_min`, `bias_signals_max` | 8, 10 | plan section 6, guess to be checked at Gate 2 |
| `pref_signals` | 3 | plan section 6, guess |
| `bias_mode_weights` | revealed 0.65, stated 0.175, contradiction 0.10 | plan section 4 midpoints |
| `pref_mode_weights` | stated 0.65, revealed 0.35 | plan section 4, guess |
| `retracted_share` | 0.075 | plan section 4 midpoint |
| `third_party_share` | 0.10 | plan section 4 |
| `claim_lead_days` | [10, 40] | guess: far enough back to be a separate session, inside a quarter |
| `ledger_session_percentile` | 75 | guess: "large trades" in plan section 4 has no number |
| `signal_session_cap` | 0.40 | plan section 4 |
| `filler_silence_share` | 0.5 | guess |
| `drift_min_per_side` | 6 | plan section 4 |

## Stage (`signals/stage.py`)

CLI subcommand `plan`, registered like the other stages. Reads stages 1-3 output and `market/regimes` (for the trading-day axis), writes `signals` and `skeletons`, refuses to overwrite without `--force`. Seeded through `rng.stream(root_seed, "plan", pm_id, ...)`, so a rerun is identical.

## Errors

`PlanError` names the PM and trait or idea: a `trait_id` in `drift_events` not in `traits`, a carrier idea missing from `ideas`, a stance entry missing for a (param, entry) the plan needs, a slot the carrier cannot fill. Shortfalls (revealed below quota, drift side below minimum, cap not met) are warnings, never errors.

## Testing

- Engine: `chased_trend` equals `extrapolation.entered_after_run` at entry; per PM, the count of true values equals the `entries_after_run` counter.
- Catalogue: `stances.yaml` loads; every bias and preference group has every entry the skeleton pass can request; no line contains a bias name or its parameter word (loss aversion, disposition, anchor, extrapolat, herd, overconfiden, conviction, exit deficiency).
- Carriers: each flag maps to the right trait; multi-flag rows give one carrier per flag; `position_days` yields the first flagged day per idea only; dormant windows drop carriers; `chased_trend` counts only for an active extrapolation PM.
- Quotas: counts sum exactly; retracted and third-party rows are additional; drift raises counts to the per-side minimum.
- Assembly: every revealed signal's date equals its carrier's date; no two signals of one trait share a session; signal sessions are at most 40% or a warning is logged; ledger rows above the percentile are all in a session; claim sessions precede their carrier.
- Skeleton: slots filled, forbidden sets correct, at most one violation per session.
- Stage: runs on the shared small fixture, deterministic across reruns, multi-asset PMs skipped and counted.
- `test_end_to_end` runs `plan` after `gate1`.

## Limitations

- **Stance lines are few and fixed.** 2-4 lines per (param, entry) means repeated phrasings across PMs; the narrator paraphrases, but a memory system could key on the bank's wording. Accepted for the pilot; the bank grows if Gate 2 or human review flags sameness.
- **Carrier-thin biases stay thin.** Decision 4 means some active biases, most on real seed R1, carry fewer revealed signals than the mix asks for. The alternative skews the mix; Gate 2 will say whether the trait is still recoverable.
- **The ledger-session threshold is a per-PM percentile, not a risk rule.** It guarantees about a quarter of the PM's orders are mentioned, not that every "material" trade is; the plan gives no number and stage 7 checks consistency only on trades a session mentions.
- **Filler dates are uniform.** Real check-ins cluster around news; uniform filler is simpler and does not affect ground truth.
- **Retracted and third-party counts are per PM, not per trait.** A trait may get none; per-trait shares at 3 preference signals would round to zero anyway.
- **Drift announcement is explicit.** A real PM may change without saying so. Stating it gives the governance probes a fair ceiling; a silent-drift variant can be added later as a config switch.

## Living docs impact

README: new "Signal plan" section after "Gate 1": what `plan` reads and writes, that `skeletons` is hidden in full, that carriers are engine-flagged only, and that shortfalls are warnings. Usage block gains `uv run pm-traitbench plan --config configs/demo.yaml --data-dir data`. The engine paragraph adds `chased_trend` to the hidden `ideas` columns.
