**Tier:** heavy
**Escalation threshold:** n/a (heavy)
**Supersedes:** none (changes `docs/pm-dataset-plan.md` section 9.1 "Sameness" and the section 9 stage 6 paragraph; no earlier spec's decisions change)

# Design: stage 6, dialogue

Date: 2026-09-26. Source of requirements: `docs/pm-dataset-plan.md`, sections 4 (dialogue layer, two-agent loop, simulated advisor, narrator inputs), 6 (scale and budget), 8 (`sessions.jsonl`), 9 (stage 6 row and paragraph), 9.1 (implementation risks). Builds on `docs/specs/2026-09-25-signal-plan-design.md`, whose `skeletons` table is this stage's contract.

## Scope

Sixth sub-project. Delivers **stage 6, dialogue**: for every skeleton stage 5 wrote, generate one dated PM-copilot session with two LLM agents, a PM narrator and a simulated advisor, and write the public `sessions` table plus a hidden `dialogue_logs` table.

In scope: `dialogue/` package (client, tools, voices, turns, prompts, session, stage), two output tables, a `dialogue` config section, two authored catalogues (`voices.yaml`, `avoid.yaml`), a placeholder advisor prompt file, `DialogueError` and `DialogueBudgetError`, the `anthropic` dependency, CLI subcommand, README section, updates to `docs/pm-dataset-plan.md`, tests.

Out of scope: stage 7 validate (leakage, ledger consistency, forbidden traits, regeneration queue); Gate 2, including the cross-PM n-gram overlap report; probes; the later replay of advisor turns with the deployed copilot; multi-asset PMs (no skeletons exist for them).

## Repo rule that binds all code

Unchanged: no file in the repo may mention `docs/`, a spec, a plan, task numbers or a coordinator. Docstrings state the rule itself. Citing a published paper or a named public series as the basis of a value is allowed.

## Decisions taken

1. **Stage 6 only; validate gets its own spec and branch.** Stage 6 pins everything validate reads (`mentions`, the per-turn log, a `feedback` hook for regeneration), so stage 7 only consumes it. Stage 7 is its own design problem (two-model judges, agreement logging, attempt cap, re-planning dropped sessions) whose thresholds need real pilot output to tune. Stage 6 is fully testable offline with a fake client. Rejected: stages 6 and 7 together (large heavy spec; holds stage 6 back on judge tuning). Cost accepted: a stage 6 run is not a usable corpus until stage 7 exists, which is fine because nothing downstream exists yet.
2. **One narrator model, `claude-opus-5-5`; no model rotation.** Differences between PMs must come from what the generator plants (traits, rules, mandate, self-description) plus a recorded, trait-independent voice. Rotating narrator models per PM makes signal recoverability, leakage rate, forbidden-trait rate and regression toward the stereotype on anti-typical PMs depend on which model narrated the PM; random balanced assignment removes the bias in expectation but not the variance, and the pilot has about 6 active PMs per bias, too few to separate a narrator effect from a trait effect. The sameness risk in plan 9.1 (stance paraphrases reading alike across PMs, so a phrasing could be learned as a trait marker) is addressed by several lines per stance entry in `stances.yaml`, per-PM voices (decision 6), and a measured cross-PM n-gram overlap (decision 4). A second narrator is added later only if overlap is high, assigned balanced across asset class x typicality x drift and recorded as a hidden per-PM column, so Gate 2 can report it as a slice. Rejected: two narrators rotated per PM (the plan 9.1 text, changed by this spec) and per session (breaks one PM's voice).
3. **Advisor model `claude-opus-5-5`.** Same model as the narrator: one API behaviour to handle, and the two sides still differ by system prompt and inputs.
4. **Cross-PM n-gram overlap is a Gate 2 report-only row.** It follows Gate 1's pattern of judged report-only rows that never block. It does not fit validate, which passes or fails individual sessions and sends failures back; a corpus-level overlap cannot be attributed to one session. Built in the Gate 2 spec, not here.
5. **Advisor system prompt is an authored file supplied by the user.** Until it arrives, a placeholder ships at `src/pm_traitbench/catalogues/advisor_prompt.md` and the code treats it as the real prompt: no placeholder flag. Its sha256 is recorded in run metadata as provenance. Swapping in the real prompt is a one-file edit or a config path change.
6. **Stage 6 owns the PM voice.** `catalogues/voices.yaml` (authored, about 8 entries, each an id and a short instruction line such as "terse trader shorthand, drops articles, levels without units") is drawn once per PM on `stream(root, "dialogue", pm_id, "voice")`, independent of traits, typicality and preferences, so voice can never become a trait signal. The drawn `voice_id` is recorded on every `dialogue_logs` row and per PM in run metadata. Rejected: a new hidden persona field sampled in stage 1 (cleaner in concept, but changes the stage 1 schema, sampler and tests for something only the narrator reads) and balanced assignment across cells (only worth it if Gate 2 slices by voice). The `register` preference param in `preferences.yaml` is unrelated: it is what the PM wants from the copilot, not how the PM talks.
7. **Live Messages API, async, bounded concurrency.** Sessions run concurrently under a semaphore and advance turn by turn. Rejected: Message Batches in lockstep rounds (half price, but every turn and tool round-trip becomes a separate batch round with minutes-to-hours latency, a cross-round state machine, and late failure; saves roughly $130 on the pilot and $0.3-1k on the full run). The `LlmClient` protocol that the offline fake client needs also leaves a batch backend possible later without touching session logic.
8. **Deterministic turn counts.** Each session's turn total is drawn from a per-kind range on the session's own stream, then raised to what its stances need. Reproducible, plannable cost, and the narrator cannot end a session before its stances are delivered. Rejected: the narrator decides when to stop.
9. **Advisor reads market data through live tools.** Five read-only lookups over the stage 2 tables, capped at the session date. Matches the plan's hard requirement (9.1 "Advisor as tool user") and mirrors how the deployed copilot works, which keeps a later real-copilot replay faithful. Rejected: a market snapshot injected into the advisor prompt (cheaper, but the advisor can only cite pre-selected facts and would free-recall anything else) and snapshot plus tools (two data paths to keep consistent).
10. **Tool calls never enter the transcript.** Three views: `sessions.turns` holds only `{role, text}`; `dialogue_logs` holds mentions, tool calls and results, models, hashes and usage; the advisor's own API context holds its earlier text turns and its own tool exchanges in this session so it does not re-fetch and stays consistent with numbers it already quoted. The narrator only ever sees the advisor's final text.
11. **Orchestration in code with a manual advisor tool loop.** A per-session driver alternates narrator and advisor calls; the advisor's tool loop is request, run tools, resend, capped by `max_tool_rounds`. Every request passes through one seam (`LlmClient.send`), where the cache key, logging, budget and the test fake live. Rejected: the SDK Tool Runner (a beta helper that makes its own requests, splitting cache, logging and budget across two paths) and one call writing both sides (the advisor could not be regenerated separately, would see the PM's stance instructions, and would make no independent tool use).
12. **The response cache is the resume manifest.** Prompts are deterministic, so a crash-resume, a widened PM filter or a retry after failures is a rerun with `--force`: every finished call replays from the disk cache and only missing calls reach the API. Rejected: a separate manifest file (duplicates what the cache already records).
13. **All or nothing per run.** The stage attempts every session, then raises listing the failures and writes no tables if any session failed. Matches the stage runner's contract that a stage writes all of its tables or none; the cache keeps every success, so a rerun only redoes what failed.
14. **Per-turn narrator directions are mid-conversation `system` messages.** Keeps each agent's history append-only, which Opus 5.5's thinking blocks require and prompt caching rewards, and keeps directions from appearing as advisor text.
15. **Effort `low` for both agents.** Thinking cannot be disabled on `claude-opus-5-5`; effort is the only control on its billed thinking tokens, and a narrated turn is short and heavily constrained by the skeleton. A config field, to be revisited after the pilot's measured cost and Gate 2.
16. **Both agents are memoryless across sessions** (plan section 4). The advisor sees only the current session's transcript, exactly as the deployed copilot would; the narrator's continuity (a contradiction pointing back at an earlier claim) is carried by the skeleton's stance lines. Sessions are therefore independent, and concurrency is per session.

## Package layout

```
src/pm_traitbench/dialogue/__init__.py
src/pm_traitbench/dialogue/client.py    # LlmClient protocol, AnthropicClient, CachedClient
src/pm_traitbench/dialogue/tools.py     # advisor tool schemas and executors
src/pm_traitbench/dialogue/voices.py    # per-PM voice draw
src/pm_traitbench/dialogue/turns.py     # turn plan and stance placement
src/pm_traitbench/dialogue/prompts.py   # narrator and advisor request builders
src/pm_traitbench/dialogue/session.py   # one-session async driver
src/pm_traitbench/dialogue/stage.py     # stage 6 entry point
src/pm_traitbench/catalogues/voices.yaml
src/pm_traitbench/catalogues/avoid.yaml
src/pm_traitbench/catalogues/advisor_prompt.md
```

## Client (`client.py`)

- `LlmClient` protocol: `async send(request: dict) -> dict`. The request is the full Messages API body (model, system, messages, tools, `output_config`); the response is the SDK message serialised to a dict.
- `AnthropicClient`: wraps `AsyncAnthropic` behind an `asyncio.Semaphore(max_concurrency)`. SDK retries handle 429, 5xx and connection errors; errors are handled by typed exception class, never by message text. Built lazily so a fully cached run needs no credentials.
- `CachedClient(inner, cache_dir, token_budget)`: `send(request, *, scope)` keys on sha256 of the session id (the scope) plus the canonical request JSON (sorted keys, no whitespace; model id is part of the body), so two sessions that happen to build byte-identical requests never share a reply. Hit: read `<cache_dir>/<key[:2]>/<key>.json`. Miss: call `inner`, and write only after the caller accepts the response (`commit(key, response)`), via a temp file and rename, so an invalid or refused response is never cached and a crash never leaves a partial entry. Counts calls, hits and fresh `input`, `output` and cache-read tokens; raises `DialogueBudgetError` before a fresh call once fresh input plus output tokens reach `token_budget`; the budget is a soft stop, since calls already in flight can overshoot it by up to the concurrency limit. An unreadable cache entry (for example after a power loss) is treated as a miss and rewritten on commit.
- Cache root: `<data-dir>/cache/llm/`, under the already gitignored `data/`.

## Tools (`tools.py`)

Five tools, `strict: true`, each taking `instrument` (an `instrument_id` or an instrument `name`) and, where relevant, a window. Every lookup reads the PM's `market_seed` and returns nothing dated after the session date.

| Tool | Returns | Source table |
|---|---|---|
| `get_quote` | price and, for a credit issuer, `spread_bp` on the session date | `market/prices` |
| `get_curve` | levels by tenor for a sovereign or commodity curve | `market/curves` |
| `get_consensus` | `street_score`, `street_view`, `positioning_pct`, `positioning` | `market/consensus` |
| `get_calendar` | events for the instrument (and market-wide rows) in a window of at most 20 trading days back and 20 forward; forward rows carry date and event, with `surprise` always null | `market/calendar` |
| `get_history` | the trailing `n` (at most 60) trading days of price or spread | `market/prices` |

An unknown instrument, a missing field or an out-of-range window returns an `is_error` tool result listing the valid choices; the advisor carries on. Forward calendar rows are the schedule a real desk knows in advance; their surprise is future information and is withheld.

## Voices (`voices.py`)

`draw_voice(root, pm_id, voices) -> Voice`, one uniform draw per PM (decision 6). The catalogue loader validates `voices.yaml` (unique ids, non-empty lines, at least 6 entries so a voice is not a near-unique PM fingerprint in a small pilot).

## Turn plan (`turns.py`)

- The total turn count is even; the PM opens and the advisor replies last.
- Totals drawn uniformly per kind (config `turns_by_kind`, **guess**): `silence` {2, 4}, `check_in` {2, 4, 6}, `decision` {4, 6, 8}; mean about 4.5. Within the plan's 2-8 range (section 4).
- Raised to the stances' minimum: at least one PM turn per stance; a `revealed_reaction` stance sits on PM turn 2 or later, so its session has at least 4 turns. If the raised total would exceed 8, the stage raises `DialogueError` naming the session, since stage 5 caps a session at 2 signals and this cannot happen with a valid skeleton.
- Stances are placed on PM turns in a random order drawn on `stream(root, "dialogue", pm_id, session_id)`; a `revealed_reaction` is placed first among turns 2 and later, and the skeleton's `advisor_violation` line goes on the advisor reply immediately before it. A `claim` stance (the earlier half of a contradiction) is an ordinary stance on its own session.
- The session day's ledger rows for the session's ideas go on PM turn 1, each with its leg's tenor when set.
- Output: `TurnPlan(n_turns, pm_directives: tuple[PmDirective, ...], violation_turn: int | None)`, where each `PmDirective` carries the turn's stance line (or none), the ledger rows to mention (turn 1 only), and the session-opening instruction (turn 1 only).

Opening instruction by kind, when turn 1 carries no stance that already sets the topic: `decision` opens on the day's trades; `check_in` opens as a routine check on the PM's open positions on that date (from `position_days`), or, when none are open, as a routine check-in about the markets the PM trades without naming a position; `silence` opens with a pure market or factual question about one instrument drawn on the session stream from the PM's asset-class universe, and the directive forbids talking about the PM's own positions, rules or habits (the silence set of plan section 4).

## Prompts (`prompts.py`)

**Narrator request.** The narrator is the `assistant`; the advisor's texts arrive as `user` messages. The first message is a fixed `user` line opening the session.
- `system` (stable for the whole session, cache-marked): role instruction (write only the PM's side, in the given voice, never name a psychological bias or trait); mandate facts; `stated_profile.self_description`; PM-scope rule texts and the idea-scope rule texts of the session's ideas; those ideas (instrument name, side, entry, target, stop, thesis template); the voice line; the forbidden list rendered from `avoid.yaml` for the skeleton's `forbidden_trait_ids` (by the trait's param) and `forbidden_pref_params`; the instruction to mention only the trades listed in a turn's directive and never to invent a trade.
- Per PM turn: a mid-conversation `system` message with that turn's directive (stance line, trades to mention with their `trade_idea_id`, side, size, instrument and price, or opening instruction; "continue naturally" when the turn carries nothing).
- `output_config.format`: JSON schema `{text: string, mentions: [Mention]}`.
- The narrator never sees a trait id, bias param, bias trait value, mode, or which directive lines are signals (plan section 4). A preference value reaches it only inside the stance line that plants it, since that stance is the PM stating or reacting to the preference.

**Advisor request.**
- `system`: the advisor prompt file, then "Today is {date}." (cache-marked).
- `tools`: the five tools. `messages`: PM texts as `user`; each advisor response's full content (text, thinking and tool blocks) appended unchanged, followed by the `tool_result` blocks in one `user` message.
- On `violation_turn`: a mid-conversation `system` message carrying the skeleton's `advisor_violation` line before that reply.
- `output_config.format`: the same `{text, mentions}` schema.
- The advisor sees no persona, rules, ledger, ideas or skeleton data.

**Mention schema** (checked by stage 7, never parsed from prose):
- `trade`: `trade_idea_id, instrument_id, side, size` (PM only).
- `level`: `instrument_id, tenor (string or null), field, value`, where `field` is a column of `market/prices`, `market/curves` or `market/consensus` (PM and advisor).

**Catalogue `avoid.yaml`:** one line per bias param (e.g. disposition: "do not describe selling winners quickly or holding on to losers") and one per preference param in `preferences.yaml` (e.g. "do not ask for street positioning"). The loader checks every bias param and every catalogue preference param has a line.

## Session driver (`session.py`)

`async narrate_session(ctx, client, *, feedback: str | None = None) -> tuple[Session, DialogueLog]`

1. Build the turn plan.
2. For each PM turn: send the narrator request; validate the response against the turn schema (`stop_reason` `end_turn`, parsable JSON matching the schema); whether mentions match the ledger is stage 7's check (decision 1); on failure retry, up to `max_retries` fresh attempts; on success `commit` it to the cache.
3. For each advisor turn: send; while `stop_reason` is `tool_use` and rounds are below `max_tool_rounds`, run every requested tool, append all `tool_result` blocks in one `user` message and resend. At the cap, append a text block to that last tool-result message telling the advisor lookups are over and to answer in text, naming what it could not look up, then resend once with `tool_choice: {"type": "none"}` and record a warning. Validate and commit the final response as in step 2.
4. Return the public row and the hidden log.

`feedback`, when set, is appended to the narrator's `system` prompt. Stage 6 never sets it; it is the hook stage 7 uses to regenerate one session, and the changed request gives it a new cache key.

## Tables

| Table | Key | Columns | Visibility |
|---|---|---|---|
| `sessions` | `(pm_id, session_id)` | `date, kind, trade_idea_ids, turns` (tuple of `{role, text}`, `role` in `pm`, `advisor`) | Public (plan section 8) |
| `dialogue_logs` | `(pm_id, session_id)` | `voice_id, turns`: one entry per turn with `role, text, mentions, directive` (the stance or violation line carried, or null), `scripted_violation` (bool), `tool_calls` (tuple of `{name, input_json, result_json, is_error}`, input and result as canonical JSON strings so the table writes as parquet too), `model, request_hashes, usage` | Hidden in full; added to `HIDDEN_COLUMNS` |

`session_id`, `date`, `kind` and `trade_idea_ids` are copied from the skeleton. The stage checks one `sessions` row and one `dialogue_logs` row per skeleton in the filtered PM set before writing.

## Config: `dialogue`

| Field | Default | Basis |
|---|---|---|
| `narrator_model` | `claude-opus-5-5` | design (decision 2) |
| `advisor_model` | `claude-opus-5-5` | design (decision 3) |
| `effort` | `low` | design (decision 15) |
| `advisor_prompt_path` | null (the packaged `advisor_prompt.md`) | design (decision 5) |
| `turns_by_kind` | silence {2, 4}, check_in {2, 4, 6}, decision {4, 6, 8} | guess |
| `max_tool_rounds` | 3 | guess: one round covers most answers with parallel tool calls |
| `max_retries` | 3 | guess |
| `max_concurrency` | 8 | guess: stays well inside default API rate limits |
| `max_output_tokens` | 4000 | design: room for low-effort thinking plus a short turn |
| `token_budget` | null (no cap) | design |
| `pm_filter` | all PMs; optional `split`, `typicality`, `drift` (`static` or `drift`), `pm_ids` | design: lets a first run measure cost on 2-3 PMs and follows plan section 9's order (static pilot PMs, then Gate 2, then drift PMs) |

## Stage (`dialogue/stage.py`)

Reads `personas`, `rules`, `traits`, `drift_events`, the engine tables, `signals`, `skeletons` and the market tables for the PMs' seeds, plus the plan stage's run metadata. Filters PMs, draws voices, builds one context per skeleton, runs sessions concurrently with at most `max_concurrency` in flight, and on success writes `sessions` and `dialogue_logs` in key order, so a fully cached rerun writes byte-identical tables. Registered as stage 6 `dialogue` with CLI subcommand `dialogue`, same arguments as the other stages.

Run metadata extras: `narrator_model`, `advisor_model`, `advisor_prompt_sha256`, `voices` (per PM), `sessions` (count per kind), `calls`, `cache_hits`, fresh `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `usage_by_model` (the same four counts and `cost_usd` per request model), `cost_usd` (the stage's fresh usage at `prices.models` list prices; `unpriced_models` names any model the table lacks), and `warnings` (tool-round caps, in PM and session order).

## Errors

| Failure | Handling |
|---|---|
| Plan run metadata missing | `DialogueError`, like stage 5's check on engine metadata; missing tables are caught by the stage runner |
| Advisor prompt file missing | `DialogueError` before any call |
| Cache miss with no credentials | `DialogueError` naming `ant auth login` and `ANTHROPIC_API_KEY` |
| 429, 5xx, network | SDK retries |
| `refusal`, `max_tokens`, schema-invalid output | fresh retry up to `max_retries`; nothing cached until valid |
| Tool error | `is_error` tool result; the advisor continues |
| Tool-round cap | final call with `tool_choice` none; warning |
| Session failed after retries | collected; after all sessions, `DialogueError` listing them; no tables written |
| Token budget reached | `DialogueBudgetError`; in-flight calls finish into the cache; no tables written |

## Cost estimate (to be measured at pilot time)

cost = sessions x calls per session x (input tokens x $4/M + output tokens x $20/M), at `claude-opus-5-5` list prices.

| Input | Low / mid / high | Source |
|---|---|---|
| Direct-asset PMs | pilot 24, full 108 | config defaults (4 asset classes x 4 cells x seeds x per-cell = 32 and 144), multi-asset (25%) skipped |
| Sessions per PM | about 80 | stage 5 measured median 78, 90th percentile 100 |
| Calls per session | 4 / 7 / 12 | turn ranges above plus about 1.5 tool round-trips per session (assumed) |
| Input tokens per call | 2k / 3k / 4k | assumed prompt sizes; the advisor prompt's size is unknown until supplied |
| Output tokens per call | 300 / 500 / 800 | about 150-200 visible plus low-effort thinking (assumed) |

Per session $0.06 / $0.15 / $0.38; pilot (about 1,900 sessions) $110 / $290 / $720; full (about 8,600) $480 / $1,300 / $3,300; mid plus about 20% regeneration from stage 7: pilot about $350, full about $1,550. Excludes stage 7 judges and Gate 2. Assumes no prompt caching, as an upper bound: the newest models cache prefixes from 512 tokens, so the stable system prompts and each session's growing history should cache, cutting input cost further; the first run's `cache_read_tokens` confirms it. This supersedes plan section 6's 300-600 output tokens per session, which counted one call per session and no thinking. Every figure is an assumption until the first filtered run records `usage`.

## Testing

All offline: no network, no credentials.

- `FakeClient` implements `LlmClient`, scripted per role (narrator or advisor, told apart by the request's system prompt), returning canned text, tool calls, refusals or schema-invalid output, and recording every request.
- `turns.py`: ranges per kind; raising to the stances' minimum; reaction on turn 2 or later with the violation on the advisor reply before it; ledger rows on turn 1; same seed gives the same plan; a skeleton needing more than 8 turns raises.
- `voices.py`: deterministic per PM; unchanged when a trait value changes.
- `tools.py`: each lookup against a small market fixture; nothing after the session date; forward calendar rows carry no surprise; id and name resolution; error results.
- `prompts.py`: the narrator request contains the stance line and voice and no trait id, param or value; the advisor request contains none of the persona, rules, ledger, ideas or skeleton text; directives are mid-conversation `system` messages; history is append-only across turns.
- `client.py`: hit and miss; atomic write; key stable across dict ordering; a model change gives a new key; uncommitted responses are not cached; budget error.
- `session.py`: full session with tool rounds; tool-round cap; retry on refusal and on schema-invalid output; `feedback` changes the narrator request.
- Stage (copying `tests/signals/test_stage.py::test_plan_stage_writes_byte_identical_tables_on_a_second_run` and its neighbours): writes both tables 1:1 with the filtered skeletons; PM filter; a failed session means no tables and a `DialogueError`; a second run fully from cache gives byte-identical tables and zero calls to the inner client; missing plan metadata raises.
- `tests/test_end_to_end.py`: extend the CLI chain through `dialogue` with the fake client injected.
- One `@pytest.mark.network` smoke test: a single real 2-turn session on `claude-opus-5-5`, confirming structured output, tools and mid-conversation `system` messages work together. Skipped unless `--run-network`.

## Limitations

- **Scripted violations do not survive a real-copilot replay.** When advisor turns are later regenerated by the deployed copilot, it will not carry out the scripted violation, so the frozen PM reaction that follows may answer something the new reply did not do. Accepted because the replay is a later refresh outside this build, only `revealed_reaction` sessions are affected (a minority of preference signals), and `dialogue_logs` marks scripted turns so the replay can keep them frozen or re-inject the violation line.
- **One narrator model.** Phrasing diversity comes only from stance lines, voices and sampling; the sameness risk is measured by Gate 2's n-gram overlap, not prevented here. Accepted because rotation would confound recoverability with narrator model (decision 2).
- **Placeholder advisor prompt until the user supplies the real one.** A corpus narrated before the swap carries a stand-in advisor; its prompt hash in run metadata identifies it. Accepted because the plan already accepts a simulated advisor (section 4) and the swap needs no code change.
- **Not reproducible at the generation level.** A rerun without the cache can return different text. Reproducibility is at the artefact level: the cache plus frozen tables. Accepted per plan 9.1.
- **No validation yet.** Leakage, ledger consistency and forbidden traits are unchecked until stage 7; a stage 6 run is not a usable corpus. Accepted per decision 1.
- **Advisor derives numbers itself.** Tools return raw levels; carry, spreads between instruments and returns are computed by the model and can be wrong. Stage 7 can check quoted levels against logged tool results but not every derived figure. Accepted because a tool per derived analytic is open-ended, and the copilot's arithmetic is not ground truth.
- **Turn ranges, tool-round cap, concurrency and effort are guesses.** Logged as config fields with a `basis`; to be revisited after the pilot's measured cost and Gate 2.
- **Cost figures are unmeasured.** See the cost section; the first filtered run replaces them.

## Living docs impact

- `README.md`: usage block gains the `dialogue` command; new "Dialogue" section after "Signal plan" describing inputs, the two tables and what is hidden, the resume-by-rerun model, the cache location, credentials (needed only on cache misses), the advisor prompt file and how to swap it, and the advice to measure cost on a filtered run of 2-3 PMs first.
- `docs/pm-dataset-plan.md`: section 9.1 "Sameness" now says one narrator model with per-PM voices, the cross-PM n-gram overlap reported by Gate 2, and a second narrator only if overlap is high, assigned balanced and recorded; section 9 stage 6 paragraph says the response cache serves as the manifest and turn counts are deterministic per session kind; section 6's token-cost bullet points at the per-call estimate above (several calls per session, thinking billed as output).
- No `ARCHITECTURE.md` (README stays the living doc, as in the stage 5 spec, decision 12).
