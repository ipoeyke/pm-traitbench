# Stage progress for API-billed stages

**Tier:** light
**Escalation threshold:** 11 files
**Supersedes:** none

## Goal

Print progress while the `dialogue`, `validate` and `gate2` stages run, so an
operator can see how far a long, API-billed run has got, what it has spent,
and whether it is heading for the token budget. Today these stages print
nothing until they finish or fail.

## Decisions

1. **One plain line per finished unit, on stderr.** Chosen over a single
   line rewritten with `\r` and over a `tqdm` bar: both garble log files,
   `nohup` output and piped output, which is how a sandboxed run is watched.
   stderr keeps stdout free for command results and needs no new dependency.
2. **Hooked once, in the shared `run_bounded` helper** (`dialogue/stage.py`).
   All three stages already send every API-billed unit through it and through
   one `CachedClient`, whose `UsageTotals` hold running call and token counts.
   Per-stage wiring was rejected as three copies of the same logic.
3. **gate2 is included**, since it shares the helper and bills the API; its
   lines count units (one PM recovery or one session classification each).
4. **No switch to turn progress off** (YAGNI): the lines go to stderr and cost
   nothing when unread.
5. **Tokens shown are fresh tokens only** (input and output, cache hits
   excluded), the same count `UsageTotals` keeps and the token budget checks,
   so the line and a budget stop agree.

## Design

### Output

A start line when `run_bounded` begins, then one line each time a unit
finishes, success or failure:

```
[dialogue] 96 sessions, concurrency 8
[dialogue] 1/96 sessions (0 failed) | 4 calls, 0 cached | 18k in, 2k out tokens | 0m21s
[dialogue] 2/96 sessions (0 failed) | 9 calls, 1 cached | 37k in, 5k out tokens (42k/5.0M budget) | 0m40s
```

- `[<label>]` is the stage name: `dialogue`, `validate` or `gate2`.
- `<done>/<total> <unit>`: `unit` is `sessions` for dialogue and validate,
  `units` for gate2.
- `(<n> failed)`: units whose worker raised; `raise_on_failure` still reports
  each failure in full after the run, unchanged.
- `<calls> calls, <hits> cached`: `UsageTotals.calls` and `cache_hits`.
- Token counts: `UsageTotals.input_tokens` and `output_tokens`. When the
  client has a token budget, `(<in+out>/<budget> budget)` follows, since the
  budget checks the sum of both.
- Elapsed wall time since the start line: `MmSSs` under an hour, `HhMMm`
  from an hour on, since full runs can take hours.
- Number format: below 1,000 as is; otherwise whole thousands with `k`
  (`182k`) while that rounds below `1000k`; from there millions to one
  decimal with `M` (`5.0M`), so no count prints as `1000k`.
- Each line is flushed, so it appears immediately in a log file.
- With zero units only the start line prints.

### Components

- **New `src/pm_traitbench/dialogue/progress.py`:**
  - `format_count(n: int) -> str` and `format_elapsed(seconds: float) -> str`,
    pure.
  - `format_line(label, unit, done, total, failed, totals: UsageTotals,
    budget: int | None, elapsed: float) -> str`, pure.
  - `Progress`: built with `label`, `unit`, `total`, `totals: UsageTotals`,
    `budget: int | None` and `clock: Callable[[], float] = time.monotonic`.
    `start(max_concurrency: int)` records the start time and prints the start
    line; `finish(failed: bool)` increments counts and
    prints one line. Prints to `sys.stderr` looked up at call time, so pytest's
    `capsys` captures it.
- **`CachedClient`** (`dialogue/client.py`): a read-only `token_budget`
  property returning the budget it was built with.
- **`run_bounded`** (`dialogue/stage.py`): two new required keyword arguments,
  `label: str` and `unit: str`. It builds a `Progress` from `client.totals`
  and `client.token_budget`, calls `start()` before gathering, and in the
  bounded wrapper calls `finish(failed=False)` after the worker returns or
  `finish(failed=True)` before re-raising any exception. Everything else
  (ordering, returned exceptions, closing the client) is unchanged.
- **Callers:** `dialogue/stage.py` passes `label="dialogue", unit="sessions"`;
  `dialogue/validate/stage.py` passes `label="validate", unit="sessions"`;
  `gates/gate2/stage.py` passes `label="gate2", unit="units"`.

### Error handling

Progress never raises on its own; a unit's failure is only counted, and the
existing all-or-nothing error reporting after the run is untouched. A budget
stop shows as a run of failed units, then the existing `DialogueBudgetError`.

## Limitations

- Counts are per unit, not per API call: a validate session that regenerates
  several times prints one line when it finishes, so a slow session looks
  like a pause. Acceptable because calls and tokens on the next line still
  show the work done.
- Totals are read from shared `UsageTotals` when a unit finishes, so a line
  includes tokens from units still in flight. Acceptable: the line reports
  what has been spent, which is what a budget watch needs.
- No ETA: unit durations vary with regenerations and cache hits, so a
  rate-based estimate would mislead.
- A full run prints one line per session, thousands of lines. Acceptable in a
  log file; throttling was left out as YAGNI.

## Living docs impact

- No `ARCHITECTURE.md` is created: the user chose to keep the README as the
  only system description.
- `README.md` Usage section: one sentence saying the API-billed stages print
  a progress line per finished session or unit to stderr.

## Implementation notes

Files:

- Create `src/pm_traitbench/dialogue/progress.py`.
- Modify `src/pm_traitbench/dialogue/client.py` (`token_budget` property).
- Modify `src/pm_traitbench/dialogue/stage.py` (`run_bounded` signature and
  wrapper; its own call site).
- Modify `src/pm_traitbench/dialogue/validate/stage.py` (call site).
- Modify `src/pm_traitbench/gates/gate2/stage.py` (call site).
- Create `tests/dialogue/test_progress.py`.
- Modify `tests/dialogue/test_stage.py` (a `run_bounded` progress test).
- Modify `README.md`.

Test intent:

- `format_count`: 999, 1,000, 182,400, 999,499, 999,500, 5,000,000 map to
  `999`, `1k`, `182k`, `999k`, `1.0M`, `5.0M`.
- `format_elapsed`: 21 s, 3,599 s, 3,720 s map to `0m21s`, `59m59s`, `1h02m`.
- `format_line`: exact string with and without a budget.
- `Progress` with a fake clock: start line, then one line per `finish`, the
  failed count rising only on `finish(failed=True)`.
- `run_bounded` over three items where one worker raises: stderr holds the
  start line and three progress lines, the last showing `3/3` and
  `(1 failed)`; results still come back in item order with the exception in
  place.
- Existing dialogue, validate and gate2 stage tests pass unchanged.

Tests to copy:

- Progress formatting: none - first test for this path.
- `run_bounded`: `test_session_concurrency_never_exceeds_max_concurrency` in
  `tests/dialogue/test_stage.py`.

Scoped test command:

```
uv run pytest -q tests/dialogue/test_progress.py tests/dialogue/test_stage.py tests/dialogue/validate tests/gates/gate2
```
