---
name: run-report
description: Present the pipeline's stage results (data/run_metadata/*.json and eval runs) as a concise HTML report that flags gate and stat-test failures, stale stages and warnings, with a root-cause analysis and suggested fix for each failure. Use when the user asks to review, summarise, report on or check the results of a pipeline run or stage, or asks why a stage or gate failed.
---

# Run report

A script extracts findings from run metadata deterministically. Your job is the part a
script cannot do: find why each failure happened and what to change, then render both
into one HTML report.

## 1. Collect findings

```sh
uv run python .claude/skills/run-report/report.py --data-dir data --json
```

Pass the user's `--data-dir` if they name one. Each finding has a stable `id` (e.g.
`gate1-1`), a `level` (`error`, `warn`, `info`), the stage's `created_at` and
`git_commit`. The script also writes a first report without analysis.

## 2. Investigate every `error` and every non-trivial `warn`

Root-cause each one from evidence, not from the finding text alone. Group findings that
share a cause and analyse them once. Skip `info` unless it explains an error.

Where to look:

- **Gate rows** (`gate1-*`, `gate2-*` failed or insufficient): read the blocking row and
  its per-class rows in `data/gate1_cells.jsonl` / `gate2_cells.jsonl`, the per-PM
  statistics in `gate1_pm` / `gate2_pm`, the estimator in
  `src/pm_traitbench/gates/gate1/estimators/` or `gates/gate2/`, and the thresholds in
  the config. Decide whether the planted signal is too weak (engine or sampling
  parameters), the estimator is noisy or biased (low `n`, outliers, missing PMs), or the
  threshold sits at the margin. Check which PMs drag the statistic, e.g. active PMs with
  low recovered values versus their planted strength.
- **Stale stage** (`stale: older than upstream ...`): compare `created_at` and
  `git_commit` across stages and `git log <old>..<new> -- src/` to see whether the
  upstream change alters what the stale stage reads. The fix is usually rerunning from
  that stage with `scripts/generate.sh --from <stage> --force`.
- **Market calibration**: `src/pm_traitbench/market/check.py` and the regime targets in
  the config.
- **Plan warnings** (dropped signals, drift sides below minimum): per-PM counts in
  `run_metadata/plan.json` `pms`, plus the engine's `opportunities` for those PMs. A PM
  with few ideas or exits has few carrier sessions.
- **Dialogue / validate**: `fails_by_layer`, `dropped_session_ids`, the hidden
  `validation` and `dialogue_logs` tables for the failing sessions.
- **Eval**: the traceback in `pms_failed` of `data/eval/<run>/run_metadata/eval-run.json`.
- **Recent changes**: `git log --oneline -- <relevant src path>` for commits that line up
  with the regression.

Each cause should name the mechanism and the evidence (numbers, PM ids, file:line). Each
fix should be one concrete action: a config key and value, a code change at a location,
or a command to rerun. Say so when the evidence is inconclusive rather than guessing.

## 3. Write the analysis and render

Write `analysis.json` in the scratchpad (not the repo):

```json
{
  "summary": "Two to four sentences: overall verdict, the blocking issue, what to do first.",
  "findings": {
    "gate1-1": {
      "cause": "Mechanism, one or two sentences.",
      "evidence": "Numbers, PM ids, file:line.",
      "fix": "One concrete action."
    }
  }
}
```

Keys must be finding ids from step 1; the script rejects unknown ids. Findings sharing a
cause may repeat the same entry. Then render:

```sh
uv run python .claude/skills/run-report/report.py --data-dir data --analysis <path>/analysis.json
```

The report lands at `data/run_report.html` (override with `--out`). Open it with
`open <path>` on macOS.

## 4. Reply

Give the report path, the overall verdict, and each error with its cause and fix in one
line each. Keep the reply short; the detail is in the report.
