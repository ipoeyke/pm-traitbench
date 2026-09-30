"""Render every stage's run metadata under a data directory as one concise HTML report.

Reads `<data-dir>/run_metadata/*.json`, each eval run's metadata under `<data-dir>/eval/`,
and the gate cell tables for detail on failed rows. `--json` prints the findings with
stable ids; `--analysis` merges a root-cause file keyed by those ids into the report.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Pipeline order, and the stages each one reads from, for the staleness check.
STAGES = ("sample", "market", "engine", "gate1", "plan", "dialogue", "validate", "gate2", "probes")
UPSTREAM = {
    "market": (),
    "sample": (),
    "engine": ("sample", "market"),
    "gate1": ("engine",),
    "plan": ("engine",),
    "dialogue": ("plan",),
    "validate": ("dialogue",),
    "gate2": ("validate",),
    "probes": ("validate",),
}
ERROR, WARN, INFO = "error", "warn", "info"
_RANK = {ERROR: 0, WARN: 1, INFO: 2}


@dataclass
class Issue:
    level: str
    text: str
    id: str = ""


@dataclass
class Section:
    """One stage's report: its headline numbers, issues and detail blocks (pre-escaped HTML)."""

    name: str
    meta: dict[str, Any]
    facts: list[tuple[str, str]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    details: list[tuple[str, str]] = field(default_factory=list)

    def add(self, level: str, text: str) -> None:
        self.issues.append(Issue(level, text))

    @property
    def status(self) -> str:
        levels = {i.level for i in self.issues}
        return ERROR if ERROR in levels else WARN if WARN in levels else "ok"


def esc(value: Any) -> str:
    return html.escape(str(value))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(fmt(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_PATTERNS = (
    (re.compile(r"\bpm_\d+\b"), "<pm>"),
    (re.compile(r"\bt_\d+\b"), "<trait>"),
    (re.compile(r"\bs_pm\d+_[\d-]+_[a-z]\b"), "<session>"),
    (re.compile(r"\bsg_\d+\b"), "<signal>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
    (re.compile(r"(?<![\w<])\d+(\.\d+)?(?![\w>])"), "<n>"),
)


def pattern(message: str) -> str:
    """A warning with its ids, dates and numbers masked, so repeats group together."""
    for regex, token in _PATTERNS:
        message = regex.sub(token, message)
    return message


def group_warnings(section: Section, warnings: list[str], level: str = WARN) -> None:
    """Add one issue per warning pattern and a detail block listing examples."""
    if not warnings:
        return
    groups: dict[str, list[str]] = {}
    for w in warnings:
        groups.setdefault(pattern(w), []).append(w)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for pat, items in ranked[:6]:
        section.add(level, f"{len(items)} × {pat}")
    if len(ranked) > 6:
        rest = sum(len(items) for _, items in ranked[6:])
        section.add(level, f"{rest} more warnings in {len(ranked) - 6} other patterns")
    rows = [[len(items), pat, "; ".join(items[:3])] for pat, items in ranked]
    section.details.append(
        (f"Warnings ({len(warnings)})", table(["count", "pattern", "examples"], rows))
    )


def usage(section: Section, meta: dict[str, Any]) -> None:
    if "calls" not in meta:
        return
    section.facts.append(("calls", f"{fmt(meta['calls'])} ({fmt(meta['cache_hits'])} cached)"))
    tokens = meta["input_tokens"] + meta["output_tokens"]
    section.facts.append(("fresh tokens", fmt(tokens)))
    if meta.get("rejected_replies"):
        section.add(WARN, f"{meta['rejected_replies']} model replies rejected and retried")


def read_table(data_dir: Path, name: str) -> list[dict[str, Any]] | None:
    jsonl = data_dir / f"{name}.jsonl"
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    parquet = data_dir / f"{name}.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None
        return pq.read_table(parquet).to_pylist()
    return None


# Per-stage summarisers.


def s_sample(sec: Section, data_dir: Path) -> None:
    pop = sec.meta["config"].get("population", {})
    sec.facts.append(("asset classes", ", ".join(pop.get("asset_classes", []))))
    seeds = [*pop.get("market_seeds", []), *pop.get("pilot_market_seeds", [])]
    sec.facts.append(("market seeds", ", ".join(seeds)))


def s_market(sec: Section, data_dir: Path) -> None:
    metrics = [m for seed in sec.meta.get("check", {}).values() for m in seed["metrics"]]
    failed = [m for m in metrics if not m["passed"]]
    sec.facts.append(("calibration checks", f"{len(metrics) - len(failed)}/{len(metrics)} pass"))
    files = sec.meta.get("raw_manifest", {}).get("files")
    if files is not None:
        sec.facts.append(("raw files", fmt(files if isinstance(files, int) else len(files))))
    for m in failed[:8]:
        where = "/".join(str(m[k]) for k in ("seed", "regime", "family") if m[k])
        sec.add(
            ERROR,
            f"market check failed {where} {m['metric']}: realised {fmt(m['realised'])} "
            f"vs target {fmt(m['target'])} (tol {fmt(m['tolerance'])})",
        )
    if len(failed) > 8:
        sec.add(ERROR, f"{len(failed) - 8} more market checks failed")


def s_engine(sec: Section, data_dir: Path) -> None:
    ideas = sec.meta.get("ideas_per_pm", {})
    counts = sorted(ideas.values())
    sec.facts.append(("PMs run", fmt(len(ideas))))
    if counts:
        sec.facts.append(
            (
                "ideas per PM",
                f"min {counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}",
            )
        )
    skipped = sec.meta.get("skipped", [])
    if skipped:
        sec.add(INFO, f"{len(skipped)} PMs skipped (multi-asset, no adapter yet)")
    zero = sorted(pm for pm, n in ideas.items() if n == 0)
    if zero:
        sec.add(WARN, f"{len(zero)} PMs produced no ideas: {', '.join(zero[:10])}")


_GATE1_CHECKS = (
    ("gap_ok", "active_share_past_floor", "gap_fraction", "active share past floor"),
    ("rank_ok", "active_rank_corr", "min_rank_corr", "active rank corr"),
    ("pop_ok", "pop_z", "min_pop_z", "pop z"),
)


def s_gate1(sec: Section, data_dir: Path) -> None:
    th = sec.meta.get("thresholds", {})
    failed = sec.meta.get("failed", [])
    cells = read_table(data_dir, "gate1_cells") or []
    blocking = [c for c in cells if c["blocking"]]
    by_id = {f"all/{c['param']}": c for c in blocking}
    if blocking:
        passed = sum(c["verdict"] == "pass" for c in blocking)
        sec.facts.append(("blocking rows", f"{passed}/{len(blocking)} pass"))
    sec.facts.append(("verdict", "FAIL" if failed else "pass"))
    for fid in failed:
        cell = by_id.get(fid)
        if cell is None:
            sec.add(ERROR, f"gate1 failed: {fid}")
            continue
        why = [
            f"{label} {fmt(cell[value])} < {fmt(th.get(knob))}"
            for ok, value, knob, label in _GATE1_CHECKS
            if cell["test"] == "per_pm" or ok != "rank_ok"
            if not cell[ok]
        ]
        if cell.get("count_ok") is False:
            why.append("count shortfall")
        sec.add(
            ERROR, f"gate1 failed: {fid} ({cell['test']}) - {'; '.join(why) or cell['verdict']}"
        )
    # Warnings read `seed/asset_class/param: count_shortfall`; group them by param.
    shortfalls: dict[str, list[str]] = {}
    for w in sec.meta.get("warnings", []):
        cell, _, _ = w.partition(": ")
        seed, asset_class, param = cell.split("/")
        shortfalls.setdefault(param, []).append(f"{seed}/{asset_class}")
    for param, where in sorted(shortfalls.items()):
        sec.add(WARN, f"{param} count shortfall in {len(where)} cell(s): {', '.join(where)}")
    if blocking:
        rows = [
            [
                c["param"],
                c["test"],
                c["verdict"],
                c["n_neutral"],
                c["n_active"],
                c["active_share_past_floor"],
                c["active_rank_corr"],
                c["pop_z"],
            ]
            for c in blocking
        ]
        headers = [
            "param",
            "test",
            "verdict",
            "n neutral",
            "n active",
            "share past floor",
            "active rank corr",
            "pop z",
        ]
        sec.details.insert(0, ("Blocking rows", table(headers, rows)))
    judged = [c for c in cells if not c["blocking"] and c["verdict"] == "fail"]
    if judged:
        sec.add(INFO, f"{len(judged)} non-blocking gate1 rows fail (per-class or report-only)")


def s_plan(sec: Section, data_dir: Path) -> None:
    pms = sec.meta.get("pms", {})
    sec.facts.append(("PMs planned", fmt(len(pms))))
    planned = sum(sum(p["revealed_planned"].values()) for p in pms.values())
    placed = sum(sum(p["revealed_placed"].values()) for p in pms.values())
    if planned:
        sec.facts.append(
            ("revealed signals placed", f"{placed}/{planned} ({placed / planned:.0%})")
        )
    shortfalls = [
        [pm, t, p["revealed_placed"].get(t, 0), n]
        for pm, p in pms.items()
        for t, n in p["revealed_planned"].items()
        if p["revealed_placed"].get(t, 0) < n / 2
    ]
    if shortfalls:
        sec.add(WARN, f"{len(shortfalls)} planted traits placed under half their revealed signals")
        sec.details.append(
            (
                "Revealed-signal shortfalls (placed < half planned)",
                table(["pm", "trait", "placed", "planned"], shortfalls),
            )
        )
    group_warnings(sec, sec.meta.get("warnings", []))


def s_dialogue(sec: Section, data_dir: Path) -> None:
    sec.facts.append(("PMs", fmt(len(sec.meta.get("pms", [])))))
    sessions = sec.meta.get("sessions", {})
    sec.facts.append(("sessions", ", ".join(f"{k} {v}" for k, v in sorted(sessions.items()))))
    sec.facts.append(
        ("models", f"{sec.meta.get('narrator_model')} / {sec.meta.get('advisor_model')}")
    )
    usage(sec, sec.meta)
    group_warnings(sec, sec.meta.get("warnings", []))


def s_validate(sec: Section, data_dir: Path) -> None:
    m = sec.meta
    checked = m.get("sessions_checked", 0)
    sec.facts.append(("sessions checked", fmt(checked)))
    sec.facts.append(("regenerated", fmt(m.get("regenerated"))))
    sec.facts.append(("dropped", fmt(m.get("dropped"))))
    fails = {k: v for k, v in m.get("fails_by_layer", {}).items() if v}
    if fails:
        sec.facts.append(("fails by layer", ", ".join(f"{k} {v}" for k, v in fails.items())))
    if m.get("dropped"):
        voids = len(m.get("void_signal_ids", []))
        sec.add(WARN, f"{m['dropped']} sessions dropped, voiding {voids} signals")
    usage(sec, m)
    group_warnings(sec, m.get("warnings", []))


def s_gate2(sec: Section, data_dir: Path) -> None:
    m = sec.meta
    failed, insufficient = m.get("failed", []), m.get("insufficient", [])
    sec.facts.append(("PMs", fmt(len(m.get("pms", [])))))
    sec.facts.append(("verdict", "FAIL" if failed else "pass"))
    alpha = m.get("thresholds", {}).get("alpha")
    cells = read_table(data_dir, "gate2_cells") or []
    blocking = [c for c in cells if c["blocking"]]

    def cid(c: dict[str, Any]) -> str:
        return "all/preferences" if c["param"] is None else f"{c['slice_value']}/{c['param']}"

    by_id = {cid(c): c for c in blocking}
    for fid in failed:
        c = by_id.get(fid)
        extra = (
            f" - rate {fmt(c['rate'])} vs chance {fmt(c['chance'])}, p {fmt(c['p'])} "
            f"> alpha {fmt(alpha)} (n {c['n']})"
            if c
            else ""
        )
        sec.add(ERROR, f"gate2 failed: {fid}{extra}")
    if insufficient:
        sec.add(WARN, f"{len(insufficient)} blocking rows insufficient: {', '.join(insufficient)}")
    if m.get("pms_without_sessions"):
        sec.add(WARN, f"{len(m['pms_without_sessions'])} PMs have no sessions")
    overlap = m.get("overlap")
    if isinstance(overlap, dict):
        sec.facts.append(("n-gram overlap", ", ".join(f"{k} {fmt(v)}" for k, v in overlap.items())))
    usage(sec, m)
    group_warnings(sec, m.get("warnings", []))
    if blocking:
        rows = [[cid(c), c["verdict"], c["n"], c["rate"], c["chance"], c["p"]] for c in blocking]
        sec.details.insert(
            0, ("Blocking rows", table(["row", "verdict", "n", "rate", "chance", "p"], rows))
        )


def s_probes(sec: Section, data_dir: Path) -> None:
    m = sec.meta
    by_type: Counter[str] = Counter()
    for counts in m.get("probes_by_type", {}).values():
        by_type.update(counts)
    sec.facts.append(("PMs", fmt(len(m.get("pms", [])))))
    sec.facts.append(("probes", ", ".join(f"{k} {v}" for k, v in sorted(by_type.items()))))
    skipped = {k: v for k, v in m.get("skipped_probes", {}).items() if v}
    if skipped:
        sec.add(INFO, "probes skipped: " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    if m.get("skipped_checkpoints"):
        sec.add(INFO, f"{len(m['skipped_checkpoints'])} checkpoints skipped (no sessions yet)")
    if m.get("pms_without_sessions"):
        sec.add(WARN, f"{len(m['pms_without_sessions'])} PMs have no sessions")


def s_eval(sec: Section, data_dir: Path) -> None:
    m = sec.meta
    sec.facts.append(("system", str(m.get("sut"))))
    sec.facts.append(("status", str(m.get("status"))))
    sec.facts.append(("PMs completed", fmt(len(m.get("pms_completed", [])))))
    if m.get("status") != "finished":
        sec.add(ERROR, f"eval run status is '{m.get('status')}': interrupted or still running")
    for pm, trace in m.get("pms_failed", {}).items():
        last = trace.strip().splitlines()[-1] if trace.strip() else "no traceback"
        sec.add(ERROR, f"{pm} failed: {last}")
    score = sec.meta.get("_score")
    if score is None:
        sec.add(INFO, "not scored yet")
    else:
        sec.facts.append(("scored responses", fmt(score.get("n_scored"))))
        if score.get("probes_sha256") != m.get("probes_sha256"):
            sec.add(WARN, "score was computed against different probes than the run")


SUMMARISERS = {
    "sample": s_sample,
    "market": s_market,
    "engine": s_engine,
    "gate1": s_gate1,
    "plan": s_plan,
    "dialogue": s_dialogue,
    "validate": s_validate,
    "gate2": s_gate2,
    "probes": s_probes,
}


def when(meta: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(meta["created_at"])


def cross_checks(sections: dict[str, Section]) -> None:
    """Flag stages older than a stage they read from, and config drift between stages."""
    for name, sec in sections.items():
        newer = [
            up
            for up in UPSTREAM.get(name, ())
            if up in sections and when(sections[up].meta) > when(sec.meta)
        ]
        if newer:
            sec.add(WARN, f"stale: older than upstream {', '.join(newer)}; rerun from {name}")
    ordered = [sections[s] for s in STAGES if s in sections]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        a, b = prev.meta.get("config", {}), cur.meta.get("config", {})
        diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        if diff:
            cur.add(INFO, f"config differs from {prev.name} in: {', '.join(diff)}")
        if prev.meta.get("root_seed") != cur.meta.get("root_seed"):
            cur.add(ERROR, f"root seed differs from {prev.name}")
    shas = {
        s: sections[s].meta.get("sessions_sha256") for s in ("gate2", "probes") if s in sections
    }
    if len(set(shas.values())) > 1:
        sections["probes"].add(WARN, "gate2 and probes were built from different sessions")


def load(data_dir: Path) -> dict[str, Section]:
    sections: dict[str, Section] = {}
    meta_dir = data_dir / "run_metadata"
    for name in STAGES:
        path = meta_dir / f"{name}.json"
        if path.exists():
            sections[name] = Section(name, json.loads(path.read_text(encoding="utf-8")))
    for path in sorted((data_dir / "eval").glob("*/run_metadata/eval-run.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        score_path = path.with_name("eval-score.json")
        if score_path.exists():
            meta["_score"] = json.loads(score_path.read_text(encoding="utf-8"))
        sections[f"eval:{path.parent.parent.name}"] = Section(f"eval: {meta['run_name']}", meta)
    return sections


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mute:#6b6b66;--line:#e3e2dd;--card:#fff;
--err:#b42318;--err-bg:#fdecea;--warn:#8a5a00;--warn-bg:#fff5db;--ok:#1f7a3d;--ok-bg:#e7f5ec;
--info:#35607f;--info-bg:#eaf1f7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;
--mute:#9b9a94;--line:#2d2d2a;--card:#1e1e1c;--err:#ff8a7a;--err-bg:#3a1b17;--warn:#f2c14e;
--warn-bg:#352a10;--ok:#6fd08c;--ok-bg:#15301f;--info:#8fbbe0;--info-bg:#18283a}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--mute:#9b9a94;--line:#2d2d2a;--card:#1e1e1c;
--err:#ff8a7a;--err-bg:#3a1b17;--warn:#f2c14e;--warn-bg:#352a10;--ok:#6fd08c;--ok-bg:#15301f;
--info:#8fbbe0;--info-bg:#18283a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0}
.sub{color:var(--mute);font-size:12px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;
margin:12px 0}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:10px;
text-transform:uppercase;letter-spacing:.03em}
.error{color:var(--err);background:var(--err-bg)}.warn{color:var(--warn);background:var(--warn-bg)}
.ok{color:var(--ok);background:var(--ok-bg)}.info{color:var(--info);background:var(--info-bg)}
ul.issues{list-style:none;padding:0;margin:8px 0 0}
ul.issues li{padding:3px 0;display:flex;gap:8px;align-items:baseline}
ul.issues li .badge{flex:none;min-width:52px;text-align:center}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.meta{color:var(--mute);font-size:12px;margin-left:auto}
dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:10px 0 0;font-size:13px}
dt{color:var(--mute)}dd{margin:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}
th,td{text-align:left;padding:3px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute);font-size:13px}
.overview td:first-child{font-weight:600}
code{font-family:ui-monospace,monospace;font-size:12px}
.mute{color:var(--mute)}
.rca{margin:4px 0 4px;padding:6px 10px;border-left:3px solid var(--line);font-size:13px}
.rca b{color:var(--mute);font-weight:600}
.summary p{margin:6px 0 0}
"""


def badge(level: str) -> str:
    return f'<span class="badge {level}">{esc(level)}</span>'


def assign_ids(sections: dict[str, Section]) -> None:
    """Number each stage's issues in severity order, e.g. `gate1-1`, for the analysis file."""
    for key, sec in sections.items():
        sec.issues.sort(key=lambda i: _RANK[i.level])
        for n, issue in enumerate(sec.issues, 1):
            issue.id = f"{key.replace(':', '-')}-{n}"


def rca(issue: Issue, analysis: dict[str, Any]) -> str:
    """The cause and fix recorded for an issue, or a marker when an error has none."""
    entry = analysis.get("findings", {}).get(issue.id)
    if not entry:
        missing = issue.level == ERROR and analysis
        return '<div class="rca mute">no root-cause analysis recorded</div>' if missing else ""
    parts = [
        f"<div><b>{label}</b> {esc(entry[k])}</div>"
        for k, label in (("cause", "Cause:"), ("fix", "Fix:"), ("evidence", "Evidence:"))
        if entry.get(k)
    ]
    return f'<div class="rca">{"".join(parts)}</div>'


def issue_list(issues: list[Issue]) -> str:
    items = "".join(
        f'<li>{badge(i.level)}<span>{esc(i.text)} <code class="mute">{esc(i.id)}</code></span></li>'
        for i in issues
    )
    return f'<ul class="issues">{items}</ul>' if items else ""


def render(sections: dict[str, Section], data_dir: Path, analysis: dict[str, Any]) -> str:
    attention = [
        (s.name, i) for s in sections.values() for i in s.issues if i.level in (ERROR, WARN)
    ]
    attention.sort(key=lambda p: _RANK[p[1].level])
    n_err = sum(i.level == ERROR for _, i in attention)
    n_warn = len(attention) - n_err
    overall = ERROR if n_err else WARN if n_warn else "ok"

    rows = "".join(
        f"<tr><td>{esc(s.name)}</td><td>{badge(s.status)}</td>"
        f"<td>{esc(when(s.meta).strftime('%Y-%m-%d %H:%M'))}</td>"
        f"<td><code>{esc(s.meta.get('git_commit', '')[:7])}</code></td>"
        f"<td>{sum(i.level == ERROR for i in s.issues)}</td>"
        f"<td>{sum(i.level == WARN for i in s.issues)}</td></tr>"
        for s in sections.values()
    )
    overview = (
        '<div class="card scroll"><table class="overview"><thead><tr><th>stage</th>'
        "<th>status</th><th>run at (UTC)</th><th>commit</th><th>errors</th><th>warnings</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )
    missing = [s for s in STAGES if s not in sections]
    att_items = "".join(
        f"<li>{badge(i.level)}<span><b>{esc(n)}</b>: {esc(i.text)}{rca(i, analysis)}</span></li>"
        for n, i in attention
    )
    att = (
        f'<div class="card"><h2>Needs attention</h2><ul class="issues">{att_items}</ul></div>'
        if attention
        else '<div class="card"><h2>Needs attention</h2><p>Nothing flagged.</p></div>'
    )
    if analysis.get("summary"):
        att = (
            f'<div class="card summary"><h2>Summary</h2><p>{esc(analysis["summary"])}</p></div>'
            + att
        )
    cards = []
    for s in sections.values():
        facts = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in s.facts)
        det = "".join(
            f'<details><summary>{esc(t)}</summary><div class="scroll">{body}</div></details>'
            for t, body in s.details
        )
        cards.append(
            f'<div class="card"><div class="head"><h2>{esc(s.name)}</h2>{badge(s.status)}'
            f'<span class="meta">{esc(s.meta["created_at"][:19])} · '
            f"<code>{esc(s.meta.get('git_commit', '')[:7])}</code></span></div>"
            f"<dl>{facts}</dl>{issue_list(s.issues)}{det}</div>"
        )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    not_run = f" · not run: {', '.join(missing)}" if missing else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Pipeline Run Report</title><style>{CSS}</style></head><body><main>"
        f"<h1>Pipeline run report {badge(overall)}</h1>"
        f'<div class="sub">{esc(data_dir)} · generated {stamp} · {n_err} errors, '
        f"{n_warn} warnings{esc(not_run)}</div>"
        f"{att}{overview}{''.join(cards)}</main></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, help="default: <data-dir>/run_report.html")
    parser.add_argument("--analysis", type=Path, help="root-cause JSON keyed by finding id")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    sections = load(data_dir)
    if not sections:
        raise SystemExit(f"no run metadata under {data_dir / 'run_metadata'}")
    for key, sec in sections.items():
        summarise = s_eval if key.startswith("eval:") else SUMMARISERS[key]
        summarise(sec, data_dir)
    cross_checks(sections)
    assign_ids(sections)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8")) if args.analysis else {}
    unknown = set(analysis.get("findings", {})) - {
        i.id for s in sections.values() for i in s.issues
    }
    if unknown:
        raise SystemExit(f"analysis names unknown finding id(s): {', '.join(sorted(unknown))}")
    out = args.out or data_dir / "run_report.html"
    out.write_text(render(sections, data_dir, analysis), encoding="utf-8")

    if args.json:
        findings = [
            {
                "id": i.id,
                "stage": s.name,
                "level": i.level,
                "text": i.text,
                "created_at": s.meta["created_at"],
                "git_commit": s.meta.get("git_commit"),
            }
            for s in sections.values()
            for i in s.issues
        ]
        print(json.dumps({"report": str(out.resolve()), "findings": findings}, indent=1))
        return
    print(f"report: {out.resolve()}")
    for sec in sections.values():
        print(f"{sec.name}: {sec.status}")
        for i in sec.issues:
            if i.level != INFO:
                print(f"  {i.id} [{i.level}] {i.text}")


if __name__ == "__main__":
    main()
