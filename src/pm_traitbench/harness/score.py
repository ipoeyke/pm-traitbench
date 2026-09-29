"""Deterministic scoring of replies: option letters and routine-question format checks."""

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import (
    CheckKind,
    FormatOutcome,
    Kind,
    ProbeForm,
    ProbeType,
    Scorer,
    SignalMode,
)
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.checks import (
    CHECKED_PARAMS,
    load_check_map,
    parse_routine_answer,
    run_check,
)
from pm_traitbench.harness.runner import RUN_METADATA, probes_sha256, run_dir
from pm_traitbench.tables.schema import ProbeRow, ScoreRow
from pm_traitbench.tables.specs import PROBES, RESPONSES, SCORES, SIGNALS, TRAITS
from pm_traitbench.tables.store import DataStore

SCORE_METADATA = "eval-score"
LETTER_PATTERN = r"^\(?([A-D])\)?(?=$|[\s.):])"
_LETTERS = "ABCD"


def _options(row: ProbeRow) -> list[str]:
    return [t for t in (row.option_a, row.option_b, row.option_c, row.option_d) if t is not None]


def parse_letter(response: str, n_options: int) -> str | None:
    """The option letter a reply opens with, or None.

    Parsing is strict because a lenient parser rewards a reply that lists every
    letter. The leading letter decides, so "A or B" parses to "A".
    """
    match = re.match(LETTER_PATTERN, response.strip())
    if match is None or match.group(1) not in _LETTERS[:n_options]:
        return None
    return match.group(1)


def score_option_letter(row: ProbeRow, response: str) -> ScoreRow:
    """Score a multiple-choice reply against the hidden answer letter."""
    parsed = parse_letter(response, len(_options(row)))
    return ScoreRow(
        probe_id=row.probe_id,
        pm_id=row.pm_id,
        scorer=Scorer.OPTION_LETTER,
        correct=parsed == row.answer,
        detail="parse_error" if parsed is None else None,
    )


def score_format(
    row: ProbeRow,
    response: str,
    check_map: dict[tuple[str, str], CheckKind],
    short_page_words: int,
) -> ScoreRow | None:
    """Score a routine reply on the held preferences that have a deterministic check.

    Returns None when every held preference needs a judge, so no score row exists.
    """
    held = [
        (param, value)
        for param, value in parse_routine_answer(row.answer)
        if param in CHECKED_PARAMS and check_map[(param, value)] != CheckKind.JUDGE
    ]
    if not held:
        return None
    failed = [
        f"{param}={value}"
        for param, value in held
        if run_check(check_map[(param, value)], response, short_page_words) == FormatOutcome.FAIL
    ]
    return ScoreRow(
        probe_id=row.probe_id,
        pm_id=row.pm_id,
        scorer=Scorer.FORMAT,
        correct=not failed,
        detail="; ".join(failed) or None,
    )


def evidence_type(signal_ids: Sequence[str], modes: Mapping[str, SignalMode]) -> str:
    """Whether a probe's supporting signals state the trait, reveal it, or both."""
    if not signal_ids:
        return "none"
    try:
        found = {modes[s] for s in signal_ids}
    except KeyError as exc:
        raise HarnessError(f"probe cites unknown signal {exc.args[0]}") from exc
    if found == {SignalMode.STATED}:
        return "explicit"
    if SignalMode.STATED not in found:
        return "implicit"
    return "mixed"


def _rate(scores: Sequence[ScoreRow]) -> dict[str, Any]:
    return {"n": len(scores), "accuracy": sum(s.correct for s in scores) / len(scores)}


def _slice(groups: dict[str, list[ScoreRow]]) -> dict[str, Any]:
    return {key: _rate(rows) for key, rows in sorted(groups.items())}


def summarise(
    probes: Sequence[ProbeRow],
    scores: Sequence[ScoreRow],
    kinds: Mapping[tuple[str, str], Kind],
    modes: Mapping[str, SignalMode],
) -> dict[str, Any]:
    """Aggregate scores by probe type, slice dimension and presence answer.

    Presence is reported per answer and balanced because most presence rows are
    "no", so an always-no reply would otherwise look strong.
    """
    by_id = {p.probe_id: p for p in probes}
    typed: dict[tuple[str, str, str], list[ScoreRow]] = defaultdict(list)
    dims: dict[str, dict[str, dict[str, list[ScoreRow]]]] = defaultdict(
        lambda: {
            "kind": defaultdict(list),
            "checkpoint_label": defaultdict(list),
            "evidence": defaultdict(list),
        }
    )
    presence: dict[str, list[ScoreRow]] = {"yes": [], "no": []}
    for score in scores:
        row = by_id[score.probe_id]
        typed[(row.probe_type, row.form, score.scorer)].append(score)
        if row.trait_id is None:
            kind = "none"
        elif (row.pm_id, row.trait_id) in kinds:
            kind = str(kinds[(row.pm_id, row.trait_id)])
        else:
            raise HarnessError(f"probe {row.probe_id} names unknown trait {row.trait_id}")
        d = dims[score.scorer]
        d["kind"][kind].append(score)
        d["checkpoint_label"][row.checkpoint_label].append(score)
        d["evidence"][evidence_type(row.supporting_signal_ids, modes)].append(score)
        if row.probe_type == ProbeType.TRAIT_PRESENCE:
            text = getattr(row, f"option_{row.answer.lower()}")
            presence["yes" if text == "yes" else "no"].append(score)

    by_type = []
    for (probe_type, form, scorer), rows in sorted(typed.items()):
        chance = None
        if scorer == Scorer.OPTION_LETTER:
            option_counts = [len(_options(by_id[s.probe_id])) for s in rows]
            chance = sum(1 / n for n in option_counts) / len(rows)
        by_type.append(
            {
                "probe_type": probe_type,
                "form": form,
                "scorer": scorer,
                "n": len(rows),
                "correct": sum(s.correct for s in rows),
                "accuracy": sum(s.correct for s in rows) / len(rows),
                "chance": chance,
                "parse_errors": sum(s.detail == "parse_error" for s in rows),
            }
        )

    scored = {s.probe_id for s in scores if s.scorer == Scorer.FORMAT}
    routine = [p for p in probes if p.probe_type == ProbeType.ROUTINE_QUESTION]
    awaiting = {
        "trait_mcq/open": _count(probes, ProbeType.TRAIT_MCQ, ProbeForm.OPEN),
        "in_situ/open": _count(probes, ProbeType.IN_SITU, ProbeForm.OPEN),
        "governance/open": _count(probes, ProbeType.GOVERNANCE, ProbeForm.OPEN),
        "routine_question/intrusion": len(routine),
        "routine_question/format_judge_only": sum(p.probe_id not in scored for p in routine),
    }
    accuracies = {k: (_rate(v)["accuracy"] if v else None) for k, v in presence.items()}
    both = all(presence.values())
    return {
        "by_type": by_type,
        "awaiting_judge": awaiting,
        "slices": {
            scorer: {name: _slice(groups) for name, groups in d.items()}
            for scorer, d in sorted(dims.items())
        },
        "presence": {
            "yes": {"n": len(presence["yes"]), "accuracy": accuracies["yes"]},
            "no": {"n": len(presence["no"]), "accuracy": accuracies["no"]},
            "balanced_accuracy": (accuracies["yes"] + accuracies["no"]) / 2 if both else None,
        },
    }


def _count(probes: Sequence[ProbeRow], probe_type: ProbeType, form: ProbeForm) -> int:
    return sum(p.probe_type == probe_type and p.form == form for p in probes)


def _check_run(config: Config, store: DataStore, run_store: DataStore, run_name: str) -> None:
    meta = run_store.read_run_metadata(RUN_METADATA)
    if meta is None:
        raise HarnessError(f"run '{run_name}' has no run metadata; run the evaluation first")
    if meta.get("pms_failed"):
        names = ", ".join(sorted(meta["pms_failed"]))
        raise HarnessError(
            f"run '{run_name}' has failed PMs ({names}); a partial run's accuracy covers a "
            "biased subset, so rerun it"
        )
    if meta.get("status") != "finished":
        raise HarnessError(f"run '{run_name}' is not finished; rerun the evaluation to completion")
    if meta.get("probes_sha256") != probes_sha256(store):
        raise HarnessError(f"probes changed since run '{run_name}' was made; rerun it with --force")


def score_run(config: Config, store: DataStore, run_name: str) -> dict[str, Any]:
    """Score a finished run's responses, write scores and summary.json, and return the summary."""
    rd = run_dir(store.data_dir, run_name)
    run_store = DataStore(rd, config.output)
    _check_run(config, store, run_store, run_name)

    responses = {r.probe_id: r.response for r in run_store.read(RESPONSES)}
    probes = store.read(PROBES)
    check_map = load_check_map(load_catalogue())
    scores: list[ScoreRow] = []
    for row in probes:
        if row.probe_id not in responses:
            raise HarnessError(f"run '{run_name}' has no response for probe {row.probe_id}")
        reply = responses[row.probe_id]
        score = None
        if row.probe_type == ProbeType.TRAIT_PRESENCE or (
            row.probe_type == ProbeType.TRAIT_MCQ and row.form == ProbeForm.MCQ
        ):
            score = score_option_letter(row, reply)
        elif row.probe_type == ProbeType.ROUTINE_QUESTION:
            score = score_format(row, reply, check_map, config.harness.short_page_words)
        if score is not None:
            scores.append(score)

    kinds = {(t.pm_id, t.trait_id): t.kind for t in store.read(TRAITS)}
    modes = {s.signal_id: s.mode for s in store.read(SIGNALS)}
    summary = summarise(probes, scores, kinds, modes)
    run_store.write(SCORES, scores)
    (rd / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    run_store.write_run_metadata(
        SCORE_METADATA,
        config,
        {"run_name": run_name, "probes_sha256": probes_sha256(store), "n_scored": len(scores)},
    )
    return summary
