"""Scoring of replies: option letters, format checks and judge verdicts, and the summary."""

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import ProbeBank
from pm_traitbench.config import Config
from pm_traitbench.enums import (
    CheckKind,
    EvidenceType,
    FormatOutcome,
    Judge,
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
from pm_traitbench.harness.judge import (
    JudgeInputs,
    governance_kind,
    in_situ_case,
    select_items,
    split_detail,
)
from pm_traitbench.harness.runner import check_scorable, probes_sha256, run_dir
from pm_traitbench.tables.schema import DriftEvent, JudgementRow, ProbeRow, ScoreRow, Trait
from pm_traitbench.tables.specs import (
    DRIFT_EVENTS,
    JUDGEMENTS,
    PROBES,
    RESPONSES,
    SCORES,
    SIGNALS,
    TRAITS,
)
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


def checkable_values(
    row: ProbeRow, check_map: Mapping[tuple[str, str], CheckKind]
) -> list[tuple[str, str]]:
    """The routine probe's held (param, value) pairs that have a deterministic check."""
    held = []
    for param, value in parse_routine_answer(row.answer):
        if param not in CHECKED_PARAMS:
            continue
        if (param, value) not in check_map:
            raise HarnessError(f"probe {row.probe_id}: no check for {param}={value}")
        if check_map[(param, value)] != CheckKind.JUDGE:
            held.append((param, value))
    return held


def score_format(
    row: ProbeRow,
    response: str,
    check_map: Mapping[tuple[str, str], CheckKind],
    short_page_words: int,
) -> ScoreRow | None:
    """Score a routine reply on the held preferences that have a deterministic check.

    Returns None, so no score row exists, when no held value has a check or every
    check is not applicable to the reply. An empty reply is incorrect.
    """
    held = checkable_values(row, check_map)
    if not held:
        return None
    if not response.strip():
        return _format_row(row, False, "empty_reply")
    outcomes = [
        (f"{param}={value}", run_check(check_map[(param, value)], response, short_page_words))
        for param, value in held
    ]
    if all(outcome == FormatOutcome.NOT_APPLICABLE for _, outcome in outcomes):
        return None
    failed = [name for name, outcome in outcomes if outcome == FormatOutcome.FAIL]
    return _format_row(row, not failed, "; ".join(failed) or None)


def _format_row(row: ProbeRow, correct: bool, detail: str | None) -> ScoreRow:
    return ScoreRow(
        probe_id=row.probe_id,
        pm_id=row.pm_id,
        scorer=Scorer.FORMAT,
        correct=correct,
        detail=detail,
    )


_AWAITING_KEYS: dict[Judge, str] = {
    Judge.OPEN: "trait_mcq/open",
    Judge.IN_SITU: "in_situ/open",
    Judge.GOVERNANCE: "governance/open",
    Judge.INTRUSION: "routine_question/intrusion",
    Judge.FORMAT: "routine_question/format_judge",
}


def judgement_score(row: JudgementRow) -> ScoreRow | None:
    """The score row of a judgement, or None when a format judge found nothing to check."""
    detail = None
    if row.detail == "empty_reply":
        detail = row.detail
    elif row.judge == Judge.FORMAT:
        outcomes = split_detail(row.detail, row.judge)
        if all(o == FormatOutcome.NOT_APPLICABLE for o in outcomes.values()):
            return None
        detail = "; ".join(k for k, o in outcomes.items() if o == FormatOutcome.FAIL) or None
    elif not row.correct:
        fields = split_detail(row.detail, row.judge)
        match row.judge:
            case Judge.OPEN:
                detail = f"chose={fields['choice']}"
            case Judge.INTRUSION:
                detail = f"evidence={fields['evidence']}"
            case _:
                detail = "; ".join(k for k, v in fields.items() if v == "false")
    return ScoreRow(
        probe_id=row.probe_id,
        pm_id=row.pm_id,
        scorer=Scorer(row.judge.value),
        correct=row.correct,
        detail=detail,
    )


def awaiting_counts(
    probes: Sequence[ProbeRow],
    judgements: Sequence[JudgementRow],
    check_map: Mapping[tuple[str, str], CheckKind],
    traits: Sequence[Trait],
    drift_events: Sequence[DriftEvent],
    bank: ProbeBank,
) -> dict[str, int]:
    """Per judge item type, the items that have no judgement row yet.

    Items come from the same selection the judge pass uses, so a routine question with
    no active bias or no judge-only value is never awaiting.
    """
    inputs = JudgeInputs(
        probes,
        dict.fromkeys((p.probe_id for p in probes), ""),
        traits,
        drift_events,
        bank,
        check_map,
    )
    judged = {(j.probe_id, j.judge) for j in judgements}
    counts = dict.fromkeys(_AWAITING_KEYS.values(), 0)
    for item in select_items(inputs).items:
        if (item.probe.probe_id, item.judge) not in judged:
            counts[_AWAITING_KEYS[item.judge]] += 1
    return counts


def evidence_type(signal_ids: Sequence[str], modes: Mapping[str, SignalMode]) -> EvidenceType:
    """Whether a probe's supporting signals state the trait, reveal it, or both."""
    if not signal_ids:
        return EvidenceType.NONE
    try:
        found = {modes[s] for s in signal_ids}
    except KeyError as exc:
        raise HarnessError(f"probe cites unknown signal {exc.args[0]}") from exc
    if found == {SignalMode.STATED}:
        return EvidenceType.EXPLICIT
    if SignalMode.STATED not in found:
        return EvidenceType.IMPLICIT
    return EvidenceType.MIXED


def _rate(scores: Sequence[ScoreRow]) -> dict[str, Any]:
    return {"n": len(scores), "accuracy": sum(s.correct for s in scores) / len(scores)}


def _slice(groups: dict[str, list[ScoreRow]]) -> dict[str, Any]:
    return {key: _rate(rows) for key, rows in sorted(groups.items())}


def summarise(
    probes: Sequence[ProbeRow],
    scores: Sequence[ScoreRow],
    kinds: Mapping[tuple[str, str], Kind],
    modes: Mapping[str, SignalMode],
    check_map: Mapping[tuple[str, str], CheckKind],
    *,
    judgements: Sequence[JudgementRow] = (),
    traits: Sequence[Trait] = (),
    drift_events: Sequence[DriftEvent] = (),
    bank: ProbeBank | None = None,
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
            "case": defaultdict(list),
            "answer_kind": defaultdict(list),
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
        if score.scorer == Scorer.JUDGE_IN_SITU:
            d["case"][in_situ_case(row.answer)].append(score)
        elif score.scorer == Scorer.JUDGE_GOVERNANCE:
            d["answer_kind"][governance_kind(row.answer)].append(score)
        if row.probe_type == ProbeType.TRAIT_PRESENCE:
            text = getattr(row, f"option_{row.answer.lower()}")
            presence["yes" if text == "yes" else "no"].append(score)

    by_type = []
    for (probe_type, form, scorer), rows in sorted(typed.items()):
        correct = sum(s.correct for s in rows)
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
                "correct": correct,
                "accuracy": correct / len(rows),
                "chance": chance,
                "parse_errors": sum(s.detail == "parse_error" for s in rows),
            }
        )

    awaiting = awaiting_counts(
        probes,
        judgements,
        check_map,
        traits,
        drift_events,
        bank if bank is not None else load_catalogue().probes,
    )
    accuracies = {k: (_rate(v)["accuracy"] if v else None) for k, v in presence.items()}
    both = all(presence.values())
    return {
        "by_type": by_type,
        "awaiting_judge": awaiting,
        "slices": {
            scorer: {name: _slice(groups) for name, groups in d.items() if groups}
            for scorer, d in sorted(dims.items())
        },
        "presence": {
            "yes": {"n": len(presence["yes"]), "accuracy": accuracies["yes"]},
            "no": {"n": len(presence["no"]), "accuracy": accuracies["no"]},
            "balanced_accuracy": (accuracies["yes"] + accuracies["no"]) / 2 if both else None,
        },
    }


def score_run(config: Config, store: DataStore, run_name: str) -> dict[str, Any]:
    """Score a finished run's responses, write scores and summary.json, and return the summary."""
    rd = run_dir(store.data_dir, run_name)
    run_store = DataStore(rd, config.output)
    check_scorable(store, run_store, run_name)

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

    judgements = run_store.read(JUDGEMENTS) if run_store.exists(JUDGEMENTS) else []
    scores += [s for s in map(judgement_score, judgements) if s is not None]

    traits = store.read(TRAITS)
    kinds = {(t.pm_id, t.trait_id): t.kind for t in traits}
    modes = {s.signal_id: s.mode for s in store.read(SIGNALS)}
    summary = summarise(
        probes,
        scores,
        kinds,
        modes,
        check_map,
        judgements=judgements,
        traits=traits,
        drift_events=store.read(DRIFT_EVENTS),
        bank=load_catalogue().probes,
    )
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
