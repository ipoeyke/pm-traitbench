"""Blind human sample of judged replies, and judge-human agreement."""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import InSituCase, Judge
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.judge import JudgeItem, load_judge_inputs, select_items
from pm_traitbench.harness.runner import check_run_name, check_scorable, run_dir
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import JudgementRow
from pm_traitbench.tables.specs import JUDGEMENTS
from pm_traitbench.tables.store import DataStore

SAMPLE_FILE = "human_sample.csv"
SAMPLE_COLUMNS = (
    "sample_id",
    "judge",
    "probe_id",
    "pm_id",
    "case",
    "question",
    "response",
    "brief",
    "human_correct",
    "human_note",
)
IN_SITU_WEIGHTS = {InSituCase.COUNTERACT: 0.4, InSituCase.DECLINE: 0.4, InSituCase.COMPLY: 0.2}
_EMPTY_REPLY = "empty_reply"
_YES_NO = {"yes": True, "no": False}


def _cap_and_redistribute[K](shares: dict[K, int], counts: Mapping[K, int]) -> dict[K, int]:
    """Cap each share at its count and hand the shortfall on, in order, one at a time."""
    result = {key: min(share, counts[key]) for key, share in shares.items()}
    shortfall = sum(shares.values()) - sum(result.values())
    while shortfall > 0:
        moved = False
        for key in result:
            if shortfall > 0 and result[key] < counts[key]:
                result[key] += 1
                shortfall -= 1
                moved = True
        if not moved:
            break
    return result


def allocate(counts: Mapping[Judge, int], size: int) -> dict[Judge, int]:
    """Split `size` evenly over the judges, capped at each judge's item count."""
    base, extra = divmod(size, len(Judge))
    shares = {judge: base + (i < extra) for i, judge in enumerate(Judge)}
    return _cap_and_redistribute(shares, counts)


def allocate_in_situ(counts: Mapping[InSituCase, int], size: int) -> dict[InSituCase, int]:
    """Split `size` over in-situ cases by weight, capped at each case's item count."""
    cases = list(IN_SITU_WEIGHTS)
    shares = {case: round(size * IN_SITU_WEIGHTS[case]) for case in cases[:-1]}
    shares[cases[-1]] = size - sum(shares.values())
    return _cap_and_redistribute(shares, counts)


def _pick(items: Sequence[JudgeItem], share: int, rng: np.random.Generator) -> list[JudgeItem]:
    ordered = sorted(items, key=lambda i: (i.probe.pm_id, i.probe.probe_id))
    return [ordered[i] for i in rng.choice(len(ordered), share, replace=False)]


def draw_sample(
    items: Sequence[JudgeItem],
    judgements: Sequence[JudgementRow],
    size: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    """Draw up to `size` rows to rate; verdicts and rationales are withheld from raters.

    Items whose reply was empty are never drawn, as they need no rater.
    """
    empty = {(j.pm_id, j.probe_id, j.judge) for j in judgements if j.detail == _EMPTY_REPLY}
    eligible = [i for i in items if (i.probe.pm_id, i.probe.probe_id, i.judge) not in empty]
    by_judge: dict[Judge, list[JudgeItem]] = {judge: [] for judge in Judge}
    for item in eligible:
        by_judge[item.judge].append(item)
    shares = allocate({judge: len(v) for judge, v in by_judge.items()}, size)

    drawn: list[JudgeItem] = []
    for judge in Judge:
        pool = by_judge[judge]
        if judge != Judge.IN_SITU:
            drawn += _pick(pool, shares[judge], rng)
            continue
        by_case = {case: [i for i in pool if i.case == case.value] for case in IN_SITU_WEIGHTS}
        case_shares = allocate_in_situ({c: len(v) for c, v in by_case.items()}, shares[judge])
        for case, case_pool in by_case.items():
            drawn += _pick(case_pool, case_shares[case], rng)

    return [
        {
            "sample_id": f"s_{n:04d}",
            "judge": item.judge.value,
            "probe_id": item.probe.probe_id,
            "pm_id": item.probe.pm_id,
            "case": item.case,
            "question": item.probe.question,
            "response": item.response,
            "brief": item.brief,
            "human_correct": "",
            "human_note": "",
        }
        for n, item in enumerate(drawn, 1)
    ]


def write_sample(config: Config, store: DataStore, run_name: str, size: int | None = None) -> Path:
    """Write the run's blind human sample CSV and return its path."""
    check_run_name(run_name)
    rd = run_dir(store.data_dir, run_name)
    run_store = DataStore(rd, config.output)
    check_scorable(store, run_store, run_name)
    if not run_store.exists(JUDGEMENTS):
        raise HarnessError(f"run '{run_name}' has no judgements; run eval judge first")
    items = select_items(load_judge_inputs(store, run_store)).items
    rng = stream(config.seed.root, "eval_sample", run_name)
    rows = draw_sample(
        items,
        run_store.read(JUDGEMENTS),
        config.judge.sample_size if size is None else size,
        rng,
    )
    path = rd / SAMPLE_FILE
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@dataclass(frozen=True)
class Rating:
    """One human verdict on a sampled judgement."""

    sample_id: str
    pm_id: str
    probe_id: str
    judge: Judge
    human_correct: bool


def read_ratings(path: Path) -> list[Rating]:
    """The rated rows of a sample CSV; rows with a blank `human_correct` are skipped."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in SAMPLE_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise HarnessError(f"{path.name}: missing column {', '.join(missing)}")
        rows = list(reader)
    ratings = []
    for row in rows:
        value = (row["human_correct"] or "").strip().lower()
        if not value:
            continue
        if value not in _YES_NO:
            raise HarnessError(f"sample {row['sample_id']}: human_correct must be yes or no")
        try:
            judge = Judge(row["judge"])
        except ValueError as exc:
            raise HarnessError(
                f"sample {row['sample_id']}: unknown judge {row['judge']!r}"
            ) from exc
        ratings.append(
            Rating(row["sample_id"], row["pm_id"], row["probe_id"], judge, _YES_NO[value])
        )
    return ratings


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa of (human, judge) verdict pairs (Cohen, 1960).

    None when chance agreement is total, i.e. both raters used one and the same label only.
    """
    n = len(pairs)
    po = sum(h == j for h, j in pairs) / n
    p_h = sum(h for h, _ in pairs) / n
    p_j = sum(j for _, j in pairs) / n
    pe = p_h * p_j + (1 - p_h) * (1 - p_j)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def agreement(
    ratings: Sequence[Rating], judgements: Sequence[JudgementRow]
) -> dict[str, dict[str, Any]]:
    """Per judge, the rated count, the agreement rate and Cohen's kappa against the judge."""
    verdicts = {(j.pm_id, j.probe_id, j.judge): j.correct for j in judgements}
    pairs: dict[Judge, list[tuple[bool, bool]]] = {}
    for rating in ratings:
        key = (rating.pm_id, rating.probe_id, rating.judge)
        if key not in verdicts:
            raise HarnessError(f"sample {rating.sample_id}: no judgement for {key}")
        pairs.setdefault(rating.judge, []).append((rating.human_correct, verdicts[key]))
    return {
        judge.value: {
            "n": len(judge_pairs),
            "agreement": sum(h == j for h, j in judge_pairs) / len(judge_pairs),
            "kappa": cohen_kappa(judge_pairs),
        }
        for judge, judge_pairs in sorted(pairs.items(), key=lambda kv: kv[0].value)
    }
