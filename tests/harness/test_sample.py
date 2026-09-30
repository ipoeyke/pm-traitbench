"""Tests for the blind human sample and judge-human agreement."""

import csv
from collections import Counter
from datetime import date

import pytest

from pm_traitbench.enums import InSituCase, Judge
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.judge import JudgeItem
from pm_traitbench.harness.runner import run_dir
from pm_traitbench.harness.sample import (
    SAMPLE_COLUMNS,
    SAMPLE_FILE,
    Rating,
    agreement,
    allocate,
    allocate_in_situ,
    cohen_kappa,
    draw_sample,
    read_ratings,
    write_sample,
)
from pm_traitbench.rng import stream
from pm_traitbench.tables.schema import JudgementRow
from pm_traitbench.tables.specs import DRIFT_EVENTS, JUDGEMENTS, TRAITS
from pm_traitbench.tables.store import DataStore
from tests.engine.fixtures import stage_config
from tests.harness.judge_fixtures import (
    governance_row,
    in_situ_row,
    open_pair,
    traits_for,
    write_run,
)

DAY = date(2026, 2, 2)
ORDER = list(Judge)


def test_allocate_even_with_remainder() -> None:
    counts = dict.fromkeys(Judge, 100)

    result = allocate(counts, 12)

    assert [result[j] for j in Judge] == [3, 3, 2, 2, 2]


def test_allocate_caps_short_judge() -> None:
    counts = dict.fromkeys(Judge, 100)
    counts[Judge.GOVERNANCE] = 0

    result = allocate(counts, 12)

    assert result[Judge.GOVERNANCE] == 0
    assert sum(result.values()) == 12
    assert all(result[j] <= counts[j] for j in Judge)


def test_allocate_total_is_capped_by_available() -> None:
    counts = {j: 1 for j in Judge}

    assert sum(allocate(counts, 50).values()) == 5


def test_allocate_in_situ_weights() -> None:
    counts = dict.fromkeys(InSituCase, 100)

    ten = allocate_in_situ(counts, 10)
    seven = allocate_in_situ(counts, 7)

    assert (ten[InSituCase.COUNTERACT], ten[InSituCase.DECLINE], ten[InSituCase.COMPLY]) == (
        4,
        4,
        2,
    )
    assert sum(seven.values()) == 7


def test_allocate_in_situ_caps_and_redistributes() -> None:
    counts = {InSituCase.COUNTERACT: 1, InSituCase.DECLINE: 100, InSituCase.COMPLY: 100}

    result = allocate_in_situ(counts, 10)

    assert result[InSituCase.COUNTERACT] == 1
    assert sum(result.values()) == 10


def _jrow(probe, judge: Judge, correct: bool, detail: str = "d") -> JudgementRow:
    return JudgementRow(
        probe_id=probe.probe_id,
        pm_id=probe.pm_id,
        judge=judge,
        correct=correct,
        detail=detail,
        rationale="secret rationale",
    )


def _item(probe, judge: Judge, case: str = "") -> JudgeItem:
    return JudgeItem(probe, judge, f"reply {probe.probe_id}", f"brief {case}", case, (), None, ())


def _sample_inputs():
    items, rows = [], []
    for n in range(1, 13):
        probe = in_situ_row(n, ("comply", "counteract", "decline")[n % 3])
        case = ("comply", "counteract", "decline")[n % 3]
        items.append(_item(probe, Judge.IN_SITU, case))
        rows.append(_jrow(probe, Judge.IN_SITU, n % 2 == 0))
    for n in range(20, 26):
        probe = governance_row(n, "update")
        items.append(_item(probe, Judge.GOVERNANCE, "update"))
        rows.append(_jrow(probe, Judge.GOVERNANCE, True, "empty_reply" if n == 20 else "d"))
    return items, rows


def test_draw_sample_blind_and_deterministic() -> None:
    items, rows = _sample_inputs()
    empty_id = "p_pm001_0020"

    first = draw_sample(items, rows, 8, stream(1, "s"))
    second = draw_sample(items, rows, 8, stream(1, "s"))

    assert first == second
    assert len(first) == 8
    assert all(tuple(r) == SAMPLE_COLUMNS for r in first)
    assert not {"correct", "rationale", "detail"} & set(first[0])
    assert [r["sample_id"] for r in first] == [f"s_{i:04d}" for i in range(1, 9)]
    assert all(r["human_correct"] == "" and r["human_note"] == "" for r in first)
    assert all(r["probe_id"] != empty_id for r in first)
    assert {r["judge"] for r in first} == {"judge_in_situ", "judge_governance"}
    # 8 over 5 judges is 2,2,2,1,1; open, intrusion and format are empty, so their 4 go to
    # in_situ and governance (4 each). In-situ 4 splits 0.4/0.4/0.2: round(1.6)=2 counteract,
    # 2 decline, and comply takes the remainder 0.
    situ = [r["case"] for r in first if r["judge"] == Judge.IN_SITU.value]
    assert Counter(situ) == {"counteract": 2, "decline": 2}
    assert all(r["brief"].startswith("brief") and r["question"] for r in first)


def test_write_sample_requires_judgements(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    probes = list(open_pair("pm_001", 1, DAY, answer="A"))
    write_run(tmp_path, config, store, probes, {p.probe_id: "B" for p in probes})

    with pytest.raises(HarnessError, match="eval judge"):
        write_sample(config, store, "r1")


def test_write_sample_writes_blind_csv(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    mcq, twin = open_pair("pm_001", 1, DAY, answer="A")
    situ = in_situ_row(3, "comply")
    gov = governance_row(4, "update")
    probes = [mcq, twin, situ, gov]
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    run_store = write_run(tmp_path, config, store, probes, {p.probe_id: "reply" for p in probes})
    run_store.write(
        JUDGEMENTS,
        [
            _jrow(twin, Judge.OPEN, True, "choice=B"),
            _jrow(situ, Judge.IN_SITU, True, "honours=true"),
            _jrow(gov, Judge.GOVERNANCE, False, "rejects_premise=false; corrects=false"),
        ],
    )

    path = write_sample(config, store, "r1", size=3)

    assert path == run_dir(tmp_path, "r1") / SAMPLE_FILE
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert tuple(rows[0]) == SAMPLE_COLUMNS
    assert {r["judge"] for r in rows} == {"judge_open", "judge_in_situ", "judge_governance"}
    assert "secret rationale" not in path.read_text(encoding="utf-8")


def test_write_sample_refuses_overwrite_without_force(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    mcq, twin = open_pair("pm_001", 1, DAY, answer="A")
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    run_store = write_run(
        tmp_path, config, store, [mcq, twin], {mcq.probe_id: "r", twin.probe_id: "r"}
    )
    run_store.write(JUDGEMENTS, [_jrow(twin, Judge.OPEN, True, "choice=B")])
    path = write_sample(config, store, "r1", size=1)
    path.write_text("rated", encoding="utf-8")

    with pytest.raises(HarnessError, match="--force"):
        write_sample(config, store, "r1", size=1)
    assert path.read_text(encoding="utf-8") == "rated"

    write_sample(config, store, "r1", size=1, force=True)
    assert path.read_text(encoding="utf-8") != "rated"


def test_read_ratings_accepts_utf8_bom(tmp_path) -> None:
    path = tmp_path / "s.csv"
    _write_csv(path, [_rated("s_0001", "yes")])
    path.write_text("\ufeff" + path.read_text(encoding="utf-8"), encoding="utf-8")

    assert [r.sample_id for r in read_ratings(path)] == ["s_0001"]


def _write_csv(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: "" for c in SAMPLE_COLUMNS} | row)


def _rated(sample_id: str, value: str, judge: str = "judge_open") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "judge": judge,
        "probe_id": "p_pm001_0001",
        "pm_id": "pm_001",
        "human_correct": value,
    }


def test_read_ratings_values_and_errors(tmp_path) -> None:
    path = tmp_path / "s.csv"
    _write_csv(path, [_rated("s_0001", "Yes"), _rated("s_0002", " no "), _rated("s_0003", "  ")])

    ratings = read_ratings(path)

    assert [(r.sample_id, r.human_correct) for r in ratings] == [
        ("s_0001", True),
        ("s_0002", False),
    ]
    assert ratings[0] == Rating("s_0001", "pm_001", "p_pm001_0001", Judge.OPEN, True)

    _write_csv(path, [_rated("s_0009", "maybe")])
    with pytest.raises(HarnessError, match="s_0009"):
        read_ratings(path)

    path.write_text("sample_id,judge\ns_0001,open\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="column"):
        read_ratings(path)

    _write_csv(path, [_rated("s_0001", "yes", judge="nonsense")])
    with pytest.raises(HarnessError, match="nonsense"):
        read_ratings(path)


def _pairs_fixture(pairs):
    ratings, judgements = [], []
    for i, (human, judged) in enumerate(pairs, 1):
        pid = f"p_pm001_{i:04d}"
        ratings.append(Rating(f"s_{i:04d}", "pm_001", pid, Judge.INTRUSION, human))
        judgements.append(
            JudgementRow(
                probe_id=pid,
                pm_id="pm_001",
                judge=Judge.INTRUSION,
                correct=judged,
                detail="d",
                rationale="",
            )
        )
    return ratings, judgements


def test_agreement_and_kappa() -> None:
    # 6 of 8 agree (po 0.75); humans and judge each say True half the time (pe 0.5).
    pairs = [(True, True)] * 3 + [(False, False)] * 3 + [(True, False), (False, True)]
    ratings, judgements = _pairs_fixture(pairs)

    result = agreement(ratings, judgements)

    assert result == {"judge_intrusion": {"n": 8, "agreement": 0.75, "kappa": 0.5}}
    assert cohen_kappa(pairs) == pytest.approx(0.5)


def test_kappa_none_when_one_label_only() -> None:
    ratings, judgements = _pairs_fixture([(True, True), (True, True), (True, True)])

    assert agreement(ratings, judgements)["judge_intrusion"]["kappa"] is None


def test_agreement_unmatched_rating_raises() -> None:
    ratings, judgements = _pairs_fixture([(True, True)])

    with pytest.raises(HarnessError, match="s_0001"):
        agreement(ratings, [])
