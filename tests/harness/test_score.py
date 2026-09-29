"""Tests for option-letter and format scoring, the summary and the score_run guards."""

import json
from datetime import date

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.enums import Kind, OptionSource, ProbeForm, ProbeType, Scorer, SignalMode
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.checks import load_check_map
from pm_traitbench.harness.runner import RUN_METADATA, probes_sha256, run_dir, run_sut
from pm_traitbench.harness.score import (
    SCORE_METADATA,
    evidence_type,
    parse_letter,
    score_format,
    score_option_letter,
    score_run,
    summarise,
)
from pm_traitbench.harness.views import opaque_probe_id
from pm_traitbench.tables.schema import ProbeRow, ScoreRow, probe_id
from pm_traitbench.tables.specs import PROBES, RESPONSES, SCORES
from pm_traitbench.tables.store import DataStore
from tests.engine.fixtures import stage_config
from tests.harness.fixtures import (
    probe_row,
    recording_factory,
    validated_corpus_with_probes,
)

DAY = date(2026, 1, 13)
BULLETS = "- one\n- two"
PROSE = "Rates look steady and the book is unchanged for now."


@pytest.mark.parametrize("text", ["A", "A.", "(B)", " C) because", "D: yes", "B\n"])
def test_parse_letter_accepts(text: str) -> None:
    assert parse_letter(text, 4) == text.strip().strip("()")[0]


@pytest.mark.parametrize(
    "text,n", [("Absolutely", 4), ("a", 4), ("The answer is A", 4), ("C", 2), ("", 4)]
)
def test_parse_letter_rejects(text: str, n: int) -> None:
    assert parse_letter(text, n) is None


def test_parse_letter_first_token_decides() -> None:
    assert parse_letter("A or B", 4) == "A"


def test_option_letter_score_and_parse_error() -> None:
    row = probe_row("pm_001", 1, DAY, options=("x", "y", "z"))

    right = score_option_letter(row, "A) because")
    wrong = score_option_letter(row, "B")
    bad = score_option_letter(row, "no idea")

    assert (right.correct, right.detail, right.scorer) == (True, None, Scorer.OPTION_LETTER)
    assert (wrong.correct, wrong.detail) == (False, None)
    assert (bad.correct, bad.detail) == (False, "parse_error")


def _routine(answer: str):
    return probe_row(
        "pm_001",
        1,
        DAY,
        form=ProbeForm.OPEN,
        probe_type=ProbeType.ROUTINE_QUESTION,
        options=(),
        answer=answer,
        trait_id=None,
    )


@pytest.fixture(scope="module")
def check_map():
    return load_check_map(load_catalogue())


def test_format_score_all_pass_fail_and_na(check_map) -> None:
    row = _routine(
        "format: response_format=short bullets; "
        "number_language=quote moves in basis points; intrusion: none"
    )

    ok = score_format(row, BULLETS, check_map, 400)
    bad = score_format(row, PROSE, check_map, 400)

    assert ok is not None and (ok.correct, ok.detail, ok.scorer) == (True, None, Scorer.FORMAT)
    assert bad is not None and (bad.correct, bad.detail) == (False, "response_format=short bullets")


def test_format_score_none_when_only_judge_values(check_map) -> None:
    row = _routine(
        "format: register=terse; "
        "hedging_language=flag uncertainty once, then commit to a view; intrusion: none"
    )
    assert score_format(row, BULLETS, check_map, 400) is None


def test_evidence_type() -> None:
    modes = {
        "a": SignalMode.STATED,
        "b": SignalMode.REVEALED,
        "c": SignalMode.CONTRADICTION,
    }
    assert evidence_type([], modes) == "none"
    assert evidence_type(["a"], modes) == "explicit"
    assert evidence_type(["b", "c"], modes) == "implicit"
    assert evidence_type(["a", "b"], modes) == "mixed"


def _score(row, correct: bool, scorer=Scorer.OPTION_LETTER) -> ScoreRow:
    return ScoreRow(
        probe_id=row.probe_id, pm_id=row.pm_id, scorer=scorer, correct=correct, detail=None
    )


def _variant(row: ProbeRow, **update) -> ProbeRow:
    """A validated copy of `row` with fields replaced."""
    return ProbeRow.model_validate({**row.model_dump(), **update})


def _presence(n: int, answer: str, trait_id: str | None = "t_01") -> ProbeRow:
    """A presence probe; source_a is current exactly when the answer is A."""
    base = probe_row("pm_001", n, DAY, options=("yes", "no", "z"))
    return _variant(
        base,
        probe_type=ProbeType.TRAIT_PRESENCE,
        trait_id=trait_id,
        option_c=None,
        source_b=None,
        source_c=None,
        answer=answer,
        source_a=OptionSource.CURRENT if answer == "A" else OptionSource.PRE_UPDATE,
    )


def test_summary_chance_and_presence_balance() -> None:
    presence = [_presence(1, "B"), _presence(2, "B"), _presence(3, "B"), _presence(4, "A")]
    mcq4 = probe_row("pm_001", 5, DAY, options=("a", "b", "c", "d"))
    mcq3 = probe_row("pm_001", 6, DAY, options=("a", "b", "c"))
    probes = [*presence, mcq4, mcq3]
    scores = [_score(p, c) for p, c in zip(presence, [True, True, True, False], strict=True)]
    scores += [_score(mcq4, True), _score(mcq3, False)]
    kinds = {("pm_001", "t_01"): Kind.BIAS}

    summary = summarise(probes, scores, kinds, {})

    assert summary["presence"] == {
        "yes": {"n": 1, "accuracy": 0.0},
        "no": {"n": 3, "accuracy": 1.0},
        "balanced_accuracy": 0.5,
    }
    by_type = {(e["probe_type"], e["form"]): e for e in summary["by_type"]}
    assert by_type[("trait_presence", "mcq")]["chance"] == 0.5
    mcq = by_type[("trait_mcq", "mcq")]
    assert mcq["chance"] == pytest.approx((0.25 + 1 / 3) / 2)
    assert (mcq["n"], mcq["correct"], mcq["accuracy"]) == (2, 1, 0.5)
    assert summary["slices"]["option_letter"]["kind"]["bias"] == {"n": 6, "accuracy": 4 / 6}


def test_summary_slices_and_parse_errors() -> None:
    a = probe_row("pm_001", 1, DAY, trait_id="t_01")
    b = _presence(2, "B", trait_id=None)
    sig = _variant(a, supporting_signal_ids=("sg_001",))
    scores = [
        _score(sig, True),
        ScoreRow(
            probe_id=b.probe_id,
            pm_id="pm_001",
            scorer=Scorer.OPTION_LETTER,
            correct=False,
            detail="parse_error",
        ),
    ]

    summary = summarise(
        [sig, b], scores, {("pm_001", "t_01"): Kind.PREFERENCE}, {"sg_001": SignalMode.STATED}
    )

    sl = summary["slices"]["option_letter"]
    assert sl["kind"] == {
        "preference": {"n": 1, "accuracy": 1.0},
        "none": {"n": 1, "accuracy": 0.0},
    }
    assert sl["evidence"] == {
        "explicit": {"n": 1, "accuracy": 1.0},
        "none": {"n": 1, "accuracy": 0.0},
    }
    assert sl["checkpoint_label"] == {"week4": {"n": 2, "accuracy": 0.5}}
    assert sum(e["parse_errors"] for e in summary["by_type"]) == 1


def test_evidence_type_unknown_signal_raises() -> None:
    with pytest.raises(HarnessError, match="sg_009"):
        evidence_type(["sg_009"], {})


def test_format_score_unmapped_value_raises(check_map) -> None:
    row = _routine("format: response_format=telepathy; intrusion: none")
    with pytest.raises(HarnessError, match="telepathy"):
        score_format(row, BULLETS, check_map, 400)


def test_summary_unknown_trait_raises() -> None:
    a = probe_row("pm_001", 1, DAY, trait_id="t_09")
    with pytest.raises(HarnessError, match="t_09"):
        summarise([a], [_score(a, True)], {}, {})


def test_awaiting_judge_counts() -> None:
    def open_probe(n, kind):
        return probe_row(
            "pm_001",
            n,
            DAY,
            form=ProbeForm.OPEN,
            probe_type=kind,
            options=(),
            answer="x",
            trait_id=None if kind == ProbeType.ROUTINE_QUESTION else "t_01",
        )

    probes = [
        open_probe(1, ProbeType.TRAIT_MCQ),
        open_probe(2, ProbeType.IN_SITU),
        open_probe(3, ProbeType.GOVERNANCE),
        open_probe(4, ProbeType.ROUTINE_QUESTION),
        open_probe(5, ProbeType.ROUTINE_QUESTION),
    ]
    scores = [_score(probes[3], True, Scorer.FORMAT)]

    summary = summarise(probes, scores, {}, {})

    assert summary["awaiting_judge"] == {
        "trait_mcq/open": 1,
        "in_situ/open": 1,
        "governance/open": 1,
        "routine_question/intrusion": 2,
        "routine_question/format_judge_only": 1,
    }


def _run_store(tmp_path, config, **meta):
    run_store = DataStore(run_dir(tmp_path, "r1"), config.output)
    body = {"status": "finished", "probes_sha256": None, "pms_failed": {}, "run_name": "r1"}
    body.update(meta)
    run_store.write_run_metadata(RUN_METADATA, config, body)
    return run_store


def _corpus(tmp_path):
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    store.write(PROBES, [probe_row("pm_001", 1, DAY)])
    return config, store


def test_score_run_refuses_missing_run(tmp_path) -> None:
    config, store = _corpus(tmp_path)
    with pytest.raises(HarnessError, match="r1"):
        score_run(config, store, "r1")


def test_score_run_refuses_failed_run(tmp_path) -> None:
    config, store = _corpus(tmp_path)
    _run_store(tmp_path, config, probes_sha256=probes_sha256(store), pms_failed={"pm_001": "boom"})
    with pytest.raises(HarnessError, match="pm_001"):
        score_run(config, store, "r1")


def test_score_run_refuses_unfinished_run(tmp_path) -> None:
    config, store = _corpus(tmp_path)
    _run_store(tmp_path, config, probes_sha256=probes_sha256(store), status="running")
    with pytest.raises(HarnessError, match="finished"):
        score_run(config, store, "r1")


def test_score_run_refuses_changed_probes(tmp_path) -> None:
    config, store = _corpus(tmp_path)
    _run_store(tmp_path, config, probes_sha256="0" * 64)
    with pytest.raises(HarnessError, match="probes"):
        score_run(config, store, "r1")


def test_score_run_refuses_missing_response(tmp_path) -> None:
    config, store = _corpus(tmp_path)
    run_store = _run_store(tmp_path, config, probes_sha256=probes_sha256(store))
    run_store.write(RESPONSES, [])
    with pytest.raises(HarnessError, match=probe_id("pm_001", 1)):
        score_run(config, store, "r1")


def test_score_run_end_to_end(tmp_path, fixture_market, neutral_pm, monkeypatch) -> None:
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    root = config.seed.root
    answers = {opaque_probe_id(root, p.probe_id): p.answer for p in store.read(PROBES)}

    def answer(as_of, probe):
        if probe.form == ProbeForm.MCQ:
            return answers[probe.probe_id]
        return BULLETS

    factory, _ = recording_factory(answer_fn=answer)
    run_sut(config, store, factory, sut_name="rec", run_name="r1")

    summary = score_run(config, store, "r1")

    rd = run_dir(store.data_dir, "r1")
    run_store = DataStore(rd, config.output)
    assert run_store.exists(SCORES)
    assert json.loads((rd / "summary.json").read_text("utf-8")) == summary
    letter = [e for e in summary["by_type"] if e["scorer"] == "option_letter"]
    assert letter and all(e["accuracy"] == 1.0 for e in letter)
    metadata = run_store.read_run_metadata(SCORE_METADATA)
    assert metadata["n_scored"] == len(run_store.read(SCORES))
