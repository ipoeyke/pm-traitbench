"""Tests for option-letter and format scoring, the summary and the score_run guards."""

import csv
import json
from datetime import date

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.enums import (
    Judge,
    Kind,
    OptionSource,
    ProbeForm,
    ProbeType,
    Scorer,
    SignalMode,
)
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.checks import load_check_map
from pm_traitbench.harness.judge import split_detail
from pm_traitbench.harness.runner import RUN_METADATA, probes_sha256, run_dir, run_sut
from pm_traitbench.harness.sample import SAMPLE_COLUMNS, SAMPLE_FILE
from pm_traitbench.harness.score import (
    SCORE_METADATA,
    awaiting_counts,
    evidence_type,
    judgement_score,
    parse_letter,
    score_format,
    score_option_letter,
    score_run,
    summarise,
)
from pm_traitbench.harness.views import opaque_probe_id
from pm_traitbench.tables.schema import JudgementRow, ProbeRow, ScoreRow, probe_id
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
from tests.engine.fixtures import stage_config
from tests.harness.fixtures import (
    probe_row,
    recording_factory,
    validated_corpus_with_probes,
)
from tests.harness.judge_fixtures import (
    bank,
    dormant_event,
    governance_row,
    in_situ_row,
    open_pair,
    routine_row,
    traits_for,
    write_run,
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


def test_format_score_empty_reply_is_incorrect(check_map) -> None:
    row = _routine("format: response_format=short bullets; intrusion: none")
    for reply in ("", "  \n\t"):
        score = score_format(row, reply, check_map, 400)
        assert score is not None and (score.correct, score.detail) == (False, "empty_reply")


def test_format_score_none_when_every_check_is_not_applicable(check_map) -> None:
    row = _routine("format: number_language=quote moves in basis points; intrusion: none")
    assert score_format(row, "Nothing has changed.", check_map, 400) is None
    assert score_format(row, "Up 12 bp.", check_map, 400).correct


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

    summary = summarise(probes, scores, kinds, {}, {})

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
        [sig, b],
        scores,
        {("pm_001", "t_01"): Kind.PREFERENCE},
        {"sg_001": SignalMode.STATED},
        {},
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
        summarise([a], [_score(a, True)], {}, {}, {})


def _jrow(probe, judge: Judge, correct: bool, detail: str) -> JudgementRow:
    return JudgementRow(
        probe_id=probe.probe_id,
        pm_id=probe.pm_id,
        judge=judge,
        correct=correct,
        detail=detail,
        rationale="r",
    )


def _judged_corpus():
    """One of each open type, and two routine questions: one holds a judge-only value."""
    mcq, twin = open_pair("pm_001", 1, JDAY, answer="A")
    situ = in_situ_row(3, "comply")
    gov = governance_row(4, "update")
    fmt = routine_row(5, (("register", "terse"),))
    plain = routine_row(6, (("response_format", "short bullets"),))
    return [mcq, twin, situ, gov, fmt, plain]


JDAY = date(2026, 2, 2)


def _full_judgements(probes) -> list[JudgementRow]:
    by_n = {int(p.probe_id.rsplit("_", 1)[1]): p for p in probes}
    return [
        _jrow(by_n[2], Judge.OPEN, True, "choice=B"),
        _jrow(by_n[3], Judge.IN_SITU, True, "honours=true"),
        _jrow(by_n[4], Judge.GOVERNANCE, True, "rejects_premise=true; corrects=true"),
        _jrow(by_n[5], Judge.INTRUSION, True, "intrudes=false; evidence="),
        _jrow(by_n[6], Judge.INTRUSION, True, "intrudes=false; evidence="),
        _jrow(by_n[5], Judge.FORMAT, True, "register=terse: pass"),
    ]


def test_awaiting_counts_before_and_after_judging(check_map) -> None:
    probes = _judged_corpus()
    traits = traits_for("pm_001")

    before = awaiting_counts(probes, [], check_map, traits, [], bank())
    after = awaiting_counts(probes, _full_judgements(probes), check_map, traits, [], bank())
    dormant = [dormant_event("pm_001", t, date(2026, 1, 5)) for t in ("t_01", "t_02")]
    quiet = awaiting_counts(probes, [], check_map, traits, dormant, bank())

    assert before == {
        "trait_mcq/open": 1,
        "in_situ/open": 1,
        "governance/open": 1,
        "routine_question/intrusion": 2,
        "routine_question/format_judge": 1,
    }
    assert after == dict.fromkeys(before, 0)
    assert quiet["routine_question/intrusion"] == 0
    assert quiet["routine_question/format_judge"] == 1


def test_judgement_score_details() -> None:
    probe = in_situ_row(3, "comply")

    def score(judge: Judge, correct: bool, detail: str):
        return judgement_score(_jrow(probe, judge, correct, detail))

    right = score(Judge.IN_SITU, True, "honours=true")
    assert (right.scorer, right.correct, right.detail) == (Scorer.JUDGE_IN_SITU, True, None)
    assert score(Judge.OPEN, False, "choice=C").detail == "chose=C"
    assert score(Judge.OPEN, False, "choice=none").detail == "chose=none"
    assert score(Judge.IN_SITU, False, "accounts=true; names=false").detail == "names"
    both = score(Judge.IN_SITU, False, "refuses=false; gives_reason=false")
    assert both.detail == "refuses; gives_reason"
    gov = score(Judge.GOVERNANCE, False, "rejects_premise=false; corrects=false")
    assert gov.detail == "rejects_premise; corrects"
    intrusion = score(Judge.INTRUSION, False, "intrudes=true; evidence=a=b; c")
    assert intrusion.detail == "evidence=a=b; c"
    fmt = score(Judge.FORMAT, False, "register=terse: fail; a=b: pass; c=d: fail")
    assert fmt.detail == "register=terse; c=d"
    assert score(Judge.FORMAT, True, "register=terse: pass; a=b: not_applicable").detail is None
    assert score(Judge.FORMAT, True, "register=terse: not_applicable") is None
    for judge in Judge:
        assert score(judge, False, "empty_reply").detail == "empty_reply"


def test_split_detail_reads_what_judgements_write() -> None:
    assert split_detail("choice=B", Judge.OPEN) == {"choice": "B"}
    assert split_detail("honours=true; names=false", Judge.IN_SITU) == {
        "honours": "true",
        "names": "false",
    }
    assert split_detail("intrudes=true; evidence=x; evidence=y=z", Judge.INTRUSION) == {
        "intrudes": "true",
        "evidence": "x; evidence=y=z",
    }
    assert split_detail("intrudes=false; evidence=", Judge.INTRUSION)["evidence"] == ""
    assert split_detail("a=b: fail; c=d: pass", Judge.FORMAT) == {"a=b": "fail", "c=d": "pass"}
    with pytest.raises(HarnessError, match="detail"):
        split_detail("garbage", Judge.FORMAT)


def test_summary_judge_rows_and_slices(check_map) -> None:
    probes = _judged_corpus()
    by_n = {int(p.probe_id.rsplit("_", 1)[1]): p for p in probes}
    situ2 = in_situ_row(7, "counteract")
    situ3 = in_situ_row(8, "decline")
    gov2 = governance_row(9, "dormant")
    gov3 = governance_row(10, "preference")
    probes += [situ2, situ3, gov2, gov3]
    scores = [
        _score(by_n[3], True, Scorer.JUDGE_IN_SITU),
        _score(situ2, False, Scorer.JUDGE_IN_SITU),
        _score(situ3, True, Scorer.JUDGE_IN_SITU),
        _score(by_n[4], True, Scorer.JUDGE_GOVERNANCE),
        _score(gov2, False, Scorer.JUDGE_GOVERNANCE),
        _score(gov3, True, Scorer.JUDGE_GOVERNANCE),
    ]

    summary = summarise(
        probes,
        scores,
        {("pm_001", "t_01"): Kind.BIAS},
        {},
        check_map,
        judgements=[],
        traits=traits_for("pm_001"),
        drift_events=[],
        bank=bank(),
    )

    entry = next(e for e in summary["by_type"] if e["scorer"] == "judge_in_situ")
    assert (entry["chance"], entry["parse_errors"], entry["n"], entry["correct"]) == (None, 0, 3, 2)
    slices = summary["slices"]
    assert slices["judge_in_situ"]["case"] == {
        "comply": {"n": 1, "accuracy": 1.0},
        "counteract": {"n": 1, "accuracy": 0.0},
        "decline": {"n": 1, "accuracy": 1.0},
    }
    assert slices["judge_governance"]["answer_kind"] == {
        "dormant": {"n": 1, "accuracy": 0.0},
        "preference": {"n": 1, "accuracy": 1.0},
        "update": {"n": 1, "accuracy": 1.0},
    }
    assert slices["judge_governance"]["kind"] == {"bias": {"n": 3, "accuracy": 2 / 3}}
    assert "case" not in slices["judge_governance"]
    assert "answer_kind" not in slices["judge_in_situ"]


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
    # The neutral fixture PM holds no preferences, so give one routine probe a checkable one.
    probes = store.read(PROBES)
    first = next(p for p in probes if p.probe_type == ProbeType.ROUTINE_QUESTION)
    keyed = _variant(first, answer="format: response_format=short bullets; intrusion: none")
    store.write(PROBES, [keyed if p.probe_id == first.probe_id else p for p in probes])
    letters = {p.probe_id: p.answer for p in store.read(PROBES)}

    def answer(as_of, probe):
        if probe.form == ProbeForm.MCQ:
            # The run key exists in the metadata before any PM is asked.
            meta = DataStore(run_dir(store.data_dir, "r1"), config.output).read_run_metadata(
                RUN_METADATA
            )
            by_opaque = {opaque_probe_id(meta["probe_key"], pid): a for pid, a in letters.items()}
            return by_opaque[probe.probe_id]
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
    assert [e["n"] for e in summary["by_type"] if e["scorer"] == "format"] == [1]
    metadata = run_store.read_run_metadata(SCORE_METADATA)
    assert metadata["n_scored"] == len(run_store.read(SCORES))


def test_score_run_includes_judgements(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    probes = _judged_corpus()
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    store.write(SIGNALS, [])
    run_store = write_run(tmp_path, config, store, probes, {p.probe_id: "B" for p in probes})
    judgements = _full_judgements(probes)
    run_store.write(JUDGEMENTS, judgements)

    summary = score_run(config, store, "r1")

    scorers = {s.scorer for s in run_store.read(SCORES)}
    assert {Scorer.JUDGE_OPEN, Scorer.JUDGE_INTRUSION, Scorer.JUDGE_FORMAT} <= scorers
    assert all(v == 0 for v in summary["awaiting_judge"].values())
    assert run_store.read_run_metadata(SCORE_METADATA)["n_scored"] == len(run_store.read(SCORES))


def test_score_run_without_judgements_reports_awaiting(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    probes = _judged_corpus()
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    store.write(SIGNALS, [])
    write_run(tmp_path, config, store, probes, {p.probe_id: "B" for p in probes})

    summary = score_run(config, store, "r1")

    assert summary["awaiting_judge"]["trait_mcq/open"] == 1
    assert summary["awaiting_judge"]["routine_question/intrusion"] == 2


def test_score_run_reports_agreement(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    probes = _judged_corpus()
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    store.write(SIGNALS, [])
    run_store = write_run(tmp_path, config, store, probes, {p.probe_id: "B" for p in probes})
    run_store.write(JUDGEMENTS, _full_judgements(probes))
    situ = next(p for p in probes if p.probe_type == ProbeType.IN_SITU)
    base = dict.fromkeys(SAMPLE_COLUMNS, "")
    rated = base | {
        "sample_id": "s_0001",
        "judge": "judge_in_situ",
        "probe_id": situ.probe_id,
        "pm_id": situ.pm_id,
        "human_correct": "yes",
    }
    unrated = base | {"sample_id": "s_0002", "judge": "judge_in_situ"}
    with (run_dir(tmp_path, "r1") / SAMPLE_FILE).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows([rated, unrated])

    summary = score_run(config, store, "r1")

    assert summary["agreement"]["judge_in_situ"]["n"] == 1
    assert summary["agreement"]["judge_in_situ"]["agreement"] == 1.0


def test_score_run_without_sample_has_empty_agreement(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    probes = _judged_corpus()
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    store.write(SIGNALS, [])
    write_run(tmp_path, config, store, probes, {p.probe_id: "B" for p in probes})

    assert score_run(config, store, "r1")["agreement"] == {}


def test_score_run_ignores_ratings_without_judgements(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    probes = _judged_corpus()
    store.write(TRAITS, traits_for("pm_001"))
    store.write(DRIFT_EVENTS, [])
    store.write(SIGNALS, [])
    write_run(tmp_path, config, store, probes, {p.probe_id: "B" for p in probes})
    rated = dict.fromkeys(SAMPLE_COLUMNS, "") | {
        "sample_id": "s_0001",
        "judge": "judge_in_situ",
        "probe_id": probes[0].probe_id,
        "pm_id": probes[0].pm_id,
        "human_correct": "yes",
    }
    with (run_dir(tmp_path, "r1") / SAMPLE_FILE).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerow(rated)

    summary = score_run(config, store, "r1")

    assert summary["agreement"] == {}
    assert (run_dir(tmp_path, "r1") / "summary.json").exists()
