"""Tests for the gate 2 stage: table I/O, run metadata and the per-PM recovery loop, wired
through a real (but scripted) validated dialogue corpus.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import Gate2Slice, Gate2Verdict
from pm_traitbench.errors import DialogueBudgetError, Gate2Error
from pm_traitbench.gates.gate2.aggregate import blocking_id
from pm_traitbench.gates.gate2.stage import make_stage
from pm_traitbench.signals.assemble import session_id as build_session_id
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import (
    GATE2_CELLS,
    GATE2_PM,
    GATE2_SIGNALS,
    GATE2_TABLES,
    GATE2_TRAITS,
    PERSONAS,
    SESSIONS,
    SIGNALS,
)
from pm_traitbench.tables.store import DataStore
from tests.dialogue.fixtures import (
    FakeClient,
    fake_message,
    raising_factory,
    run_engine_and_plan,
    with_section,
)
from tests.dialogue.validate.test_stage import _clean_validate_stage, faithful_dialogue_stage
from tests.engine.fixtures import stage_config
from tests.gates.gate2.fixtures import (
    PLANTED_PMS,
    is_classify_request,
    is_recovery_request,
    truthful_responder,
    write_planted_corpus,
    wrong_responder,
)


def _run_validated_corpus(tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch):
    """Engine, plan, a faithful dialogue stage and a clean validate stage, chained."""
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    run_stage(_clean_validate_stage(store), config, store)
    return config, store


def test_truthful_corpus_writes_four_tables_one_row_per_unit_and_passes_or_is_insufficient(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)

    catalogue = load_catalogue()
    personas = {p.pm_id: p for p in store.read(PERSONAS)}
    sessions = store.read(SESSIONS)
    pms_with_sessions = sorted({s.pm_id for s in sessions})
    expected_trait_rows = sum(
        len(BIAS_PARAMS) + len(catalogue.preferences_for(personas[pm].mandate.asset_class))
        for pm in pms_with_sessions
    )
    session_ids = {s.session_id for s in sessions}
    surviving_signal_ids = {
        sig.signal_id for sig in store.read(SIGNALS) if sig.session_id in session_ids
    }

    gate2_stage = make_stage(lambda c: FakeClient(truthful_responder(store)))

    run_stage(gate2_stage, config, store)

    trait_rows = store.read(GATE2_TRAITS)
    assert len(trait_rows) == expected_trait_rows

    signal_rows = store.read(GATE2_SIGNALS)
    assert {r.signal_id for r in signal_rows} == surviving_signal_ids

    pm_rows = store.read(GATE2_PM)
    assert {r.pm_id for r in pm_rows} == set(pms_with_sessions)

    cells = store.read(GATE2_CELLS)
    all_bias_rows = [c for c in cells if c.slice == Gate2Slice.ALL and c.param in BIAS_PARAMS]
    assert len(all_bias_rows) == len(BIAS_PARAMS)
    assert all(c.verdict in (Gate2Verdict.PASS, Gate2Verdict.INSUFFICIENT) for c in all_bias_rows)

    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert set(metadata) >= {
        "model",
        "pms",
        "pms_without_sessions",
        "sessions_sha256",
        "failed",
        "insufficient",
        "overlap",
        "warnings",
        "calls",
        "cache_hits",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "rejected_replies",
        "thresholds",
    }
    assert metadata["pms"] == pms_with_sessions
    expected_sha256 = hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    assert metadata["sessions_sha256"] == expected_sha256
    # This fixture corpus plants no active bias and holds no preference, so every
    # blocking row comes back insufficient, and insufficient never blocks.
    assert metadata["failed"] == []
    assert metadata["insufficient"] == sorted(blocking_id(c) for c in cells if c.blocking)


def test_wrong_responder_on_neutral_corpus_still_reports_insufficient_after_writing_tables(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This corpus plants no active bias and holds no preference, so every blocking row's
    class sizes are structurally below `min_class` no matter what the responder predicts:
    every row comes back insufficient, and insufficient never blocks.
    """
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    gate2_stage = make_stage(lambda c: FakeClient(wrong_responder(store)))

    run_stage(gate2_stage, config, store)

    for spec in GATE2_TABLES:
        assert store.exists(spec)

    cells = store.read(GATE2_CELLS)
    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert metadata["failed"] == []
    assert metadata["insufficient"] == sorted(blocking_id(c) for c in cells if c.blocking)


def test_forced_second_run_is_byte_identical_and_makes_no_inner_calls(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    gate2_stage = make_stage(lambda c: FakeClient(truthful_responder(store)))

    run_stage(gate2_stage, config, store)

    before = {spec.name: store.path(spec).read_bytes() for spec in GATE2_TABLES}

    # A forced rerun with a client that raises on any call must still succeed, reading
    # every request back from the cache built by the first run.
    run_stage(make_stage(raising_factory), config, store, force=True)

    for spec in GATE2_TABLES:
        assert store.path(spec).read_bytes() == before[spec.name]


def test_stale_validate_metadata_raises_before_any_call(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)

    meta_path = tmp_path / "run_metadata" / "validate.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = "2000-01-01T00:00:00+00:00"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(Gate2Error, match="rerun validate"):
        run_stage(make_stage(raising_factory), config, store)

    for spec in GATE2_TABLES:
        assert not store.exists(spec)


def test_missing_validate_metadata_raises(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    (tmp_path / "run_metadata" / "validate.json").unlink()

    with pytest.raises(Gate2Error, match="validate run metadata is missing"):
        run_stage(make_stage(raising_factory), config, store)

    for spec in GATE2_TABLES:
        assert not store.exists(spec)


def test_budget_error_writes_nothing(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    config = with_section(config, "gate2", token_budget=1)
    gate2_stage = make_stage(lambda c: FakeClient(truthful_responder(store)))

    with pytest.raises(DialogueBudgetError, match="gate2"):
        run_stage(gate2_stage, config, store)

    for spec in GATE2_TABLES:
        assert not store.exists(spec)


def test_pm_with_all_sessions_dropped_is_listed_not_scored(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    sessions = store.read(SESSIONS)
    target_pm = sorted({s.pm_id for s in sessions})[0]
    store.write(SESSIONS, [s for s in sessions if s.pm_id != target_pm])

    gate2_stage = make_stage(lambda c: FakeClient(truthful_responder(store)))
    run_stage(gate2_stage, config, store)

    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert target_pm in metadata["pms_without_sessions"]
    assert target_pm not in metadata["pms"]

    assert all(r.pm_id != target_pm for r in store.read(GATE2_TRAITS))
    assert all(r.pm_id != target_pm for r in store.read(GATE2_PM))


def _write_planted(tmp_path: Path) -> tuple[Config, DataStore]:
    """A hand-seeded gate 2 corpus with a real active bias and a real held preference,
    written straight into a fresh store, bypassing every upstream stage.
    """
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_planted_corpus(store, config)
    return config, store


def _cell(cells, slice_, slice_value: str, param: str | None):
    return next(
        c for c in cells if c.slice == slice_ and c.slice_value == slice_value and c.param == param
    )


def test_planted_corpus_truthful_responder_passes_disposition_and_pooled_preferences(
    tmp_path: Path,
) -> None:
    config, store = _write_planted(tmp_path)
    gate2_stage = make_stage(lambda c: FakeClient(truthful_responder(store)))

    run_stage(gate2_stage, config, store)

    cells = store.read(GATE2_CELLS)
    disposition = _cell(cells, Gate2Slice.ALL, "all", "disposition_ratio")
    assert disposition.verdict == Gate2Verdict.PASS
    assert disposition.p == pytest.approx(1 / 35)

    pooled_preferences = _cell(cells, Gate2Slice.ALL, "all", None)
    assert pooled_preferences.verdict == Gate2Verdict.PASS

    pref_rows = [r for r in store.read(GATE2_TRAITS) if r.param == "response_format"]
    assert pref_rows and all(r.trait_id == "t_09" for r in pref_rows)

    other_bias_ids = sorted(f"all/{param}" for param in BIAS_PARAMS if param != "disposition_ratio")
    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert metadata["insufficient"] == other_bias_ids
    assert metadata["failed"] == []

    signal_rows = store.read(GATE2_SIGNALS)
    assert len(signal_rows) == len(PLANTED_PMS)
    for row in signal_rows:
        assert row.classified is True
        assert row.kind_ok is True
        assert row.cited is True
        assert row.recovered is True

    stated_mode = _cell(cells, Gate2Slice.MODE, "stated", None)
    assert stated_mode.n == len(PLANTED_PMS)
    assert stated_mode.n_positive == len(PLANTED_PMS)


def test_planted_corpus_wrong_responder_fails_disposition_and_preferences(
    tmp_path: Path,
) -> None:
    config, store = _write_planted(tmp_path)
    gate2_stage = make_stage(lambda c: FakeClient(wrong_responder(store)))

    with pytest.raises(Gate2Error, match="^gate 2 failed for:"):
        run_stage(gate2_stage, config, store)

    cells = store.read(GATE2_CELLS)
    disposition = _cell(cells, Gate2Slice.ALL, "all", "disposition_ratio")
    assert disposition.verdict == Gate2Verdict.FAIL

    pooled_preferences = _cell(cells, Gate2Slice.ALL, "all", None)
    assert pooled_preferences.verdict == Gate2Verdict.FAIL

    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert "all/disposition_ratio" in metadata["failed"]
    assert "all/preferences" in metadata["failed"]

    signal_rows = store.read(GATE2_SIGNALS)
    assert len(signal_rows) == len(PLANTED_PMS)
    assert all(row.recovered is False for row in signal_rows)


def test_budget_error_wins_over_a_failed_recovery_unit(tmp_path: Path) -> None:
    """A recovery unit that fails after retries must not hide a budget error tripped on a
    later classification unit: the budget error is what a resumed run needs to see.
    """
    config, store = _write_planted(tmp_path)
    truthful = truthful_responder(store)

    def responder(request: dict) -> dict:
        if is_recovery_request(request) and "s_pm001_" in request["messages"][0]["content"]:
            return fake_message([{"type": "text", "text": "not valid json"}])
        return truthful(request)

    # Ten recovery calls (four failed attempts on pm_001, one each on the other six PMs)
    # spend exactly 1500 fake tokens; the budget then trips on the very first
    # classification call, before a plain recovery failure ever gets raised.
    config = with_section(config, "gate2", token_budget=1500)
    gate2_stage = make_stage(lambda c: FakeClient(responder))

    with pytest.raises(DialogueBudgetError, match="gate2"):
        run_stage(gate2_stage, config, store)

    for spec in GATE2_TABLES:
        assert not store.exists(spec)


def test_stage_names_a_failed_unit_on_each_half_when_neither_hits_budget(tmp_path: Path) -> None:
    """A recovery unit and a classification unit that both stay unparsable after retries,
    with no budget error, must both be named in one `Gate2Error`, and write nothing.
    """
    config, store = _write_planted(tmp_path)
    truthful = truthful_responder(store)
    fail_session_id = build_session_id("pm_002", date(2026, 1, 5), 0)

    def responder(request: dict) -> dict:
        if is_recovery_request(request) and "s_pm001_" in request["messages"][0]["content"]:
            return fake_message([{"type": "text", "text": "not valid json"}])
        if is_classify_request(request) and fail_session_id in request["messages"][0]["content"]:
            return fake_message([{"type": "text", "text": "not valid json"}])
        return truthful(request)

    gate2_stage = make_stage(lambda c: FakeClient(responder))

    with pytest.raises(Gate2Error) as exc_info:
        run_stage(gate2_stage, config, store)

    message = str(exc_info.value)
    assert "pm pm_001" in message
    assert fail_session_id in message

    for spec in GATE2_TABLES:
        assert not store.exists(spec)
