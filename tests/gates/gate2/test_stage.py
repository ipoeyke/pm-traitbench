"""Tests for the gate 2 stage: table I/O, run metadata and the per-PM recovery loop, wired
through a real (but scripted) validated dialogue corpus.
"""

import hashlib
import json
from pathlib import Path

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import Gate2Slice, Gate2Verdict
from pm_traitbench.errors import DialogueBudgetError, Gate2Error
from pm_traitbench.gates.gate2.stage import make_stage
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
from tests.dialogue.fixtures import FakeClient, raising_factory, run_engine_and_plan, with_section
from tests.dialogue.validate.test_stage import _clean_validate_stage, faithful_dialogue_stage
from tests.gates.gate2.fixtures import truthful_responder, wrong_responder


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

    with pytest.raises(Gate2Error, match="^gate 2 failed for:"):
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


def test_wrong_responder_fails_blocking_rows_after_writing_tables(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    gate2_stage = make_stage(lambda c: FakeClient(wrong_responder(store)))

    with pytest.raises(Gate2Error, match="^gate 2 failed for:") as exc_info:
        run_stage(gate2_stage, config, store)

    for spec in GATE2_TABLES:
        assert store.exists(spec)

    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert metadata["failed"]
    # Insufficient rows block too, so every insufficient id is also a failed id.
    assert set(metadata["insufficient"]) <= set(metadata["failed"])
    for failed_id in metadata["failed"]:
        assert failed_id in str(exc_info.value)


def test_forced_second_run_is_byte_identical_and_makes_no_inner_calls(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    gate2_stage = make_stage(lambda c: FakeClient(truthful_responder(store)))

    with pytest.raises(Gate2Error):
        run_stage(gate2_stage, config, store)

    before = {spec.name: store.path(spec).read_bytes() for spec in GATE2_TABLES}

    with pytest.raises(Gate2Error):
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
    with pytest.raises(Gate2Error):
        run_stage(gate2_stage, config, store)

    metadata = store.read_run_metadata("gate2")
    assert metadata is not None
    assert target_pm in metadata["pms_without_sessions"]
    assert target_pm not in metadata["pms"]

    assert all(r.pm_id != target_pm for r in store.read(GATE2_TRAITS))
    assert all(r.pm_id != target_pm for r in store.read(GATE2_PM))
