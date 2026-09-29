"""Tests for the probes stage: table I/O, run metadata and the per-PM checkpoint loop, wired
through a real (but scripted) validated dialogue corpus.
"""

import hashlib
import json
from pathlib import Path

import pytest

from pm_traitbench.cli import build_parser
from pm_traitbench.enums import Kind, ProbeType
from pm_traitbench.errors import ProbesError
from pm_traitbench.pipeline import STAGES
from pm_traitbench.probes.stage import PROBES_STAGE
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import PROBES, SESSIONS, SIGNALS, TRAITS
from tests.dialogue.fixtures import run_engine_and_plan
from tests.dialogue.validate.test_stage import _clean_validate_stage, faithful_dialogue_stage


def _run_validated_corpus(tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch):
    """Engine, plan, a faithful dialogue stage and a clean validate stage, chained."""
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    run_stage(_clean_validate_stage(store), config, store)
    return config, store


def test_stage_writes_probes_for_every_narrated_pm(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)

    run_stage(PROBES_STAGE, config, store)

    rows = store.read(PROBES)
    sessions = store.read(SESSIONS)
    pms_with_sessions = sorted({s.pm_id for s in sessions})
    assert rows
    assert {r.pm_id for r in rows} <= set(pms_with_sessions)

    surviving = {s.session_id for s in sessions}
    surviving_signals = {
        sig.signal_id for sig in store.read(SIGNALS) if sig.session_id in surviving
    }
    for row in rows:
        assert set(row.supporting_signal_ids) <= surviving_signals

    traits = store.read(TRAITS)
    for pm_id in {r.pm_id for r in rows}:
        pm_rows = [r for r in rows if r.pm_id == pm_id]
        last = max(r.checkpoint_date for r in pm_rows)
        inactive = {
            t.trait_id for t in traits if t.pm_id == pm_id and t.kind == Kind.BIAS and not t.active
        }
        assert inactive
        presence = {
            r.trait_id: r.answer
            for r in pm_rows
            if r.checkpoint_date == last
            and r.probe_type == ProbeType.TRAIT_PRESENCE
            and r.trait_id in inactive
        }
        assert set(presence) == inactive
        assert set(presence.values()) == {"B"}

    metadata = store.read_run_metadata("probes")
    assert metadata is not None
    assert metadata["mcq_horizon"] == {"loss_aversion_lambda": 34, "disposition_ratio": 14}
    assert metadata["pms"] == pms_with_sessions
    assert (
        metadata["sessions_sha256"] == hashlib.sha256(store.path(SESSIONS).read_bytes()).hexdigest()
    )
    assert set(metadata) >= {
        "pms_without_sessions",
        "probes_by_type",
        "skipped_checkpoints",
        "skipped_probes",
    }


def test_forced_rerun_is_byte_identical(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    run_stage(PROBES_STAGE, config, store)
    before = store.path(PROBES).read_bytes()

    run_stage(PROBES_STAGE, config, store, force=True)

    assert store.path(PROBES).read_bytes() == before


def test_stale_validate_metadata_raises_probes_error_and_writes_nothing(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    meta_path = tmp_path / "run_metadata" / "validate.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = "2000-01-01T00:00:00+00:00"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ProbesError, match="rerun validate"):
        run_stage(PROBES_STAGE, config, store)

    assert not store.exists(PROBES)


def test_pm_missing_a_bias_trait_raises_probes_error(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _run_validated_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    pm_id = min(s.pm_id for s in store.read(SESSIONS))
    traits = [
        t
        for t in store.read(TRAITS)
        if not (t.pm_id == pm_id and t.kind == Kind.BIAS and t.param == "herding_weight")
    ]
    store.write(TRAITS, traits)

    with pytest.raises(
        ProbesError, match=rf"pm {pm_id}: missing bias trait\(s\) \['herding_weight'\]"
    ):
        run_stage(PROBES_STAGE, config, store)


def test_stage_is_registered() -> None:
    assert PROBES_STAGE in STAGES
    assert (PROBES_STAGE.number, PROBES_STAGE.name) == (9, "probes")
    assert build_parser(STAGES).parse_args(["probes"]).command == "probes"
