"""Tests for the validated-corpus check shared by stages that read the narrated corpus."""

import json
from pathlib import Path

import pytest

from pm_traitbench.config import Config
from pm_traitbench.corpus_checks import check_validated_corpus
from pm_traitbench.errors import CorpusError
from pm_traitbench.tables.store import DataStore

_OLD = "2000-01-01T00:00:00+00:00"
_NEW = "2001-01-01T00:00:00+00:00"


def _store(tmp_path: Path, *, validate: dict | None, dialogue: dict | None) -> DataStore:
    config = Config()
    store = DataStore(tmp_path, config.output)
    for name, meta in (("validate", validate), ("dialogue", dialogue)):
        if meta is not None:
            path = tmp_path / "run_metadata" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meta), encoding="utf-8")
    return store


def test_current_validate_metadata_is_returned(tmp_path: Path) -> None:
    validate = {"created_at": _NEW, "pms": ["pm_01", "pm_02"]}
    store = _store(tmp_path, validate=validate, dialogue={"created_at": _OLD})

    assert check_validated_corpus(store, ["pm_01"]) == validate


def test_stale_validate_metadata_raises(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        validate={"created_at": _OLD, "pms": ["pm_01"]},
        dialogue={"created_at": _NEW},
    )

    with pytest.raises(CorpusError, match="rerun validate"):
        check_validated_corpus(store, ["pm_01"])


def test_missing_validate_metadata_raises(tmp_path: Path) -> None:
    store = _store(tmp_path, validate=None, dialogue={"created_at": _OLD})

    with pytest.raises(CorpusError, match="validate run metadata is missing"):
        check_validated_corpus(store, ["pm_01"])


def test_missing_dialogue_metadata_raises(tmp_path: Path) -> None:
    store = _store(tmp_path, validate={"created_at": _NEW, "pms": ["pm_01"]}, dialogue=None)

    with pytest.raises(CorpusError, match="dialogue run metadata is missing"):
        check_validated_corpus(store, ["pm_01"])


def test_pm_with_sessions_absent_from_validate_pms_raises(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        validate={"created_at": _NEW, "pms": ["pm_01"]},
        dialogue={"created_at": _OLD},
    )

    with pytest.raises(CorpusError, match="pm_02"):
        check_validated_corpus(store, ["pm_01", "pm_02"])
