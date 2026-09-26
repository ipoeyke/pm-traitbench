"""Tests for the dialogue stage: table I/O, run metadata and its response cache."""

from pathlib import Path

import pytest

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config, DialogueConfig, PmFilter
from pm_traitbench.dialogue.client import LlmClient
from pm_traitbench.dialogue.stage import make_stage
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.errors import DialogueBudgetError, DialogueError
from pm_traitbench.signals.stage import PLAN_STAGE
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import DIALOGUE_LOGS, SESSIONS, SKELETONS
from pm_traitbench.tables.store import DataStore
from tests.dialogue.conftest import FakeClient, default_responder, fake_message
from tests.engine.conftest import (  # noqa: F401
    MULTI_ASSET_PM_ID,
    fixture_market,
    neutral_pm,
    stage_config,
    write_stage_inputs,
)

_TARGET_PM_ID = "pm_003"


def _run_engine_and_plan(
    tmp_path: Path, fixture_market: dict, neutral_pm
) -> tuple[Config, DataStore]:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)

    run_stage(ENGINE_STAGE, config, store)
    run_stage(PLAN_STAGE, config, store)
    return config, store


def _run_full(
    tmp_path: Path, fixture_market: dict, neutral_pm, *, dialogue: DialogueConfig | None = None
) -> tuple[Config, DataStore]:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    if dialogue is not None:
        config = config.model_copy(update={"dialogue": dialogue})
    run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)
    return config, store


def _raising_factory(config: Config) -> LlmClient:
    raise AssertionError("the inner client must not be constructed on a fully cached rerun")


def test_dialogue_stage_writes_one_session_and_log_per_skeleton(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    _, store = _run_full(tmp_path, fixture_market, neutral_pm)

    skeleton_keys = {(s.pm_id, s.session_id) for s in store.read(SKELETONS)}
    session_keys = {(s.pm_id, s.session_id) for s in store.read(SESSIONS)}
    log_keys = {(log.pm_id, log.session_id) for log in store.read(DIALOGUE_LOGS)}

    assert skeleton_keys
    assert session_keys == skeleton_keys
    assert log_keys == skeleton_keys


def test_pm_filter_limits_the_sessions_written(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    dialogue = DialogueConfig(pm_filter=PmFilter(pm_ids=(_TARGET_PM_ID,)))
    _, store = _run_full(tmp_path, fixture_market, neutral_pm, dialogue=dialogue)

    sessions = store.read(SESSIONS)
    assert sessions
    assert {s.pm_id for s in sessions} == {_TARGET_PM_ID}


def test_failed_session_writes_no_tables_and_raises(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = config.model_copy(
        update={
            "dialogue": config.dialogue.model_copy(
                update={"pm_filter": PmFilter(pm_ids=(_TARGET_PM_ID,))}
            )
        }
    )
    target_skeletons = sorted(
        (s for s in store.read(SKELETONS) if s.pm_id == _TARGET_PM_ID), key=lambda s: s.session_id
    )
    assert len(target_skeletons) >= 2  # a session unaffected by the refusal must still exist
    target = target_skeletons[-1]
    target_marker = f"Today is {target.date.isoformat()}."

    def refusing_responder(request):
        system = request.get("system") or ""
        if target_marker in system:
            return fake_message([{"type": "text", "text": "no"}], stop_reason="refusal")
        return default_responder(request)

    with pytest.raises(DialogueError, match=target.session_id):
        run_stage(make_stage(lambda c: FakeClient(refusing_responder)), config, store)

    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_budget_error_writes_no_tables(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = config.model_copy(
        update={"dialogue": config.dialogue.model_copy(update={"token_budget": 1})}
    )

    with pytest.raises(DialogueBudgetError):
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_second_run_from_cache_is_byte_identical_with_zero_inner_calls(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = _run_full(tmp_path, fixture_market, neutral_pm)
    sessions_before = store.path(SESSIONS).read_bytes()
    logs_before = store.path(DIALOGUE_LOGS).read_bytes()

    run_stage(make_stage(_raising_factory), config, store, force=True)

    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_missing_plan_metadata_raises(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    (tmp_path / "run_metadata" / "plan.json").unlink()

    with pytest.raises(DialogueError, match="plan run metadata is missing"):
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)


def test_run_metadata_records_models_prompt_hash_voices_and_usage(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = _run_full(tmp_path, fixture_market, neutral_pm)

    metadata = store.read_run_metadata("dialogue")
    assert metadata is not None
    assert metadata["narrator_model"] == config.dialogue.narrator_model
    assert metadata["advisor_model"] == config.dialogue.advisor_model
    assert len(metadata["advisor_prompt_sha256"]) == 64
    assert metadata["skipped"] == [MULTI_ASSET_PM_ID]
    pm_ids = {s.pm_id for s in store.read(SKELETONS)}
    assert set(metadata["pms"]) == pm_ids
    assert set(metadata["voices"]) == pm_ids
    assert metadata["calls"] > 0
    assert 0 <= metadata["cache_hits"] <= metadata["calls"]
    assert metadata["input_tokens"] > 0
    assert metadata["output_tokens"] > 0
    assert sum(metadata["sessions"].values()) == len(store.read(SESSIONS))
    assert isinstance(metadata["warnings"], list)
