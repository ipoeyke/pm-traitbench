"""Tests for the dialogue stage: table I/O, run metadata and its response cache."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config, DialogueConfig, PmFilter
from pm_traitbench.dialogue.client import CachedClient, Reply, request_key
from pm_traitbench.dialogue.stage import make_stage, raise_on_failure
from pm_traitbench.errors import DialogueBudgetError, DialogueError
from pm_traitbench.stages import run_stage
from pm_traitbench.tables.specs import DIALOGUE_LOGS, SESSIONS, SKELETONS
from pm_traitbench.tables.store import DataStore
from tests.dialogue.fixtures import (
    SESSION_ID,
    FakeClient,
    default_responder,
    fake_message,
    patch_send,
    raising_factory,
    run_engine_and_plan,
    with_section,
)
from tests.engine.fixtures import MULTI_ASSET_PM_ID

_TARGET_PM_ID = "pm_003"
_MANY_SESSIONS_PM_ID = "pm_001"


class _ConcurrencyTrackingClient:
    """A fake `LlmClient` that yields control on every send, to exercise real interleaving.

    Without an `await` inside `send`, asyncio never has a chance to let a
    second coroutine's send overlap this one's, even if many are scheduled.
    """

    def __init__(self) -> None:
        self.max_in_flight = 0
        self._in_flight = 0

    async def send(self, request: dict) -> dict:
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        await asyncio.sleep(0)
        try:
            return default_responder(request)
        finally:
            self._in_flight -= 1


def _intercept_by_scope(monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, Any]) -> None:
    """Patch `CachedClient.send` so a call scoped to a session id in `outcomes` is replaced.

    An `Exception` value is raised as-is; any other value is returned as an
    uncached `Reply`. Every other scope goes through the real cache and inner
    client unchanged. This targets one exact session regardless of whether
    its rendered request body happens to collide with another session's.
    """

    async def hook(request: dict, scope: str, send: Callable[[], Awaitable[Reply]]) -> Reply:
        outcome = outcomes.get(scope)
        if outcome is None:
            return await send()
        if isinstance(outcome, BaseException):
            raise outcome
        return Reply(key=request_key(request, scope), response=outcome, cached=False)

    patch_send(monkeypatch, hook)


def _refusal(text: str = "no") -> dict:
    return fake_message([{"type": "text", "text": text}], stop_reason="refusal")


def _run_full(
    tmp_path: Path, fixture_market: dict, neutral_pm, *, dialogue: DialogueConfig | None = None
) -> tuple[Config, DataStore]:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    if dialogue is not None:
        config = config.model_copy(update={"dialogue": dialogue})
    run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)
    return config, store


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


def _sorted_skeleton_ids(store: DataStore, pm_id: str) -> list[str]:
    return sorted(s.session_id for s in store.read(SKELETONS) if s.pm_id == pm_id)


def test_failed_session_writes_no_tables_and_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=(_TARGET_PM_ID,)))
    session_ids = _sorted_skeleton_ids(store, _TARGET_PM_ID)
    assert len(session_ids) >= 2  # a session unaffected by the refusal must still exist
    target_id = session_ids[-1]
    _intercept_by_scope(monkeypatch, {target_id: _refusal()})

    with pytest.raises(DialogueError, match=f"session {target_id}: the reply was refused"):
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_several_failed_sessions_are_listed_in_session_order(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=(_MANY_SESSIONS_PM_ID,)))
    session_ids = _sorted_skeleton_ids(store, _MANY_SESSIONS_PM_ID)
    assert len(session_ids) >= 3  # at least one session must survive unrefused
    first_id, second_id = session_ids[0], session_ids[1]
    _intercept_by_scope(monkeypatch, {first_id: _refusal(), second_id: _refusal()})

    with pytest.raises(DialogueError) as excinfo:
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    # One reason, shared by both sessions, collapses onto a single line naming both ids
    # in session order.
    assert str(excinfo.value) == f"sessions {first_id}, {second_id}: the reply was refused"
    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_client_raised_error_is_prefixed_with_its_session_id(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error the inner client raises directly (missing credentials, a 400) carries no
    'session {id}: ' prefix of its own; the stage must add one so a rerun can be pointed
    at the right session.
    """
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=(_TARGET_PM_ID,)))
    session_ids = _sorted_skeleton_ids(store, _TARGET_PM_ID)
    target_id = session_ids[0]
    credentials_error = DialogueError("no Anthropic credentials: run `ant auth login`")
    _intercept_by_scope(monkeypatch, {target_id: credentials_error})

    expected = f"session {target_id}: no Anthropic credentials: run `ant auth login`"
    with pytest.raises(DialogueError, match=re.escape(expected)):
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    assert not store.exists(SESSIONS)


def test_budget_error_takes_precedence_over_a_plain_dialogue_error(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(
        config, "dialogue", pm_filter=PmFilter(pm_ids=(_TARGET_PM_ID,)), token_budget=1
    )
    session_ids = _sorted_skeleton_ids(store, _TARGET_PM_ID)
    assert len(session_ids) >= 2
    # This session's every send bypasses the real budget-tracking client entirely, so
    # it always fails with a plain refusal; the other session runs for real and is
    # certain to exceed the token_budget of 1 on its second call.
    _intercept_by_scope(monkeypatch, {session_ids[0]: _refusal()})

    with pytest.raises(DialogueBudgetError):
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_non_dialogue_exception_is_re_raised_and_writes_no_tables(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=(_TARGET_PM_ID,)))
    session_ids = _sorted_skeleton_ids(store, _TARGET_PM_ID)
    _intercept_by_scope(monkeypatch, {session_ids[0]: ValueError("boom")})

    with pytest.raises(ValueError, match="boom"):
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_session_concurrency_never_exceeds_max_concurrency(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", max_concurrency=2)
    tracker = _ConcurrencyTrackingClient()

    run_stage(make_stage(lambda c: tracker), config, store)

    # Exact, not just bounded: single-threaded asyncio plus `send`'s own yield means two
    # sessions always overlap here, since there are far more than two to run.
    assert tracker.max_in_flight == 2


def test_budget_error_writes_no_tables(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", token_budget=1)

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

    run_stage(make_stage(raising_factory), config, store, force=True)

    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_pm_filter_matching_no_pm_raises_before_any_call_and_writes_nothing(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=("no_such_pm",)))

    with pytest.raises(DialogueError, match="dialogue.pm_filter selects no PMs"):
        run_stage(make_stage(raising_factory), config, store)

    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_run_metadata_totals_rejected_replies_across_sessions(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    calls = {"n": 0}

    def responder(request: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return _refusal()
        return default_responder(request)

    run_stage(make_stage(lambda c: FakeClient(responder)), config, store)

    metadata = store.read_run_metadata("dialogue")
    assert metadata is not None
    assert metadata["rejected_replies"] == 1


def test_transient_and_rejected_failures_are_both_listed(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient SDK error, already mapped to a `DialogueError` by the client, must be
    grouped by session alongside a plain rejected-reply failure, not silently dropped.
    """
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=(_MANY_SESSIONS_PM_ID,)))
    session_ids = _sorted_skeleton_ids(store, _MANY_SESSIONS_PM_ID)
    assert len(session_ids) >= 3
    first_id, second_id = session_ids[0], session_ids[1]
    transient_error = DialogueError("APIStatusError (status 529): overloaded")
    _intercept_by_scope(monkeypatch, {first_id: transient_error, second_id: _refusal()})

    with pytest.raises(DialogueError) as excinfo:
        run_stage(make_stage(lambda c: FakeClient(default_responder)), config, store)

    message = str(excinfo.value)
    assert f"session {first_id}: APIStatusError (status 529): overloaded" in message
    assert f"session {second_id}: the reply was refused" in message
    assert not store.exists(SESSIONS)
    assert not store.exists(DIALOGUE_LOGS)


def test_client_is_closed_after_a_successful_run(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    fake = FakeClient(default_responder)
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)

    run_stage(make_stage(lambda c: fake), config, store)

    assert fake.closed is True


def test_client_is_closed_even_when_the_run_fails(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient(default_responder)
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    config = with_section(config, "dialogue", pm_filter=PmFilter(pm_ids=(_TARGET_PM_ID,)))
    session_ids = _sorted_skeleton_ids(store, _TARGET_PM_ID)
    _intercept_by_scope(monkeypatch, {session_ids[0]: _refusal()})

    with pytest.raises(DialogueError):
        run_stage(make_stage(lambda c: fake), config, store)

    assert fake.closed is True


def test_missing_plan_metadata_raises(
    tmp_path: Path, fixture_market: dict, neutral_pm, catalogue: Catalogue
) -> None:
    config, store = run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
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
    assert metadata["cache_hits"] == 0
    assert metadata["input_tokens"] > 0
    assert metadata["output_tokens"] > 0
    assert sum(metadata["sessions"].values()) == len(store.read(SESSIONS))
    assert isinstance(metadata["warnings"], list)


def test_raise_on_failure_groups_by_label_and_strips_the_prefix(tmp_path: Path) -> None:
    client = CachedClient(lambda: FakeClient(default_responder), tmp_path, token_budget=None)
    results = [DialogueError("pm pm_001: bad"), DialogueError("bad")]

    with pytest.raises(DialogueError) as excinfo:
        raise_on_failure(("pm_001", "pm_002"), results, client, label="pm")

    assert str(excinfo.value) == "pms pm_001, pm_002: bad"


def test_raise_on_failure_keeps_session_wording_by_default(tmp_path: Path) -> None:
    client = CachedClient(lambda: FakeClient(default_responder), tmp_path, token_budget=None)

    with pytest.raises(DialogueError, match=f"^session {SESSION_ID}: "):
        raise_on_failure((SESSION_ID,), [DialogueError("boom")], client)
