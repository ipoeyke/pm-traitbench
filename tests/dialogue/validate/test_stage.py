"""Tests for the validate stage: table I/O, run metadata and the per-session attempt loop
wired through a real (but scripted) dialogue corpus.
"""

import json
from collections import Counter
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest

from pm_traitbench import pipeline
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import CachedClient, UsageTotals
from pm_traitbench.dialogue.stage import make_stage
from pm_traitbench.dialogue.validate.stage import _run_metadata
from pm_traitbench.dialogue.validate.stage import make_stage as make_validate_stage
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.enums import Ownership, SignalMode, Typicality, Valence, ValidationStatus
from pm_traitbench.errors import DialogueBudgetError, ValidateError
from pm_traitbench.signals.stage import PLAN_STAGE
from pm_traitbench.stages import Stage, run_stage
from pm_traitbench.tables.schema import Signal, ValidationRow, to_record
from pm_traitbench.tables.specs import (
    DIALOGUE_LOGS,
    LEDGER,
    SESSIONS,
    SIGNALS,
    SKELETONS,
    VALIDATION,
)
from pm_traitbench.tables.store import DataStore
from tests.dialogue.conftest import FakeClient, default_responder, fake_message, turn_text
from tests.dialogue.validate.conftest import (
    advisor_turn,
    forbidden_reply,
    is_forbidden_request,
    is_leak_request,
    leak_reply,
    log_of,
    pm_turn,
    trade_mention,
)
from tests.engine.conftest import (  # noqa: F401
    fixture_market,
    neutral_pm,
    stage_config,
    write_stage_inputs,
)

_CLEAN_TEXT = "all clear on the book"
# A responder must never see this: it means `CachedClient.send` was called without
# `_intercept_scope` installed first, so failing loudly beats a silently empty lookup.
_NO_SCOPE = "<no-scope-intercepted>"
_current_scope: ContextVar[str] = ContextVar("validate_test_scope", default=_NO_SCOPE)


def _mentions_by_session(store: DataStore) -> dict[str, list[dict]]:
    """Every session's PM-turn trade mentions, built exactly from its own ledger day-trades."""
    ledger = store.read(LEDGER)
    skeletons = store.read(SKELETONS)
    result: dict[str, list[dict]] = {}
    for sk in skeletons:
        rows = [
            row
            for row in ledger
            if row.pm_id == sk.pm_id
            and row.date == sk.date
            and row.trade_idea_id in sk.trade_idea_ids
        ]
        result[sk.session_id] = [trade_mention(row).model_dump(mode="json") for row in rows]
    return result


def _intercept_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch `CachedClient.send` to record its scope for the current task.

    Every other behaviour (caching, budget, usage totals) is untouched: the
    patched method only stashes `scope` before delegating to the real one, so
    a scope-blind `LlmClient` responder can still answer per session.
    """
    real_send = CachedClient.send

    async def patched_send(self: CachedClient, request: dict, *, scope: str, refresh: bool = False):
        _current_scope.set(scope)
        return await real_send(self, request, scope=scope, refresh=refresh)

    monkeypatch.setattr(CachedClient, "send", patched_send)


def _faithful_responder(
    store: DataStore, bad_text_by_session: dict[str, str] | None = None
) -> Callable[[dict], dict]:
    """A responder whose narrator turns mention exactly the session's own ledger day-trades.

    A session named in `bad_text_by_session` gets that text on its first
    narration only; a regeneration (its request carries validator feedback)
    always gets clean text, so a session can be made to fail once and then
    pass without needing a second override.
    """
    mentions_by_session = _mentions_by_session(store)
    bad_text_by_session = bad_text_by_session or {}

    def responder(request: dict) -> dict:
        if is_leak_request(request):
            return leak_reply(False, None)
        if is_forbidden_request(request):
            return forbidden_reply([])
        if "tools" in request:
            return default_responder(request)
        scope = _current_scope.get()
        if scope == _NO_SCOPE:
            raise AssertionError(
                "CachedClient.send was not intercepted; call faithful_dialogue_stage (or "
                "_intercept_scope) before running a stage that narrates or judges a session"
            )
        is_regeneration = "A validator rejected the previous version" in request.get("system", "")
        text = bad_text_by_session.get(scope, _CLEAN_TEXT) if not is_regeneration else _CLEAN_TEXT
        mentions = mentions_by_session.get(scope, [])
        return fake_message([turn_text(text, mentions)])

    return responder


def faithful_dialogue_stage(
    monkeypatch: pytest.MonkeyPatch,
    store: DataStore,
    *,
    bad_text_by_session: dict[str, str] | None = None,
) -> Stage:
    """The dialogue stage wired to a responder whose sessions all match their own ledger."""
    _intercept_scope(monkeypatch)
    return make_stage(lambda c: FakeClient(_faithful_responder(store, bad_text_by_session)))


def _clean_validate_stage(store: DataStore) -> Stage:
    """The validate stage wired to a responder that judges every session clean."""
    return make_validate_stage(lambda c: FakeClient(_faithful_responder(store)))


def _run_engine_and_plan(
    tmp_path: Path, fixture_market: dict, neutral_pm
) -> tuple[Config, DataStore]:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    write_stage_inputs(store, fixture_market, neutral_pm)
    run_stage(ENGINE_STAGE, config, store)
    run_stage(PLAN_STAGE, config, store)
    return config, store


def _run_clean_corpus(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> tuple[Config, DataStore]:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    run_stage(_clean_validate_stage(store), config, store)
    return config, store


def _raising_factory(_config: Config):
    raise AssertionError("the client must not be constructed here")


def test_clean_run_writes_one_pass_row_per_session_and_keeps_tables(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    sessions_before = [to_record(s) for s in store.read(SESSIONS)]
    logs_before = [to_record(log) for log in store.read(DIALOGUE_LOGS)]

    run_stage(_clean_validate_stage(store), config, store)

    validation = store.read(VALIDATION)
    session_ids = {s.session_id for s in store.read(SKELETONS)}
    assert {row.session_id for row in validation} == session_ids
    assert all(row.status == ValidationStatus.PASS for row in validation)
    assert all(n == 1 for n in Counter(row.session_id for row in validation).values())

    assert [to_record(s) for s in store.read(SESSIONS)] == sessions_before
    assert [to_record(log) for log in store.read(DIALOGUE_LOGS)] == logs_before


def test_grep_failure_regenerates_and_replaces_the_session(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    target_id = sorted(sk.session_id for sk in store.read(SKELETONS))[0]
    dialogue_stage = faithful_dialogue_stage(
        monkeypatch, store, bad_text_by_session={target_id: "my herding_weight is showing today"}
    )
    run_stage(dialogue_stage, config, store)
    turns_before = {s.session_id: s.turns for s in store.read(SESSIONS)}[target_id]

    run_stage(_clean_validate_stage(store), config, store)

    rows = [r for r in store.read(VALIDATION) if r.session_id == target_id]
    assert [r.status for r in rows] == [ValidationStatus.REGENERATE, ValidationStatus.PASS]
    turns_after = {s.session_id: s.turns for s in store.read(SESSIONS)}[target_id]
    assert turns_after != turns_before

    metadata = store.read_run_metadata("validate")
    assert metadata is not None
    assert metadata["fails_by_layer"]["grep"] == 1
    total_sessions = metadata["sessions_checked"]
    assert metadata["regeneration_rate_by_typicality"] == {"typical": 1 / total_sessions}


def _setup_drop_scenario(
    tmp_path: Path, fixture_market: dict, neutral_pm, monkeypatch: pytest.MonkeyPatch
) -> tuple[Config, DataStore, str]:
    """Run engine, plan and a faithful dialogue stage, with one carrier session (and a
    signal planted on it) primed to fail every attempt.

    Returns the one-attempt config that will force the drop, the store and the
    session id that will drop, all before validate has run.
    """
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    skeleton = sorted(store.read(SKELETONS), key=lambda s: s.session_id)[0]
    target_id = skeleton.session_id
    signal = Signal(
        signal_id="sg_001",
        pm_id=skeleton.pm_id,
        session_id=target_id,
        date=skeleton.date,
        trait_id="t_01",
        mode=SignalMode.STATED,
        trade_idea_id=None,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        third_party_value=None,
        claim_session_id=None,
    )
    store.write(SIGNALS, [signal])

    dialogue_stage = faithful_dialogue_stage(
        monkeypatch, store, bad_text_by_session={target_id: "my herding_weight is showing today"}
    )
    run_stage(dialogue_stage, config, store)

    config_one_attempt = config.model_copy(
        update={"validation": config.validation.model_copy(update={"max_attempts": 1})}
    )
    return config_one_attempt, store, target_id


def test_drop_removes_session_and_log_and_reports_void_signals(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_one_attempt, store, target_id = _setup_drop_scenario(
        tmp_path, fixture_market, neutral_pm, monkeypatch
    )
    other_session_ids = {s.session_id for s in store.read(SESSIONS)} - {target_id}
    other_rows_before = {
        s.session_id: to_record(s) for s in store.read(SESSIONS) if s.session_id != target_id
    }

    run_stage(_clean_validate_stage(store), config_one_attempt, store)

    session_ids_after = {s.session_id for s in store.read(SESSIONS)}
    log_ids_after = {log.session_id for log in store.read(DIALOGUE_LOGS)}
    assert target_id not in session_ids_after
    assert target_id not in log_ids_after
    assert other_session_ids <= session_ids_after
    other_rows_after = {
        s.session_id: to_record(s) for s in store.read(SESSIONS) if s.session_id != target_id
    }
    assert other_rows_after == other_rows_before

    metadata = store.read_run_metadata("validate")
    assert metadata is not None
    assert metadata["dropped_session_ids"] == [target_id]
    assert metadata["void_signal_ids"] == ["sg_001"]
    target_pm_id = next(s.pm_id for s in store.read(SKELETONS) if s.session_id == target_id)
    assert metadata["void_signals_by_pm"] == {target_pm_id: 1}


def test_rerun_after_drop_skips_the_dropped_session_and_carries_its_rows(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_one_attempt, store, target_id = _setup_drop_scenario(
        tmp_path, fixture_market, neutral_pm, monkeypatch
    )
    run_stage(_clean_validate_stage(store), config_one_attempt, store)
    validation_rows_before = [r for r in store.read(VALIDATION) if r.session_id == target_id]
    sessions_before = store.path(SESSIONS).read_bytes()
    logs_before = store.path(DIALOGUE_LOGS).read_bytes()

    run_stage(make_validate_stage(_raising_factory), config_one_attempt, store, force=True)

    metadata = store.read_run_metadata("validate")
    assert metadata is not None
    assert metadata["dropped_session_ids"] == [target_id]
    assert metadata["void_signal_ids"] == ["sg_001"]
    validation_rows_after = [r for r in store.read(VALIDATION) if r.session_id == target_id]
    assert validation_rows_after == validation_rows_before
    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_rerun_after_full_pm_drop_still_lists_the_pm_and_carries_its_rows(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun must still visit a PM whose sessions all dropped, not silently drop it too."""
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    skeletons = store.read(SKELETONS)
    target_pm_id = sorted({sk.pm_id for sk in skeletons})[0]
    target_session_ids = tuple(
        sorted(sk.session_id for sk in skeletons if sk.pm_id == target_pm_id)
    )
    target_skeleton = next(sk for sk in skeletons if sk.session_id == target_session_ids[0])
    signal = Signal(
        signal_id="sg_001",
        pm_id=target_pm_id,
        session_id=target_session_ids[0],
        date=target_skeleton.date,
        trait_id="t_01",
        mode=SignalMode.STATED,
        trade_idea_id=None,
        valence=Valence.CONFIRM,
        ownership=Ownership.SELF,
        third_party_value=None,
        claim_session_id=None,
    )
    store.write(SIGNALS, [signal])

    dialogue_stage = faithful_dialogue_stage(
        monkeypatch,
        store,
        bad_text_by_session=dict.fromkeys(target_session_ids, "my herding_weight is showing today"),
    )
    run_stage(dialogue_stage, config, store)

    config_one_attempt = config.model_copy(
        update={"validation": config.validation.model_copy(update={"max_attempts": 1})}
    )
    run_stage(_clean_validate_stage(store), config_one_attempt, store)
    assert target_pm_id not in {s.pm_id for s in store.read(SESSIONS)}
    validation_rows_before = [
        r for r in store.read(VALIDATION) if r.session_id in target_session_ids
    ]
    assert all(r.status == ValidationStatus.DROPPED for r in validation_rows_before)
    sessions_before = store.path(SESSIONS).read_bytes()
    logs_before = store.path(DIALOGUE_LOGS).read_bytes()

    run_stage(make_validate_stage(_raising_factory), config_one_attempt, store, force=True)

    metadata = store.read_run_metadata("validate")
    assert metadata is not None
    assert target_pm_id in metadata["pms"]
    assert set(target_session_ids) <= set(metadata["dropped_session_ids"])
    assert metadata["void_signal_ids"] == ["sg_001"]
    assert metadata["void_signals_by_pm"] == {target_pm_id: 1}
    validation_rows_after = [
        r for r in store.read(VALIDATION) if r.session_id in target_session_ids
    ]
    assert validation_rows_after == validation_rows_before
    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_missing_dialogue_metadata_raises_validate_error_before_any_call(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    (tmp_path / "run_metadata" / "dialogue.json").unlink()

    with pytest.raises(ValidateError, match="dialogue run metadata is missing"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_advisor_prompt_hash_mismatch_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)

    other_prompt = tmp_path / "other_prompt.md"
    other_prompt.write_text("A different advisor system prompt.", encoding="utf-8")
    config = config.model_copy(
        update={
            "dialogue": config.dialogue.model_copy(update={"advisor_prompt_path": other_prompt})
        }
    )

    with pytest.raises(ValidateError, match="advisor prompt"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_missing_voices_key_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)

    meta_path = tmp_path / "run_metadata" / "dialogue.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["voices"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValidateError, match="voices"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_missing_advisor_prompt_sha256_key_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)

    meta_path = tmp_path / "run_metadata" / "dialogue.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["advisor_prompt_sha256"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValidateError, match="advisor_prompt_sha256"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_voice_mismatch_raises_validate_error(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)

    meta_path = tmp_path / "run_metadata" / "dialogue.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tampered_pm_id = next(iter(meta["voices"]))
    meta["voices"][tampered_pm_id] = "not-a-real-voice-id"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValidateError, match="voice draw"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_one_missing_row_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skeleton missing exactly one of its session/log rows reflects a corrupted data
    dir, not a legitimate drop, and must raise.
    """
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    target_id = sorted(sk.session_id for sk in store.read(SKELETONS))[0]
    store.write(SESSIONS, [s for s in store.read(SESSIONS) if s.session_id != target_id])

    with pytest.raises(ValidateError, match=f"session '{target_id}' has no narrated"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_missing_session_and_log_with_no_validation_table_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skeleton missing both its session and log rows, with no `validation` table at
    all, has never been checked and never dropped: it must raise, not be skipped.
    """
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    target_id = sorted(sk.session_id for sk in store.read(SKELETONS))[0]
    store.write(SESSIONS, [s for s in store.read(SESSIONS) if s.session_id != target_id])
    store.write(
        DIALOGUE_LOGS, [log for log in store.read(DIALOGUE_LOGS) if log.session_id != target_id]
    )
    assert not store.exists(VALIDATION)

    with pytest.raises(ValidateError, match=f"session '{target_id}' has no narrated"):
        run_stage(make_validate_stage(_raising_factory), config, store)


def test_confirmed_dropped_id_with_only_one_row_missing_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session id an earlier run's `validation` table confirms `DROPPED` is only
    skipped while both its rows stay missing; one reappearing is still an inconsistency.
    """
    config_one_attempt, store, target_id = _setup_drop_scenario(
        tmp_path, fixture_market, neutral_pm, monkeypatch
    )
    run_stage(_clean_validate_stage(store), config_one_attempt, store)
    dropped_rows = {
        r.session_id for r in store.read(VALIDATION) if r.status == ValidationStatus.DROPPED
    }
    assert target_id in dropped_rows

    skeleton = next(sk for sk in store.read(SKELETONS) if sk.session_id == target_id)
    stray_log = log_of(target_id, skeleton.pm_id, [pm_turn("filler"), advisor_turn("noted")])
    store.write(DIALOGUE_LOGS, [*store.read(DIALOGUE_LOGS), stray_log])

    with pytest.raises(ValidateError, match=f"session '{target_id}' has no narrated"):
        run_stage(make_validate_stage(_raising_factory), config_one_attempt, store, force=True)


def test_missing_rows_for_a_passed_session_raises(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `validation` table that exists but holds no `DROPPED` row for a missing session
    (here, an earlier `PASS`) must not be mistaken for an earlier drop.
    """
    config, store = _run_clean_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    target_id = next(
        row.session_id for row in store.read(VALIDATION) if row.status == ValidationStatus.PASS
    )
    store.write(SESSIONS, [s for s in store.read(SESSIONS) if s.session_id != target_id])
    store.write(
        DIALOGUE_LOGS, [log for log in store.read(DIALOGUE_LOGS) if log.session_id != target_id]
    )

    with pytest.raises(ValidateError, match=f"session '{target_id}' has no narrated"):
        run_stage(make_validate_stage(_raising_factory), config, store, force=True)


def test_pm_with_no_skeletons_raises_and_leaves_sessions_untouched(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PM the dialogue metadata's `voices` names but that now has no skeletons must
    raise rather than have its narrated `sessions`/`dialogue_logs` rows silently dropped.
    """
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    target_pm_id = sorted({sk.pm_id for sk in store.read(SKELETONS)})[0]
    store.write(SKELETONS, [sk for sk in store.read(SKELETONS) if sk.pm_id != target_pm_id])
    sessions_before = store.path(SESSIONS).read_bytes()

    with pytest.raises(ValidateError, match=target_pm_id):
        run_stage(make_validate_stage(_raising_factory), config, store)

    assert store.path(SESSIONS).read_bytes() == sessions_before


def test_budget_error_writes_nothing(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    sessions_before = store.path(SESSIONS).read_bytes()
    logs_before = store.path(DIALOGUE_LOGS).read_bytes()

    config = config.model_copy(
        update={"validation": config.validation.model_copy(update={"token_budget": 1})}
    )

    with pytest.raises(DialogueBudgetError):
        run_stage(_clean_validate_stage(store), config, store)

    assert not store.exists(VALIDATION)
    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_unparsable_forbidden_judge_raises_validate_error_and_writes_nothing(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forbidden judge that never returns parsable JSON must surface as the validate
    stage's own error, not the dialogue stage's, and leave every table untouched.
    """
    config, store = _run_engine_and_plan(tmp_path, fixture_market, neutral_pm)
    run_stage(faithful_dialogue_stage(monkeypatch, store), config, store)
    sessions_before = store.path(SESSIONS).read_bytes()
    logs_before = store.path(DIALOGUE_LOGS).read_bytes()

    faithful = _faithful_responder(store)

    def responder(request: dict) -> dict:
        if is_forbidden_request(request):
            return fake_message([{"type": "text", "text": "not json"}])
        return faithful(request)

    stage = make_validate_stage(lambda c: FakeClient(responder))

    with pytest.raises(ValidateError, match="the judge reply was unparsable"):
        run_stage(stage, config, store)

    assert not store.exists(VALIDATION)
    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_rerun_from_cache_is_byte_identical_with_zero_inner_calls(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_clean_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)
    validation_before = store.path(VALIDATION).read_bytes()
    sessions_before = store.path(SESSIONS).read_bytes()
    logs_before = store.path(DIALOGUE_LOGS).read_bytes()

    run_stage(make_validate_stage(_raising_factory), config, store, force=True)

    assert store.path(VALIDATION).read_bytes() == validation_before
    assert store.path(SESSIONS).read_bytes() == sessions_before
    assert store.path(DIALOGUE_LOGS).read_bytes() == logs_before


def test_run_metadata_keys_and_typicality_rate(
    tmp_path: Path,
    fixture_market: dict,
    neutral_pm,
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _run_clean_corpus(tmp_path, fixture_market, neutral_pm, monkeypatch)

    metadata = store.read_run_metadata("validate")
    assert metadata is not None
    for key in (
        "judge_model",
        "pms",
        "sessions_checked",
        "fails_by_layer",
        "regenerated",
        "dropped",
        "dropped_session_ids",
        "void_signal_ids",
        "void_signals_by_pm",
        "regeneration_rate_by_typicality",
        "warnings",
        "calls",
        "cache_hits",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "rejected_replies",
    ):
        assert key in metadata

    assert metadata["regeneration_rate_by_typicality"] == {"typical": 0.0}
    assert metadata["dropped"] == 0
    assert metadata["regenerated"] == 0


def _unit_of(pm_id: str, typicality: Typicality) -> SimpleNamespace:
    """A stand-in for `_Unit`, carrying only the attributes `_run_metadata` reads."""
    skeleton = SimpleNamespace(pm_id=pm_id)
    return SimpleNamespace(ctx=SimpleNamespace(skeleton=skeleton), typicality=typicality)


def _validation_row(
    pm_id: str, session_id: str, attempt: int, status: ValidationStatus
) -> ValidationRow:
    passed = status == ValidationStatus.PASS
    return ValidationRow(
        pm_id=pm_id,
        session_id=session_id,
        attempt=attempt,
        status=status,
        ledger_ok=True,
        grep_ok=True,
        leak_judged=False,
        leak_ok=True,
        forbidden_ok=passed,
        level_warnings=0,
        reasons=() if passed else ("forbidden: test reason",),
        judge_model="judge-test",
    )


def test_regeneration_rate_counts_a_twice_regenerated_session_once() -> None:
    """A session regenerated twice before passing must count once toward the rate, not
    twice, so the rate stays a share of sessions rather than a count of attempts.
    """
    units = (
        _unit_of("pm_001", Typicality.TYPICAL),
        _unit_of("pm_002", Typicality.TYPICAL),
    )
    rows = [
        _validation_row("pm_001", "s_pm001_2026-01-05_a", 1, ValidationStatus.REGENERATE),
        _validation_row("pm_001", "s_pm001_2026-01-05_a", 2, ValidationStatus.REGENERATE),
        _validation_row("pm_001", "s_pm001_2026-01-05_a", 3, ValidationStatus.PASS),
        _validation_row("pm_002", "s_pm002_2026-01-05_a", 1, ValidationStatus.PASS),
    ]

    metadata = _run_metadata(
        Config(),
        units,
        rows,
        session_warnings=(),
        rejected_replies=0,
        dropped_session_ids=(),
        signals=(),
        pm_ids={"pm_001", "pm_002"},
        totals=UsageTotals(),
    )

    assert metadata["regeneration_rate_by_typicality"] == {"typical": 0.5}
    assert metadata["regenerated"] == 2


def test_pipeline_lists_validate_seventh() -> None:
    assert pipeline.STAGES[-1].number == 7
    assert pipeline.STAGES[-1].name == "validate"
