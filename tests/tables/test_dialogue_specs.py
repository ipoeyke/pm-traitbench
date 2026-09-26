import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pm_traitbench.config import OutputConfig
from pm_traitbench.enums import AdvisorTool, MentionKind, SessionKind, Side, TurnRole
from pm_traitbench.tables.schema import (
    CallUsage,
    DialogueLog,
    Mention,
    Session,
    ToolCall,
    Turn,
    TurnLog,
    canonical_json,
)
from pm_traitbench.tables.specs import DIALOGUE_LOGS, HIDDEN_COLUMNS, SESSIONS
from pm_traitbench.tables.store import DataStore

_HASH = "a" * 64


def _turn(**overrides: object) -> Turn:
    fields = {"role": TurnRole.PM, "text": "How are the books looking today?"}
    fields.update(overrides)
    return Turn(**fields)


def _session(**overrides: object) -> Session:
    fields = {
        "session_id": "s_pm001_2026-03-02_a",
        "pm_id": "pm_001",
        "date": datetime.date(2026, 3, 2),
        "kind": SessionKind.DECISION,
        "trade_idea_ids": ("ti_001",),
        "turns": (
            _turn(role=TurnRole.PM, text="How are the books looking today?"),
            _turn(role=TurnRole.ADVISOR, text="Steady, no surprises."),
        ),
    }
    fields.update(overrides)
    return Session(**fields)


def _usage(**overrides: object) -> CallUsage:
    fields = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    fields.update(overrides)
    return CallUsage(**fields)


def _tool_call(**overrides: object) -> ToolCall:
    fields = {
        "name": AdvisorTool.GET_QUOTE,
        "input_json": '{"instrument":"EQ-001"}',
        "result_json": '{"price":100.0}',
        "is_error": False,
    }
    fields.update(overrides)
    return ToolCall(**fields)


def _turn_log(**overrides: object) -> TurnLog:
    fields = {
        "role": TurnRole.PM,
        "text": "How are the books looking today?",
        "mentions": (),
        "directive": None,
        "scripted_violation": False,
        "tool_calls": (),
        "model": "claude-opus-5-5",
        "request_hashes": (_HASH,),
        "usage": _usage(),
    }
    fields.update(overrides)
    return TurnLog(**fields)


def _dialogue_log(**overrides: object) -> DialogueLog:
    fields = {
        "session_id": "s_pm001_2026-03-02_a",
        "pm_id": "pm_001",
        "voice_id": "terse_trader",
        "turns": (
            _turn_log(role=TurnRole.PM, text="How are the books looking today?"),
            _turn_log(role=TurnRole.ADVISOR, text="Steady, no surprises."),
        ),
    }
    fields.update(overrides)
    return DialogueLog(**fields)


def _mention(**overrides: object) -> Mention:
    fields = {
        "kind": MentionKind.TRADE,
        "instrument_id": "EQ-001",
        "trade_idea_id": "ti_001",
        "tenor": None,
        "side": Side.BUY,
        "size": 10.0,
        "field": None,
        "value": None,
    }
    fields.update(overrides)
    return Mention(**fields)


def test_session_rejects_odd_turn_count() -> None:
    with pytest.raises(ValidationError):
        _session(
            turns=(
                _turn(role=TurnRole.PM),
                _turn(role=TurnRole.ADVISOR),
                _turn(role=TurnRole.PM),
            )
        )


def test_session_rejects_turns_not_alternating_from_pm() -> None:
    with pytest.raises(ValidationError):
        _session(
            turns=(
                _turn(role=TurnRole.ADVISOR),
                _turn(role=TurnRole.PM),
            )
        )


def test_session_rejects_more_than_eight_turns() -> None:
    turns = tuple(_turn(role=TurnRole.PM if i % 2 == 0 else TurnRole.ADVISOR) for i in range(10))
    with pytest.raises(ValidationError):
        _session(turns=turns)


def test_session_rejects_session_id_of_another_date() -> None:
    with pytest.raises(ValidationError):
        _session(session_id="s_pm001_2026-03-03_a")


def test_trade_mention_requires_side_and_size_and_no_level_fields() -> None:
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, side=None)
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, size=None)
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, size=-1.0)
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, trade_idea_id=None)
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, field="price")
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, value=100.0)
    assert _mention(kind=MentionKind.TRADE).kind == MentionKind.TRADE


def test_level_mention_requires_field_and_value_and_no_trade_fields() -> None:
    level_fields = {
        "kind": MentionKind.LEVEL,
        "trade_idea_id": None,
        "side": None,
        "size": None,
        "field": "price",
        "value": 101.5,
    }
    with pytest.raises(ValidationError):
        _mention(**{**level_fields, "field": None})
    with pytest.raises(ValidationError):
        _mention(**{**level_fields, "value": None})
    with pytest.raises(ValidationError):
        _mention(**{**level_fields, "trade_idea_id": "ti_001"})
    with pytest.raises(ValidationError):
        _mention(**{**level_fields, "side": Side.BUY})
    with pytest.raises(ValidationError):
        _mention(**{**level_fields, "size": 10.0})
    assert _mention(**level_fields).kind == MentionKind.LEVEL


def test_dialogue_logs_is_hidden_in_full() -> None:
    non_key = tuple(name for name in DialogueLog.model_fields if name not in set(DIALOGUE_LOGS.key))
    assert HIDDEN_COLUMNS["dialogue_logs"] == non_key
    assert "sessions" not in HIDDEN_COLUMNS


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_session_round_trips_through_the_store(tmp_path: Path, format_name: str) -> None:
    store = DataStore(tmp_path, OutputConfig(format=format_name))
    session = _session()
    dialogue_log = _dialogue_log(
        turns=(
            _turn_log(role=TurnRole.PM),
            _turn_log(role=TurnRole.ADVISOR, tool_calls=(_tool_call(),)),
        )
    )
    store.write(SESSIONS, [session])
    store.write(DIALOGUE_LOGS, [dialogue_log])
    assert store.read(SESSIONS) == [session]
    assert store.read(DIALOGUE_LOGS) == [dialogue_log]


def test_tool_call_and_dialogue_log_build() -> None:
    tool_call = _tool_call()
    log = _dialogue_log(
        turns=(
            _turn_log(role=TurnRole.PM),
            _turn_log(role=TurnRole.ADVISOR, tool_calls=(tool_call,)),
        )
    )
    assert log.turns[1].tool_calls[0].name == AdvisorTool.GET_QUOTE


def test_dialogue_log_rejects_session_id_of_another_pm() -> None:
    with pytest.raises(ValidationError):
        _dialogue_log(session_id="s_pm002_2026-03-02_a")


def test_canonical_json_sorts_keys_and_strips_whitespace() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_tool_call_rejects_non_json_input() -> None:
    with pytest.raises(ValidationError):
        _tool_call(input_json="not json")


def test_tool_call_rejects_result_json_with_extra_whitespace() -> None:
    with pytest.raises(ValidationError):
        _tool_call(result_json='{"price": 100.0}')


def test_tool_call_rejects_input_json_with_unsorted_keys() -> None:
    with pytest.raises(ValidationError):
        _tool_call(input_json='{"b":1,"a":2}')


def test_dialogue_log_rejects_odd_turn_count() -> None:
    with pytest.raises(ValidationError):
        _dialogue_log(
            turns=(
                _turn_log(role=TurnRole.PM),
                _turn_log(role=TurnRole.ADVISOR),
                _turn_log(role=TurnRole.PM),
            )
        )


def test_dialogue_log_rejects_more_than_eight_turns() -> None:
    turns = tuple(
        _turn_log(role=TurnRole.PM if i % 2 == 0 else TurnRole.ADVISOR) for i in range(10)
    )
    with pytest.raises(ValidationError):
        _dialogue_log(turns=turns)


def test_dialogue_log_rejects_turns_not_alternating_from_pm() -> None:
    with pytest.raises(ValidationError):
        _dialogue_log(
            turns=(
                _turn_log(role=TurnRole.ADVISOR),
                _turn_log(role=TurnRole.PM),
            )
        )


def test_turn_log_request_hashes_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        _turn_log(request_hashes=("g" * 64,))


def test_turn_log_request_hashes_rejects_uppercase() -> None:
    with pytest.raises(ValidationError):
        _turn_log(request_hashes=("A" * 64,))


def test_turn_log_request_hashes_rejects_empty_tuple() -> None:
    with pytest.raises(ValidationError):
        _turn_log(request_hashes=())


def test_session_rejects_zero_turns() -> None:
    with pytest.raises(ValidationError):
        _session(turns=())


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"],
)
def test_call_usage_rejects_negative_values(field: str) -> None:
    with pytest.raises(ValidationError):
        _usage(**{field: -1})


def test_turn_log_pm_turn_forbids_tool_calls() -> None:
    with pytest.raises(ValidationError):
        _turn_log(role=TurnRole.PM, tool_calls=(_tool_call(),))


def test_turn_log_pm_turn_forbids_scripted_violation() -> None:
    with pytest.raises(ValidationError):
        _turn_log(role=TurnRole.PM, scripted_violation=True)


def test_dialogue_log_rejects_more_than_one_scripted_violation() -> None:
    with pytest.raises(ValidationError):
        _dialogue_log(
            turns=(
                _turn_log(role=TurnRole.PM),
                _turn_log(role=TurnRole.ADVISOR, scripted_violation=True),
                _turn_log(role=TurnRole.PM),
                _turn_log(role=TurnRole.ADVISOR, scripted_violation=True),
            )
        )


def test_session_rejects_unsorted_trade_idea_ids() -> None:
    with pytest.raises(ValidationError):
        _session(trade_idea_ids=("ti_002", "ti_001"))


def test_session_rejects_duplicate_trade_idea_ids() -> None:
    with pytest.raises(ValidationError):
        _session(trade_idea_ids=("ti_001", "ti_001"))


def test_session_rejects_trade_idea_id_failing_pattern() -> None:
    with pytest.raises(ValidationError):
        _session(trade_idea_ids=("bad_id",))


def test_mention_trade_idea_id_must_match_pattern() -> None:
    with pytest.raises(ValidationError):
        _mention(kind=MentionKind.TRADE, trade_idea_id="bad_id")
