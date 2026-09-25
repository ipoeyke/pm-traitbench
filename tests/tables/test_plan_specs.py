import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pm_traitbench.config import OutputConfig
from pm_traitbench.enums import Ownership, SessionKind, SignalMode, StanceEntry, Valence
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import Signal, Skeleton, Stance, to_record
from pm_traitbench.tables.specs import HIDDEN_COLUMNS, PLAN_TABLES, SIGNALS, SKELETONS
from pm_traitbench.tables.store import DataStore


def _signal(**overrides: object) -> Signal:
    fields = {
        "signal_id": "sg_001",
        "pm_id": "pm_001",
        "session_id": "s_pm001_2026-03-02_a",
        "date": datetime.date(2026, 3, 2),
        "trait_id": "t_01",
        "mode": SignalMode.STATED,
        "trade_idea_id": None,
        "valence": Valence.CONFIRM,
        "ownership": Ownership.SELF,
        "third_party_value": None,
        "claim_session_id": None,
    }
    fields.update(overrides)
    return Signal(**fields)


def _stance(**overrides: object) -> Stance:
    fields = {
        "signal_id": "sg_001",
        "trait_id": "t_01",
        "mode": SignalMode.STATED,
        "entry": StanceEntry.STATED,
        "stance": "I focus on quality names.",
    }
    fields.update(overrides)
    return Stance(**fields)


def _skeleton(**overrides: object) -> Skeleton:
    fields = {
        "session_id": "s_pm001_2026-03-02_a",
        "pm_id": "pm_001",
        "date": datetime.date(2026, 3, 2),
        "kind": SessionKind.DECISION,
        "trade_idea_ids": ("ti_001", "ti_002"),
        "stances": (_stance(),),
        "advisor_violation": None,
        "forbidden_trait_ids": (),
        "forbidden_pref_params": (),
    }
    fields.update(overrides)
    return Skeleton(**fields)


def test_valid_signal_builds() -> None:
    signal = _signal()
    assert signal.signal_id == "sg_001"


def test_valid_skeleton_builds() -> None:
    skeleton = _skeleton()
    assert skeleton.session_id == "s_pm001_2026-03-02_a"


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_signal(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    row = _signal(
        mode=SignalMode.CONTRADICTION,
        trade_idea_id="ti_001",
        claim_session_id="s_pm001_2026-02-01_a",
    )
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], Signal, path)
    records = fmt.read(path, Signal)
    assert [Signal.model_validate(record) for record in records] == [row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_round_trip_skeleton_with_nested_stances(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    row = _skeleton(
        stances=(
            _stance(),
            _stance(
                signal_id="sg_002",
                trait_id="t_02",
                entry=StanceEntry.REVEALED_REACTION,
                stance="Flagged an advisor violation.",
            ),
        ),
        advisor_violation="Advisor violated a mandate constraint.",
    )
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], Skeleton, path)
    records = fmt.read(path, Skeleton)
    assert [Skeleton.model_validate(record) for record in records] == [row]


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_data_store_round_trip_signals_and_skeletons(tmp_path: Path, format_name: str) -> None:
    store = DataStore(tmp_path, OutputConfig(format=format_name))
    signal = _signal()
    skeleton = _skeleton()
    store.write(SIGNALS, [signal])
    store.write(SKELETONS, [skeleton])
    assert store.read(SIGNALS) == [signal]
    assert store.read(SKELETONS) == [skeleton]


def test_wrong_session_prefix_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(session_id="s_pm999_2026-03-02_a")


def test_contradiction_without_claim_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(mode=SignalMode.CONTRADICTION, trade_idea_id="ti_001", claim_session_id=None)


def test_claim_on_non_contradiction_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(mode=SignalMode.STATED, claim_session_id="s_pm001_2026-02-01_a")


def test_claim_dated_on_or_after_signal_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(
            mode=SignalMode.CONTRADICTION,
            trade_idea_id="ti_001",
            claim_session_id="s_pm001_2026-03-02_a",
        )


def test_claim_belonging_to_another_pm_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(
            mode=SignalMode.CONTRADICTION,
            trade_idea_id="ti_001",
            claim_session_id="s_pm002_2026-02-01_a",
        )


def test_contradiction_without_idea_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(
            mode=SignalMode.CONTRADICTION,
            trade_idea_id=None,
            claim_session_id="s_pm001_2026-02-01_a",
        )


def test_third_party_with_mode_revealed_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(mode=SignalMode.REVEALED, ownership=Ownership.COLLEAGUE)


def test_third_party_retracted_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(mode=SignalMode.STATED, ownership=Ownership.COLLEAGUE, valence=Valence.RETRACTED)


def test_third_party_value_with_ownership_self_raises() -> None:
    with pytest.raises(ValidationError):
        _signal(ownership=Ownership.SELF, third_party_value="growth investing")


def test_silence_with_a_stance_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(kind=SessionKind.SILENCE, trade_idea_ids=(), stances=(_stance(),))


def test_duplicate_stance_trait_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(
            stances=(
                _stance(),
                _stance(signal_id="sg_002", stance="Same trait again."),
            )
        )


def test_advisor_violation_without_revealed_reaction_stance_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(advisor_violation="Advisor violated a mandate constraint.")


def test_revealed_reaction_stance_without_advisor_violation_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(
            stances=(_stance(entry=StanceEntry.REVEALED_REACTION),),
            advisor_violation=None,
        )


def test_unsorted_trade_idea_ids_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(trade_idea_ids=("ti_002", "ti_001"))


def test_trade_idea_id_failing_pattern_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(trade_idea_ids=("bad_id",))


def test_duplicate_trade_idea_ids_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(trade_idea_ids=("ti_001", "ti_001"))


def test_silence_with_trade_idea_ids_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(kind=SessionKind.SILENCE, stances=(), trade_idea_ids=("ti_001",))


def test_silence_with_advisor_violation_raises() -> None:
    with pytest.raises(ValidationError):
        _skeleton(
            kind=SessionKind.SILENCE,
            stances=(),
            trade_idea_ids=(),
            advisor_violation="Advisor violated a mandate constraint.",
        )


def test_hidden_columns_skeletons_equals_non_key_fields() -> None:
    non_key = tuple(name for name in Skeleton.model_fields if name not in set(SKELETONS.key))
    assert HIDDEN_COLUMNS["skeletons"] == non_key


def test_signals_has_no_hidden_columns() -> None:
    assert "signals" not in HIDDEN_COLUMNS


def test_plan_tables_names_and_keys() -> None:
    assert [spec.name for spec in PLAN_TABLES] == ["signals", "skeletons"]
    assert SIGNALS.key == ("pm_id", "signal_id")
    assert SKELETONS.key == ("pm_id", "session_id")
