import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pm_traitbench.config import OutputConfig
from pm_traitbench.enums import CheckpointLabel, OptionSource, ProbeForm, ProbeType
from pm_traitbench.tables.schema import ProbeRow, probe_id
from pm_traitbench.tables.specs import HIDDEN_COLUMNS, PROBES, PROBES_TABLES
from pm_traitbench.tables.store import DataStore

_NO_OPTIONS = {
    "option_a": None,
    "option_b": None,
    "option_c": None,
    "option_d": None,
    "source_a": None,
    "source_b": None,
    "source_c": None,
    "source_d": None,
}


def _row(**overrides: object) -> ProbeRow:
    fields = {
        "probe_id": "p_pm001_0001",
        "pm_id": "pm_001",
        "checkpoint_date": datetime.date(2026, 3, 2),
        "checkpoint_label": CheckpointLabel.WEEK4,
        "probe_type": ProbeType.IN_SITU,
        "trait_id": "t_01",
        "form": ProbeForm.OPEN,
        "question": "What do you do?",
        "answer": "Hold the position.",
        "supporting_signal_ids": (),
        "context_chars": 0,
        **_NO_OPTIONS,
    }
    fields.update(overrides)
    return ProbeRow(**fields)


def _presence(**overrides: object) -> ProbeRow:
    fields = {
        "probe_type": ProbeType.TRAIT_PRESENCE,
        "form": ProbeForm.MCQ,
        "option_a": "yes",
        "option_b": "no",
        "source_a": OptionSource.CURRENT,
        "answer": "A",
        "supporting_signal_ids": ("sg_001",),
    }
    fields.update(overrides)
    return _row(**fields)


def _mcq(**overrides: object) -> ProbeRow:
    fields = {
        "probe_type": ProbeType.TRAIT_MCQ,
        "form": ProbeForm.MCQ,
        "option_a": "hold",
        "option_b": "sell",
        "option_c": "add",
        "option_d": "trim",
        "source_a": OptionSource.PRE_UPDATE,
        "source_b": OptionSource.CURRENT,
        "source_c": OptionSource.STATED_PROFILE,
        "source_d": OptionSource.THIRD_PARTY,
        "answer": "B",
    }
    fields.update(overrides)
    return _row(**fields)


def test_probe_id_format() -> None:
    assert probe_id("pm_001", 1) == "p_pm001_0001"


def test_valid_rows_build() -> None:
    _presence()
    _presence(option_a="yes", source_a=OptionSource.NONE, answer="B")
    _mcq()
    _mcq(
        option_d=None,
        source_d=None,
        source_c=OptionSource.STATED_PROFILE,
    )
    _mcq(
        form=ProbeForm.OPEN,
        option_a=None,
        option_b=None,
        option_c=None,
        option_d=None,
        source_a=None,
        source_b=None,
        source_c=None,
        source_d=None,
        answer="sell",
    )
    _row()
    _row(probe_type=ProbeType.ROUTINE_QUESTION, trait_id=None)
    _row(probe_type=ProbeType.GOVERNANCE)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"probe_id": "p_pm002_0001"}, "probe_id must start with"),
        ({"supporting_signal_ids": ("bad",)}, "supporting_signal_ids must match"),
        ({"supporting_signal_ids": ("sg_002", "sg_001")}, "sorted and unique"),
        ({"supporting_signal_ids": ("sg_001", "sg_001")}, "sorted and unique"),
        ({"probe_type": ProbeType.TRAIT_MCQ, "option_a": " "}, "options must not be blank"),
        ({"option_a": "x"}, "open probe must have no options"),
        ({"source_a": OptionSource.CURRENT}, "open probe must have no options"),
        ({"probe_type": ProbeType.IN_SITU, "form": ProbeForm.MCQ}, "must be open"),
    ],
)
def test_general_rules_raise(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _row(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"form": ProbeForm.OPEN, "option_a": None, "option_b": None, "source_a": None},
            "presence probe must be mcq",
        ),
        ({"option_a": "maybe"}, "presence probe options must be yes and no"),
        ({"option_c": "x"}, "presence probe options must be yes and no"),
        ({"source_a": None}, "presence probe must set source_a only"),
        ({"source_b": OptionSource.NONE}, "presence probe must set source_a only"),
        ({"answer": "C"}, "presence answer must be A or B"),
        ({"answer": "B"}, "presence answer must be A exactly when source_a is current"),
        ({"source_a": OptionSource.NONE}, "presence answer must be A exactly when"),
    ],
)
def test_presence_rules_raise(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _presence(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "option_b": None,
                "source_b": None,
                "option_c": None,
                "source_c": None,
                "option_d": None,
                "source_d": None,
                "answer": "A",
            },
            "mcq must have 3 or 4 options",
        ),
        ({"option_c": None, "source_c": None}, "mcq options must be contiguous"),
        ({"option_b": "hold"}, "mcq option texts must be unique"),
        ({"source_d": None}, "mcq source must be set exactly where an option is"),
        ({"source_a": OptionSource.CURRENT}, "mcq must have exactly one current source"),
        ({"answer": "A"}, "mcq answer must be the letter of the current option"),
    ],
)
def test_mcq_rules_raise(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _mcq(**overrides)


def test_trait_id_rules_raise() -> None:
    with pytest.raises(ValidationError, match="trait_id may be null only"):
        _row(probe_type=ProbeType.GOVERNANCE, trait_id=None)
    with pytest.raises(ValidationError, match="trait_id may be null only"):
        _mcq(trait_id=None)
    with pytest.raises(ValidationError, match="routine_question requires a null trait_id"):
        _row(probe_type=ProbeType.ROUTINE_QUESTION, trait_id="t_01")
    _presence(trait_id=None)


def test_hidden_columns() -> None:
    assert HIDDEN_COLUMNS["probes"] == (
        "answer",
        "source_a",
        "source_b",
        "source_c",
        "source_d",
        "supporting_signal_ids",
    )
    assert PROBES_TABLES == (PROBES,)


@pytest.mark.parametrize("format_name", ["jsonl", "parquet"])
def test_data_store_round_trip_probes(tmp_path: Path, format_name: str) -> None:
    store = DataStore(tmp_path, OutputConfig(format=format_name))
    rows = [_presence(), _mcq(probe_id="p_pm001_0002"), _row(probe_id="p_pm001_0003")]
    store.write(PROBES, rows)
    assert store.read(PROBES) == rows
