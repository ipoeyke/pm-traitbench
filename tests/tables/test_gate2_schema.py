import pytest
from pydantic import ValidationError

from pm_traitbench.enums import (
    AssetClass,
    DriftStatus,
    Gate2Slice,
    Gate2Verdict,
    Kind,
    Ownership,
    SignalMode,
    Typicality,
    Valence,
)
from pm_traitbench.tables.schema import Gate2CellRow, Gate2PmRow, Gate2SignalRow, Gate2TraitRow
from pm_traitbench.tables.specs import (
    GATE2_CELLS,
    GATE2_PM,
    GATE2_SIGNALS,
    GATE2_TABLES,
    GATE2_TRAITS,
    HIDDEN_COLUMNS,
)


def _bias_trait_row(**overrides: object) -> Gate2TraitRow:
    fields = {
        "pm_id": "pm_001",
        "param": "loss_aversion_lambda",
        "trait_id": "t_01",
        "kind": Kind.BIAS,
        "truth_active": True,
        "truth_value": None,
        "predicted_active": True,
        "predicted_value": None,
        "correct": True,
        "cited_session_ids": ("s_pm001_2026-01-05_a",),
        "false_attribution_ids": (),
    }
    fields.update(overrides)
    return Gate2TraitRow(**fields)


def _pref_trait_row(**overrides: object) -> Gate2TraitRow:
    fields = {
        "pm_id": "pm_001",
        "param": "instrument_preference",
        "trait_id": None,
        "kind": Kind.PREFERENCE,
        "truth_active": None,
        "truth_value": None,
        "predicted_active": None,
        "predicted_value": None,
        "correct": True,
        "cited_session_ids": (),
        "false_attribution_ids": (),
    }
    fields.update(overrides)
    return Gate2TraitRow(**fields)


def _signal_row(**overrides: object) -> Gate2SignalRow:
    fields = {
        "pm_id": "pm_001",
        "signal_id": "sg_001",
        "session_id": "s_pm001_2026-01-05_a",
        "trait_id": "t_01",
        "param": "loss_aversion_lambda",
        "kind": Kind.BIAS,
        "mode": SignalMode.STATED,
        "valence": Valence.CONFIRM,
        "ownership": Ownership.SELF,
        "pre_update": False,
        "cited": True,
        "recovered": True,
        "classified": True,
        "kind_predicted": Kind.BIAS,
        "kind_ok": True,
    }
    fields.update(overrides)
    return Gate2SignalRow(**fields)


def _pm_row(**overrides: object) -> Gate2PmRow:
    fields = {
        "pm_id": "pm_001",
        "asset_class": AssetClass.EQUITIES,
        "typicality": Typicality.TYPICAL,
        "drift": DriftStatus.STATIC,
        "seed": "A",
        "sessions": 10,
        "context_chars": 1000,
        "ngram_containment": 0.1,
        "biases_correct": 4,
        "preferences_held": 3,
        "preferences_correct": 2,
        "stated_signals": 5,
        "stated_kind_ok": 4,
    }
    fields.update(overrides)
    return Gate2PmRow(**fields)


def _cell_row(**overrides: object) -> Gate2CellRow:
    fields = {
        "slice": Gate2Slice.ALL,
        "slice_value": "all",
        "param": "loss_aversion_lambda",
        "n": 10,
        "n_positive": 5,
        "rate": 0.5,
        "chance": 0.3,
        "p": 0.02,
        "verdict": Gate2Verdict.PASS,
        "blocking": True,
    }
    fields.update(overrides)
    return Gate2CellRow(**fields)


def test_bias_trait_row_requires_active_fields_and_null_values() -> None:
    row = _bias_trait_row()
    assert row.kind == Kind.BIAS
    with pytest.raises(ValidationError):
        _bias_trait_row(truth_value="x")
    with pytest.raises(ValidationError):
        _bias_trait_row(predicted_active=None)
    with pytest.raises(ValidationError):
        _bias_trait_row(trait_id=None)


def test_preference_trait_row_requires_null_active_fields() -> None:
    row = _pref_trait_row()
    assert row.kind == Kind.PREFERENCE
    with pytest.raises(ValidationError):
        _pref_trait_row(truth_active=True)


def test_preference_trait_row_requires_trait_id_exactly_when_held() -> None:
    held = _pref_trait_row(trait_id="t_09", truth_value="short bullets")
    assert held.trait_id == "t_09"
    with pytest.raises(ValidationError):
        _pref_trait_row(trait_id="t_09")
    with pytest.raises(ValidationError):
        _pref_trait_row(truth_value="short bullets")


def test_trait_row_cited_ids_sorted_unique_and_own_pm() -> None:
    with pytest.raises(ValidationError):
        _bias_trait_row(
            cited_session_ids=("s_pm001_2026-02-01_a", "s_pm001_2026-01-05_a"),
        )
    with pytest.raises(ValidationError):
        _bias_trait_row(
            cited_session_ids=("s_pm001_2026-01-05_a", "s_pm001_2026-01-05_a"),
        )
    with pytest.raises(ValidationError):
        _bias_trait_row(cited_session_ids=("s_pm002_2026-01-05_a",))
    with pytest.raises(ValidationError):
        _bias_trait_row(
            cited_session_ids=("s_pm001_2026-01-05_a",),
            false_attribution_ids=("s_pm001_2026-02-01_a",),
        )
    with pytest.raises(ValidationError):
        _bias_trait_row(
            cited_session_ids=("s_pm001_2026-01-05_a", "s_pm001_2026-02-01_a"),
            false_attribution_ids=("s_pm001_2026-02-01_a", "s_pm001_2026-01-05_a"),
        )


def test_signal_row_recovered_implies_cited() -> None:
    row = _signal_row()
    assert row.recovered is True
    with pytest.raises(ValidationError):
        _signal_row(recovered=True, cited=False)


def test_signal_row_classification_fields_are_consistent() -> None:
    with pytest.raises(ValidationError):
        _signal_row(classified=False, kind_predicted=None, kind_ok=False)
    with pytest.raises(ValidationError):
        _signal_row(kind_predicted=None, kind_ok=True)
    with pytest.raises(ValidationError):
        _signal_row(kind_predicted=Kind.BIAS, kind=Kind.BIAS, kind_ok=False)
    with pytest.raises(ValidationError):
        _signal_row(classified=False, kind_predicted=Kind.BIAS, kind_ok=None)
    with pytest.raises(ValidationError):
        _signal_row(kind_predicted=Kind.PREFERENCE, kind=Kind.BIAS, kind_ok=True)
    with pytest.raises(ValidationError):
        _signal_row(pre_update=True)


def test_signal_row_session_id_must_belong_to_own_pm() -> None:
    with pytest.raises(ValidationError):
        _signal_row(session_id="s_pm002_2026-01-05_a")


def test_pm_row_counts_bounded() -> None:
    row = _pm_row()
    assert row.preferences_correct <= row.preferences_held
    with pytest.raises(ValidationError):
        _pm_row(preferences_correct=4, preferences_held=3)
    with pytest.raises(ValidationError):
        _pm_row(biases_correct=9)
    with pytest.raises(ValidationError):
        _pm_row(ngram_containment=1.5)
    with pytest.raises(ValidationError):
        _pm_row(stated_kind_ok=6, stated_signals=5)


def test_cell_row_verdict_and_p_consistency() -> None:
    row = _cell_row()
    assert row.verdict == Gate2Verdict.PASS
    with pytest.raises(ValidationError):
        _cell_row(p=None)
    with pytest.raises(ValidationError):
        _cell_row(verdict=Gate2Verdict.INSUFFICIENT, p=0.1)
    with pytest.raises(ValidationError):
        _cell_row(verdict=None, p=0.1)
    with pytest.raises(ValidationError):
        _cell_row(slice=Gate2Slice.KIND, slice_value="bias", blocking=True)
    with pytest.raises(ValidationError):
        _cell_row(n=5, n_positive=6)
    with pytest.raises(ValidationError):
        _cell_row(rate=1.5)
    with pytest.raises(ValidationError):
        _cell_row(p=-0.1)


def test_gate2_specs_keys() -> None:
    assert GATE2_TRAITS.key == ("pm_id", "param")
    assert GATE2_SIGNALS.key == ("pm_id", "signal_id")
    assert GATE2_PM.key == ("pm_id",)
    assert GATE2_CELLS.key == ("slice", "slice_value", "param")
    assert GATE2_TABLES == (GATE2_TRAITS, GATE2_SIGNALS, GATE2_PM, GATE2_CELLS)
    for spec in GATE2_TABLES:
        assert spec.name not in HIDDEN_COLUMNS
