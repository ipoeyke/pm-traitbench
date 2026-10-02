import pytest
from pydantic import ValidationError

from pm_traitbench.enums import ValidationStatus
from pm_traitbench.tables.schema import ValidationRow
from pm_traitbench.tables.specs import HIDDEN_COLUMNS, VALIDATION


def _row(**overrides: object) -> ValidationRow:
    fields = {
        "pm_id": "pm_001",
        "session_id": "s_pm001_2026-03-02_a",
        "attempt": 1,
        "status": ValidationStatus.PASS,
        "ledger_ok": True,
        "level_ok": True,
        "grep_ok": True,
        "leak_judged": True,
        "leak_ok": True,
        "forbidden_ok": True,
        "stance_ok": True,
        "level_warnings": 0,
        "reasons": (),
        "judge_model": "claude-opus-5-5",
        "fallback_model": None,
        "fallback_judges": (),
    }
    fields.update(overrides)
    return ValidationRow(**fields)


def test_pass_row_requires_all_layers_ok_and_no_reasons() -> None:
    assert _row().status == ValidationStatus.PASS
    for layer in ("ledger", "level", "grep", "leak", "forbidden", "stance"):
        with pytest.raises(ValidationError):
            _row(**{f"{layer}_ok": False})
    with pytest.raises(ValidationError):
        _row(reasons=("some reason",))


@pytest.mark.parametrize("status", [ValidationStatus.REGENERATE, ValidationStatus.DROPPED])
def test_non_pass_status_with_all_ok_raises(status: ValidationStatus) -> None:
    with pytest.raises(ValidationError):
        _row(status=status, reasons=("some reason",))


def test_failed_row_requires_reasons() -> None:
    with pytest.raises(ValidationError):
        _row(status=ValidationStatus.REGENERATE, ledger_ok=False, reasons=())
    row = _row(status=ValidationStatus.REGENERATE, ledger_ok=False, reasons=("ledger mismatch",))
    assert row.status == ValidationStatus.REGENERATE


def test_unjudged_leak_must_be_ok() -> None:
    with pytest.raises(ValidationError):
        _row(leak_judged=False, leak_ok=False)
    row = _row(leak_judged=False, leak_ok=True)
    assert row.leak_judged is False


def test_session_id_must_belong_to_pm() -> None:
    with pytest.raises(ValidationError):
        _row(pm_id="pm_001", session_id="s_pm002_2026-01-05_a")


def test_attempt_is_at_least_one() -> None:
    with pytest.raises(ValidationError):
        _row(attempt=0)


def test_validation_spec_key_and_hidden_columns() -> None:
    assert VALIDATION.key == ("pm_id", "session_id", "attempt")
    assert set(HIDDEN_COLUMNS["validation"]) == set(ValidationRow.model_fields) - {
        "pm_id",
        "session_id",
        "attempt",
    }


def test_fallback_model_is_set_exactly_when_a_judge_fell_back() -> None:
    row = _row(fallback_model="claude-sonnet-5-5", fallback_judges=("stance turn 2",))
    assert row.fallback_judges == ("stance turn 2",)
    with pytest.raises(ValidationError):
        _row(fallback_model="claude-sonnet-5-5", fallback_judges=())
    with pytest.raises(ValidationError):
        _row(fallback_model=None, fallback_judges=("leak",))
