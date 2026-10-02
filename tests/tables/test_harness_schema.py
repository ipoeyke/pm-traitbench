from pathlib import Path

import pytest
from pydantic import ValidationError

from pm_traitbench.enums import Judge, Scorer
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import JudgementRow, ResponseRow, ScoreRow, to_record
from pm_traitbench.tables.specs import JUDGEMENTS, RESPONSES, SCORES, parts_spec


def _response(**overrides: object) -> ResponseRow:
    fields = {
        "probe_id": "p_pm001_0001",
        "pm_id": "pm_001",
        "response": "B",
        "latency_ms": 1200,
    }
    fields.update(overrides)
    return ResponseRow(**fields)


def _score(**overrides: object) -> ScoreRow:
    fields = {
        "probe_id": "p_pm001_0001",
        "pm_id": "pm_001",
        "scorer": Scorer.OPTION_LETTER,
        "correct": True,
        "detail": None,
    }
    fields.update(overrides)
    return ScoreRow(**fields)


@pytest.mark.parametrize("format_name", sorted(FORMATS))
def test_response_row_round_trips(tmp_path: Path, format_name: str) -> None:
    fmt = FORMATS[format_name]
    row = _response()
    path = tmp_path / f"table.{fmt.extension}"
    fmt.write([to_record(row)], ResponseRow, path)
    records = fmt.read(path, ResponseRow)
    assert [ResponseRow.model_validate(record) for record in records] == [row]


def test_response_row_rejects_foreign_probe_prefix() -> None:
    with pytest.raises(ValidationError):
        _response(probe_id="p_pm002_0001")


def test_response_row_allows_empty_response() -> None:
    assert _response(response="").response == ""


def test_score_row_rejects_blank_detail() -> None:
    with pytest.raises(ValidationError):
        _score(detail="")


def test_score_row_rejects_foreign_probe_prefix() -> None:
    with pytest.raises(ValidationError):
        _score(probe_id="p_pm002_0001")


def test_parts_spec_names_pm() -> None:
    spec = parts_spec("pm_001")
    assert spec.name == "parts/pm_001"
    assert spec.key == RESPONSES.key


def test_scores_key_includes_scorer() -> None:
    assert SCORES.key == ("pm_id", "probe_id", "scorer")


def test_judgements_spec() -> None:
    assert JUDGEMENTS.name == "judgements"
    assert JUDGEMENTS.model is JudgementRow
    assert JUDGEMENTS.key == ("pm_id", "probe_id", "judge")


def _judgement(**overrides: object) -> JudgementRow:
    fields = {
        "probe_id": "p_pm001_0001",
        "pm_id": "pm_001",
        "judge": Judge.OPEN,
        "correct": True,
        "detail": "verdict=right",
        "rationale": "matches the ground truth",
    }
    fields.update(overrides)
    return JudgementRow(**fields)


def test_judgement_row_round_trip() -> None:
    row = _judgement()
    assert JudgementRow.model_validate(row.model_dump()) == row
    with pytest.raises(ValidationError):
        _judgement(detail="")
    assert _judgement(rationale="").rationale == ""


def test_judge_and_scorer_share_values() -> None:
    assert {j.value for j in Judge} <= {s.value for s in Scorer}
