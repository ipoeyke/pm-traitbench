from pathlib import Path

import pytest
from pydantic import ValidationError

from pm_traitbench.enums import Scorer
from pm_traitbench.tables.formats import FORMATS
from pm_traitbench.tables.schema import ResponseRow, ScoreRow, to_record
from pm_traitbench.tables.specs import RESPONSES, parts_spec


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
