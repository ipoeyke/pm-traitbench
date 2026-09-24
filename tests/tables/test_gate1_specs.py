import pytest
from pydantic import ValidationError

from pm_traitbench.tables.schema import Gate1CellRow, Gate1PmRow
from pm_traitbench.tables.specs import GATE1_CELLS, GATE1_PM, GATE1_TABLES, HIDDEN_COLUMNS


def test_gate1_tables_names_and_keys() -> None:
    actual = {spec.name: spec.key for spec in GATE1_TABLES}
    assert actual == {
        "gate1_pm": ("pm_id", "param", "split"),
        "gate1_cells": ("seed_group", "asset_class", "param", "split"),
    }


def test_gate1_tables_contains_exactly_the_two_specs() -> None:
    assert GATE1_TABLES == (GATE1_PM, GATE1_CELLS)


def test_gate1_tables_have_no_hidden_columns() -> None:
    assert "gate1_pm" not in HIDDEN_COLUMNS
    assert "gate1_cells" not in HIDDEN_COLUMNS


def test_gate1_pm_row_rejects_unknown_field(gate1_pm_row: Gate1PmRow) -> None:
    with pytest.raises(ValidationError):
        Gate1PmRow(**{**gate1_pm_row.model_dump(), "unknown_field": "nope"})


def test_gate1_pm_row_rejects_negative_n(gate1_pm_row: Gate1PmRow) -> None:
    with pytest.raises(ValidationError):
        Gate1PmRow(**{**gate1_pm_row.model_dump(), "n": -1})


def test_gate1_cell_row_rejects_unknown_field(gate1_cell_row: Gate1CellRow) -> None:
    with pytest.raises(ValidationError):
        Gate1CellRow(**{**gate1_cell_row.model_dump(), "unknown_field": "nope"})


def test_gate1_cell_row_rejects_negative_n_neutral(gate1_cell_row: Gate1CellRow) -> None:
    with pytest.raises(ValidationError):
        Gate1CellRow(**{**gate1_cell_row.model_dump(), "n_neutral": -1})


def test_gate1_cell_row_rejects_negative_n_active(gate1_cell_row: Gate1CellRow) -> None:
    with pytest.raises(ValidationError):
        Gate1CellRow(**{**gate1_cell_row.model_dump(), "n_active": -1})


def test_gate1_cell_row_rejects_negative_n_missing(gate1_cell_row: Gate1CellRow) -> None:
    with pytest.raises(ValidationError):
        Gate1CellRow(**{**gate1_cell_row.model_dump(), "n_missing": -1})
