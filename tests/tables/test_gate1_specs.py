import pytest
from pydantic import ValidationError

from pm_traitbench.tables.schema import Gate1CellRow
from pm_traitbench.tables.specs import GATE1_CELLS, GATE1_PM, GATE1_TABLES, HIDDEN_COLUMNS
from tests.tables.test_formats import _gate1_cell_row


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


def test_gate1_cell_row_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Gate1CellRow(**{**_gate1_cell_row().model_dump(), "unknown_field": "nope"})


def test_gate1_cell_row_rejects_negative_n_neutral() -> None:
    with pytest.raises(ValidationError):
        Gate1CellRow(**{**_gate1_cell_row().model_dump(), "n_neutral": -1})
