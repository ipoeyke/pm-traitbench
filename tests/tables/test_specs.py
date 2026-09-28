from pm_traitbench.tables.schema import PositionDay
from pm_traitbench.tables.specs import (
    DIALOGUE_TABLES,
    ENGINE_TABLES,
    HIDDEN_COLUMNS,
    PLAN_TABLES,
    POSITION_DAYS,
    VALIDATE_TABLES,
)

_EXPECTED_ENGINE_TABLES = {
    "ideas": ("pm_id", "trade_idea_id"),
    "ledger": ("pm_id", "date", "trade_idea_id", "instrument_id", "tenor", "side"),
    "rule_events": ("pm_id", "rule_id", "trade_idea_id", "date_fired"),
    "position_days": ("pm_id", "date", "trade_idea_id"),
}


def test_engine_tables_names_and_keys() -> None:
    actual = {spec.name: spec.key for spec in ENGINE_TABLES}
    assert actual == _EXPECTED_ENGINE_TABLES


def test_hidden_columns_names_real_columns() -> None:
    models_by_table = {
        spec.name: spec.model
        for spec in (*ENGINE_TABLES, *PLAN_TABLES, *DIALOGUE_TABLES, *VALIDATE_TABLES)
    }
    for table_name, hidden in HIDDEN_COLUMNS.items():
        model = models_by_table[table_name]
        for column in hidden:
            assert column in model.model_fields, f"{column} is not a field of {model.__name__}"


def test_position_days_hidden_columns_equal_non_key_columns() -> None:
    expected = set(PositionDay.model_fields) - set(POSITION_DAYS.key)
    assert set(HIDDEN_COLUMNS["position_days"]) == expected


def test_chased_trend_is_a_hidden_ideas_column() -> None:
    assert "chased_trend" in HIDDEN_COLUMNS["ideas"]
