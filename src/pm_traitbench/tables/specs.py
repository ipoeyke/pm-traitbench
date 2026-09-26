"""Table specifications: the row model and storage key for each table."""

from dataclasses import dataclass

from pydantic import BaseModel

from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    DriftEvent,
    Gate1CellRow,
    Gate1PmRow,
    Idea,
    Instrument,
    LedgerRow,
    Persona,
    PositionDay,
    Price,
    RegimeSpan,
    Rule,
    RuleEvent,
    Signal,
    Skeleton,
    Trait,
)


@dataclass(frozen=True)
class TableSpec:
    """A table's row model and the field names that make its rows unique."""

    name: str
    model: type[BaseModel]
    key: tuple[str, ...]


PERSONAS = TableSpec("personas", Persona, ("pm_id",))
TRAITS = TableSpec("traits", Trait, ("pm_id", "trait_id"))
RULES = TableSpec("rules", Rule, ("pm_id", "rule_id"))
DRIFT_EVENTS = TableSpec("drift_events", DriftEvent, ("pm_id", "date", "trait_id", "event"))

MARKET_INSTRUMENTS = TableSpec("market/instruments", Instrument, ("instrument_id",))
MARKET_PRICES = TableSpec("market/prices", Price, ("seed", "date", "instrument_id"))
MARKET_CURVES = TableSpec("market/curves", CurvePoint, ("seed", "date", "curve_id", "tenor"))
MARKET_CONSENSUS = TableSpec("market/consensus", ConsensusRow, ("seed", "date", "instrument_id"))
MARKET_CALENDAR = TableSpec(
    "market/calendar", CalendarEvent, ("seed", "date", "instrument_id", "event")
)
MARKET_REGIMES = TableSpec("market/regimes", RegimeSpan, ("seed", "date_start"))

MARKET_TABLES: tuple[TableSpec, ...] = (
    MARKET_INSTRUMENTS,
    MARKET_PRICES,
    MARKET_CURVES,
    MARKET_CONSENSUS,
    MARKET_CALENDAR,
    MARKET_REGIMES,
)

IDEAS = TableSpec("ideas", Idea, ("pm_id", "trade_idea_id"))
LEDGER = TableSpec(
    "ledger", LedgerRow, ("pm_id", "date", "trade_idea_id", "instrument_id", "tenor", "side")
)
RULE_EVENTS = TableSpec(
    "rule_events", RuleEvent, ("pm_id", "rule_id", "trade_idea_id", "date_fired")
)
POSITION_DAYS = TableSpec("position_days", PositionDay, ("pm_id", "date", "trade_idea_id"))

ENGINE_TABLES: tuple[TableSpec, ...] = (IDEAS, LEDGER, RULE_EVENTS, POSITION_DAYS)

GATE1_PM = TableSpec("gate1_pm", Gate1PmRow, ("pm_id", "param", "split"))
GATE1_CELLS = TableSpec(
    "gate1_cells", Gate1CellRow, ("seed_group", "asset_class", "param", "split")
)
GATE1_TABLES: tuple[TableSpec, ...] = (GATE1_PM, GATE1_CELLS)

SIGNALS = TableSpec("signals", Signal, ("pm_id", "signal_id"))
SKELETONS = TableSpec("skeletons", Skeleton, ("pm_id", "session_id"))
PLAN_TABLES: tuple[TableSpec, ...] = (SIGNALS, SKELETONS)

HIDDEN_COLUMNS: dict[str, tuple[str, ...]] = {
    "ledger": ("bias_flag", "rule_id"),
    "ideas": (
        "own_signal",
        "forecast",
        "interval_lo",
        "interval_hi",
        "street_view_at_entry",
        "conflict",
        "followed_street",
        "conviction",
        "size_rank",
        "chased_trend",
    ),
    "position_days": tuple(
        name for name in PositionDay.model_fields if name not in set(POSITION_DAYS.key)
    ),
    "skeletons": tuple(name for name in Skeleton.model_fields if name not in set(SKELETONS.key)),
}


def _check_hidden_columns() -> None:
    """Fail at import time if a hidden column no longer exists on its model."""
    models_by_table = {spec.name: spec.model for spec in (*ENGINE_TABLES, *PLAN_TABLES)}
    for table_name, hidden in HIDDEN_COLUMNS.items():
        model = models_by_table[table_name]
        for column in hidden:
            if column not in model.model_fields:
                raise ValueError(f"hidden column '{column}' is not a field of {model.__name__}")


_check_hidden_columns()
