"""Table specifications: the row model and storage key for each table."""

from dataclasses import dataclass

from pydantic import BaseModel

from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    DialogueLog,
    DriftEvent,
    Gate1CellRow,
    Gate1PmRow,
    Gate2CellRow,
    Gate2PmRow,
    Gate2SignalRow,
    Gate2TraitRow,
    Idea,
    Instrument,
    LedgerRow,
    Persona,
    PositionDay,
    Price,
    ProbeRow,
    RegimeSpan,
    ResponseRow,
    Rule,
    RuleEvent,
    ScoreRow,
    Session,
    Signal,
    Skeleton,
    Trait,
    ValidationRow,
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

GATE2_TRAITS = TableSpec("gate2_traits", Gate2TraitRow, ("pm_id", "param"))
GATE2_SIGNALS = TableSpec("gate2_signals", Gate2SignalRow, ("pm_id", "signal_id"))
GATE2_PM = TableSpec("gate2_pm", Gate2PmRow, ("pm_id",))
GATE2_CELLS = TableSpec("gate2_cells", Gate2CellRow, ("slice", "slice_value", "param"))
GATE2_TABLES: tuple[TableSpec, ...] = (GATE2_TRAITS, GATE2_SIGNALS, GATE2_PM, GATE2_CELLS)

SIGNALS = TableSpec("signals", Signal, ("pm_id", "signal_id"))
SKELETONS = TableSpec("skeletons", Skeleton, ("pm_id", "session_id"))
PLAN_TABLES: tuple[TableSpec, ...] = (SIGNALS, SKELETONS)

SESSIONS = TableSpec("sessions", Session, ("pm_id", "session_id"))
DIALOGUE_LOGS = TableSpec("dialogue_logs", DialogueLog, ("pm_id", "session_id"))
DIALOGUE_TABLES: tuple[TableSpec, ...] = (SESSIONS, DIALOGUE_LOGS)

VALIDATION = TableSpec("validation", ValidationRow, ("pm_id", "session_id", "attempt"))
VALIDATE_TABLES: tuple[TableSpec, ...] = (VALIDATION,)

PROBES = TableSpec("probes", ProbeRow, ("pm_id", "probe_id"))
PROBES_TABLES: tuple[TableSpec, ...] = (PROBES,)

RESPONSES = TableSpec("responses", ResponseRow, ("pm_id", "probe_id"))
SCORES = TableSpec("scores", ScoreRow, ("pm_id", "probe_id"))


def parts_spec(pm_id: str) -> TableSpec:
    """Spec for one PM's partial responses, written as that PM finishes."""
    return TableSpec(f"parts/{pm_id}", ResponseRow, ("pm_id", "probe_id"))


HIDDEN_COLUMNS: dict[str, tuple[str, ...]] = {
    "personas": ("typicality",),
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
    "dialogue_logs": tuple(
        name for name in DialogueLog.model_fields if name not in set(DIALOGUE_LOGS.key)
    ),
    "validation": tuple(
        name for name in ValidationRow.model_fields if name not in set(VALIDATION.key)
    ),
    "probes": ("answer", "source_a", "source_b", "source_c", "source_d", "supporting_signal_ids"),
}


def _check_hidden_columns() -> None:
    """Fail at import time if a hidden column no longer exists on its model."""
    models_by_table = {
        spec.name: spec.model
        for spec in (
            PERSONAS,
            *ENGINE_TABLES,
            *PLAN_TABLES,
            *DIALOGUE_TABLES,
            *VALIDATE_TABLES,
            *PROBES_TABLES,
        )
    }
    for table_name, hidden in HIDDEN_COLUMNS.items():
        model = models_by_table[table_name]
        for column in hidden:
            if column not in model.model_fields:
                raise ValueError(f"hidden column '{column}' is not a field of {model.__name__}")


_check_hidden_columns()
