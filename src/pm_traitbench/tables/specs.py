"""Table specifications: the row model and storage key for each table."""

from dataclasses import dataclass

from pydantic import BaseModel

from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    DriftEvent,
    Instrument,
    Persona,
    Price,
    RegimeSpan,
    Rule,
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
