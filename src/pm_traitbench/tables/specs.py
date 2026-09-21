"""Table specifications: the row model and storage key for each table."""

from dataclasses import dataclass

from pydantic import BaseModel

from pm_traitbench.tables.schema import DriftEvent, Persona, Rule, Trait


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
