"""Table specifications: model, storage key and struct-nesting for each table."""

from dataclasses import dataclass

from pydantic import BaseModel

from pm_traitbench.tables.introspect import columns
from pm_traitbench.tables.schema import DriftEvent, Persona, Rule, Trait


@dataclass(frozen=True)
class TableSpec:
    """A table's row model and the field names that make its rows unique."""

    name: str
    model: type[BaseModel]
    key: tuple[str, ...]

    @property
    def nested(self) -> bool:
        """Whether any column holds a nested model, ruling out CSV storage."""
        return any(col.kind == "struct" for col in columns(self.model))


PERSONAS = TableSpec("personas", Persona, ("pm_id",))
TRAITS = TableSpec("traits", Trait, ("pm_id", "trait_id"))
RULES = TableSpec("rules", Rule, ("pm_id", "rule_id"))
DRIFT_EVENTS = TableSpec("drift_events", DriftEvent, ("pm_id", "date", "trait_id", "event"))
