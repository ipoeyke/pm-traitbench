"""Row models for the dataset's tables.

Re-exports pm_traitbench.enums so table code has a single import path.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from pm_traitbench.enums import (
    Action,
    AssetClass,
    DriftEventType,
    Kind,
    Op,
    Regime,
    RuleScope,
    RuleSource,
    Split,
    Typicality,
)
from pm_traitbench.numeric import parse_number

__all__ = [
    "AssetClass",
    "Split",
    "Typicality",
    "Kind",
    "RuleSource",
    "RuleScope",
    "Op",
    "Action",
    "DriftEventType",
    "Regime",
    "Mandate",
    "StatedProfile",
    "Persona",
    "Trait",
    "Rule",
    "DriftEvent",
    "to_record",
    "multiplier_field",
]

_PM_ID_PATTERN = r"^pm_\d{3,}$"
_TRAIT_ID_PATTERN = r"^t_\d{2,}$"
_RULE_ID_PATTERN = r"^r_\d{2,}$"


def _coerce_numeric_str(value: Any) -> Any:
    """Parse a string as a float when possible, else leave it as a string.

    CSV and parquet store mixed numeric/text columns as text; this recovers
    numbers without misreading genuinely textual levels.
    """
    if isinstance(value, str):
        parsed = parse_number(value)
        return value if parsed is None else parsed
    return value


class Mandate(BaseModel):
    """A PM's investment mandate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_class: AssetClass = Field(description="Asset class the mandate covers.")
    sub_style: str = Field(description="Sub-style within the asset class, e.g. 'value'.")
    book_size: float = Field(description="Notional book size the PM manages.")
    risk_unit: str = Field(description="Unit that rule levels for this mandate are expressed in.")
    benchmark: str = Field(description="Benchmark the mandate is measured against.")


class StatedProfile(BaseModel):
    """A PM's self-reported description of how they invest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    self_description: str = Field(description="Free-text self-description of the PM's approach.")


class Persona(BaseModel):
    """A synthetic PM persona: identity, mandate and stated profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(pattern=_PM_ID_PATTERN, description="Unique identifier for the PM.")
    market_seed: str = Field(description="Market data seed the PM's scenario draws from.")
    split: Split = Field(description="Dataset split the PM belongs to.")
    mandate: Mandate = Field(description="The PM's investment mandate.")
    stated_profile: StatedProfile = Field(description="The PM's stated self-description.")
    typicality: Typicality = Field(description="Whether the PM is typical or anti-typical.")


class Trait(BaseModel):
    """A behavioural bias or preference trait assigned to a PM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM this trait belongs to."
    )
    trait_id: str = Field(pattern=_TRAIT_ID_PATTERN, description="Unique identifier for the trait.")
    kind: Kind = Field(description="Whether the trait is a bias or a preference.")
    param: str = Field(description="Name of the underlying trait parameter.")
    value: float | str = Field(
        description="Trait value: a float for biases, a string for preferences."
    )
    active: bool = Field(description="Whether the trait is currently active.")
    mult_range: float | None = Field(description="Multiplier applied in a range regime.")
    mult_risk_off: float | None = Field(description="Multiplier applied in a risk-off regime.")
    mult_risk_on: float | None = Field(description="Multiplier applied in a risk-on regime.")

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_bias_value(cls, value: Any, info: ValidationInfo) -> Any:
        if info.data.get("kind") == Kind.BIAS and isinstance(value, str):
            return float(value)
        return value

    @model_validator(mode="after")
    def _check_kind_invariants(self) -> "Trait":
        multipliers = (self.mult_range, self.mult_risk_off, self.mult_risk_on)
        if self.kind == Kind.BIAS:
            if not isinstance(self.value, float):
                raise ValueError("bias trait requires a float value")
            if any(m is None or m <= 0 for m in multipliers):
                raise ValueError("bias trait requires all multipliers to be non-null and positive")
        else:
            if not isinstance(self.value, str):
                raise ValueError("preference trait requires a string value")
            if self.active is not True:
                raise ValueError("preference trait requires active to be True")
            if any(m is not None for m in multipliers):
                raise ValueError("preference trait requires all multipliers to be null")
        return self


class Rule(BaseModel):
    """A mandate- or self-imposed trading rule for a PM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM this rule belongs to."
    )
    rule_id: str = Field(pattern=_RULE_ID_PATTERN, description="Unique identifier for the rule.")
    source: RuleSource = Field(description="Where the rule originates from.")
    scope: RuleScope = Field(
        description="Whether the rule applies to the PM or to a single trade idea."
    )
    trade_idea_id: str | None = Field(
        description="Trade idea the rule applies to; null unless scope is 'idea'."
    )
    param: str = Field(description="Name of the underlying rule parameter.")
    field: str = Field(description="Data field the rule's condition is evaluated on.")
    op: Op = Field(description="Comparison operator used by the rule's condition.")
    level: float | str = Field(description="Threshold level: numeric where applicable, else text.")
    unit: str | None = Field(description="Unit the level is expressed in, if any.")
    window: int = Field(ge=1, description="Number of periods the rule is evaluated over.")
    action: Action = Field(description="Action taken when the rule's condition triggers.")
    text: str = Field(description="Human-readable statement of the rule.")

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, value: Any) -> Any:
        return _coerce_numeric_str(value)

    @model_validator(mode="after")
    def _check_scope(self) -> "Rule":
        if self.scope == RuleScope.PM and self.trade_idea_id is not None:
            raise ValueError("scope 'pm' requires trade_idea_id to be null")
        if self.scope == RuleScope.IDEA and self.trade_idea_id is None:
            raise ValueError("scope 'idea' requires trade_idea_id to be set")
        return self


class DriftEvent(BaseModel):
    """A change to a PM's trait over the course of the simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM this event belongs to."
    )
    date: datetime.date = Field(description="Date the drift event occurred.")
    event: DriftEventType = Field(description="Type of drift event.")
    trait_id: str = Field(pattern=_TRAIT_ID_PATTERN, description="Trait affected by the event.")
    from_value: float | str | None = Field(
        alias="from", description="Trait value before the event; null for dormant and revive."
    )
    to_value: float | str | None = Field(
        alias="to", description="Trait value after the event; null for dormant and revive."
    )

    @field_validator("from_value", "to_value", mode="before")
    @classmethod
    def _coerce_endpoint(cls, value: Any) -> Any:
        return _coerce_numeric_str(value)

    @model_validator(mode="after")
    def _check_event(self) -> "DriftEvent":
        both_set = self.from_value is not None and self.to_value is not None
        both_null = self.from_value is None and self.to_value is None
        if self.event == DriftEventType.UPDATE and not both_set:
            raise ValueError("update event requires both from and to values to be set")
        if self.event in (DriftEventType.DORMANT, DriftEventType.REVIVE) and not both_null:
            raise ValueError(
                f"{self.event.value} event requires both from and to values to be null"
            )
        return self


def to_record(row: BaseModel) -> dict[str, Any]:
    """Serialise a row model to a JSON-safe dict, keyed by field alias."""
    return row.model_dump(mode="json", by_alias=True)


def multiplier_field(regime: Regime) -> str:
    """Return the Trait multiplier field name for a regime."""
    return "mult_" + regime.value
