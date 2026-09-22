"""Row models for the dataset's tables.

Re-exports pm_traitbench.enums so table code has a single import path.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm_traitbench.enums import (
    MARKET_WIDE_EVENTS,
    NULL_SURPRISE_EVENTS,
    Action,
    AssetClass,
    CommodityGroup,
    DriftEventType,
    EventType,
    ExpiryRule,
    Family,
    InstrumentKind,
    Kind,
    Op,
    Positioning,
    RatingBand,
    Regime,
    RuleScope,
    RuleSource,
    Split,
    StreetView,
    Tenor,
    Typicality,
)

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
    "Family",
    "InstrumentKind",
    "EventType",
    "StreetView",
    "Positioning",
    "Tenor",
    "CommodityGroup",
    "RatingBand",
    "ExpiryRule",
    "Mandate",
    "StatedProfile",
    "Persona",
    "Trait",
    "Rule",
    "DriftEvent",
    "Instrument",
    "Price",
    "CurvePoint",
    "ConsensusRow",
    "CalendarEvent",
    "RegimeSpan",
    "to_record",
    "multiplier_field",
]

_PM_ID_PATTERN = r"^pm_\d{3,}$"
_TRAIT_ID_PATTERN = r"^t_\d{2,}$"
_RULE_ID_PATTERN = r"^r_\d{2,}$"


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


_INSTRUMENT_OPTIONAL_FIELDS = (
    "sector",
    "rating_band",
    "commodity_group",
    "duration_years",
    "beta",
    "expiry_rule",
)

_INSTRUMENT_KIND_OPTIONAL_FIELDS: dict[InstrumentKind, frozenset[str]] = {
    InstrumentKind.EQUITY: frozenset({"sector", "beta"}),
    InstrumentKind.CREDIT_ISSUER: frozenset({"sector", "rating_band", "duration_years"}),
    InstrumentKind.SOVEREIGN_CURVE: frozenset(),
    InstrumentKind.COMMODITY: frozenset({"commodity_group", "expiry_rule"}),
    InstrumentKind.FX_PAIR: frozenset(),
}

_INSTRUMENT_KIND_FAMILY: dict[InstrumentKind, Family] = {
    InstrumentKind.EQUITY: Family.EQUITIES,
    InstrumentKind.CREDIT_ISSUER: Family.CREDIT,
    InstrumentKind.SOVEREIGN_CURVE: Family.RATES,
    InstrumentKind.COMMODITY: Family.COMMODITIES,
    InstrumentKind.FX_PAIR: Family.FX,
}


class Instrument(BaseModel):
    """A tradable instrument in the market universe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str = Field(description="Unique identifier for the instrument.")
    family: Family = Field(description="Asset family the instrument belongs to.")
    kind: InstrumentKind = Field(description="Instrument's specific kind within its family.")
    name: str = Field(description="Human-readable name of the instrument.")
    currency: str = Field(description="Currency the instrument is priced in.")
    sector: str | None = Field(description="Sector; set for equities and credit issuers only.")
    rating_band: RatingBand | None = Field(
        description="Credit rating band; set for credit issuers only."
    )
    commodity_group: CommodityGroup | None = Field(
        description="Commodity group; set for commodities only."
    )
    duration_years: float | None = Field(
        description="Duration in years; set for credit issuers only."
    )
    beta: float | None = Field(description="Equity beta; set for equities only.")
    expiry_rule: ExpiryRule | None = Field(
        description="Contract expiry rule; set for commodities only."
    )

    @model_validator(mode="after")
    def _check_kind_invariants(self) -> "Instrument":
        expected_family = _INSTRUMENT_KIND_FAMILY[self.kind]
        if self.family != expected_family:
            raise ValueError(
                f"kind '{self.kind.value}' requires family '{expected_family.value}', "
                f"got '{self.family.value}'"
            )
        expected_fields = _INSTRUMENT_KIND_OPTIONAL_FIELDS[self.kind]
        actual_fields = {
            name for name in _INSTRUMENT_OPTIONAL_FIELDS if getattr(self, name) is not None
        }
        if actual_fields != expected_fields:
            raise ValueError(
                f"kind '{self.kind.value}' requires optional fields {sorted(expected_fields)}, "
                f"got {sorted(actual_fields)}"
            )
        return self


class Price(BaseModel):
    """A daily observed price for an instrument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: str = Field(description="Market seed the price belongs to.")
    date: datetime.date = Field(description="Date the price was observed.")
    instrument_id: str = Field(description="Instrument the price belongs to.")
    price: float = Field(gt=0, description="Observed price, or level for spread-quoted credit.")
    spread_bp: float | None = Field(
        description="Credit spread in basis points; set for credit instruments only."
    )

    @model_validator(mode="after")
    def _check_spread(self) -> "Price":
        is_credit = self.instrument_id.startswith("CR-")
        if is_credit and self.spread_bp is None:
            raise ValueError("credit instrument requires spread_bp to be set")
        if not is_credit and self.spread_bp is not None:
            raise ValueError("non-credit instrument requires spread_bp to be null")
        if self.spread_bp is not None and self.spread_bp <= 0:
            raise ValueError("spread_bp must be positive when set")
        return self


class CurvePoint(BaseModel):
    """A single tenor point on a sovereign or futures curve on a given day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: str = Field(description="Market seed the curve point belongs to.")
    date: datetime.date = Field(description="Date the curve point was observed.")
    curve_id: str = Field(description="Curve the point belongs to.")
    tenor: Tenor = Field(description="Tenor of the curve point.")
    level: float = Field(description="Curve level at this tenor: a yield or futures price.")


class ConsensusRow(BaseModel):
    """The street's consensus view and positioning on an instrument on a given day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: str = Field(description="Market seed the consensus row belongs to.")
    date: datetime.date = Field(description="Date the consensus was observed.")
    instrument_id: str = Field(description="Instrument the consensus applies to.")
    street_score: float = Field(ge=-1, le=1, description="Continuous consensus score.")
    street_view: StreetView = Field(description="Categorical consensus view.")
    positioning_pct: float = Field(
        ge=0, le=100, description="Positioning percentile among the street."
    )
    positioning: Positioning = Field(description="Categorical crowding of positioning.")


class CalendarEvent(BaseModel):
    """A scheduled or realised market event on a given day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: str = Field(description="Market seed the event belongs to.")
    date: datetime.date = Field(description="Date the event occurred.")
    instrument_id: str | None = Field(
        description="Instrument the event applies to; null for market-wide events."
    )
    event: EventType = Field(description="Type of calendar event.")
    surprise: float | None = Field(
        ge=-1,
        le=1,
        description="Surprise magnitude relative to consensus; null where the event has none.",
    )
    affected: str = Field(
        description="Family affected by the event, or 'all' for a market-wide event."
    )

    @model_validator(mode="after")
    def _check_event(self) -> "CalendarEvent":
        surprise_is_null = self.event in NULL_SURPRISE_EVENTS
        if surprise_is_null and self.surprise is not None:
            raise ValueError(f"{self.event.value} event requires surprise to be null")
        if not surprise_is_null and self.surprise is None:
            raise ValueError(f"{self.event.value} event requires surprise to be set")

        instrument_is_null = self.event in MARKET_WIDE_EVENTS
        if instrument_is_null and self.instrument_id is not None:
            raise ValueError(f"{self.event.value} event requires instrument_id to be null")
        if not instrument_is_null and self.instrument_id is None:
            raise ValueError(f"{self.event.value} event requires instrument_id to be set")

        valid_affected = {family.value for family in Family} | {"all"}
        if self.affected not in valid_affected:
            raise ValueError(f"affected must be a family or 'all', got '{self.affected}'")
        if (self.affected == "all") != (self.instrument_id is None):
            raise ValueError("affected must be 'all' exactly when instrument_id is null")

        if self.event == EventType.RATING_DOWNGRADE and (
            self.surprise is None or self.surprise >= 0
        ):
            raise ValueError("rating_downgrade event requires surprise < 0")
        if self.event == EventType.RATING_UPGRADE and (self.surprise is None or self.surprise <= 0):
            raise ValueError("rating_upgrade event requires surprise > 0")
        return self


class RegimeSpan(BaseModel):
    """A contiguous date span in which a single market regime applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: str = Field(description="Market seed the regime span belongs to.")
    regime: Regime = Field(description="Regime in effect over the span.")
    date_start: datetime.date = Field(description="First date of the span.")
    date_end: datetime.date = Field(description="Last date of the span.")

    @model_validator(mode="after")
    def _check_span(self) -> "RegimeSpan":
        if self.date_start > self.date_end:
            raise ValueError("date_start must not be after date_end")
        return self


def to_record(row: BaseModel) -> dict[str, Any]:
    """Serialise a row model to a JSON-safe dict, keyed by field alias."""
    return row.model_dump(mode="json", by_alias=True)


def multiplier_field(regime: Regime) -> str:
    """Return the Trait multiplier field name for a regime."""
    return "mult_" + regime.value
