"""Row models for the dataset's tables.

Re-exports pm_traitbench.enums so table code has a single import path.
"""

import datetime
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pm_traitbench.enums import (
    MARKET_WIDE_EVENTS,
    MULTI_LEG_FORMS,
    NULL_SURPRISE_EVENTS,
    Action,
    AdvisorTool,
    AssetClass,
    CheckpointLabel,
    CommodityGroup,
    DriftEventType,
    DriftStatus,
    EventType,
    ExpiryRule,
    Expression,
    Family,
    Gate1Split,
    Gate1Test,
    Gate1Verdict,
    Gate2Slice,
    Gate2Verdict,
    InstrumentKind,
    Kind,
    MentionKind,
    Op,
    OptionSource,
    Ownership,
    PnlState,
    PositionAction,
    Positioning,
    ProbeForm,
    ProbeType,
    RatingBand,
    Regime,
    RuleResponse,
    RuleScope,
    RuleSource,
    Scorer,
    SeedGroupKind,
    SessionKind,
    Side,
    SignalMode,
    Split,
    StanceEntry,
    StreetView,
    Tenor,
    TurnRole,
    Typicality,
    Valence,
    ValidationStatus,
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
    "Side",
    "Expression",
    "RuleResponse",
    "PositionAction",
    "PnlState",
    "Gate1Verdict",
    "Gate1Split",
    "SeedGroupKind",
    "Gate2Verdict",
    "Gate2Slice",
    "SignalMode",
    "Valence",
    "Ownership",
    "SessionKind",
    "StanceEntry",
    "TurnRole",
    "MentionKind",
    "AdvisorTool",
    "ProbeType",
    "Scorer",
    "ProbeForm",
    "CheckpointLabel",
    "OptionSource",
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
    "Leg",
    "Idea",
    "LedgerRow",
    "RuleEvent",
    "PositionDay",
    "Gate1PmRow",
    "Gate1CellRow",
    "Signal",
    "Stance",
    "Skeleton",
    "Turn",
    "Session",
    "Mention",
    "ToolCall",
    "CallUsage",
    "TurnLog",
    "DialogueLog",
    "ValidationRow",
    "Gate2TraitRow",
    "Gate2SignalRow",
    "Gate2PmRow",
    "Gate2CellRow",
    "ProbeRow",
    "ResponseRow",
    "ScoreRow",
    "probe_id",
    "to_record",
    "multiplier_field",
]

_PM_ID_PATTERN = r"^pm_\d{3,}$"
_TRAIT_ID_PATTERN = r"^t_\d{2,}$"
_RULE_ID_PATTERN = r"^r_\d{2,}$"
_IDEA_ID_PATTERN = r"^ti_\d{3,}$"
_BIAS_FLAG_PATTERN = r"^[a-z_]+:[a-z_]+(;[a-z_]+:[a-z_]+)*$"


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


class Leg(BaseModel):
    """A single leg of a trade idea."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str = Field(description="Instrument traded by this leg.")
    tenor: Tenor | None = Field(
        description="Tenor traded by this leg; set for curve and calendar-spread legs only."
    )
    side: Side = Field(description="Direction of this leg.")
    weight: float = Field(gt=0, description="Relative weight of this leg within the idea.")


class Idea(BaseModel):
    """A PM's trade idea: its thesis, sizing context and eventual outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM who owns the idea."
    )
    trade_idea_id: str = Field(
        pattern=_IDEA_ID_PATTERN, description="Unique identifier for the trade idea."
    )
    instrument_id: str = Field(description="Primary instrument the idea trades.")
    expression: Expression = Field(description="Structural form of the idea.")
    side: Side = Field(description="Overall direction of the idea.")
    legs: tuple[Leg, ...] = Field(min_length=1, description="Legs making up the idea.")
    entry_date: datetime.date = Field(description="Date the idea was entered.")
    exit_date: datetime.date | None = Field(
        description="Date the idea was exited; null while the idea is open."
    )
    entry_level: float = Field(description="Level of the primary instrument at entry.")
    target_level: float = Field(description="Target level for the idea.")
    stop_level: float = Field(description="Stop level for the idea.")
    thesis: str = Field(description="Free-text statement of the idea's thesis.")
    outcome: str | None = Field(
        description="Free-text outcome of the idea; null while the idea is open."
    )
    own_signal: float = Field(description="PM's own signal strength for the idea.")
    forecast: float = Field(description="PM's point forecast for the primary instrument.")
    interval_lo: float = Field(description="Lower bound of the PM's forecast interval.")
    interval_hi: float = Field(description="Upper bound of the PM's forecast interval.")
    street_view_at_entry: StreetView = Field(
        description="Street's consensus view when the idea was entered."
    )
    conflict: bool = Field(description="Whether the idea conflicts with the street's view.")
    followed_street: bool | None = Field(
        description="Whether the PM followed the street; set only when conflict is true."
    )
    conviction: int = Field(ge=1, le=5, description="PM's stated conviction level.")
    size_rank: int = Field(
        ge=1, le=5, description="Idea's size rank relative to the PM's other ideas."
    )
    chased_trend: bool = Field(
        description=(
            "Whether the entry is on the side of a trailing move already past one "
            "standard deviation of a horizon move."
        )
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> "Idea":
        if not self.interval_lo < self.interval_hi:
            raise ValueError("interval_lo must be less than interval_hi")
        if self.conflict != (self.followed_street is not None):
            raise ValueError("followed_street must be set exactly when conflict is true")
        if (self.exit_date is None) != (self.outcome is None):
            raise ValueError("outcome must be null exactly when exit_date is null")
        if self.exit_date is not None and self.exit_date < self.entry_date:
            raise ValueError("exit_date must not be before entry_date")

        is_multi_leg = self.expression in MULTI_LEG_FORMS
        expected_legs = 2 if is_multi_leg else 1
        if len(self.legs) != expected_legs:
            raise ValueError(
                f"expression '{self.expression.value}' requires {expected_legs} leg(s), "
                f"got {len(self.legs)}"
            )
        requires_tenor = self.expression in (Expression.CURVE, Expression.CALENDAR_SPREAD)
        for leg in self.legs:
            if requires_tenor and leg.tenor is None:
                raise ValueError(
                    f"expression '{self.expression.value}' requires every leg to have a tenor"
                )
            if not requires_tenor and leg.tenor is not None:
                raise ValueError(
                    f"expression '{self.expression.value}' requires every leg to have no tenor"
                )
        return self


class LedgerRow(BaseModel):
    """A single order placed against a trade idea."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM who placed the order."
    )
    date: datetime.date = Field(description="Date the order was placed.")
    trade_idea_id: str = Field(
        pattern=_IDEA_ID_PATTERN, description="Trade idea the order belongs to."
    )
    instrument_id: str = Field(description="Instrument traded by the order.")
    tenor: Tenor | None = Field(
        description="Tenor traded by the order; set for curve and calendar-spread instruments only."
    )
    instrument_type: InstrumentKind = Field(description="Kind of instrument traded.")
    side: Side = Field(description="Direction of the order.")
    size: float = Field(gt=0, description="Size of the order.")
    risk_amount: float = Field(gt=0, description="Risk amount consumed by the order.")
    price_or_yield: float = Field(description="Execution price or yield.")
    stated_conviction: int = Field(
        ge=1, le=5, description="Conviction stated at the time of the order."
    )
    bias_flag: str | None = Field(
        pattern=_BIAS_FLAG_PATTERN,
        description="Semicolon-separated bias:pattern tags detected on this order.",
    )
    rule_id: str | None = Field(
        pattern=_RULE_ID_PATTERN, description="Rule that drove this order, if any."
    )


class RuleEvent(BaseModel):
    """A rule firing and how the PM responded to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the rule fired for."
    )
    rule_id: str = Field(pattern=_RULE_ID_PATTERN, description="Rule that fired.")
    trade_idea_id: str = Field(
        pattern=_IDEA_ID_PATTERN, description="Trade idea the rule fired against."
    )
    date_fired: datetime.date = Field(description="Date the rule's condition was met.")
    response: RuleResponse = Field(description="How the PM responded to the rule firing.")
    response_date: datetime.date = Field(description="Date the PM's response was recorded.")

    @model_validator(mode="after")
    def _check_response_date(self) -> "RuleEvent":
        if self.response_date < self.date_fired:
            raise ValueError("response_date must not be before date_fired")
        return self


class PositionDay(BaseModel):
    """A trade idea's daily mark-to-market and position state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM holding the position."
    )
    date: datetime.date = Field(description="Date of this position snapshot.")
    trade_idea_id: str = Field(
        pattern=_IDEA_ID_PATTERN, description="Trade idea this snapshot belongs to."
    )
    pnl_unit: float = Field(description="Mark-to-market P&L in the mandate's risk unit.")
    pnl_z: float = Field(description="Mark-to-market P&L expressed as a z-score.")
    pnl_state: PnlState = Field(description="Categorical P&L state derived from pnl_z.")
    sessions_held: int = Field(ge=0, description="Number of sessions the idea has been held.")
    triggers_fired: int = Field(
        ge=0, description="Rule triggers fired on this idea so far, this day included."
    )
    trigger_pending: bool = Field(description="Whether a triggered rule is awaiting a response.")
    action: PositionAction = Field(description="Position action taken on this day.")
    bias_flag: str | None = Field(
        pattern=_BIAS_FLAG_PATTERN,
        description="Semicolon-separated bias:pattern tags detected on this day.",
    )
    anchor_level: float | None = Field(
        description="The idea's fixed round-level anchor; null if none qualified at entry."
    )
    effective_exit_level: float | None = Field(
        description="The idea's anchor when anchored, else its target level."
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> "PositionDay":
        if self.effective_exit_level is None and self.anchor_level is not None:
            raise ValueError("anchor_level requires effective_exit_level to be set")
        is_flat = abs(self.pnl_z) < 1e-9
        if is_flat != (self.pnl_state == PnlState.FLAT):
            raise ValueError("pnl_state must be 'flat' exactly when abs(pnl_z) < 1e-9")
        return self


class Gate1PmRow(BaseModel):
    """A single PM's recovered statistic for one bias parameter and split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(pattern=_PM_ID_PATTERN, description="Identifier of the PM the row covers.")
    param: str = Field(description="Name of the bias parameter the statistic estimates.")
    split: Gate1Split = Field(description="Window of the ledger the statistic was computed over.")
    seed: str = Field(description="Market seed the row's ledger data comes from.")
    asset_class: AssetClass = Field(description="Asset class of the PM's mandate.")
    statistic: float | None = Field(
        description="Estimator's recovered value; null when the data was too thin to compute."
    )
    n: int = Field(
        ge=0, description="Number of ledger observations the statistic was computed from."
    )
    planted: float = Field(description="Trait's sampled value that was planted on the PM.")
    active: bool = Field(description="Whether the bias was active on the PM.")
    drifted: bool = Field(description="Whether the trait had a drift event during the run.")


class Gate1CellRow(BaseModel):
    """A neutral-versus-active comparison for one bias parameter, asset class (or every
    direct asset class pooled) and split.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed_group: str = Field(description="Identifier of the seed group the cell aggregates over.")
    seed_group_kind: SeedGroupKind = Field(
        description="Kind of seed group the cell aggregates over."
    )
    asset_class: AssetClass | None = Field(
        description="Asset class the cell covers; null pools every direct asset class."
    )
    param: str = Field(description="Name of the bias parameter the cell compares.")
    split: Gate1Split = Field(description="Window of the ledger the cell was computed over.")
    n_neutral: int = Field(ge=0, description="Number of neutral PMs contributing to the cell.")
    n_active: int = Field(ge=0, description="Number of active PMs contributing to the cell.")
    n_missing: int = Field(
        ge=0, description="Number of PMs excluded from the cell for lacking a usable statistic."
    )
    neutral_mean: float | None = Field(description="Mean statistic across neutral PMs.")
    neutral_sd: float | None = Field(
        description="Standard deviation of the statistic across neutral PMs."
    )
    active_mean: float | None = Field(description="Mean statistic across active PMs.")
    floor: float | None = Field(
        description=(
            "Neutral mean plus or minus floor_se neutral sds, in the strengthening direction."
        )
    )
    active_share_past_floor: float | None = Field(
        description="Share of active PMs whose statistic clears the floor."
    )
    rank_corr: float | None = Field(
        description=(
            "Rank correlation between planted strength and the recovered statistic, "
            "over neutral and active PMs together; report-only."
        )
    )
    active_rank_corr: float | None = Field(
        description=(
            "Rank correlation between planted strength and the recovered statistic "
            "over active PMs only; the value the rank check uses."
        )
    )
    count_p10: float | None = Field(
        description="10th percentile of the cell's PMs' estimator opportunity counts (Estimate.n)."
    )
    calibration: float | None = Field(
        description=(
            "Pooled Pearson correlation of entry direction with the trailing move in horizon "
            "sds; extrapolation only, null for every other parameter."
        )
    )
    test: Gate1Test = Field(description="Which rule judged the cell: per-PM or population.")
    gap_ok: bool = Field(description="Whether the neutral-to-active gap check passed.")
    rank_ok: bool = Field(description="Whether the rank correlation check passed.")
    pop_z: float | None = Field(
        description=(
            "Standard errors of the difference by which the active mean exceeds the neutral "
            "mean, in the parameter's own direction; null when either set has fewer than 2 "
            "values or the standard error is 0."
        )
    )
    pop_ok: bool = Field(
        description="Whether the population check (pop_z at or above its minimum) passed."
    )
    count_ok: bool | None = Field(
        description="Whether the observation count check passed; null when not evaluated."
    )
    count_shortfall: bool = Field(
        description=(
            "For a single-seed cell, whether its own count fell short of the estimator's "
            "minimum; for a pooled cell, whether any single synthetic seed's count did."
        )
    )
    verdict: Gate1Verdict = Field(description="The cell's recovery verdict.")
    blocking: bool = Field(
        description=(
            "Whether a failing verdict on this cell blocks the gate; always false for a "
            "report-only parameter."
        )
    )


_SIGNAL_ID_PATTERN = r"^sg_\d{3,}$"
_SESSION_ID_PATTERN = r"^s_pm\d{3,}_\d{4}-\d{2}-\d{2}_[a-z]$"
_IDEA_ID_RE = re.compile(_IDEA_ID_PATTERN)


def _session_pm_prefix(pm_id: str) -> str:
    return f"s_{pm_id.replace('_', '')}_"


def _session_prefix(pm_id: str, date: datetime.date) -> str:
    return f"{_session_pm_prefix(pm_id)}{date.isoformat()}_"


def _check_idea_id_tuple(trade_idea_ids: tuple[str, ...]) -> None:
    """Raise unless every id matches the idea-id pattern and the tuple is sorted and unique."""
    for trade_idea_id in trade_idea_ids:
        if not _IDEA_ID_RE.fullmatch(trade_idea_id):
            raise ValueError(
                f"trade_idea_ids must match {_IDEA_ID_PATTERN!r}, got '{trade_idea_id}'"
            )
    if list(trade_idea_ids) != sorted(set(trade_idea_ids)):
        raise ValueError("trade_idea_ids must be sorted and unique")


def _session_date(session_id: str) -> datetime.date:
    """Parse the YYYY-MM-DD segment out of a session id."""
    return datetime.date.fromisoformat(session_id.split("_")[2])


class Signal(BaseModel):
    """A single trait signal planted on a dated session for a later narrator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(
        pattern=_SIGNAL_ID_PATTERN, description="Unique identifier for the signal."
    )
    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the signal belongs to."
    )
    session_id: str = Field(pattern=_SESSION_ID_PATTERN, description="Session the signal sits on.")
    date: datetime.date = Field(description="Date of the session the signal sits on.")
    trait_id: str = Field(pattern=_TRAIT_ID_PATTERN, description="Trait the signal expresses.")
    mode: SignalMode = Field(description="How the signal expresses its trait.")
    trade_idea_id: str | None = Field(
        pattern=_IDEA_ID_PATTERN,
        description="Trade idea the signal points at; required for a contradiction.",
    )
    valence: Valence = Field(description="Whether the signal confirms or retracts its trait.")
    ownership: Ownership = Field(description="Who the signal is attributed to.")
    third_party_value: str | None = Field(
        description="The third party's stated value; set only when ownership is not 'self'."
    )
    claim_session_id: str | None = Field(
        pattern=_SESSION_ID_PATTERN,
        description="Earlier session a contradiction claims against; null otherwise.",
    )

    @model_validator(mode="after")
    def _check_session(self) -> "Signal":
        if not self.session_id.startswith(_session_prefix(self.pm_id, self.date)):
            raise ValueError("session_id must sit on the signal's own pm and date")
        return self

    @model_validator(mode="after")
    def _check_claim(self) -> "Signal":
        has_claim = self.claim_session_id is not None
        if has_claim != (self.mode == SignalMode.CONTRADICTION):
            raise ValueError("claim_session_id must be set exactly when mode is 'contradiction'")
        if has_claim:
            if not self.claim_session_id.startswith(_session_pm_prefix(self.pm_id)):
                raise ValueError("claim_session_id must belong to the signal's own pm")
            if _session_date(self.claim_session_id) >= self.date:
                raise ValueError("claim_session_id must be dated earlier than date")
        return self

    @model_validator(mode="after")
    def _check_trade_idea(self) -> "Signal":
        if self.mode == SignalMode.CONTRADICTION and self.trade_idea_id is None:
            raise ValueError("mode 'contradiction' requires trade_idea_id to be set")
        return self

    @model_validator(mode="after")
    def _check_ownership(self) -> "Signal":
        if self.ownership != Ownership.SELF and (
            self.mode != SignalMode.STATED or self.valence != Valence.CONFIRM
        ):
            raise ValueError(
                "ownership other than 'self' requires mode 'stated' and valence 'confirm'"
            )
        if self.third_party_value is not None and self.ownership == Ownership.SELF:
            raise ValueError("third_party_value requires ownership other than 'self'")
        return self


class Stance(BaseModel):
    """A single rendered stance line making up part of a session's skeleton."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(pattern=_SIGNAL_ID_PATTERN, description="Signal this stance renders.")
    trait_id: str = Field(pattern=_TRAIT_ID_PATTERN, description="Trait the stance expresses.")
    mode: SignalMode = Field(description="How the stance expresses its trait.")
    entry: StanceEntry = Field(description="Kind of stance entry.")
    stance: str = Field(min_length=1, description="Free-text rendering of the stance.")


class Skeleton(BaseModel):
    """The planted shape of a single PM session: its stances and the ideas it raises."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(
        pattern=_SESSION_ID_PATTERN, description="Unique identifier for the session."
    )
    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the session belongs to."
    )
    date: datetime.date = Field(description="Date of the session.")
    kind: SessionKind = Field(description="Kind of session.")
    trade_idea_ids: tuple[str, ...] = Field(
        description="Trade ideas raised in the session, sorted and unique."
    )
    stances: tuple[Stance, ...] = Field(description="Stances rendered in the session.")
    advisor_violation: str | None = Field(
        description="Free-text advisor violation; set exactly when a stance reveals one."
    )
    forbidden_trait_ids: tuple[str, ...] = Field(
        description="Traits the session must not surface a stance for."
    )
    forbidden_pref_params: tuple[str, ...] = Field(
        description="Preference params the session must not surface a stance for."
    )

    @model_validator(mode="after")
    def _check_session(self) -> "Skeleton":
        if not self.session_id.startswith(_session_prefix(self.pm_id, self.date)):
            raise ValueError("session_id must sit on the session's own pm and date")
        return self

    @model_validator(mode="after")
    def _check_silence(self) -> "Skeleton":
        if self.kind == SessionKind.SILENCE and (
            self.stances != () or self.trade_idea_ids != () or self.advisor_violation is not None
        ):
            raise ValueError(
                "kind 'silence' requires no stances, trade_idea_ids or advisor_violation"
            )
        return self

    @model_validator(mode="after")
    def _check_stance_traits_unique(self) -> "Skeleton":
        trait_ids = [stance.trait_id for stance in self.stances]
        if len(trait_ids) != len(set(trait_ids)):
            raise ValueError("stance trait_ids must be unique within a session")
        return self

    @model_validator(mode="after")
    def _check_advisor_violation(self) -> "Skeleton":
        has_reaction = any(stance.entry == StanceEntry.REVEALED_REACTION for stance in self.stances)
        if has_reaction != (self.advisor_violation is not None):
            raise ValueError(
                "advisor_violation must be set exactly when a stance has entry 'revealed_reaction'"
            )
        return self

    @model_validator(mode="after")
    def _check_trade_idea_ids(self) -> "Skeleton":
        _check_idea_id_tuple(self.trade_idea_ids)
        return self


_REQUEST_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _check_turns_alternate(turns: "tuple[Turn, ...] | tuple[TurnLog, ...]") -> None:
    """Raise unless turns has an even length in [2, 8] and alternates pm, advisor, pm, ..."""
    if len(turns) % 2 != 0 or not (2 <= len(turns) <= 8):
        raise ValueError("turns must have an even length between 2 and 8")
    for index, turn in enumerate(turns):
        expected = TurnRole.PM if index % 2 == 0 else TurnRole.ADVISOR
        if turn.role != expected:
            raise ValueError("turns must alternate starting with pm and ending with advisor")


class Turn(BaseModel):
    """One turn of a session's public transcript: who spoke and what they said."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: TurnRole = Field(description="Who spoke this turn.")
    text: str = Field(min_length=1, description="The turn's spoken text.")


class Session(BaseModel):
    """A dated PM-copilot session: the public transcript of a two-agent dialogue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(
        pattern=_SESSION_ID_PATTERN, description="Unique identifier for the session."
    )
    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the session belongs to."
    )
    date: datetime.date = Field(description="Date of the session.")
    kind: SessionKind = Field(description="Kind of session.")
    trade_idea_ids: tuple[str, ...] = Field(
        description="Trade ideas raised in the session, sorted and unique."
    )
    turns: tuple[Turn, ...] = Field(description="The session's public transcript.")

    @model_validator(mode="after")
    def _check_session(self) -> "Session":
        if not self.session_id.startswith(_session_prefix(self.pm_id, self.date)):
            raise ValueError("session_id must sit on the session's own pm and date")
        return self

    @model_validator(mode="after")
    def _check_turns(self) -> "Session":
        _check_turns_alternate(self.turns)
        return self

    @model_validator(mode="after")
    def _check_trade_idea_ids(self) -> "Session":
        _check_idea_id_tuple(self.trade_idea_ids)
        return self


class Mention(BaseModel):
    """A single trade or market level mentioned in a turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: MentionKind = Field(description="Whether the mention is a trade or a market level.")
    instrument_id: str = Field(description="Instrument the mention refers to.")
    trade_idea_id: str | None = Field(
        pattern=_IDEA_ID_PATTERN, description="Trade idea mentioned; set only for a trade mention."
    )
    tenor: Tenor | None = Field(description="Tenor mentioned, if any.")
    side: Side | None = Field(description="Side mentioned; set only for a trade mention.")
    size: float | None = Field(gt=0, description="Size mentioned; set only for a trade mention.")
    field: str | None = Field(description="Data field mentioned; set only for a level mention.")
    value: float | None = Field(description="Value mentioned; set only for a level mention.")

    @model_validator(mode="after")
    def _check_kind(self) -> "Mention":
        if self.kind == MentionKind.TRADE:
            if self.trade_idea_id is None or self.side is None or self.size is None:
                raise ValueError("trade mention requires trade_idea_id, side and size")
            if self.field is not None or self.value is not None:
                raise ValueError("trade mention requires field and value to be null")
        else:
            if self.field is None or self.value is None:
                raise ValueError("level mention requires field and value")
            if self.trade_idea_id is not None or self.side is not None or self.size is not None:
                raise ValueError("level mention requires trade_idea_id, side and size to be null")
        return self


def canonical_json(value: Any) -> str:
    """One JSON serialisation for `value`: sorted keys, no extra whitespace.

    The dialogue session log and the `tool_result` block sent back to the
    model both serialise a tool call's input and result through this, so a
    given value always renders the same string.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ToolCall(BaseModel):
    """One advisor tool invocation and its result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: AdvisorTool = Field(description="Tool invoked.")
    input_json: str = Field(
        description=(
            "Tool call input, as canonical JSON: "
            "json.dumps(value, sort_keys=True, separators=(',', ':'))."
        )
    )
    result_json: str = Field(
        description=(
            "Tool call result, as canonical JSON: "
            "json.dumps(value, sort_keys=True, separators=(',', ':'))."
        )
    )
    is_error: bool = Field(description="Whether the tool call resulted in an error.")

    @field_validator("input_json", "result_json")
    @classmethod
    def _check_canonical_json(cls, v: str) -> str:
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"must be valid JSON: {e}") from e
        if canonical_json(parsed) != v:
            raise ValueError("must be canonical JSON (sorted keys, no extra whitespace)")
        return v


class CallUsage(BaseModel):
    """Token usage summed over a turn's API calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0, description="Fresh input tokens billed.")
    output_tokens: int = Field(ge=0, description="Fresh output tokens billed.")
    cache_read_input_tokens: int = Field(ge=0, description="Input tokens read from cache.")
    cache_creation_input_tokens: int = Field(ge=0, description="Input tokens written to cache.")


class TurnLog(BaseModel):
    """The hidden log of one turn: its mentions, directive, tool use and call metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: TurnRole = Field(description="Who spoke this turn.")
    text: str = Field(description="The turn's spoken text.")
    mentions: tuple[Mention, ...] = Field(description="Trades and levels mentioned this turn.")
    directive: str | None = Field(
        description="The stance, opening or violation line the turn carried, if any."
    )
    scripted_violation: bool = Field(
        description="Whether this is the advisor turn that carried the scripted violation."
    )
    tool_calls: tuple[ToolCall, ...] = Field(description="Tool calls made this turn.")
    model: str = Field(description="Model that generated this turn.")
    request_hashes: tuple[str, ...] = Field(
        min_length=1, description="Cache keys of the requests that produced this turn."
    )
    usage: CallUsage = Field(description="Token usage summed over the turn's calls.")

    @model_validator(mode="after")
    def _check_request_hashes(self) -> "TurnLog":
        for request_hash in self.request_hashes:
            if not _REQUEST_HASH_PATTERN.fullmatch(request_hash):
                raise ValueError(
                    f"request_hashes entries must be 64 lowercase hex characters, "
                    f"got '{request_hash}'"
                )
        return self

    @model_validator(mode="after")
    def _check_pm_turn(self) -> "TurnLog":
        if self.role == TurnRole.PM and (self.tool_calls != () or self.scripted_violation):
            raise ValueError("pm turn requires empty tool_calls and scripted_violation False")
        return self


class DialogueLog(BaseModel):
    """The hidden per-turn log of a session: voice, mentions, tool use and call metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(
        pattern=_SESSION_ID_PATTERN, description="Unique identifier for the session."
    )
    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the session belongs to."
    )
    voice_id: str = Field(description="Voice drawn for the PM this session was narrated in.")
    turns: tuple[TurnLog, ...] = Field(description="The session's hidden per-turn log.")

    @model_validator(mode="after")
    def _check_session_id(self) -> "DialogueLog":
        if not self.session_id.startswith(_session_pm_prefix(self.pm_id)):
            raise ValueError("session_id must belong to the log's own pm")
        return self

    @model_validator(mode="after")
    def _check_turns(self) -> "DialogueLog":
        _check_turns_alternate(self.turns)
        return self

    @model_validator(mode="after")
    def _check_scripted_violation_count(self) -> "DialogueLog":
        count = sum(turn.scripted_violation for turn in self.turns)
        if count > 1:
            raise ValueError("at most one turn may have scripted_violation True")
        return self


class ValidationRow(BaseModel):
    """A judged validation outcome for one narrated dialogue session attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the session belongs to."
    )
    session_id: str = Field(pattern=_SESSION_ID_PATTERN, description="Session the row validates.")
    attempt: int = Field(ge=1, description="Attempt number the row validates, starting at 1.")
    status: ValidationStatus = Field(description="Overall validation outcome for the attempt.")
    ledger_ok: bool = Field(
        description="Whether the session's mentions are consistent with the ledger."
    )
    level_ok: bool = Field(
        description="Whether every PM level mention matches the market data on the session date."
    )
    grep_ok: bool = Field(description="Whether the session passed the deterministic grep checks.")
    leak_judged: bool = Field(description="Whether a judge assessed the session for a trait leak.")
    leak_ok: bool = Field(description="Whether the session passed the leak check.")
    forbidden_ok: bool = Field(description="Whether the session avoided its rendered avoid lines.")
    stance_ok: bool = Field(description="Whether every stanced PM turn carried its stance out.")
    level_warnings: int = Field(
        ge=0, description="Number of advisor level mentions no tool result confirmed."
    )
    reasons: tuple[str, ...] = Field(description="Free-text reasons the attempt did not pass.")
    judge_model: str = Field(min_length=1, description="Model that judged the attempt.")

    @model_validator(mode="after")
    def _check_session_id(self) -> "ValidationRow":
        if not self.session_id.startswith(_session_pm_prefix(self.pm_id)):
            raise ValueError("session_id must belong to the row's own pm")
        return self

    @model_validator(mode="after")
    def _check_status(self) -> "ValidationRow":
        all_ok = (
            self.ledger_ok
            and self.level_ok
            and self.grep_ok
            and self.leak_ok
            and self.forbidden_ok
            and self.stance_ok
        )
        if (self.status == ValidationStatus.PASS) != all_ok:
            raise ValueError("status must be 'pass' exactly when every check passes")
        return self

    @model_validator(mode="after")
    def _check_reasons(self) -> "ValidationRow":
        if (self.status != ValidationStatus.PASS) != bool(self.reasons):
            raise ValueError("reasons must be non-empty exactly when status is not 'pass'")
        return self

    @model_validator(mode="after")
    def _check_unjudged_leak(self) -> "ValidationRow":
        if not self.leak_judged and not self.leak_ok:
            raise ValueError("leak_ok must be true when leak_judged is false")
        return self


class Gate2TraitRow(BaseModel):
    """A PM's per-trait ground truth versus what the recovery model predicted for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(pattern=_PM_ID_PATTERN, description="Identifier of the PM the row covers.")
    param: str = Field(description="Name of the underlying trait parameter.")
    trait_id: str | None = Field(
        pattern=_TRAIT_ID_PATTERN,
        description="Trait identifier; set for a bias, and for a preference that is held.",
    )
    kind: Kind = Field(description="Whether the trait is a bias or a preference.")
    truth_active: bool | None = Field(
        description="Whether the bias was truly active; null for a preference."
    )
    truth_value: str | None = Field(
        description="Preference's true value; null for a bias or when not held."
    )
    predicted_active: bool | None = Field(
        description="Whether the recovery model predicted the bias active; null for a preference."
    )
    predicted_value: str | None = Field(
        description="Preference value the recovery model predicted; null for a bias or when "
        "predicted not held."
    )
    correct: bool = Field(description="Whether the prediction matched the truth.")
    cited_session_ids: tuple[str, ...] = Field(
        description="Sessions the recovery model cited as evidence, sorted and unique."
    )
    false_attribution_ids: tuple[str, ...] = Field(
        description="Cited sessions that do not in fact carry evidence for this trait."
    )

    @model_validator(mode="after")
    def _check_bias_kind(self) -> "Gate2TraitRow":
        if self.kind != Kind.BIAS:
            return self
        if self.trait_id is None:
            raise ValueError("kind 'bias' requires trait_id to be set")
        if self.truth_active is None or self.predicted_active is None:
            raise ValueError("kind 'bias' requires truth_active and predicted_active to be set")
        if self.truth_value is not None or self.predicted_value is not None:
            raise ValueError("kind 'bias' requires truth_value and predicted_value to be null")
        return self

    @model_validator(mode="after")
    def _check_preference_kind(self) -> "Gate2TraitRow":
        if self.kind == Kind.PREFERENCE and (
            self.truth_active is not None or self.predicted_active is not None
        ):
            raise ValueError(
                "kind 'preference' requires truth_active and predicted_active to be null"
            )
        return self

    @model_validator(mode="after")
    def _check_preference_trait_id(self) -> "Gate2TraitRow":
        if self.kind == Kind.PREFERENCE and (self.trait_id is not None) != (
            self.truth_value is not None
        ):
            raise ValueError(
                "kind 'preference' requires trait_id to be set exactly when truth_value is set"
            )
        return self

    @model_validator(mode="after")
    def _check_cited_session_ids(self) -> "Gate2TraitRow":
        prefix = _session_pm_prefix(self.pm_id)
        for session_id in self.cited_session_ids:
            if not session_id.startswith(prefix):
                raise ValueError("cited_session_ids must belong to the row's own pm")
        if list(self.cited_session_ids) != sorted(set(self.cited_session_ids)):
            raise ValueError("cited_session_ids must be sorted and unique")
        return self

    @model_validator(mode="after")
    def _check_false_attribution_ids(self) -> "Gate2TraitRow":
        if list(self.false_attribution_ids) != sorted(set(self.false_attribution_ids)):
            raise ValueError("false_attribution_ids must be sorted and unique")
        if not set(self.false_attribution_ids) <= set(self.cited_session_ids):
            raise ValueError("false_attribution_ids must be a subset of cited_session_ids")
        return self


class Gate2SignalRow(BaseModel):
    """One planted signal and how the recovery model cited, recovered and classified it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the signal belongs to."
    )
    signal_id: str = Field(
        pattern=_SIGNAL_ID_PATTERN, description="Signal identifier the row reports on."
    )
    session_id: str = Field(pattern=_SESSION_ID_PATTERN, description="Session the signal sits on.")
    trait_id: str = Field(pattern=_TRAIT_ID_PATTERN, description="Trait the signal expresses.")
    param: str = Field(description="Name of the underlying trait parameter.")
    kind: Kind = Field(description="Whether the trait is a bias or a preference.")
    mode: SignalMode = Field(description="How the signal expresses its trait.")
    valence: Valence = Field(description="Whether the signal confirms or retracts its trait.")
    ownership: Ownership = Field(description="Who the signal is attributed to.")
    pre_update: bool = Field(description="Whether the signal predates a drift update to its trait.")
    cited: bool = Field(description="Whether the recovery model cited this signal as evidence.")
    recovered: bool = Field(
        description="Whether citing this signal contributed to a correct prediction."
    )
    classified: bool = Field(
        description="Whether the classify step assigned this signal a predicted kind."
    )
    kind_predicted: Kind | None = Field(
        description="Kind the classify step predicted for this signal; null when unclassified."
    )
    kind_ok: bool | None = Field(
        description="Whether the predicted kind is correct; null unless classified."
    )

    @model_validator(mode="after")
    def _check_recovered_implies_cited(self) -> "Gate2SignalRow":
        if self.recovered and not self.cited:
            raise ValueError("recovered requires cited to be true")
        return self

    @model_validator(mode="after")
    def _check_kind_ok_set_exactly_when_classified(self) -> "Gate2SignalRow":
        if self.classified != (self.kind_ok is not None):
            raise ValueError("kind_ok must be set exactly when classified is true")
        return self

    @model_validator(mode="after")
    def _check_kind_predicted_implies_classified(self) -> "Gate2SignalRow":
        if self.kind_predicted is not None and not self.classified:
            raise ValueError("kind_predicted requires classified to be true")
        return self

    @model_validator(mode="after")
    def _check_unpredicted_classification_fails(self) -> "Gate2SignalRow":
        if self.classified and self.kind_predicted is None and self.kind_ok is not False:
            raise ValueError("classified with no kind_predicted requires kind_ok to be false")
        return self

    @model_validator(mode="after")
    def _check_kind_ok_matches_prediction(self) -> "Gate2SignalRow":
        if self.kind_predicted is not None and self.kind_ok != (self.kind_predicted == self.kind):
            raise ValueError(
                "kind_ok must equal (kind_predicted == kind) when kind_predicted is set"
            )
        return self

    @model_validator(mode="after")
    def _check_pre_update_requires_preference(self) -> "Gate2SignalRow":
        if self.pre_update and self.kind != Kind.PREFERENCE:
            raise ValueError("pre_update requires kind to be 'preference'")
        return self

    @model_validator(mode="after")
    def _check_session_id(self) -> "Gate2SignalRow":
        if not self.session_id.startswith(_session_pm_prefix(self.pm_id)):
            raise ValueError("session_id must belong to the row's own pm")
        return self


class Gate2PmRow(BaseModel):
    """A single PM's Gate 2 recovery counts and context summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pm_id: str = Field(pattern=_PM_ID_PATTERN, description="Identifier of the PM the row covers.")
    asset_class: AssetClass = Field(description="Asset class of the PM's mandate.")
    typicality: Typicality = Field(description="Whether the PM is typical or anti-typical.")
    drift: DriftStatus = Field(description="Whether the PM's traits drifted during the run.")
    seed: str = Field(description="Market seed the PM's scenario draws from.")
    sessions: int = Field(ge=1, description="Number of sessions in the PM's context.")
    context_chars: int = Field(ge=0, description="Character count of the PM's context.")
    ngram_containment: float = Field(
        ge=0,
        le=1,
        description="Share of this PM's word n-grams that the nearest other PM's PM turns "
        "also contain.",
    )
    biases_correct: int = Field(
        ge=0, le=8, description="Number of the PM's 8 biases correctly predicted."
    )
    preferences_held: int = Field(ge=0, description="Number of preferences truly held.")
    preferences_correct: int = Field(
        ge=0, description="Number of held preferences correctly predicted."
    )
    stated_signals: int = Field(ge=0, description="Number of stated-mode signals classified.")
    stated_kind_ok: int = Field(
        ge=0, description="Number of stated-mode signals with a correct predicted kind."
    )

    @model_validator(mode="after")
    def _check_preferences_correct_bounded(self) -> "Gate2PmRow":
        if self.preferences_correct > self.preferences_held:
            raise ValueError("preferences_correct must not exceed preferences_held")
        return self

    @model_validator(mode="after")
    def _check_stated_kind_ok_bounded(self) -> "Gate2PmRow":
        if self.stated_kind_ok > self.stated_signals:
            raise ValueError("stated_kind_ok must not exceed stated_signals")
        return self


class Gate2CellRow(BaseModel):
    """A recovery-rate comparison for one slice of the PM population, optionally by parameter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice: Gate2Slice = Field(description="Which grouping of PMs the cell covers.")
    slice_value: str = Field(min_length=1, description="Value of the slice this cell covers.")
    param: str | None = Field(
        description="Trait parameter the cell covers; null when the cell pools every parameter."
    )
    n: int = Field(ge=0, description="Number of observations contributing to the cell.")
    n_positive: int = Field(ge=0, description="Number of correct or recovered observations.")
    rate: float | None = Field(
        ge=0,
        le=1,
        description="The cell's rate: balanced accuracy on a two-by-two cell, otherwise "
        "n_positive over n; null when n is 0.",
    )
    chance: float | None = Field(description="Chance-level rate the cell is tested against.")
    p: float | None = Field(
        ge=0, le=1, description="One-sided exact test p-value; set only for a judged cell."
    )
    verdict: Gate2Verdict | None = Field(description="The cell's recovery verdict, if judged.")
    blocking: bool = Field(description="Whether a failing verdict on this cell blocks the gate.")

    @model_validator(mode="after")
    def _check_n_positive_bounded(self) -> "Gate2CellRow":
        if self.n_positive > self.n:
            raise ValueError("n_positive must not exceed n")
        return self

    @model_validator(mode="after")
    def _check_verdict_p_consistency(self) -> "Gate2CellRow":
        judged = self.verdict in (Gate2Verdict.PASS, Gate2Verdict.FAIL)
        if judged != (self.p is not None):
            raise ValueError("p must be set exactly when verdict is 'pass' or 'fail'")
        return self

    @model_validator(mode="after")
    def _check_blocking_requires_all_slice(self) -> "Gate2CellRow":
        if self.blocking and self.slice != Gate2Slice.ALL:
            raise ValueError("blocking requires slice to be 'all'")
        return self


_PROBE_ID_PATTERN = r"^p_pm\d{3,}_\d{4,}$"
_OPTION_LETTERS = ("A", "B", "C", "D")
_OPEN_TYPES = (ProbeType.IN_SITU, ProbeType.ROUTINE_QUESTION, ProbeType.GOVERNANCE)


def probe_id(pm_id: str, n: int) -> str:
    """Return the identifier of a PM's n-th probe."""
    return f"p_{pm_id.replace('_', '')}_{n:04d}"


def _check_probe_prefix(probe_id: str, pm_id: str) -> None:
    prefix = f"p_{pm_id.replace('_', '')}_"
    if not probe_id.startswith(prefix):
        raise ValueError(f"probe_id must start with '{prefix}'")


class ProbeRow(BaseModel):
    """One question put to a copilot at a checkpoint, with its deterministic ground truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str = Field(pattern=_PROBE_ID_PATTERN, description="Unique identifier for the probe.")
    pm_id: str = Field(
        pattern=_PM_ID_PATTERN, description="Identifier of the PM the probe is asked about."
    )
    checkpoint_date: datetime.date = Field(description="Date of the checkpoint the probe sits at.")
    checkpoint_label: CheckpointLabel = Field(description="Which checkpoint the probe sits at.")
    probe_type: ProbeType = Field(description="Kind of question the probe asks.")
    trait_id: str | None = Field(
        pattern=_TRAIT_ID_PATTERN,
        description="Trait the probe tests; null for a presence negative or a routine question.",
    )
    form: ProbeForm = Field(description="Whether the probe is multiple choice or open.")
    question: str = Field(min_length=1, description="The question text put to the copilot.")
    option_a: str | None = Field(description="Text of option A; null for an open probe.")
    option_b: str | None = Field(description="Text of option B; null for an open probe.")
    option_c: str | None = Field(description="Text of option C; null when absent.")
    option_d: str | None = Field(description="Text of option D; null when absent.")
    answer: str = Field(
        min_length=1, description="Ground-truth answer: an option letter, or the open reference."
    )
    source_a: OptionSource | None = Field(description="Where option A's content comes from.")
    source_b: OptionSource | None = Field(description="Where option B's content comes from.")
    source_c: OptionSource | None = Field(description="Where option C's content comes from.")
    source_d: OptionSource | None = Field(description="Where option D's content comes from.")
    supporting_signal_ids: tuple[str, ...] = Field(
        description="Signals in the corpus that evidence the answer, sorted and unique."
    )
    context_chars: int = Field(
        ge=0,
        description=(
            "Length of the rendered transcript of every surviving session at or before the "
            "checkpoint, as Gate 2 counts it."
        ),
    )

    def _options(self) -> tuple[str | None, ...]:
        return (self.option_a, self.option_b, self.option_c, self.option_d)

    def _sources(self) -> tuple[OptionSource | None, ...]:
        return (self.source_a, self.source_b, self.source_c, self.source_d)

    @model_validator(mode="after")
    def _check_probe_id_prefix(self) -> "ProbeRow":
        _check_probe_prefix(self.probe_id, self.pm_id)
        return self

    @model_validator(mode="after")
    def _check_supporting_signals(self) -> "ProbeRow":
        ids = self.supporting_signal_ids
        for signal_id in ids:
            if not re.fullmatch(_SIGNAL_ID_PATTERN, signal_id):
                raise ValueError(
                    f"supporting_signal_ids must match {_SIGNAL_ID_PATTERN!r}, got '{signal_id}'"
                )
        if list(ids) != sorted(set(ids)):
            raise ValueError("supporting_signal_ids must be sorted and unique")
        return self

    @model_validator(mode="after")
    def _check_options_not_blank(self) -> "ProbeRow":
        if any(text is not None and not text.strip() for text in self._options()):
            raise ValueError("options must not be blank")
        return self

    @model_validator(mode="after")
    def _check_open_has_no_options(self) -> "ProbeRow":
        if self.form == ProbeForm.OPEN and any(
            value is not None for value in (*self._options(), *self._sources())
        ):
            raise ValueError("open probe must have no options or sources")
        return self

    @model_validator(mode="after")
    def _check_presence(self) -> "ProbeRow":
        if self.probe_type != ProbeType.TRAIT_PRESENCE:
            return self
        if self.form != ProbeForm.MCQ:
            raise ValueError("presence probe must be mcq")
        if (self.option_a, self.option_b, self.option_c, self.option_d) != (
            "yes",
            "no",
            None,
            None,
        ):
            raise ValueError("presence probe options must be yes and no")
        if self.source_a is None or any(source is not None for source in self._sources()[1:]):
            raise ValueError("presence probe must set source_a only")
        if self.answer not in ("A", "B"):
            raise ValueError("presence answer must be A or B")
        if (self.answer == "A") != (self.source_a == OptionSource.CURRENT):
            raise ValueError("presence answer must be A exactly when source_a is current")
        return self

    @model_validator(mode="after")
    def _check_trait_mcq(self) -> "ProbeRow":
        if self.probe_type != ProbeType.TRAIT_MCQ or self.form != ProbeForm.MCQ:
            return self
        present = [text is not None for text in self._options()]
        if sum(present) not in (3, 4):
            raise ValueError("mcq must have 3 or 4 options")
        if present != sorted(present, reverse=True):
            raise ValueError("mcq options must be contiguous from option_a")
        texts = [text for text in self._options() if text is not None]
        if len(set(texts)) != len(texts):
            raise ValueError("mcq option texts must be unique")
        if [source is not None for source in self._sources()] != present:
            raise ValueError("mcq source must be set exactly where an option is")
        current = [
            _OPTION_LETTERS[i]
            for i, source in enumerate(self._sources())
            if source == OptionSource.CURRENT
        ]
        if len(current) != 1:
            raise ValueError("mcq must have exactly one current source")
        if self.answer != current[0]:
            raise ValueError("mcq answer must be the letter of the current option")
        return self

    @model_validator(mode="after")
    def _check_open_types_are_open(self) -> "ProbeRow":
        if self.probe_type in _OPEN_TYPES and self.form != ProbeForm.OPEN:
            raise ValueError(f"{self.probe_type.value} probe must be open")
        return self

    @model_validator(mode="after")
    def _check_trait_id_nullability(self) -> "ProbeRow":
        nullable = (ProbeType.TRAIT_PRESENCE, ProbeType.ROUTINE_QUESTION)
        if self.trait_id is None and self.probe_type not in nullable:
            raise ValueError("trait_id may be null only for presence and routine_question probes")
        if self.probe_type == ProbeType.ROUTINE_QUESTION and self.trait_id is not None:
            raise ValueError("routine_question requires a null trait_id")
        return self


def to_record(row: BaseModel) -> dict[str, Any]:
    """Serialise a row model to a JSON-safe dict, keyed by field alias."""
    return row.model_dump(mode="json", by_alias=True)


def multiplier_field(regime: Regime) -> str:
    """Return the Trait multiplier field name for a regime."""
    return "mult_" + regime.value


class ResponseRow(BaseModel):
    """One system's reply to one probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str = Field(pattern=_PROBE_ID_PATTERN, description="Probe the reply answers.")
    pm_id: str = Field(pattern=_PM_ID_PATTERN, description="PM the probe is asked about.")
    response: str = Field(
        description="The system's reply; empty when it returned nothing, which scores as wrong."
    )
    latency_ms: int = Field(ge=0, description="Wall-clock time the system took to answer.")

    @model_validator(mode="after")
    def _check_probe_id_prefix(self) -> "ResponseRow":
        _check_probe_prefix(self.probe_id, self.pm_id)
        return self


class ScoreRow(BaseModel):
    """The verdict on one reply."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str = Field(pattern=_PROBE_ID_PATTERN, description="Probe that was scored.")
    pm_id: str = Field(pattern=_PM_ID_PATTERN, description="PM the probe is asked about.")
    scorer: Scorer = Field(description="Which scorer produced the verdict.")
    correct: bool = Field(description="Whether the reply matched the ground truth.")
    detail: str | None = Field(
        min_length=1, description="Why the reply was marked as it was; null when nothing to add."
    )

    @model_validator(mode="after")
    def _check_probe_id_prefix(self) -> "ScoreRow":
        _check_probe_prefix(self.probe_id, self.pm_id)
        return self
