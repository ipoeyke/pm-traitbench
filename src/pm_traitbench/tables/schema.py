"""Row models for the dataset's tables.

Re-exports pm_traitbench.enums so table code has a single import path.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm_traitbench.enums import (
    MARKET_WIDE_EVENTS,
    MULTI_LEG_FORMS,
    NULL_SURPRISE_EVENTS,
    Action,
    AssetClass,
    CommodityGroup,
    DriftEventType,
    EventType,
    ExpiryRule,
    Expression,
    Family,
    Gate1Split,
    Gate1Test,
    Gate1Verdict,
    InstrumentKind,
    Kind,
    Op,
    PnlState,
    PositionAction,
    Positioning,
    RatingBand,
    Regime,
    RuleResponse,
    RuleScope,
    RuleSource,
    SeedGroupKind,
    Side,
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
    "Side",
    "Expression",
    "RuleResponse",
    "PositionAction",
    "PnlState",
    "Gate1Verdict",
    "Gate1Split",
    "SeedGroupKind",
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
    """A neutral-versus-active comparison for one bias parameter, asset class and split."""

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
        description="Rank correlation between planted strength and the recovered statistic."
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
        description="Whether the population check (pop_z and a positive rank correlation) passed."
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


def to_record(row: BaseModel) -> dict[str, Any]:
    """Serialise a row model to a JSON-safe dict, keyed by field alias."""
    return row.model_dump(mode="json", by_alias=True)


def multiplier_field(regime: Regime) -> str:
    """Return the Trait multiplier field name for a regime."""
    return "mult_" + regime.value
