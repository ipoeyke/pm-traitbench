"""Catalogue content models: preferences, rule templates, mandate sub-styles, phrasings,
and signpost and thesis templates.

All frozen and reject unknown fields; catalogue content is fully specified at
load time and never mutated afterwards.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm_traitbench.enums import Action, AssetClass, Expression, Op, StanceEntry
from pm_traitbench.errors import CatalogueError, PlanError

# The (asset class, expression) pairs an adapter can build an idea in; used to
# check every cell has thesis templates without the loader importing the engine.
ADAPTER_FORMS: dict[AssetClass, tuple[Expression, ...]] = {
    AssetClass.EQUITIES: (Expression.OUTRIGHT, Expression.PAIR),
    AssetClass.RATES_CREDIT: (Expression.OUTRIGHT, Expression.CURVE),
    AssetClass.COMMODITIES: (Expression.OUTRIGHT, Expression.CALENDAR_SPREAD),
}


class PreferenceGroup(StrEnum):
    COMMUNICATION = "communication"
    INFORMATION = "information"
    WORKFLOW = "workflow"
    EXPRESSION = "expression"


class PreferenceEntry(BaseModel):
    """One preference parameter: its group, applicable asset classes and value options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    group: PreferenceGroup
    asset_classes: tuple[AssetClass, ...] = Field(min_length=1)
    values: tuple[str, ...]


class RuleVariant(BaseModel):
    """One asset-class (and optionally sub-style) variant of a rule template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_class: AssetClass
    sub_styles: tuple[str, ...] = ()
    field: str
    op: Op
    unit: str | None
    level_min: float | None = None
    level_max: float | None = None
    round_to: float | None = None
    level_choices: tuple[float | str, ...] = ()
    action: Action
    window: int = 1
    templates: tuple[str, ...]

    @model_validator(mode="after")
    def _check_level_shape(self) -> "RuleVariant":
        range_fields = (self.level_min, self.level_max, self.round_to)
        has_full_range = all(v is not None for v in range_fields)
        has_partial_range = any(v is not None for v in range_fields) and not has_full_range
        has_choices = len(self.level_choices) > 0
        if has_partial_range:
            raise ValueError("level_min, level_max and round_to must all be set together")
        if has_full_range and has_choices:
            raise ValueError("variant must not set both a level range and level_choices")
        return self


class RuleEntry(BaseModel):
    """A rule parameter with its variants per asset class.

    ``share`` is the chance a PM draws this rule, not the realised share of PMs
    holding it: repair then forces a discipline rule where none was drawn and
    expands or contracts the set to the configured rule count.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    share: float = Field(ge=0, le=1)
    mandatory: bool = False
    discipline: bool = False
    variants: tuple[RuleVariant, ...]

    @property
    def asset_classes(self) -> tuple[AssetClass, ...]:
        """Asset classes this entry has at least one variant for, in variant order."""
        seen: list[AssetClass] = []
        for variant in self.variants:
            if variant.asset_class not in seen:
                seen.append(variant.asset_class)
        return tuple(seen)

    def variant_for(self, asset_class: AssetClass, sub_style: str) -> RuleVariant:
        """Return the first variant for an asset class, preferring one naming the sub-style."""
        for variant in self.variants:
            if variant.asset_class == asset_class and (
                not variant.sub_styles or sub_style in variant.sub_styles
            ):
                return variant
        raise CatalogueError(
            f"rule '{self.param}' has no variant for asset class '{asset_class}' "
            f"and sub-style '{sub_style}'"
        )


class RuleCatalogue(BaseModel):
    """All rule parameters, including the mandate risk cap common to every PM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mandate_cap: RuleEntry
    entries: tuple[RuleEntry, ...]

    @model_validator(mode="after")
    def _check_mandate_cap_param(self) -> "RuleCatalogue":
        if self.mandate_cap.param != "max_risk_pct":
            raise ValueError("mandate_cap must have param 'max_risk_pct'")
        return self

    def entries_for(self, asset_class: AssetClass) -> tuple[RuleEntry, ...]:
        """Return rule entries applicable to an asset class, in catalogue order."""
        return tuple(entry for entry in self.entries if asset_class in entry.asset_classes)

    @model_validator(mode="after")
    def _check_level_shapes(self) -> "RuleCatalogue":
        for entry in (self.mandate_cap, *self.entries):
            is_cap = entry is self.mandate_cap
            for variant in entry.variants:
                has_full_range = all(
                    v is not None for v in (variant.level_min, variant.level_max, variant.round_to)
                )
                has_choices = len(variant.level_choices) > 0
                has_neither = not has_full_range and not has_choices
                if is_cap and not has_neither:
                    raise ValueError(
                        "mandate_cap variant must not set a level range or level_choices"
                    )
                if has_neither and not is_cap:
                    raise ValueError(
                        f"rule '{entry.param}' variant must set a level range or level_choices"
                    )
        return self


class SubStyle(BaseModel):
    """A mandate sub-style: its name, risk unit and benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    risk_unit: str
    benchmark: str


class Phrasings(BaseModel):
    """Self-description phrasings that agree or contradict a bias parameter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agree: tuple[str, ...]
    contradict: tuple[str, ...]


class SignpostTemplates(BaseModel):
    """A signpost's templates for one asset class, by kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: tuple[str, ...] = Field(min_length=2)
    level: tuple[str, ...] = Field(min_length=2)
    relative: tuple[str, ...] = Field(min_length=2)


class ThesisTemplates(BaseModel):
    """Thesis templates per (asset class, expression) and outcome templates by result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    theses: dict[AssetClass, dict[Expression, tuple[str, ...]]]
    outcomes: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def _check_outcome_keys(self) -> "ThesisTemplates":
        if set(self.outcomes) != {"win", "loss", "open"}:
            raise ValueError("outcomes keys must be exactly 'win', 'loss' and 'open'")
        return self


StanceLines = dict[str, tuple[str, ...]]
"""Lines for one stance entry, keyed by "all" or an `AssetClass` value."""


class BiasStances(BaseModel):
    """One bias parameter's stance lines, by the kind of evidence they carry.

    ``revealed`` is keyed by the engine's flagged action pattern rather than by asset
    class directly, since the line drawn must describe the specific action the engine
    took that day, not just the trait behind it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    revealed: dict[str, StanceLines]
    stated: StanceLines
    claim: StanceLines
    retract: StanceLines
    third_party: StanceLines
    drift_update: StanceLines
    drift_dormant: StanceLines
    drift_revive: StanceLines


class PreferenceStances(BaseModel):
    """One preference group's stance lines, by the kind of evidence they carry.

    ``revealed`` is empty for every group but expression, whose ideas are the
    only preference evidence the engine itself can act out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stated: StanceLines
    revealed_reaction: StanceLines
    violation: StanceLines
    retract: StanceLines
    third_party: StanceLines
    drift_update: StanceLines
    revealed: StanceLines = {}


class Stances(BaseModel):
    """The stance bank: short instruction lines a narrator turns into PM dialogue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    biases: dict[str, BiasStances]
    preferences: dict[PreferenceGroup, PreferenceStances]

    def lines(self, key: str, entry: StanceEntry, asset_class: AssetClass) -> tuple[str, ...]:
        """Asset-class lines if present, else the "all" lines.

        `key` is a bias param name or a `PreferenceGroup` value. Raises `PlanError`
        naming key and entry when the bank has no such entry. A bias's `revealed`
        entry is keyed by engine action pattern instead, so `revealed_lines` serves it.
        """
        if key in self.biases:
            if entry == StanceEntry.REVEALED:
                raise PlanError(
                    f"stance bank key '{key}' entry 'revealed' is keyed by engine "
                    "action; use revealed_lines instead"
                )
            bank: BiasStances | PreferenceStances = self.biases[key]
        elif key in self.preferences:
            bank = self.preferences[key]
        else:
            raise PlanError(f"stance bank has no entry for key '{key}' entry '{entry.value}'")
        stance_lines = getattr(bank, entry.value, None)
        if not stance_lines:
            raise PlanError(f"stance bank key '{key}' has no lines for entry '{entry.value}'")
        return stance_lines.get(asset_class.value, stance_lines.get("all", ()))

    def revealed_lines(self, param: str, pattern: str, asset_class: AssetClass) -> tuple[str, ...]:
        """Asset-class lines for one bias's engine-flagged action pattern, else "all".

        Raises `PlanError` naming `param` and `pattern` when the bank has no such pattern.
        """
        bank = self.biases.get(param)
        if bank is None:
            raise PlanError(f"stance bank has no bias entry for param '{param}'")
        pattern_lines = bank.revealed.get(pattern)
        if not pattern_lines:
            raise PlanError(
                f"stance bank param '{param}' has no revealed lines for pattern '{pattern}'"
            )
        return pattern_lines.get(asset_class.value, pattern_lines.get("all", ()))


class Catalogue(BaseModel):
    """The full reference catalogue that samplers draw from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preferences: tuple[PreferenceEntry, ...]
    rules: RuleCatalogue
    sub_styles: dict[AssetClass, tuple[SubStyle, ...]]
    self_descriptions: dict[str, Phrasings]
    signposts: dict[AssetClass, SignpostTemplates]
    theses: ThesisTemplates
    stances: Stances

    def preferences_for(self, asset_class: AssetClass) -> tuple[PreferenceEntry, ...]:
        """Return preference entries applicable to an asset class, in catalogue order."""
        return tuple(entry for entry in self.preferences if asset_class in entry.asset_classes)
