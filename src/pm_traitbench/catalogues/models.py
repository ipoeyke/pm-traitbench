"""Catalogue content models: preferences, rule templates, mandate sub-styles and phrasings.

All frozen and reject unknown fields; catalogue content is fully specified at
load time and never mutated afterwards.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm_traitbench.enums import Action, AssetClass, Op
from pm_traitbench.errors import CatalogueError


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


class Catalogue(BaseModel):
    """The full reference catalogue that samplers draw from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preferences: tuple[PreferenceEntry, ...]
    rules: RuleCatalogue
    sub_styles: dict[AssetClass, tuple[SubStyle, ...]]
    self_descriptions: dict[str, Phrasings]

    def preferences_for(self, asset_class: AssetClass) -> tuple[PreferenceEntry, ...]:
        """Return preference entries applicable to an asset class, in catalogue order."""
        return tuple(entry for entry in self.preferences if asset_class in entry.asset_classes)
