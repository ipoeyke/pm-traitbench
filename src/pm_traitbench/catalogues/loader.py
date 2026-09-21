"""Loads catalogues from YAML and checks cross-cutting consistency the models can't."""

import functools
import string
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from pm_traitbench.catalogues.models import (
    Catalogue,
    Phrasings,
    PreferenceEntry,
    PreferenceGroup,
    RuleCatalogue,
    SubStyle,
)
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import AssetClass
from pm_traitbench.errors import CatalogueError

_FILE_NAMES = ("preferences.yaml", "rules.yaml", "mandates.yaml", "self_descriptions.yaml")


class _PreferencesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preferences: tuple[PreferenceEntry, ...]


class _MandatesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sub_styles: dict[AssetClass, tuple[SubStyle, ...]]


class _SelfDescriptionsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    self_descriptions: dict[str, Phrasings]


def _read_yaml(base: Any, name: str) -> dict[str, Any]:
    try:
        text = base.joinpath(name).read_text()
    except OSError as e:
        raise CatalogueError(f"failed to read catalogue file '{name}': {e}") from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise CatalogueError(f"invalid YAML in catalogue file '{name}': {e}") from e
    if not isinstance(data, dict):
        raise CatalogueError(f"catalogue file '{name}' must contain a mapping")
    return data


def _build_catalogue(base: Any) -> Catalogue:
    raw = {name: _read_yaml(base, name) for name in _FILE_NAMES}
    try:
        preferences = _PreferencesFile.model_validate(raw["preferences.yaml"]).preferences
        rules = RuleCatalogue.model_validate(raw["rules.yaml"])
        sub_styles = _MandatesFile.model_validate(raw["mandates.yaml"]).sub_styles
        self_descriptions = _SelfDescriptionsFile.model_validate(
            raw["self_descriptions.yaml"]
        ).self_descriptions
    except ValidationError as e:
        raise CatalogueError(f"invalid catalogue content: {e}") from e
    return Catalogue(
        preferences=preferences,
        rules=rules,
        sub_styles=sub_styles,
        self_descriptions=self_descriptions,
    )


@functools.cache
def _load_packaged() -> Catalogue:
    return _build_catalogue(resources.files("pm_traitbench.catalogues"))


def load_catalogue(directory: Path | None = None) -> Catalogue:
    """Load a catalogue from a directory of YAML files, or the packaged default."""
    if directory is None:
        return _load_packaged()
    return _build_catalogue(directory)


def render_template(template: str, level: float | str, unit: str | None) -> str:
    """Render a rule template, filling its {level} and {unit} slots.

    A string level is a snake_case id (e.g. "ccc_and_below"); it renders with
    underscores replaced by spaces. The stored level itself is untouched.
    """
    rendered_level = level.replace("_", " ") if isinstance(level, str) else format(level, "g")
    return template.format(level=rendered_level, unit=unit if unit is not None else "")


def _check_preference_group_coverage(
    catalogue: Catalogue, asset_classes: Sequence[AssetClass]
) -> None:
    for asset_class in asset_classes:
        groups_present = {entry.group for entry in catalogue.preferences_for(asset_class)}
        for group in PreferenceGroup:
            if group not in groups_present:
                raise CatalogueError(
                    f"preferences: asset class '{asset_class}' has no entry in group '{group}'"
                )


def _check_preference_count(
    catalogue: Catalogue, asset_classes: Sequence[AssetClass], n_preferences_max: int
) -> None:
    for asset_class in asset_classes:
        applicable = catalogue.preferences_for(asset_class)
        if len(applicable) < n_preferences_max:
            raise CatalogueError(
                f"preferences: asset class '{asset_class}' has only {len(applicable)} "
                f"applicable params, need at least {n_preferences_max}"
            )


def _check_preference_values(catalogue: Catalogue) -> None:
    seen_params: set[str] = set()
    for entry in catalogue.preferences:
        if entry.param in seen_params:
            raise CatalogueError(f"preferences: duplicate param '{entry.param}'")
        seen_params.add(entry.param)
        if len(entry.values) < 2:
            raise CatalogueError(f"preferences: param '{entry.param}' has fewer than 2 values")
        if len(set(entry.values)) != len(entry.values):
            raise CatalogueError(f"preferences: param '{entry.param}' has duplicate values")
        for value in entry.values:
            if not value.strip():
                raise CatalogueError(f"preferences: param '{entry.param}' has an empty value")


def _all_rule_entries(catalogue: Catalogue) -> tuple:
    return (catalogue.rules.mandate_cap, *catalogue.rules.entries)


def _check_rule_coverage(catalogue: Catalogue, asset_classes: Sequence[AssetClass]) -> None:
    seen_params: set[str] = set()
    for entry in _all_rule_entries(catalogue):
        if entry.param in seen_params:
            raise CatalogueError(f"rules: duplicate param '{entry.param}'")
        seen_params.add(entry.param)

    for entry in _all_rule_entries(catalogue):
        if not entry.variants:
            raise CatalogueError(f"rules: param '{entry.param}' has no variants")
        for asset_class in entry.asset_classes:
            for sub_style in catalogue.sub_styles.get(asset_class, ()):
                variant = entry.variant_for(asset_class, sub_style.name)
                if not variant.templates:
                    raise CatalogueError(
                        f"rules: param '{entry.param}' variant for asset class "
                        f"'{asset_class}' has no templates"
                    )

    for asset_class in asset_classes:
        if asset_class not in catalogue.rules.mandate_cap.asset_classes:
            raise CatalogueError(
                f"rules: param '{catalogue.rules.mandate_cap.param}' has no variant for "
                f"asset class '{asset_class}'"
            )
    for entry in catalogue.rules.entries:
        if not (entry.mandatory or entry.discipline):
            continue
        for asset_class in asset_classes:
            if asset_class not in entry.asset_classes:
                raise CatalogueError(
                    f"rules: param '{entry.param}' has no variant for asset class '{asset_class}'"
                )


def _check_rule_tags(catalogue: Catalogue) -> None:
    if not any(entry.mandatory for entry in catalogue.rules.entries):
        raise CatalogueError("rules: no entry has mandatory set")
    if not any(entry.discipline for entry in catalogue.rules.entries):
        raise CatalogueError("rules: no entry has discipline set")


def _check_rule_templates(catalogue: Catalogue) -> None:
    formatter = string.Formatter()
    for entry in _all_rule_entries(catalogue):
        for variant in entry.variants:
            for template in variant.templates:
                try:
                    fields = {
                        field_name
                        for _, field_name, _, _ in formatter.parse(template)
                        if field_name is not None
                    }
                except ValueError as e:
                    raise CatalogueError(
                        f"rules: param '{entry.param}' template '{template}' is malformed: {e}"
                    ) from e
                unknown = fields - {"level", "unit"}
                if unknown:
                    raise CatalogueError(
                        f"rules: param '{entry.param}' template '{template}' uses "
                        f"unknown slot(s) {sorted(unknown)}"
                    )


def _check_rule_levels(catalogue: Catalogue) -> None:
    for entry in _all_rule_entries(catalogue):
        for variant in entry.variants:
            if variant.level_min is None:
                continue
            if not variant.level_min < variant.level_max:
                raise CatalogueError(
                    f"rules: param '{entry.param}' variant for asset class "
                    f"'{variant.asset_class}' has level_min not less than level_max"
                )
            if not variant.round_to > 0:
                raise CatalogueError(
                    f"rules: param '{entry.param}' variant for asset class "
                    f"'{variant.asset_class}' has non-positive round_to"
                )


def _check_sub_styles(catalogue: Catalogue, asset_classes: Sequence[AssetClass]) -> None:
    for asset_class in asset_classes:
        styles = catalogue.sub_styles.get(asset_class, ())
        if len(styles) < 1:
            raise CatalogueError(f"mandates: asset class '{asset_class}' has no sub-styles")
        names = [style.name for style in styles]
        if len(set(names)) != len(names):
            raise CatalogueError(
                f"mandates: asset class '{asset_class}' has duplicate sub-style names"
            )


def _check_self_descriptions(catalogue: Catalogue) -> None:
    actual = set(catalogue.self_descriptions)
    expected = set(BIAS_PARAMS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CatalogueError(
            f"self_descriptions: keys must equal the bias parameter set; "
            f"missing {missing}, extra {extra}"
        )
    for param, phrasings in catalogue.self_descriptions.items():
        if len(phrasings.agree) < 2:
            raise CatalogueError(
                f"self_descriptions: param '{param}' has fewer than 2 agree phrasings"
            )
        if len(phrasings.contradict) < 2:
            raise CatalogueError(
                f"self_descriptions: param '{param}' has fewer than 2 contradict phrasings"
            )


def check_catalogue(
    catalogue: Catalogue, asset_classes: Sequence[AssetClass], n_preferences_max: int
) -> None:
    """Check catalogue consistency beyond what the models validate on their own."""
    _check_preference_group_coverage(catalogue, asset_classes)
    _check_preference_count(catalogue, asset_classes, n_preferences_max)
    _check_preference_values(catalogue)
    _check_rule_coverage(catalogue, asset_classes)
    _check_rule_tags(catalogue)
    _check_rule_templates(catalogue)
    _check_rule_levels(catalogue)
    _check_sub_styles(catalogue, asset_classes)
    _check_self_descriptions(catalogue)
