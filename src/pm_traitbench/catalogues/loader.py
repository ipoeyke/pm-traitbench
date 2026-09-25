"""Loads catalogues from YAML and checks cross-cutting consistency the models can't."""

import functools
import string
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from pm_traitbench.catalogues.models import (
    ADAPTER_FORMS,
    Catalogue,
    Phrasings,
    PreferenceEntry,
    PreferenceGroup,
    RuleCatalogue,
    SignpostTemplates,
    StanceLines,
    Stances,
    SubStyle,
    ThesisTemplates,
)
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import AssetClass, Kind, StanceEntry
from pm_traitbench.errors import CatalogueError

_FILE_NAMES = (
    "preferences.yaml",
    "rules.yaml",
    "mandates.yaml",
    "self_descriptions.yaml",
    "signposts.yaml",
    "theses.yaml",
    "stances.yaml",
)

# A stance line's slots vary by (kind of trait, kind of evidence): which parts of the
# planted event a line may quote. "value" names the preference itself, so any entry
# whose slot set contains it must use it, or the line would never say what the PM wants.
STANCE_SLOTS: dict[tuple[Kind, StanceEntry], frozenset[str]] = {
    (Kind.BIAS, StanceEntry.REVEALED): frozenset({"instrument", "entry", "target", "stop"}),
    (Kind.BIAS, StanceEntry.STATED): frozenset(),
    (Kind.BIAS, StanceEntry.CLAIM): frozenset(),
    (Kind.BIAS, StanceEntry.RETRACT): frozenset(),
    (Kind.BIAS, StanceEntry.THIRD_PARTY): frozenset({"who"}),
    (Kind.BIAS, StanceEntry.DRIFT_UPDATE): frozenset(),
    (Kind.BIAS, StanceEntry.DRIFT_DORMANT): frozenset(),
    (Kind.BIAS, StanceEntry.DRIFT_REVIVE): frozenset(),
    (Kind.PREFERENCE, StanceEntry.STATED): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.REVEALED): frozenset({"value", "instrument"}),
    (Kind.PREFERENCE, StanceEntry.REVEALED_REACTION): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.VIOLATION): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.RETRACT): frozenset({"value"}),
    (Kind.PREFERENCE, StanceEntry.THIRD_PARTY): frozenset({"value", "who"}),
    (Kind.PREFERENCE, StanceEntry.DRIFT_UPDATE): frozenset({"value", "old_value"}),
}
# Words that would name the bias a stance is planted for, leaking the label the
# dataset otherwise hides; "conviction" is left out because it is ordinary desk
# vocabulary and a public ledger column.
BANNED_STANCE_WORDS: tuple[str, ...] = (
    "loss aversion",
    "loss averse",
    "disposition",
    "anchor",
    "extrapolat",
    "herd",
    "overconfiden",
    "miscalibrat",
    "exit deficiency",
    "bias",
)

# A signpost's {level} slot arrives rendered with its own unit (or as a bare price).
_SIGNPOST_SLOTS: dict[str, frozenset[str]] = {
    "event": frozenset({"event"}),
    "level": frozenset({"level", "window"}),
    "relative": frozenset({"level", "peer"}),
}
_THESIS_SLOTS = frozenset({"name", "entry", "target", "move", "unit", "horizon", "side"})
_OUTCOME_SLOTS = frozenset({"pnl", "unit", "closer"})
UNIT_DISPLAY: dict[str, str] = {"pct": "%", "bp": "bp"}

# What closed the idea, in the PM's own words; the engine passes the closing
# rule's param name, "discretionary", or "horizon_end" for a still-open idea.
CLOSER_PHRASES: dict[str, str] = {
    "stop": "the stop",
    "target": "the target",
    "signpost": "a signpost",
    "trim_at_target": "the trim",
    "roll": "the roll",
    "discretionary": "my call",
    "horizon_end": "the horizon end",
}


class _PreferencesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preferences: tuple[PreferenceEntry, ...]


class _MandatesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sub_styles: dict[AssetClass, tuple[SubStyle, ...]]


class _SelfDescriptionsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    self_descriptions: dict[str, Phrasings]


class _SignpostsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    signposts: dict[AssetClass, SignpostTemplates]


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
        signposts = _SignpostsFile.model_validate(raw["signposts.yaml"]).signposts
        theses = ThesisTemplates.model_validate(raw["theses.yaml"])
        stances = Stances.model_validate(raw["stances.yaml"])
    except ValidationError as e:
        raise CatalogueError(f"invalid catalogue content: {e}") from e
    return Catalogue(
        preferences=preferences,
        rules=rules,
        sub_styles=sub_styles,
        self_descriptions=self_descriptions,
        signposts=signposts,
        theses=theses,
        stances=stances,
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


def render_signpost(
    template: str,
    *,
    level: str | None = None,
    window: int | None = None,
    event: str | None = None,
    peer: str | None = None,
) -> str:
    """Render a signpost template, filling whichever of its slots are given.

    `level` arrives already rendered, unit included, by the engine's level formatter.
    """
    slots: dict[str, str] = {}
    if level is not None:
        slots["level"] = level
    if window is not None:
        slots["window"] = str(window)
    if event is not None:
        slots["event"] = event.replace("_", " ")
    if peer is not None:
        slots["peer"] = peer
    return template.format(**slots)


def render_thesis(template: str, **slots: Any) -> str:
    """Render a thesis or outcome template, filling whichever of its slots are given.

    ``move`` and ``pnl`` render signed to one decimal when given as floats;
    ``closer`` maps through ``CLOSER_PHRASES``; every other slot renders with ``str``.
    Every thesis template must carry ``{side}``, so direction is never implied by a sign.
    """
    rendered: dict[str, str] = {}
    for key, value in slots.items():
        if key == "unit":
            # None means a quoted price: no unit suffix, not an omitted slot.
            rendered[key] = UNIT_DISPLAY.get(value, value) if value is not None else ""
            continue
        if value is None:
            continue
        if key == "closer":
            try:
                rendered[key] = CLOSER_PHRASES[value]
            except KeyError:
                raise CatalogueError(f"render_thesis: unknown closer '{value}'") from None
        elif key in ("move", "pnl") and isinstance(value, float):
            rendered[key] = format(value, "+.1f")
        else:
            rendered[key] = str(value)
    return template.format(**rendered)


def render_stance(line: str, slots: Mapping[str, str]) -> str:
    """Render a stance line, filling its slots from a mapping of slot name to value."""
    try:
        return line.format(**slots)
    except KeyError as e:
        raise CatalogueError(f"stance line '{line}' is missing slot {e}") from e


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


def _template_fields(template: str, context: str) -> set[str]:
    try:
        return {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name is not None
        }
    except ValueError as e:
        raise CatalogueError(f"{context}: template '{template}' is malformed: {e}") from e


def _check_no_unknown_slots(
    templates: Sequence[str], allowed: frozenset[str], context: str
) -> None:
    for template in templates:
        if not template.strip():
            raise CatalogueError(f"{context}: has a blank template")
        unknown = _template_fields(template, context) - allowed
        if unknown:
            raise CatalogueError(
                f"{context}: template '{template}' uses unknown slot(s) {sorted(unknown)}"
            )


def _check_engine_templates(catalogue: Catalogue) -> None:
    for asset_class in ADAPTER_FORMS:
        if asset_class not in catalogue.signposts:
            raise CatalogueError(f"signposts: asset class '{asset_class}' has no templates")
        signpost = catalogue.signposts[asset_class]
        for kind, allowed in _SIGNPOST_SLOTS.items():
            context = f"signposts: asset class '{asset_class}' kind '{kind}'"
            _check_no_unknown_slots(getattr(signpost, kind), allowed, context)
        for expression in ADAPTER_FORMS[asset_class]:
            templates = catalogue.theses.theses.get(asset_class, {}).get(expression, ())
            if not templates:
                raise CatalogueError(
                    f"theses: asset class '{asset_class}' expression '{expression}' "
                    "has no templates"
                )
            context = f"theses: asset class '{asset_class}' expression '{expression}'"
            _check_no_unknown_slots(templates, _THESIS_SLOTS, context)
            for template in templates:
                if "side" not in _template_fields(template, context):
                    raise CatalogueError(f"{context}: template '{template}' has no {{side}} slot")
    for kind, templates in catalogue.theses.outcomes.items():
        _check_no_unknown_slots(templates, _OUTCOME_SLOTS, f"outcomes: kind '{kind}'")


def _check_stance_lines(
    kind: Kind, key: str, entry: StanceEntry, stance_lines: StanceLines
) -> None:
    context = f"stances: {kind} '{key}' entry '{entry.value}'"
    if "all" not in stance_lines:
        raise CatalogueError(f"{context} has no 'all' key")
    allowed_slots = STANCE_SLOTS[(kind, entry)]
    asset_class_values = {asset_class.value for asset_class in AssetClass}
    for lines_key, lines in stance_lines.items():
        if lines_key != "all" and lines_key not in asset_class_values:
            raise CatalogueError(f"{context} has unknown key '{lines_key}'")
        if len(lines) < 2:
            raise CatalogueError(f"{context} key '{lines_key}' has fewer than 2 lines")
        for line in lines:
            fields = _template_fields(line, context)
            unknown = fields - allowed_slots
            if unknown:
                raise CatalogueError(
                    f"{context} line '{line}' uses unknown slot(s) {sorted(unknown)}"
                )
            if "value" in allowed_slots and "value" not in fields:
                raise CatalogueError(f"{context} line '{line}' does not use the {{value}} slot")
            lowered = line.lower()
            for banned in BANNED_STANCE_WORDS:
                if banned in lowered:
                    raise CatalogueError(f"{context} line '{line}' contains banned word '{banned}'")


def check_stances(catalogue: Catalogue) -> None:
    """Check the stance bank's coverage, slot usage and banned-word list.

    A stance line describes behaviour, never the bias it plants, since the dataset
    hides trait labels from the text a narrator turns into PM dialogue.
    """
    stances = catalogue.stances
    actual_bias_keys = set(stances.biases)
    expected_bias_keys = set(BIAS_PARAMS)
    if actual_bias_keys != expected_bias_keys:
        missing = sorted(expected_bias_keys - actual_bias_keys)
        extra = sorted(actual_bias_keys - expected_bias_keys)
        raise CatalogueError(
            f"stances: biases keys must equal the bias parameter set; "
            f"missing {missing}, extra {extra}"
        )
    actual_pref_keys = set(stances.preferences)
    expected_pref_keys = set(PreferenceGroup)
    if actual_pref_keys != expected_pref_keys:
        missing = sorted(expected_pref_keys - actual_pref_keys)
        extra = sorted(actual_pref_keys - expected_pref_keys)
        raise CatalogueError(
            f"stances: preferences keys must equal the preference group set; "
            f"missing {missing}, extra {extra}"
        )

    for param, bank in stances.biases.items():
        for field_name in type(bank).model_fields:
            entry = StanceEntry(field_name)
            _check_stance_lines(Kind.BIAS, param, entry, getattr(bank, field_name))

    for group, bank in stances.preferences.items():
        for field_name in type(bank).model_fields:
            if field_name == "revealed":
                continue
            entry = StanceEntry(field_name)
            _check_stance_lines(Kind.PREFERENCE, group.value, entry, getattr(bank, field_name))
        if group == PreferenceGroup.EXPRESSION:
            if not bank.revealed:
                raise CatalogueError(
                    f"stances: preference '{group.value}' entry 'revealed' must not be empty"
                )
            _check_stance_lines(Kind.PREFERENCE, group.value, StanceEntry.REVEALED, bank.revealed)
        elif bank.revealed:
            raise CatalogueError(
                f"stances: preference '{group.value}' entry 'revealed' must be empty"
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
    _check_engine_templates(catalogue)
    check_stances(catalogue)
