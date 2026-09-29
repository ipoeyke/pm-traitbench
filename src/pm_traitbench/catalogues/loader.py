"""Loads catalogues from YAML and checks cross-cutting consistency the models can't."""

import functools
import re
import string
from collections.abc import Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from pm_traitbench.catalogues.models import (
    ADAPTER_FORMS,
    AvoidLines,
    BiasDefinitions,
    BiasLabels,
    Catalogue,
    Phrasings,
    PreferenceEntry,
    PreferenceGroup,
    ProbeBank,
    RuleCatalogue,
    SignpostTemplates,
    StanceLines,
    Stances,
    SubStyle,
    ThesisTemplates,
    Voice,
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
    "voices.yaml",
    "avoid.yaml",
    "bias_labels.yaml",
    "bias_definitions.yaml",
    "probes.yaml",
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
# Keys are the engine's per-bias action flags, so the line drawn always matches the
# specific action logged that day, not just the trait behind it.
REVEALED_PATTERNS: dict[str, tuple[str, ...]] = {
    "loss_aversion_lambda": ("add", "add_before_trigger", "hold"),
    "disposition_ratio": ("realise_gain_early", "hold_loser"),
    "anchoring_rho": ("exit_at_anchor",),
    "extrapolation_theta": ("chased_trend",),
    "herding_weight": ("followed_street",),
    "overconfidence_coverage": ("oversized",),
    "conviction_size_miscalibration": ("mis_sized",),
    "exit_deficiency": ("acked_no_action", "added", "late_roll"),
}

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


class _VoicesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    voices: tuple[Voice, ...]


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
        voices = _VoicesFile.model_validate(raw["voices.yaml"]).voices
        avoid = AvoidLines.model_validate(raw["avoid.yaml"])
        bias_labels = BiasLabels.model_validate(raw["bias_labels.yaml"])
        bias_definitions = BiasDefinitions.model_validate(raw["bias_definitions.yaml"])
        probes = ProbeBank.model_validate(raw["probes.yaml"])
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
        voices=voices,
        avoid=avoid,
        bias_labels=bias_labels,
        bias_definitions=bias_definitions,
        probes=probes,
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
    except (KeyError, IndexError, ValueError) as e:
        raise CatalogueError(f"stance line '{line}' failed to render: {e}") from e


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


def _check_key_set(context: str, actual: set[str], expected: set[str], what: str) -> None:
    """Raise unless `actual` equals `expected`, naming the missing and extra keys.

    `context` is the message lead up to "must equal", e.g. "avoid: biases keys".
    """
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CatalogueError(f"{context} must equal {what}; missing {missing}, extra {extra}")


def _check_self_descriptions(catalogue: Catalogue) -> None:
    _check_key_set(
        "self_descriptions: keys",
        set(catalogue.self_descriptions),
        set(BIAS_PARAMS),
        "the bias parameter set",
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
    context: str,
    allowed_slots: frozenset[str],
    stance_lines: StanceLines,
    required: frozenset[str] | None = None,
) -> None:
    """Check keys, line counts, slots and banned words of one lines mapping.

    `required` slots must appear in every line; by default only `{value}`, when allowed.
    """
    if required is None:
        required = allowed_slots & {"value"}
    if "all" not in stance_lines:
        raise CatalogueError(f"{context} has no 'all' key")
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
            missing = sorted(required - fields)
            if missing:
                raise CatalogueError(
                    f"{context} line '{line}' does not use the {{{missing[0]}}} slot"
                )
            banned = banned_words_in(line)
            if banned:
                raise CatalogueError(f"{context} line '{line}' contains banned word '{banned[0]}'")


def check_stances(catalogue: Catalogue) -> None:
    """Check the stance bank's coverage, slot usage and banned-word list.

    A stance line describes behaviour, never the bias it plants, since the dataset
    hides trait labels from the text a narrator turns into PM dialogue.
    """
    stances = catalogue.stances
    _check_key_set(
        "stances: biases keys", set(stances.biases), set(BIAS_PARAMS), "the bias parameter set"
    )
    _check_key_set(
        "stances: preferences keys",
        set(stances.preferences),
        set(PreferenceGroup),
        "the preference group set",
    )

    for param, bank in stances.biases.items():
        for field_name in type(bank).model_fields:
            if field_name == "revealed":
                continue
            entry = StanceEntry(field_name)
            context = f"stances: bias '{param}' entry '{entry.value}'"
            _check_stance_lines(
                context, STANCE_SLOTS[(Kind.BIAS, entry)], getattr(bank, field_name)
            )

        expected_patterns = set(REVEALED_PATTERNS[param])
        _check_key_set(
            f"stances: bias '{param}' entry 'revealed' pattern keys",
            set(bank.revealed),
            expected_patterns,
            str(sorted(expected_patterns)),
        )
        revealed_slots = STANCE_SLOTS[(Kind.BIAS, StanceEntry.REVEALED)]
        for pattern, pattern_lines in bank.revealed.items():
            context = f"stances: bias '{param}' entry 'revealed' pattern '{pattern}'"
            _check_stance_lines(context, revealed_slots, pattern_lines)

    for group, bank in stances.preferences.items():
        for field_name in type(bank).model_fields:
            if field_name == "revealed":
                continue
            entry = StanceEntry(field_name)
            context = f"stances: preference '{group.value}' entry '{entry.value}'"
            _check_stance_lines(
                context, STANCE_SLOTS[(Kind.PREFERENCE, entry)], getattr(bank, field_name)
            )
        if group == PreferenceGroup.EXPRESSION:
            if not bank.revealed:
                raise CatalogueError(
                    f"stances: preference '{group.value}' entry 'revealed' must not be empty"
                )
            context = f"stances: preference '{group.value}' entry 'revealed'"
            _check_stance_lines(
                context, STANCE_SLOTS[(Kind.PREFERENCE, StanceEntry.REVEALED)], bank.revealed
            )
        elif bank.revealed:
            raise CatalogueError(
                f"stances: preference '{group.value}' entry 'revealed' must be empty"
            )


# One text per engine outcome the MCQ distinguishes, in the order the probe stage indexes them.
PROBE_ACTION_COUNTS: dict[str, int] = {
    "loss_aversion_lambda": 4,
    "disposition_ratio": 4,
    "anchoring_rho": 4,
    "extrapolation_theta": 3,
    "herding_weight": 4,
    "overconfidence_coverage": 4,
    "conviction_size_miscalibration": 4,
    "exit_deficiency": 4,
}
# The only biases whose request can breach the one mandate rule, the position cap.
DECLINE_PARAMS: tuple[str, ...] = ("loss_aversion_lambda", "overconfidence_coverage")
# Every bias MCQ line may use these slots; the required ones carry the number or level
# the answer depends on, so a line without it asks an unanswerable question.
MCQ_SLOTS: dict[str, frozenset[str]] = {
    "loss_aversion_lambda": frozenset(
        {"instrument", "level", "entry", "stop", "target", "horizon"}
    ),
    "disposition_ratio": frozenset({"instrument", "level", "entry", "target", "horizon"}),
    "anchoring_rho": frozenset({"instrument", "level", "entry", "target", "round_level"}),
    "exit_deficiency": frozenset({"instrument", "level", "entry", "stop"}),
    "herding_weight": frozenset({"instrument", "level", "street", "own_side"}),
    "extrapolation_theta": frozenset(
        {"instrument", "level", "thesis_sd", "trailing_sd", "horizon"}
    ),
    "overconfidence_coverage": frozenset({"instrument", "level"}),
    "conviction_size_miscalibration": frozenset({"instrument", "level", "rating"}),
}
MCQ_REQUIRED: dict[str, frozenset[str]] = {
    "loss_aversion_lambda": frozenset({"horizon"}),
    "disposition_ratio": frozenset({"horizon"}),
    "anchoring_rho": frozenset({"round_level"}),
    "exit_deficiency": frozenset({"stop"}),
    "herding_weight": frozenset({"street", "own_side"}),
    "extrapolation_theta": frozenset({"thesis_sd", "trailing_sd"}),
    "overconfidence_coverage": frozenset(),
    "conviction_size_miscalibration": frozenset({"rating"}),
}
_INSTRUMENT_LEVEL = frozenset({"instrument", "level"})


def _check_probe_text(context: str, text: str) -> None:
    """Check one probe text is non-blank and leaks no label, param name or em dash."""
    if not text.strip():
        raise CatalogueError(f"{context} has a blank text")
    banned = banned_words_in(text)
    if banned:
        raise CatalogueError(f"{context} text '{text}' contains banned word '{banned[0]}'")
    matched = matched_params(text, BIAS_PARAMS)
    if matched:
        raise CatalogueError(f"{context} text '{text}' names param '{matched[0]}'")
    if "—" in text:
        raise CatalogueError(f"{context} text '{text}' contains an em dash")


def _check_probe_lines(
    context: str,
    stance_lines: StanceLines,
    allowed: frozenset[str] = frozenset(),
    required: frozenset[str] = frozenset(),
) -> None:
    _check_stance_lines(context, allowed, stance_lines, required)
    for lines in stance_lines.values():
        for line in lines:
            _check_probe_text(context, line)


def check_probes_catalogue(catalogue: Catalogue) -> None:
    """Check the probe bank's coverage, slot usage and label-free wording.

    A probe asks about behaviour, never the bias it tests, so the copilot cannot answer
    from a trait name; a request must not name the preference value, or it gives the
    answer away.
    """
    probes = catalogue.probes
    for entry in catalogue.preferences:
        if len(entry.values) not in (3, 4):
            raise CatalogueError(
                f"probes: preference '{entry.param}' has {len(entry.values)} values, need 3 or 4"
            )
    _check_key_set(
        "probes: biases keys", set(probes.biases), set(BIAS_PARAMS), "the bias parameter set"
    )
    _check_key_set(
        "probes: preferences keys",
        set(probes.preferences),
        set(PreferenceGroup),
        "the preference group set",
    )

    for param, bank in probes.biases.items():
        prefix = f"probes: bias '{param}'"
        _check_probe_lines(f"{prefix} entry 'presence'", bank.presence)
        _check_probe_lines(
            f"{prefix} entry 'mcq'",
            bank.mcq,
            MCQ_SLOTS[param],
            MCQ_REQUIRED[param],
        )
        _check_probe_lines(f"{prefix} entry 'in_situ'", bank.in_situ, _INSTRUMENT_LEVEL)
        _check_probe_lines(f"{prefix} entry 'governance'", bank.governance)
        if bool(bank.decline) != (param in DECLINE_PARAMS):
            raise CatalogueError(
                f"{prefix} entry 'decline' must be non-empty exactly for {list(DECLINE_PARAMS)}"
            )
        if bank.decline:
            _check_probe_lines(
                f"{prefix} entry 'decline'",
                bank.decline,
                frozenset({"instrument", "level", "size"}),
                frozenset({"size"}),
            )
        _check_probe_text(f"{prefix} entry 'behaviour'", bank.behaviour)
        expected = PROBE_ACTION_COUNTS[param]
        if len(bank.actions) != expected:
            raise CatalogueError(
                f"{prefix} entry 'actions' has {len(bank.actions)} entries, need {expected}"
            )
        if len(set(bank.actions)) != len(bank.actions):
            raise CatalogueError(f"{prefix} entry 'actions' has duplicate entries")
        for action in bank.actions:
            _check_probe_text(f"{prefix} entry 'actions'", action)

    for group, pref_bank in probes.preferences.items():
        prefix = f"probes: preference '{group.value}'"
        _check_probe_lines(
            f"{prefix} entry 'presence'",
            pref_bank.presence,
            frozenset({"value"}),
            frozenset({"value"}),
        )
        _check_probe_lines(f"{prefix} entry 'mcq'", pref_bank.mcq)
        _check_probe_lines(f"{prefix} entry 'in_situ'", pref_bank.in_situ, _INSTRUMENT_LEVEL)
        _check_probe_lines(
            f"{prefix} entry 'governance'",
            pref_bank.governance,
            frozenset({"old_value"}),
            frozenset({"old_value"}),
        )

    _check_probe_lines("probes: entry 'routine'", probes.routine, _INSTRUMENT_LEVEL)


def _param_forms(param: str) -> tuple[str, str]:
    """A param's lower-cased raw spelling and its underscore-replaced phrase."""
    raw = param.lower()
    return raw, raw.replace("_", " ")


def matched_params(text: str, params: Iterable[str]) -> tuple[str, ...]:
    """Every param named in text, whole-word and case-insensitive, in `params` order.

    Checks both the raw param (with underscores) and its underscore-replaced
    phrase, so "register" and "loss_aversion_lambda" are caught however the text
    spells them.
    """
    lowered = text.lower()
    return tuple(
        param
        for param in params
        if any(re.search(rf"\b{re.escape(form)}\b", lowered) for form in _param_forms(param))
    )


def banned_words_in(text: str) -> tuple[str, ...]:
    """Every `BANNED_STANCE_WORDS` entry found in text as a case-insensitive substring."""
    lowered = text.lower()
    return tuple(word for word in BANNED_STANCE_WORDS if word in lowered)


def leak_param_names(catalogue: Catalogue) -> tuple[str, ...]:
    """The bias params followed by every catalogue preference's param, in catalogue order."""
    return (*BIAS_PARAMS, *(entry.param for entry in catalogue.preferences))


def check_dialogue_catalogue(catalogue: Catalogue) -> None:
    """Check the narrator's voice bank and its forbidden-behaviour lines.

    A voice must not be so specific it becomes a near-unique PM fingerprint, and
    every avoid line must cover exactly the bias and preference params a skeleton
    can name as forbidden for a PM who does not have that trait.
    """
    if len(catalogue.voices) < 6:
        raise CatalogueError(f"voices: need at least 6 voices, got {len(catalogue.voices)}")
    seen_ids: set[str] = set()
    for voice in catalogue.voices:
        if voice.voice_id in seen_ids:
            raise CatalogueError(f"voices: duplicate voice_id '{voice.voice_id}'")
        seen_ids.add(voice.voice_id)
        if not voice.line.strip():
            raise CatalogueError(f"voices: voice '{voice.voice_id}' line is blank")
        banned = banned_words_in(voice.line)
        if banned:
            raise CatalogueError(
                f"voices: voice '{voice.voice_id}' line contains banned word '{banned[0]}'"
            )
        matched = matched_params(voice.line, leak_param_names(catalogue))
        if matched:
            raise CatalogueError(
                f"voices: voice '{voice.voice_id}' line names param '{matched[0]}'"
            )

    _check_key_set(
        "avoid: biases keys",
        set(catalogue.avoid.biases),
        set(BIAS_PARAMS),
        "the bias parameter set",
    )
    _check_key_set(
        "avoid: preferences keys",
        set(catalogue.avoid.preferences),
        {entry.param for entry in catalogue.preferences},
        "the catalogue's preference params",
    )
    for param, line in {**catalogue.avoid.biases, **catalogue.avoid.preferences}.items():
        if not line.strip():
            raise CatalogueError(f"avoid: param '{param}' has an empty line")


def check_validate_catalogue(catalogue: Catalogue) -> None:
    """Check the bias label catalogue the validation stage's leakage judge maps labels through.

    A judge never writes a raw param identifier, so any phrase equal to one is a
    labeling error under any param. A phrase equal to a different param's spaced-out
    name is filed under the wrong bias; a phrase equal to its own param's spaced-out
    name is that bias's plain-English name and is expected.
    """
    labels = catalogue.bias_labels.labels
    _check_key_set("bias_labels: keys", set(labels), set(BIAS_PARAMS), "the bias parameter set")

    forms = {param: _param_forms(param) for param in BIAS_PARAMS}
    raw_forms = {raw for raw, _ in forms.values()}
    spaced_forms = {param: spaced for param, (_, spaced) in forms.items()}

    seen: dict[str, str] = {}
    for param, phrases in labels.items():
        if not phrases:
            raise CatalogueError(f"bias_labels: param '{param}' has no phrases")
        for phrase in phrases:
            if not phrase.strip() or phrase != phrase.strip().lower():
                raise CatalogueError(
                    f"bias_labels: param '{param}' phrase '{phrase}' must be "
                    "lower-case and non-blank"
                )
            if phrase in raw_forms:
                raise CatalogueError(
                    f"bias_labels: param '{param}' phrase '{phrase}' is a raw param string"
                )
            for other, spaced in spaced_forms.items():
                if other != param and phrase == spaced:
                    raise CatalogueError(
                        f"bias_labels: param '{param}' phrase '{phrase}' equals "
                        f"the name of param '{other}'"
                    )
            if phrase in seen and seen[phrase] != param:
                raise CatalogueError(
                    f"bias_labels: phrase '{phrase}' appears under both "
                    f"'{seen[phrase]}' and '{param}'"
                )
            seen[phrase] = param


def check_gate2_catalogue(catalogue: Catalogue) -> None:
    """Check the bias definitions a Gate 2 recovery prompt shows beside each param name."""
    definitions = catalogue.bias_definitions.definitions
    _check_key_set(
        "bias_definitions: keys", set(definitions), set(BIAS_PARAMS), "the bias parameter set"
    )
    for param, line in definitions.items():
        if not line.strip():
            raise CatalogueError(f"bias_definitions: param '{param}' line is blank")
        if "—" in line:
            raise CatalogueError(f"bias_definitions: param '{param}' line contains an em dash")
        matched = matched_params(line, BIAS_PARAMS)
        if matched:
            raise CatalogueError(
                f"bias_definitions: param '{param}' line names param '{matched[0]}'"
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
    check_dialogue_catalogue(catalogue)
