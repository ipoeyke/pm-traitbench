"""Rule sampling: draws each PM's mandate cap and self-imposed trading rules."""

from numpy.random import Generator

from pm_traitbench.catalogues.loader import render_template
from pm_traitbench.catalogues.models import Catalogue, RuleEntry, RuleVariant
from pm_traitbench.config import Config
from pm_traitbench.enums import RuleScope, RuleSource
from pm_traitbench.tables.schema import Mandate, Rule


def _draw_level(variant: RuleVariant, rng: Generator) -> float | str:
    """Draw a rule level: a uniform choice, or a range value snapped to round_to."""
    if variant.level_choices:
        choice = variant.level_choices[rng.integers(len(variant.level_choices))]
        return round(choice, 4) if isinstance(choice, float) else choice
    raw = rng.uniform(variant.level_min, variant.level_max)
    snapped = variant.round_to * round(raw / variant.round_to)
    clipped = min(max(snapped, variant.level_min), variant.level_max)
    return round(clipped, 4)


def _build_rule(
    pm_id: str,
    rule_id: str,
    source: RuleSource,
    param: str,
    variant: RuleVariant,
    level: float | str,
    rng: Generator,
) -> Rule:
    template = variant.templates[rng.integers(len(variant.templates))]
    return Rule(
        pm_id=pm_id,
        rule_id=rule_id,
        source=source,
        scope=RuleScope.PM,
        trade_idea_id=None,
        param=param,
        field=variant.field,
        op=variant.op,
        level=level,
        unit=variant.unit,
        window=variant.window,
        action=variant.action,
        text=render_template(template, level, variant.unit),
    )


def _choose_inclusion(entries: tuple[RuleEntry, ...], rng: Generator) -> list[str]:
    """Draw one inclusion uniform per entry, in catalogue order."""
    included: list[str] = []
    for entry in entries:
        u = rng.uniform()
        if entry.mandatory or u < entry.share:
            included.append(entry.param)
    return included


def _repair_inclusion(
    entries: tuple[RuleEntry, ...], included: list[str], config: Config, rng: Generator
) -> list[str]:
    """Guarantee a discipline rule and clamp the self rule count into bounds."""
    by_param = {entry.param: entry for entry in entries}

    if not any(by_param[param].discipline for param in included):
        discipline_params = [entry.param for entry in entries if entry.discipline]
        included.append(discipline_params[rng.integers(len(discipline_params))])

    while len(included) > config.rules.n_self_rules_max:
        discipline_included = [param for param in included if by_param[param].discipline]
        removable = [
            param
            for param in included
            if not by_param[param].mandatory
            and not (len(discipline_included) == 1 and param == discipline_included[0])
        ]
        included.remove(removable[rng.integers(len(removable))])

    while len(included) < config.rules.n_self_rules_min:
        not_included = [entry.param for entry in entries if entry.param not in included]
        included.append(not_included[rng.integers(len(not_included))])

    return included


def sample_rules(
    pm_id: str, mandate: Mandate, config: Config, catalogue: Catalogue, rng: Generator
) -> list[Rule]:
    """Sample a PM's mandate cap plus 3-5 self rules, always a stop and a discipline rule."""
    asset_class = mandate.asset_class
    sub_style = mandate.sub_style

    cap_entry = catalogue.rules.mandate_cap
    cap_variant = cap_entry.variant_for(asset_class, sub_style)
    cap_level = round(float(rng.choice(config.rules.max_risk_pct_choices)), 4)
    rows = [
        _build_rule(pm_id, "r_01", RuleSource.MANDATE, cap_entry.param, cap_variant, cap_level, rng)
    ]

    entries = catalogue.rules.entries
    included = _choose_inclusion(entries, rng)
    included = _repair_inclusion(entries, included, config, rng)
    included_ordered = [entry for entry in entries if entry.param in included]

    for index, entry in enumerate(included_ordered, start=2):
        variant = entry.variant_for(asset_class, sub_style)
        level = _draw_level(variant, rng)
        rows.append(
            _build_rule(pm_id, f"r_{index:02d}", RuleSource.SELF, entry.param, variant, level, rng)
        )

    return rows
