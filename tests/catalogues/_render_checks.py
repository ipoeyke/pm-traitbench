"""Shared render-cleanliness checks for rule catalogue templates."""

import re

from pm_traitbench.catalogues.loader import render_template
from pm_traitbench.catalogues.models import Catalogue, RuleVariant

_REPEATED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def _representative_levels(variant: RuleVariant, is_cap: bool) -> tuple[float | str, ...]:
    """Levels worth rendering for one rule variant: min, max, and each choice."""
    if is_cap:
        return (10.0,)
    if variant.level_choices:
        return variant.level_choices
    return (variant.level_min, variant.level_max)


def assert_every_rule_template_renders_cleanly(catalogue: Catalogue) -> None:
    """Render every template of every rule variant and assert it reads as clean PM text."""
    for entry in (catalogue.rules.mandate_cap, *catalogue.rules.entries):
        is_cap = entry is catalogue.rules.mandate_cap
        for variant in entry.variants:
            for level in _representative_levels(variant, is_cap):
                for template in variant.templates:
                    rendered = render_template(template, level, variant.unit)
                    context = (entry.param, variant.asset_class, template, rendered)
                    assert "{" not in rendered and "}" not in rendered, context
                    assert "pct" not in rendered.lower(), context
                    assert "None" not in rendered, context
                    assert "  " not in rendered, context
                    assert "_" not in rendered, context
                    assert rendered == rendered.strip(), context
                    assert not _REPEATED_WORD.search(rendered), context
