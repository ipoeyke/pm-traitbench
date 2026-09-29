"""Prompt fragments shared by gate 2's recovery and classification requests."""

from collections.abc import Sequence

from pm_traitbench.enums import RuleScope
from pm_traitbench.tables.schema import Mandate, Rule


def mandate_line(mandate: Mandate) -> str:
    """One line stating the PM's mandate facts."""
    return (
        f"Mandate: asset class {mandate.asset_class.value}, sub-style {mandate.sub_style}, "
        f"book size {mandate.book_size}, risk unit {mandate.risk_unit}, "
        f"benchmark {mandate.benchmark}."
    )


def pm_rules_section(pm_rules: Sequence[Rule]) -> str:
    """The PM-scope rule texts among `pm_rules`, ordered by rule id."""
    ordered = sorted(
        (rule for rule in pm_rules if rule.scope == RuleScope.PM), key=lambda rule: rule.rule_id
    )
    lines = "\n".join(rule.text for rule in ordered)
    return f"Rules the PM is held to:\n{lines}"
