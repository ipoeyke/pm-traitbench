"""The condition grammar evaluator: whether a rule's condition holds, and its
run-counter and firing logic.

A rule's condition is one comparison (`op`) between a named field on the
current day and the rule's fixed `level`. `evaluate_day` runs every rule that
applies to a position on one day and returns the ones that fired.
"""

from collections.abc import Mapping, Sequence

from pm_traitbench.engine.state import Position
from pm_traitbench.enums import Op, RuleScope
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Rule

FieldValues = Mapping[str, float | int | str | frozenset[str]]

# PM-scope params evaluated every day, regardless of trade idea; the rest
# (stop_loss, max_risk_pct, no_add_before_trigger, max_positions, exclusion)
# are constraints or sources consulted elsewhere, not daily conditions.
DAILY_PM_PARAMS = frozenset({"trim_at_target", "min_holding_period", "roll_before_expiry"})

_NUMERIC_OPS = {Op.LE, Op.GE, Op.LT, Op.GT}


def _as_float(value: float | int | str | frozenset[str]) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def condition_holds(rule: Rule, fields: FieldValues) -> bool:
    """Whether `rule`'s condition holds against today's `fields`."""
    if rule.field not in fields:
        raise EngineError(f"rule '{rule.rule_id}' references missing field '{rule.field}'")
    value = fields[rule.field]

    if isinstance(value, frozenset):
        if rule.op == Op.EQ:
            return rule.level in value
        if rule.op == Op.NE:
            return rule.level not in value
        raise EngineError(f"rule '{rule.rule_id}' op '{rule.op.value}' is invalid on a set field")

    if rule.op in _NUMERIC_OPS:
        left, right = float(value), float(rule.level)  # type: ignore[arg-type]
        if rule.op == Op.LE:
            return left <= right
        if rule.op == Op.GE:
            return left >= right
        if rule.op == Op.LT:
            return left < right
        return left > right

    # == and !=: compare numerically when both sides parse as floats, else as strings.
    left_f, right_f = _as_float(value), _as_float(rule.level)
    if left_f is not None and right_f is not None:
        equal = left_f == right_f
    else:
        equal = str(value) == str(rule.level)
    return equal if rule.op == Op.EQ else not equal


def _is_consumable(rule: Rule) -> bool:
    return rule.scope == RuleScope.IDEA or rule.param == "trim_at_target"


def advance(position: Position, rule: Rule, holds: bool) -> tuple[Position, bool]:
    """Advance `rule`'s run counter on `position` and report whether it fired."""
    if _is_consumable(rule) and rule.rule_id in position.consumed_rule_ids:
        return position, False
    new_counter = position.counter(rule.rule_id) + 1 if holds else 0
    new_position = position.with_counter(rule.rule_id, new_counter)
    fired = holds and new_counter >= rule.window
    return new_position, fired


def _applies_today(rule: Rule, position: Position) -> bool:
    if rule.trade_idea_id is None:
        return rule.param in DAILY_PM_PARAMS
    return rule.trade_idea_id == position.trade_idea_id


def evaluate_day(
    position: Position, rules: Sequence[Rule], fields: FieldValues
) -> tuple[Position, list[Rule]]:
    """Advance every rule that applies to `position` today, in rule id order."""
    applicable = sorted(
        (rule for rule in rules if _applies_today(rule, position)), key=lambda r: r.rule_id
    )
    fired_rules: list[Rule] = []
    for rule in applicable:
        holds = condition_holds(rule, fields)
        position, fired = advance(position, rule, holds)
        if fired:
            fired_rules.append(rule)
    return position, fired_rules


def required_fields(rules: Sequence[Rule]) -> frozenset[str]:
    """Fields that `evaluate_day` would read: daily PM-scope rules plus every idea-scope rule."""
    return frozenset(
        rule.field
        for rule in rules
        if rule.trade_idea_id is not None or rule.param in DAILY_PM_PARAMS
    )
