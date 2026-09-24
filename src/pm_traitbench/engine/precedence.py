"""Action precedence: which of the rules that fired on one day for one
position wins, and what happens to the rest.

`resolve` never receives `exclude` or `cap` rules; those are pre-trade
constraints handled elsewhere. They are ranked here only so the table stays
total, and (being rank 0) would win if ever passed in error.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pm_traitbench.enums import Action
from pm_traitbench.tables.schema import Rule

RANK: dict[Action, int] = {
    Action.EXCLUDE: 0,
    Action.CAP: 0,
    Action.EXIT: 1,
    Action.SIGNPOST: 1,
    Action.ROLL: 2,
    Action.TRIM_HALF: 3,
    Action.TARGET: 3,
    Action.HOLD: 4,
    Action.NO_ADD: 5,
}


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one day's fired rules for one position."""

    winner: Rule | None
    also_acted: tuple[Rule, ...]
    overridden: tuple[Rule, ...]


def is_exit(rule: Rule) -> bool:
    """Whether `rule`'s action closes the position (exit or signpost)."""
    return rule.action in (Action.EXIT, Action.SIGNPOST)


def _pick_winner(tied: Sequence[Rule]) -> Rule:
    """Break a tie among rules sharing the minimum rank."""
    if len(tied) == 1:
        return tied[0]
    rank = RANK[tied[0].action]
    if rank == 1:
        stops = [rule for rule in tied if rule.param == "stop"]
        if stops:
            return min(stops, key=lambda rule: rule.rule_id)
    elif rank == 3:
        trims = [rule for rule in tied if rule.action == Action.TRIM_HALF]
        if trims:
            return min(trims, key=lambda rule: rule.rule_id)
    return min(tied, key=lambda rule: rule.rule_id)


def resolve(fired: Sequence[Rule]) -> Resolution:
    """Resolve one day's fired rules to a winner plus also-acted/overridden.

    Ties at the lowest rank all also-act alongside the winner; rules ranked
    above the winner are overridden, except that a winning roll leaves any
    fired target/trim_half rules also-acting, since the trim executes on the
    rolled position the same day.
    """
    if not fired:
        return Resolution(winner=None, also_acted=(), overridden=())

    min_rank = min(RANK[rule.action] for rule in fired)
    tied = [rule for rule in fired if RANK[rule.action] == min_rank]
    winner = _pick_winner(tied)

    also_acted = {rule for rule in tied if rule is not winner}
    remaining = [rule for rule in fired if rule not in tied]

    if winner.action == Action.ROLL:
        also_acted |= {rule for rule in remaining if RANK[rule.action] == 3}
        overridden = {rule for rule in remaining if RANK[rule.action] != 3}
    else:
        overridden = set(remaining)

    return Resolution(
        winner=winner,
        also_acted=tuple(sorted(also_acted, key=lambda rule: rule.rule_id)),
        overridden=tuple(sorted(overridden, key=lambda rule: rule.rule_id)),
    )
