"""Tests for action precedence: resolve, RANK, is_exit."""

import pytest

from pm_traitbench.engine.precedence import RANK, Resolution, is_exit, resolve
from pm_traitbench.enums import Action, Op, RuleScope, RuleSource
from pm_traitbench.tables.schema import Rule

_PM_ID = "pm_001"


def _rule(
    *,
    rule_id: str,
    action: Action,
    param: str = "stop",
    scope: RuleScope = RuleScope.PM,
    trade_idea_id: str | None = None,
) -> Rule:
    return Rule(
        pm_id=_PM_ID,
        rule_id=rule_id,
        source=RuleSource.SELF,
        scope=scope,
        trade_idea_id=trade_idea_id,
        param=param,
        field="pnl_from_entry",
        op=Op.LE,
        level=-10.0,
        unit=None,
        window=1,
        action=action,
        text="test rule",
    )


def _idea_rule(*, rule_id: str, action: Action, param: str) -> Rule:
    return _rule(
        rule_id=rule_id,
        action=action,
        param=param,
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
    )


def test_rank_table_matches_expected_values() -> None:
    assert RANK == {
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


def test_resolve_no_rules_returns_none_winner() -> None:
    resolution = resolve([])
    assert resolution == Resolution(winner=None, also_acted=(), overridden=())


def test_resolve_single_stop_wins_alone() -> None:
    stop = _idea_rule(rule_id="r_01", action=Action.EXIT, param="stop")

    resolution = resolve([stop])

    assert resolution.winner is stop
    assert resolution.also_acted == ()
    assert resolution.overridden == ()


def test_resolve_stop_beats_signpost_at_rank_one() -> None:
    stop = _idea_rule(rule_id="r_01", action=Action.EXIT, param="stop")
    signpost = _idea_rule(rule_id="r_02", action=Action.SIGNPOST, param="signpost")

    resolution = resolve([signpost, stop])

    assert resolution.winner is stop
    assert resolution.also_acted == (signpost,)
    assert resolution.overridden == ()


def test_resolve_two_signposts_lowest_rule_id_wins() -> None:
    signpost_a = _idea_rule(rule_id="r_02", action=Action.SIGNPOST, param="signpost")
    signpost_b = _idea_rule(rule_id="r_01", action=Action.SIGNPOST, param="signpost")

    resolution = resolve([signpost_a, signpost_b])

    assert resolution.winner is signpost_b
    assert resolution.also_acted == (signpost_a,)
    assert resolution.overridden == ()


def test_resolve_stop_beats_two_signposts() -> None:
    stop = _idea_rule(rule_id="r_03", action=Action.EXIT, param="stop")
    signpost_a = _idea_rule(rule_id="r_01", action=Action.SIGNPOST, param="signpost")
    signpost_b = _idea_rule(rule_id="r_02", action=Action.SIGNPOST, param="signpost")

    resolution = resolve([signpost_a, signpost_b, stop])

    assert resolution.winner is stop
    assert resolution.also_acted == (signpost_a, signpost_b)
    assert resolution.overridden == ()


def test_resolve_target_and_trim_half_trim_wins() -> None:
    target = _idea_rule(rule_id="r_01", action=Action.TARGET, param="target")
    trim = _rule(rule_id="r_02", action=Action.TRIM_HALF, param="trim_at_target")

    resolution = resolve([target, trim])

    assert resolution.winner is trim
    assert resolution.also_acted == (target,)
    assert resolution.overridden == ()


def test_resolve_roll_beats_trim_half_which_also_acts() -> None:
    roll = _rule(rule_id="r_01", action=Action.ROLL, param="roll_before_expiry")
    trim = _rule(rule_id="r_02", action=Action.TRIM_HALF, param="trim_at_target")

    resolution = resolve([trim, roll])

    assert resolution.winner is roll
    assert resolution.also_acted == (trim,)
    assert resolution.overridden == ()


def test_resolve_roll_beats_target_which_also_acts() -> None:
    roll = _rule(rule_id="r_01", action=Action.ROLL, param="roll_before_expiry")
    target = _idea_rule(rule_id="r_02", action=Action.TARGET, param="target")

    resolution = resolve([target, roll])

    assert resolution.winner is roll
    assert resolution.also_acted == (target,)
    assert resolution.overridden == ()


def test_resolve_roll_beats_both_trim_half_and_target_both_also_act() -> None:
    roll = _rule(rule_id="r_01", action=Action.ROLL, param="roll_before_expiry")
    trim = _rule(rule_id="r_03", action=Action.TRIM_HALF, param="trim_at_target")
    target = _idea_rule(rule_id="r_02", action=Action.TARGET, param="target")

    resolution = resolve([trim, target, roll])

    assert resolution.winner is roll
    assert resolution.also_acted == (target, trim)
    assert resolution.overridden == ()


def test_resolve_hold_alone_wins_with_nothing_else() -> None:
    hold = _rule(rule_id="r_01", action=Action.HOLD, param="min_holding_period")

    resolution = resolve([hold])

    assert resolution.winner is hold
    assert resolution.also_acted == ()
    assert resolution.overridden == ()


def test_resolve_hold_is_overridden_by_trim_half() -> None:
    hold = _rule(rule_id="r_01", action=Action.HOLD, param="min_holding_period")
    trim = _rule(rule_id="r_02", action=Action.TRIM_HALF, param="trim_at_target")

    resolution = resolve([hold, trim])

    assert resolution.winner is trim
    assert resolution.also_acted == ()
    assert resolution.overridden == (hold,)


def test_resolve_hold_is_overridden_by_exit() -> None:
    hold = _rule(rule_id="r_01", action=Action.HOLD, param="min_holding_period")
    stop = _idea_rule(rule_id="r_02", action=Action.EXIT, param="stop")

    resolution = resolve([hold, stop])

    assert resolution.winner is stop
    assert resolution.also_acted == ()
    assert resolution.overridden == (hold,)


def test_resolve_hold_is_overridden_by_roll_and_not_also_acted() -> None:
    hold = _rule(rule_id="r_02", action=Action.HOLD, param="min_holding_period")
    roll = _rule(rule_id="r_01", action=Action.ROLL, param="roll_before_expiry")

    resolution = resolve([hold, roll])

    assert resolution.winner is roll
    assert resolution.also_acted == ()
    assert resolution.overridden == (hold,)


def test_resolve_no_add_always_overridden() -> None:
    no_add = _rule(rule_id="r_01", action=Action.NO_ADD, param="no_add_before_trigger")
    stop = _idea_rule(rule_id="r_02", action=Action.EXIT, param="stop")

    resolution = resolve([no_add, stop])

    assert resolution.winner is stop
    assert resolution.also_acted == ()
    assert resolution.overridden == (no_add,)


def test_resolve_exclude_outranks_stop_since_never_passed_together() -> None:
    # exclude/cap are rank 0 and should never be passed to resolve alongside
    # other rules; if they are, they win, as RANK dictates.
    exclude = _rule(rule_id="r_02", action=Action.EXCLUDE, param="exclusion")
    stop = _idea_rule(rule_id="r_01", action=Action.EXIT, param="stop")

    resolution = resolve([stop, exclude])

    assert resolution.winner is exclude
    assert resolution.also_acted == ()
    assert resolution.overridden == (stop,)


def test_resolve_overridden_ordered_by_rule_id() -> None:
    stop = _idea_rule(rule_id="r_01", action=Action.EXIT, param="stop")
    hold = _rule(rule_id="r_03", action=Action.HOLD, param="min_holding_period")
    no_add = _rule(rule_id="r_02", action=Action.NO_ADD, param="no_add_before_trigger")

    resolution = resolve([hold, no_add, stop])

    assert resolution.winner is stop
    assert resolution.overridden == (no_add, hold)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action.EXIT, True),
        (Action.SIGNPOST, True),
        (Action.TRIM_HALF, False),
        (Action.TARGET, False),
        (Action.HOLD, False),
        (Action.ROLL, False),
        (Action.NO_ADD, False),
        (Action.EXCLUDE, False),
        (Action.CAP, False),
    ],
)
def test_is_exit(action: Action, expected: bool) -> None:
    rule = _rule(rule_id="r_01", action=action, param="p")
    assert is_exit(rule) is expected
