"""Tests for the condition grammar evaluator: condition_holds, advance, evaluate_day."""

import pytest

from pm_traitbench.engine.rules_eval import (
    advance,
    condition_holds,
    evaluate_day,
    required_fields,
)
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import Position
from pm_traitbench.enums import Action, AssetClass, Expression, Op, RuleScope, RuleSource, Side
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Leg, Rule

_PM_ID = "pm_001"


def _rule(
    *,
    rule_id: str = "r_01",
    scope: RuleScope = RuleScope.PM,
    trade_idea_id: str | None = None,
    param: str = "stop_loss",
    field: str = "pnl_from_entry",
    op: Op = Op.LE,
    level: float | str = -10.0,
    window: int = 1,
    action: Action = Action.EXIT,
) -> Rule:
    return Rule(
        pm_id=_PM_ID,
        rule_id=rule_id,
        source=RuleSource.SELF,
        scope=scope,
        trade_idea_id=trade_idea_id,
        param=param,
        field=field,
        op=op,
        level=level,
        unit=None,
        window=window,
        action=action,
        text="test rule",
    )


def _position(
    *,
    trade_idea_id: str = "ti_001",
    consumed: frozenset[str] = frozenset(),
    run_counters: tuple[tuple[str, int], ...] = (),
) -> Position:
    leg = Leg(instrument_id="EQ-0001", tenor=None, side=Side.BUY, weight=1.0)
    series = Series(
        legs=(LegRef(instrument_id="EQ-0001", tenor=None, coeff=1.0),),
        bullish_sign=1,
        unit="pct",
    )
    return Position(
        trade_idea_id=trade_idea_id,
        expression=Expression.OUTRIGHT,
        instrument_id="EQ-0001",
        legs=(leg,),
        series=series,
        side=Side.BUY,
        entry_t=0,
        entry_level=100.0,
        target_level=110.0,
        stop_level=90.0,
        sd_h_at_entry=1.0,
        forecast=105.0,
        size_pct_book=1.0,
        original_size_pct_book=1.0,
        conviction=1,
        size_rank=1,
        triggers_fired=0,
        consumed_rule_ids=consumed,
        run_counters=run_counters,
        size_changed_t=0,
    )


@pytest.mark.parametrize(
    ("op", "field_value", "level", "expected"),
    [
        (Op.LE, 5.0, 5.0, True),
        (Op.LE, 6.0, 5.0, False),
        (Op.GE, 5.0, 5.0, True),
        (Op.GE, 4.0, 5.0, False),
        (Op.LT, 4.0, 5.0, True),
        (Op.LT, 5.0, 5.0, False),
        (Op.GT, 6.0, 5.0, True),
        (Op.GT, 5.0, 5.0, False),
        (Op.EQ, 5.0, 5.0, True),
        (Op.EQ, 4.0, 5.0, False),
        (Op.NE, 4.0, 5.0, True),
        (Op.NE, 5.0, 5.0, False),
    ],
)
def test_condition_holds_numeric_ops(
    op: Op, field_value: float, level: float, expected: bool
) -> None:
    rule = _rule(op=op, level=level, field="pnl_from_entry")
    assert condition_holds(rule, {"pnl_from_entry": field_value}) is expected


def test_condition_holds_string_not_equal() -> None:
    rule = _rule(op=Op.NE, level="energy", field="sector")
    assert condition_holds(rule, {"sector": "financials"}) is True
    assert condition_holds(rule, {"sector": "energy"}) is False


def test_condition_holds_set_membership_equal() -> None:
    rule = _rule(op=Op.EQ, level="earnings", field="event")
    assert condition_holds(rule, {"event": frozenset({"earnings", "cb_meeting"})}) is True
    assert condition_holds(rule, {"event": frozenset({"cb_meeting"})}) is False


def test_condition_holds_set_absence_not_equal() -> None:
    rule = _rule(op=Op.NE, level="earnings", field="event")
    assert condition_holds(rule, {"event": frozenset({"cb_meeting"})}) is True
    assert condition_holds(rule, {"event": frozenset({"earnings"})}) is False


def test_condition_holds_set_with_other_op_raises() -> None:
    rule = _rule(op=Op.LE, level="earnings", field="event")
    with pytest.raises(EngineError):
        condition_holds(rule, {"event": frozenset({"earnings"})})


def test_condition_holds_missing_field_raises() -> None:
    rule = _rule(rule_id="r_09", field="sector", op=Op.NE, level="energy")
    with pytest.raises(EngineError, match="r_09"):
        condition_holds(rule, {})
    with pytest.raises(EngineError, match="sector"):
        condition_holds(rule, {})


def test_advance_fires_on_third_consecutive_holding_day() -> None:
    rule = _rule(rule_id="r_10", window=3, field="pnl_from_entry", op=Op.LE, level=-5.0)
    pos = _position()

    pos, fired_1 = advance(pos, rule, holds=True)
    assert fired_1 is False
    assert pos.counter("r_10") == 1

    pos, fired_2 = advance(pos, rule, holds=True)
    assert fired_2 is False
    assert pos.counter("r_10") == 2

    pos, fired_3 = advance(pos, rule, holds=True)
    assert fired_3 is True
    assert pos.counter("r_10") == 3


def test_advance_resets_counter_when_condition_breaks() -> None:
    rule = _rule(rule_id="r_10", window=3)
    pos = _position(run_counters=(("r_10", 2),))

    pos, fired = advance(pos, rule, holds=False)
    assert fired is False
    assert pos.counter("r_10") == 0


def test_advance_idea_rule_consumed_never_fires_again() -> None:
    rule = _rule(
        rule_id="r_20",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
        param="stop",
        field="pnl_from_entry",
        op=Op.LE,
        level=-5.0,
        window=1,
        action=Action.EXIT,
    )
    pos = _position(consumed=frozenset({"r_20"}), run_counters=(("r_20", 0),))

    new_pos, fired = advance(pos, rule, holds=True)
    assert fired is False
    assert new_pos.counter("r_20") == 0


def test_advance_trim_at_target_consumed_never_fires_again() -> None:
    rule = _rule(
        rule_id="r_03",
        param="trim_at_target",
        field="target_hit",
        op=Op.EQ,
        level=1.0,
        window=1,
        action=Action.TRIM_HALF,
    )
    pos = _position(consumed=frozenset({"r_03"}))

    new_pos, fired = advance(pos, rule, holds=True)
    assert fired is False
    assert new_pos.counter("r_03") == 0


def test_evaluate_day_skips_stop_loss_and_max_risk_pct(neutral_pm) -> None:
    _, _, rules = neutral_pm(AssetClass.EQUITIES, "growth")
    pos = _position()
    # pnl_from_entry (stop_loss) and size_pct_book (max_risk_pct) are absent; if
    # evaluate_day tried those rules, the missing field would raise.
    fields = {"target_hit": 0.0, "triggers_fired": 1, "sessions_held": 0, "n_positions": 0}

    _, fired = evaluate_day(pos, rules, fields)

    fired_params = {rule.param for rule in fired}
    assert "stop_loss" not in fired_params
    assert "max_risk_pct" not in fired_params


def test_evaluate_day_includes_idea_scope_rule_for_matching_idea_only() -> None:
    matching = _rule(
        rule_id="r_20",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
        param="stop",
        field="pnl_from_entry",
        op=Op.LE,
        level=-5.0,
    )
    other_idea = _rule(
        rule_id="r_21",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_999",
        param="stop",
        field="pnl_from_entry",
        op=Op.LE,
        level=-5.0,
    )
    pos = _position(trade_idea_id="ti_001")
    fields = {"pnl_from_entry": -10.0}

    _, fired = evaluate_day(pos, [other_idea, matching], fields)

    assert [rule.rule_id for rule in fired] == ["r_20"]


def test_evaluate_day_applies_rules_in_rule_id_order() -> None:
    rule_b = _rule(
        rule_id="r_02",
        param="min_holding_period",
        field="sessions_held",
        op=Op.GE,
        level=1.0,
    )
    rule_a = _rule(
        rule_id="r_01",
        param="trim_at_target",
        field="target_hit",
        op=Op.EQ,
        level=1.0,
        action=Action.TRIM_HALF,
    )
    pos = _position()
    fields = {"sessions_held": 5, "target_hit": 1.0}

    _, fired = evaluate_day(pos, [rule_b, rule_a], fields)

    assert [rule.rule_id for rule in fired] == ["r_01", "r_02"]


def test_required_fields_excludes_skipped_params() -> None:
    stop_loss_rule = _rule(
        rule_id="r_01", param="stop_loss", field="pnl_from_entry", op=Op.LE, level=-10.0
    )
    max_risk_rule = _rule(
        rule_id="r_02", param="max_risk_pct", field="size_pct_book", op=Op.LE, level=10.0
    )
    trim_rule = _rule(
        rule_id="r_03",
        param="trim_at_target",
        field="target_hit",
        op=Op.EQ,
        level=1.0,
        action=Action.TRIM_HALF,
    )
    idea_rule = _rule(
        rule_id="r_20",
        scope=RuleScope.IDEA,
        trade_idea_id="ti_001",
        param="stop",
        field="signpost",
        op=Op.GE,
        level=1.0,
    )

    fields = required_fields([stop_loss_rule, max_risk_rule, trim_rule, idea_rule])

    assert fields == frozenset({"target_hit", "signpost"})
