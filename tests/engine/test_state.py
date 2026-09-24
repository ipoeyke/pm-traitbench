"""Tests for engine position and PM state: sign helpers, counters and ordering."""

from dataclasses import FrozenInstanceError, replace

import pytest

from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position, idea_id, rule_id
from pm_traitbench.enums import Expression, Side
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Leg


def _make_position(
    *, side: Side = Side.BUY, bullish_sign: int = 1, trade_idea_id: str = "ti_001"
) -> Position:
    leg = Leg(instrument_id="EQ-0001", tenor=None, side=side, weight=1.0)
    series = Series(
        legs=(LegRef(instrument_id="EQ-0001", tenor=None, coeff=1.0),),
        bullish_sign=bullish_sign,
        unit="pct",
    )
    return Position(
        trade_idea_id=trade_idea_id,
        expression=Expression.OUTRIGHT,
        instrument_id="EQ-0001",
        legs=(leg,),
        series=series,
        side=side,
        entry_t=0,
        entry_level=100.0,
        target_level=110.0,
        stop_level=90.0,
        sd_h_at_entry=1.0,
        forecast=105.0,
        size_pct_book=1.0,
        original_size_pct_book=1.0,
        size_at_entry=1.0,
        conviction=1,
        size_rank=1,
        triggers_fired=0,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=0,
    )


@pytest.mark.parametrize(
    ("side", "bullish_sign", "expected"),
    [
        (Side.BUY, 1, -1),
        (Side.BUY, -1, 1),
        (Side.SELL, 1, 1),
        (Side.SELL, -1, -1),
    ],
)
def test_adverse_dir_sign_combinations(side: Side, bullish_sign: int, expected: int) -> None:
    pos = _make_position(side=side, bullish_sign=bullish_sign)
    assert pos.side_sign == (1 if side == Side.BUY else -1)
    assert pos.adverse_dir == expected


def test_with_counter_returns_new_object_and_leaves_old_unchanged() -> None:
    pos = _make_position()
    updated = pos.with_counter("r_01", 3)

    assert updated is not pos
    assert updated.counter("r_01") == 3
    assert pos.counter("r_01") == 0


def test_with_counter_overwrites_existing_entry() -> None:
    pos = _make_position().with_counter("r_01", 2)
    updated = pos.with_counter("r_01", 5)

    assert updated.counter("r_01") == 5
    assert len(updated.run_counters) == 1


def test_with_counter_replaces_in_place_keeping_tuple_order() -> None:
    pos = _make_position().with_counter("r_01", 1).with_counter("r_02", 2)
    updated = pos.with_counter("r_01", 9)

    assert updated.run_counters == (("r_01", 9), ("r_02", 2))


def test_with_counter_equal_regardless_of_update_order() -> None:
    # Updating r_01 in place must match a position where r_01 was inserted
    # with its final value from the start - the update must not reorder it.
    a = _make_position().with_counter("r_01", 1).with_counter("r_02", 2).with_counter("r_01", 9)
    b = _make_position().with_counter("r_01", 9).with_counter("r_02", 2)

    assert a.run_counters == b.run_counters


def test_position_is_frozen() -> None:
    pos = _make_position()
    with pytest.raises(FrozenInstanceError):
        pos.entry_level = 200.0  # type: ignore[misc]


def test_pm_state_is_frozen() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    with pytest.raises(FrozenInstanceError):
        state.next_idea = 2  # type: ignore[misc]


def test_pm_state_add_position_keeps_order_by_id() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    state = state.add_position(_make_position(trade_idea_id="ti_003"))
    state = state.add_position(_make_position(trade_idea_id="ti_001"))
    state = state.add_position(_make_position(trade_idea_id="ti_002"))

    assert [p.trade_idea_id for p in state.positions] == ["ti_001", "ti_002", "ti_003"]
    assert state.n_positions == 3


def test_pm_state_add_position_rejects_duplicate_id() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    state = state.add_position(_make_position(trade_idea_id="ti_001"))

    with pytest.raises(EngineError):
        state.add_position(_make_position(trade_idea_id="ti_001"))


def test_pm_state_add_position_leaves_original_unchanged() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    updated = state.add_position(_make_position(trade_idea_id="ti_001"))

    assert state.positions == ()
    assert updated is not state
    assert [p.trade_idea_id for p in updated.positions] == ["ti_001"]


def test_pm_state_remove_position_unknown_id_raises() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)

    with pytest.raises(EngineError, match="ti_999"):
        state.remove_position("ti_999")


def test_pm_state_replace_position_unknown_id_raises() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)

    with pytest.raises(EngineError, match="ti_999"):
        state.replace_position(_make_position(trade_idea_id="ti_999"))


def test_pm_state_replace_and_remove_position() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    state = state.add_position(_make_position(trade_idea_id="ti_001"))

    replaced = state.replace_position(_make_position(trade_idea_id="ti_001", side=Side.SELL))
    assert replaced.position("ti_001").side == Side.SELL

    removed = replaced.remove_position("ti_001")
    assert removed.n_positions == 0


def test_pm_state_replace_position_leaves_original_unchanged() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    state = state.add_position(_make_position(trade_idea_id="ti_001", side=Side.BUY))

    replaced = state.replace_position(_make_position(trade_idea_id="ti_001", side=Side.SELL))

    assert state.position("ti_001").side == Side.BUY
    assert replaced is not state
    assert replaced.position("ti_001").side == Side.SELL


def test_pm_state_remove_position_leaves_original_unchanged() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    state = state.add_position(_make_position(trade_idea_id="ti_001"))

    removed = state.remove_position("ti_001")

    assert state.n_positions == 1
    assert removed is not state
    assert removed.n_positions == 0


def test_pm_state_held_instruments() -> None:
    state = PmState(pm_id="pm_001", positions=(), next_idea=1, next_rule=1)
    state = state.add_position(_make_position(trade_idea_id="ti_001"))
    assert state.held_instruments == frozenset({"EQ-0001"})


def test_pm_state_held_instruments_include_a_pair_partner() -> None:
    base = _make_position(trade_idea_id="ti_001")
    partner = Leg(instrument_id="EQ-0002", tenor=None, side=Side.SELL, weight=1.0)
    pair = replace(base, expression=Expression.PAIR, legs=(*base.legs, partner))
    state = PmState(pm_id="pm_001", positions=(pair,), next_idea=2, next_rule=1)
    assert state.held_instruments == frozenset({"EQ-0001", "EQ-0002"})


def test_idea_id_and_rule_id_formatting() -> None:
    assert idea_id(7) == "ti_007"
    assert rule_id(7) == "r_07"
