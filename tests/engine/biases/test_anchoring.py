"""Tests for the fixed round-level anchoring bias."""

import math

import pytest

from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.adapters.rates_credit import RatesCreditAdapter
from pm_traitbench.engine.biases.anchoring import entry_anchor, evaluate, flag
from pm_traitbench.engine.constants import ANCHOR_FRACTION
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import Position
from pm_traitbench.enums import Expression, Side
from pm_traitbench.tables.schema import Leg

_HORIZON = 20


class _FakeAdapter:
    """A stub adapter whose `round_step` always returns a fixed level."""

    def __init__(self, rounded: float) -> None:
        self._rounded = rounded

    def round_step(self, series: Series, level: float) -> float:
        return self._rounded


def _params(*, active: bool) -> EffectiveParams:
    values = {param: 0.5 for param in BIAS_PARAMS}
    active_map = {param: (param == "anchoring_rho" and active) for param in BIAS_PARAMS}
    return EffectiveParams(values=values, active=active_map)


def _position(
    *, series: Series, side: Side, entry_level: float, target_level: float, anchor_level, anchored
) -> Position:
    leg = Leg(instrument_id="X", tenor=None, side=side, weight=1.0)
    return Position(
        trade_idea_id="ti_001",
        expression=Expression.OUTRIGHT,
        instrument_id="X",
        legs=(leg,),
        series=series,
        side=side,
        entry_t=0,
        entry_level=entry_level,
        target_level=target_level,
        stop_level=entry_level - (target_level - entry_level),
        sd_h_at_entry=1.0,
        forecast=0.0,
        size_pct_book=1.0,
        original_size_pct_book=1.0,
        size_at_entry=1.0,
        conviction=1,
        size_rank=1,
        triggers_fired=0,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=0,
        anchor_level=anchor_level,
        anchored=anchored,
    )


# --- entry_anchor ------------------------------------------------------------


def test_entry_anchor_for_a_long_price_series() -> None:
    adapter = EquitiesAdapter("value", _HORIZON)
    series = adapter.outright_series("EQ-0001")
    entry_level = 100.0 * math.log(100.0)
    target_level = entry_level + 200.0

    anchor = entry_anchor(adapter, series, entry_level, target_level)

    raw = entry_level + ANCHOR_FRACTION * (target_level - entry_level)
    assert anchor == pytest.approx(adapter.round_step(series, raw))
    assert entry_level < anchor < target_level


def test_entry_anchor_for_a_short_bearish_rates_outright() -> None:
    adapter = RatesCreditAdapter("sovereign_rates", _HORIZON)
    series = adapter.outright_series("RT-USD")
    assert series.bullish_sign == -1
    entry_level = 400.0
    target_level = entry_level - 100.0

    anchor = entry_anchor(adapter, series, entry_level, target_level)

    raw = entry_level + ANCHOR_FRACTION * (target_level - entry_level)
    assert anchor == pytest.approx(adapter.round_step(series, raw))
    assert target_level < anchor < entry_level


def test_entry_anchor_is_none_when_rounded_level_reaches_or_passes_the_target() -> None:
    series = Series(legs=(LegRef("X", None, 1.0),), bullish_sign=1, unit="pct")
    entry_level, target_level = 100.0, 110.0

    assert entry_anchor(_FakeAdapter(110.0), series, entry_level, target_level) is None
    assert entry_anchor(_FakeAdapter(111.0), series, entry_level, target_level) is None


def test_entry_anchor_is_none_when_rounded_level_falls_on_the_entry_side() -> None:
    series = Series(legs=(LegRef("X", None, 1.0),), bullish_sign=1, unit="pct")
    entry_level, target_level = 100.0, 110.0

    assert entry_anchor(_FakeAdapter(100.0), series, entry_level, target_level) is None
    assert entry_anchor(_FakeAdapter(99.0), series, entry_level, target_level) is None


def test_entry_anchor_valid_between_entry_and_target() -> None:
    series = Series(legs=(LegRef("X", None, 1.0),), bullish_sign=1, unit="pct")
    entry_level, target_level = 100.0, 110.0

    assert entry_anchor(_FakeAdapter(105.0), series, entry_level, target_level) == pytest.approx(
        105.0
    )


# --- evaluate ------------------------------------------------------------


def test_evaluate_anchored_position_reaches_at_the_anchor_not_one_tick_short() -> None:
    series = Series(legs=(LegRef("X", None, 1.0),), bullish_sign=1, unit="pct")
    pos = _position(
        series=series,
        side=Side.BUY,
        entry_level=100.0,
        target_level=120.0,
        anchor_level=108.0,
        anchored=True,
    )

    at_anchor = evaluate(pos, 108.0)
    short_of_anchor = evaluate(pos, 107.999)

    assert at_anchor.reached is True
    assert at_anchor.effective_exit_level == pytest.approx(108.0)
    assert short_of_anchor.reached is False


def test_evaluate_unanchored_position_never_reaches_and_targets_effective_exit() -> None:
    series = Series(legs=(LegRef("X", None, 1.0),), bullish_sign=1, unit="pct")
    pos = _position(
        series=series,
        side=Side.BUY,
        entry_level=100.0,
        target_level=120.0,
        anchor_level=108.0,
        anchored=False,
    )

    result = evaluate(pos, 500.0)

    assert result.reached is False
    assert result.effective_exit_level == pytest.approx(120.0)
    assert result.anchor_level == pytest.approx(108.0)


# --- flag ------------------------------------------------------------


def test_flag_set_only_when_active() -> None:
    assert flag(_params(active=True)) == "anchoring:exit_at_anchor"
    assert flag(_params(active=False)) is None
