"""The adapter protocol: per-asset-class behaviour the engine calls into to
build legs and series, size ideas and evaluate rule fields.

Adapters are pure and hold no mutable state: every call takes the market
view, day and PM state it needs as an argument rather than caching anything.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol

import numpy as np

from pm_traitbench.engine.constants import TRAILING_HIGH_DAYS
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.enums import (
    Action,
    AssetClass,
    Expression,
    InstrumentKind,
    Op,
    Regime,
    RuleScope,
    Side,
)
from pm_traitbench.tables.schema import Instrument, Rule


class Adapter(Protocol):
    """Per-asset-class behaviour: universe, leg construction, sizing and rule fields."""

    asset_class: AssetClass
    FIELDS: ClassVar[frozenset[str]]

    def universe(
        self, instruments: Mapping[str, Instrument], rules: Sequence[Rule]
    ) -> tuple[str, ...]: ...

    def forms(self, sub_style: str) -> tuple[Expression, ...]: ...

    def build_legs(
        self,
        form: Expression,
        instrument_id: str,
        view: MarketView,
        t: int,
        universe: Sequence[str],
        held: frozenset[str],
        rng: np.random.Generator,
    ) -> tuple[LegRef, ...] | None: ...

    def series(self, form: Expression, legs: tuple[LegRef, ...]) -> Series: ...

    def outright_series(self, instrument_id: str) -> Series: ...

    def stop_distance(self, rule: Rule, series: Series) -> float: ...

    def position_fields(
        self, pos: Position, view: MarketView, t: int, state: PmState, pnl_unit: float
    ) -> dict[str, float | int | str | frozenset[str]]: ...

    def size_and_risk(
        self,
        size_pct_book: float,
        legs: tuple[LegRef, ...],
        view: MarketView,
        t: int,
        book_size: float,
    ) -> tuple[float, float]: ...

    def leg_risk_amount(
        self, leg: LegRef, size: float, view: MarketView, risk_amount: float
    ) -> float: ...

    def leg_price(self, leg: LegRef, view: MarketView, t: int) -> float: ...

    def instrument_type(self, instrument_id: str, view: MarketView) -> InstrumentKind: ...

    def anchors(self, pos: Position, view: MarketView, t: int) -> tuple[float, ...]: ...

    def peer_ids(
        self, instrument_id: str, instruments: Mapping[str, Instrument]
    ) -> tuple[str, ...]: ...

    def peer_label(self, instrument_id: str, instruments: Mapping[str, Instrument]) -> str: ...

    def round_step(self, series: Series, level: float) -> float: ...

    def leg_bullish(self, leg: LegRef) -> int: ...


def pnl_unit(pos: Position, level_now: float) -> float:
    """Mark-to-market P&L in series units, signed so a positive value favours the position."""
    return pos.series.bullish_sign * pos.side_sign * (level_now - pos.entry_level)


def tracked_level(pos: Position, view: MarketView, t: int) -> float:
    """The series level on day `t`, re-based into the position's pre-roll tenor frame.

    Mid-roll the position's legs read a different tenor; `rolled_offset` is the accumulated
    raw-level gap from every retag, so subtracting it recovers a level comparable with
    `entry_level`/`stop_level`/`target_level`, which are never themselves shifted.
    """
    return view.level(pos.series, t) - pos.rolled_offset


def leg_side(pos_side_sign: int, bullish_sign: int, coeff: float, leg_bullish: int) -> Side:
    """The ledger side for one leg, from the position's direction and the leg's own sign."""
    return Side.BUY if pos_side_sign * bullish_sign * coeff * leg_bullish > 0 else Side.SELL


def target_reached(pos: Position, level_now: float) -> bool:
    """Whether the series has reached or passed the position's target level."""
    return (level_now - pos.target_level) * (-pos.adverse_dir) >= 0


def common_fields(
    pos: Position, state: PmState, t: int, pnl_unit: float, target_hit_now: bool
) -> dict[str, float | int]:
    """Rule fields every adapter shares.

    The current level is recovered algebraically from `pnl_unit` rather than
    read from a market view, so this stays a pure function of position state.
    """
    level_now = pos.entry_level + pnl_unit * pos.series.bullish_sign * pos.side_sign
    return {
        "sessions_held": t - pos.entry_t,
        "triggers_fired": pos.triggers_fired,
        "n_positions": state.n_positions,
        "size_pct_book": pos.size_pct_book,
        "target_hit": 1 if target_hit_now else 0,
        "level": level_now,
    }


def relative_move(view: MarketView, adapter: Adapter, pos: Position, t: int, h: int) -> float:
    """How far the position's series has moved relative to its peer group over `h` days.

    Positive when the relative move favours the PM's position.
    """
    peers = adapter.peer_ids(pos.instrument_id, view.instruments)
    peer_move = view.peer_move(peers, adapter.outright_series, t, h)
    own_move = view.trailing_move(pos.series, t, h)
    return pos.series.bullish_sign * pos.side_sign * (own_move - peer_move)


def excluded_values(rules: Sequence[Rule], field: str) -> frozenset[str]:
    """Values a PM's exclusion rules bar for `field`, read by an adapter's universe filter."""
    return frozenset(
        rule.level
        for rule in rules
        if rule.scope == RuleScope.PM
        and rule.param == "exclusion"
        and rule.action == Action.EXCLUDE
        and rule.op == Op.NE
        and rule.field == field
    )


def standard_anchors(
    adapter: Adapter, pos: Position, view: MarketView, t: int
) -> tuple[float, ...]:
    """Entry level; the round level in range regime; the trailing extreme against the position.

    All three in the position's tracked (pre-roll) frame: `entry_level` is never shifted, so it
    is used as-is; the round level and trailing extreme read the raw series, so each is re-based
    by `-rolled_offset` (the round level via `tracked_level` as its input) before being returned.
    """
    values = [pos.entry_level]
    if view.regime(t) == Regime.RANGE:
        values.append(adapter.round_step(pos.series, tracked_level(pos, view, t)))
    if pos.adverse_dir < 0:
        values.append(view.trailing_high(pos.series, t, TRAILING_HIGH_DAYS) - pos.rolled_offset)
    else:
        values.append(view.trailing_low(pos.series, t, TRAILING_HIGH_DAYS) - pos.rolled_offset)
    return tuple(values)
