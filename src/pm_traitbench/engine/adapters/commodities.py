"""Commodities adapter: outright futures and calendar spread ideas."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from pm_traitbench.engine.adapters.base import (
    common_fields,
    excluded_values,
    relative_move,
    standard_anchors,
    target_reached,
)
from pm_traitbench.engine.constants import CALENDAR_BACK_TENORS
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.enums import FUTURES_TENORS, AssetClass, Expression, InstrumentKind, Tenor
from pm_traitbench.errors import EngineError
from pm_traitbench.market.constants import CONTRACT_MULTIPLIER
from pm_traitbench.market.levels import log_grid_step, nearest_level
from pm_traitbench.tables.schema import Instrument, Rule

_FUTURES_DIRECTIONAL = "commodity_futures_directional"
_CURVE_AND_SPREAD = "curve_and_spread"
_SUB_STYLES = (_FUTURES_DIRECTIONAL, _CURVE_AND_SPREAD)

_STOP_FIELD = "pnl_from_entry"
_M1 = Tenor.M1
# No expiry on record: large enough that a roll_before_expiry rule never fires.
_NO_EXPIRY_DAYS = 10_000


def _next_tenor(tenor: Tenor) -> Tenor:
    """The next futures month out from `tenor`."""
    idx = FUTURES_TENORS.index(tenor)
    if idx + 1 >= len(FUTURES_TENORS):
        raise EngineError(f"tenor {tenor.value} has no month past M12 to roll into")
    return FUTURES_TENORS[idx + 1]


@dataclass(frozen=True)
class CommoditiesAdapter:
    """Commodity futures: single-contract outrights and calendar spreads."""

    sub_style: str
    horizon_days: int
    asset_class: AssetClass = field(default=AssetClass.COMMODITIES, init=False)

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "pnl_from_entry",
            "commodity_group",
            "days_to_expiry",
            "price",
            "n_positions",
            "sessions_held",
            "size_pct_book",
            "triggers_fired",
            "target_hit",
            "level",
            "event",
            "relative_move",
        }
    )

    def __post_init__(self) -> None:
        if self.sub_style not in _SUB_STYLES:
            raise EngineError(f"commodities adapter has no sub-style '{self.sub_style}'")

    def universe(
        self, instruments: Mapping[str, Instrument], rules: Sequence[Rule]
    ) -> tuple[str, ...]:
        excluded_groups = excluded_values(rules, "commodity_group")
        candidates = (
            instrument_id
            for instrument_id, instrument in instruments.items()
            if instrument.kind == InstrumentKind.COMMODITY
            and instrument.commodity_group.value not in excluded_groups
        )
        return tuple(sorted(candidates))

    def forms(self, sub_style: str) -> tuple[Expression, ...]:
        if sub_style == _FUTURES_DIRECTIONAL:
            return (Expression.OUTRIGHT,)
        if sub_style == _CURVE_AND_SPREAD:
            return (Expression.OUTRIGHT, Expression.CALENDAR_SPREAD)
        raise EngineError(f"commodities adapter has no forms for sub-style '{sub_style}'")

    def build_legs(
        self,
        form: Expression,
        instrument_id: str,
        view: MarketView,
        t: int,
        universe: Sequence[str],
        held: frozenset[str],
        rng: np.random.Generator,
    ) -> tuple[LegRef, ...] | None:
        if form == Expression.OUTRIGHT:
            return (LegRef(instrument_id, _M1, 1.0),)
        if form == Expression.CALENDAR_SPREAD:
            idx = int(rng.integers(len(CALENDAR_BACK_TENORS)))
            back_tenor = CALENDAR_BACK_TENORS[idx]
            return (LegRef(instrument_id, _M1, 1.0), LegRef(instrument_id, back_tenor, -1.0))
        raise EngineError(f"commodities adapter cannot build legs for form '{form.value}'")

    def series(self, form: Expression, legs: tuple[LegRef, ...]) -> Series:
        return Series(legs=legs, bullish_sign=1, unit="pct")

    def outright_series(self, instrument_id: str) -> Series:
        return Series(legs=(LegRef(instrument_id, _M1, 1.0),), bullish_sign=1, unit="pct")

    def stop_distance(self, rule: Rule, series: Series) -> float:
        if rule.field != _STOP_FIELD:
            raise EngineError(f"commodities adapter cannot evaluate stop rule field '{rule.field}'")
        return abs(float(rule.level))

    def position_fields(
        self, pos: Position, view: MarketView, t: int, state: PmState, pnl_unit: float
    ) -> dict[str, float | int | str | frozenset[str]]:
        level_now = view.level(pos.series, t)
        reached = target_reached(pos, level_now)
        fields: dict[str, float | int | str | frozenset[str]] = dict(
            common_fields(pos, state, t, pnl_unit, reached)
        )
        days_to_expiry = view.days_to_expiry(pos.instrument_id, t)
        fields["pnl_from_entry"] = pnl_unit
        fields["days_to_expiry"] = days_to_expiry if days_to_expiry is not None else _NO_EXPIRY_DAYS
        fields["price"] = math.exp(view.raw_level(pos.instrument_id, _M1, t) / 100.0)
        fields["commodity_group"] = view.instruments[pos.instrument_id].commodity_group.value
        fields["event"] = view.events_on(pos.instrument_id, t)
        fields["relative_move"] = relative_move(view, self, pos, t, self.horizon_days)
        return fields

    def size_and_risk(
        self,
        size_pct_book: float,
        legs: tuple[LegRef, ...],
        view: MarketView,
        t: int,
        book_size: float,
    ) -> tuple[float, float]:
        risk_amount = book_size * size_pct_book / 100
        instrument_id = legs[0].instrument_id
        price_m1 = math.exp(view.raw_level(instrument_id, _M1, t) / 100.0)
        code = instrument_id.removeprefix("CM-")
        size = max(1, math.floor(risk_amount / (price_m1 * CONTRACT_MULTIPLIER[code])))
        return float(size), risk_amount

    def leg_price(self, leg: LegRef, view: MarketView, t: int) -> float:
        raw = view.raw_level(leg.instrument_id, leg.tenor, t)
        return math.exp(raw / 100.0)

    def instrument_type(self, instrument_id: str, view: MarketView) -> InstrumentKind:
        return view.instruments[instrument_id].kind

    def round_step(self, series: Series, level: float) -> float:
        price = math.exp(level / 100.0)
        step = log_grid_step(price)
        return 100.0 * math.log(nearest_level(price, step))

    def anchors(self, pos: Position, view: MarketView, t: int) -> tuple[float, ...]:
        return standard_anchors(self, pos, view, t)

    def peer_ids(
        self, instrument_id: str, instruments: Mapping[str, Instrument]
    ) -> tuple[str, ...]:
        group = instruments[instrument_id].commodity_group
        return tuple(
            sorted(
                iid
                for iid, instrument in instruments.items()
                if instrument.kind == InstrumentKind.COMMODITY
                and instrument.commodity_group == group
            )
        )

    def peer_label(self, instrument_id: str) -> str:
        return "commodity group"

    def leg_bullish(self, leg: LegRef) -> int:
        return 1

    def roll_legs(
        self, pos: Position, view: MarketView, t: int
    ) -> tuple[tuple[tuple[LegRef, float], tuple[LegRef, float]], ...]:
        """Each of the position's legs paired with itself one futures month out.

        An outright yields one pair (M1 -> M2); a calendar spread's two legs
        each roll a month out and cannot be folded into a single pair, so this
        returns one pair per leg.
        """
        pairs = []
        for leg in pos.series.legs:
            next_tenor = _next_tenor(leg.tenor)
            current = LegRef(leg.instrument_id, leg.tenor, leg.coeff)
            rolled = LegRef(leg.instrument_id, next_tenor, leg.coeff)
            level_current = view.raw_level(leg.instrument_id, leg.tenor, t)
            level_rolled = view.raw_level(leg.instrument_id, next_tenor, t)
            pairs.append(((current, level_current), (rolled, level_rolled)))
        return tuple(pairs)

    def roll_shift(self, pos: Position, view: MarketView, t: int) -> float:
        """Change in the position's series level when every leg moves one month out."""
        total = 0.0
        for leg in pos.series.legs:
            next_tenor = _next_tenor(leg.tenor)
            level_current = view.raw_level(leg.instrument_id, leg.tenor, t)
            level_rolled = view.raw_level(leg.instrument_id, next_tenor, t)
            total += leg.coeff * (level_rolled - level_current)
        return total
