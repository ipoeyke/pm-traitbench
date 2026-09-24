"""Rates and credit adapter: sovereign outright/curve ideas and single-name credit outrights."""

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
from pm_traitbench.engine.constants import CURVE_PAIRS, DV01_PER_MILLION, OUTRIGHT_TENOR
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.enums import AssetClass, Expression, InstrumentKind, Tenor
from pm_traitbench.errors import EngineError
from pm_traitbench.market.levels import CURVE_STEP_BP, SPREAD_STEP_BP, YIELD_STEP_PCT, nearest_level
from pm_traitbench.tables.schema import Instrument, Rule

_SOVEREIGN_RATES = "sovereign_rates"
_LONG_SHORT_CREDIT = "long_short_credit"
_SUB_STYLES = (_SOVEREIGN_RATES, _LONG_SHORT_CREDIT)


@dataclass(frozen=True)
class RatesCreditAdapter:
    """Sovereign rates outright and curve ideas, and single-name credit outrights."""

    sub_style: str
    horizon_days: int
    asset_class: AssetClass = field(default=AssetClass.RATES_CREDIT, init=False)

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "adverse_yield_move_bp",
            "adverse_spread_move_bp",
            "pnl_from_entry",
            "rating_band",
            "n_positions",
            "sessions_held",
            "size_pct_book",
            "triggers_fired",
            "target_hit",
            "level",
            "event",
            "relative_move",
            "spread_bp",
        }
    )

    def __post_init__(self) -> None:
        if self.sub_style not in _SUB_STYLES:
            raise EngineError(f"rates and credit adapter has no sub-style '{self.sub_style}'")

    def _is_sovereign(self) -> bool:
        return self.sub_style == _SOVEREIGN_RATES

    def _dv01(self, tenor: Tenor | None, instrument_id: str, view: MarketView) -> float:
        """DV01 per $1mm notional: the tenor table for a rates leg, duration for a credit leg."""
        if tenor is not None:
            return DV01_PER_MILLION[tenor]
        duration_years = view.instruments[instrument_id].duration_years
        return duration_years * 100

    def universe(
        self, instruments: Mapping[str, Instrument], rules: Sequence[Rule]
    ) -> tuple[str, ...]:
        if self._is_sovereign():
            candidates = (
                instrument_id
                for instrument_id, instrument in instruments.items()
                if instrument.kind == InstrumentKind.SOVEREIGN_CURVE
            )
            return tuple(sorted(candidates))
        excluded_bands = excluded_values(rules, "rating_band")
        candidates = (
            instrument_id
            for instrument_id, instrument in instruments.items()
            if instrument.kind == InstrumentKind.CREDIT_ISSUER
            and instrument.rating_band.value not in excluded_bands
        )
        return tuple(sorted(candidates))

    def forms(self, sub_style: str) -> tuple[Expression, ...]:
        if sub_style == _SOVEREIGN_RATES:
            return (Expression.OUTRIGHT, Expression.CURVE)
        if sub_style == _LONG_SHORT_CREDIT:
            return (Expression.OUTRIGHT,)
        raise EngineError(f"rates and credit adapter has no forms for sub-style '{sub_style}'")

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
            if self._is_sovereign():
                return (LegRef(instrument_id, OUTRIGHT_TENOR, 1.0),)
            return (LegRef(instrument_id, None, 1.0),)
        if form == Expression.CURVE:
            if not self._is_sovereign():
                raise EngineError("rates and credit adapter cannot build a curve for credit")
            idx = int(rng.integers(len(CURVE_PAIRS)))
            short_tenor, long_tenor = CURVE_PAIRS[idx]
            return (
                LegRef(instrument_id, long_tenor, 1.0),
                LegRef(instrument_id, short_tenor, -1.0),
            )
        raise EngineError(f"rates and credit adapter cannot build legs for form '{form.value}'")

    def series(self, form: Expression, legs: tuple[LegRef, ...]) -> Series:
        bullish_sign = 1 if form == Expression.CURVE else -1
        return Series(legs=legs, bullish_sign=bullish_sign, unit="bp")

    def outright_series(self, instrument_id: str) -> Series:
        if self._is_sovereign():
            return Series(
                legs=(LegRef(instrument_id, OUTRIGHT_TENOR, 1.0),), bullish_sign=-1, unit="bp"
            )
        return Series(legs=(LegRef(instrument_id, None, 1.0),), bullish_sign=-1, unit="bp")

    def stop_distance(self, rule: Rule, series: Series) -> float:
        expected_field = (
            "adverse_yield_move_bp" if self._is_sovereign() else "adverse_spread_move_bp"
        )
        if rule.field != expected_field:
            raise EngineError(
                f"rates and credit adapter cannot evaluate stop rule field '{rule.field}'"
            )
        return abs(float(rule.level))

    def position_fields(
        self, pos: Position, view: MarketView, t: int, state: PmState, pnl_unit: float
    ) -> dict[str, float | int | str | frozenset[str]]:
        level_now = view.level(pos.series, t)
        reached = target_reached(pos, level_now)
        fields: dict[str, float | int | str | frozenset[str]] = dict(
            common_fields(pos, state, t, pnl_unit, reached)
        )
        leg0 = pos.legs[0]
        dv01 = self._dv01(leg0.tenor, leg0.instrument_id, view)
        is_sovereign = self._is_sovereign()
        fields["adverse_yield_move_bp"] = max(-pnl_unit, 0.0) if is_sovereign else 0.0
        fields["adverse_spread_move_bp"] = 0.0 if is_sovereign else max(-pnl_unit, 0.0)
        fields["pnl_from_entry"] = pnl_unit * dv01 / 10000
        fields["rating_band"] = (
            "" if is_sovereign else view.instruments[pos.instrument_id].rating_band.value
        )
        fields["spread_bp"] = level_now
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
        long_leg = legs[0]
        dv01 = self._dv01(long_leg.tenor, long_leg.instrument_id, view)
        size = risk_amount / 1e6 * dv01
        return size, risk_amount

    def leg_risk_amount(
        self, leg: LegRef, size: float, view: MarketView, risk_amount: float
    ) -> float:
        """Risk the ledger carries for one leg, scaled by that leg's own DV01.

        A curve's short leg has a different notional to its long leg so both
        legs carry the same DV01 exposure; `risk_amount` is unused here since
        the DV01 ratio alone determines each leg's share.
        """
        return size / self._dv01(leg.tenor, leg.instrument_id, view) * 1e6

    def leg_price(self, leg: LegRef, view: MarketView, t: int) -> float:
        raw = view.raw_level(leg.instrument_id, leg.tenor, t)
        return raw / 100.0 if leg.tenor is not None else raw

    def instrument_type(self, instrument_id: str, view: MarketView) -> InstrumentKind:
        return view.instruments[instrument_id].kind

    def round_step(self, series: Series, level: float) -> float:
        if len(series.legs) == 2:
            return float(nearest_level(level, CURVE_STEP_BP))
        leg = series.legs[0]
        if leg.tenor is not None:
            return float(nearest_level(level / 100.0, YIELD_STEP_PCT) * 100.0)
        return float(nearest_level(level, SPREAD_STEP_BP))

    def anchors(self, pos: Position, view: MarketView, t: int) -> tuple[float, ...]:
        return standard_anchors(self, pos, view, t)

    def peer_ids(
        self, instrument_id: str, instruments: Mapping[str, Instrument]
    ) -> tuple[str, ...]:
        if self._is_sovereign():
            return (instrument_id,)
        band = instruments[instrument_id].rating_band
        return tuple(
            sorted(
                iid
                for iid, instrument in instruments.items()
                if instrument.kind == InstrumentKind.CREDIT_ISSUER
                and instrument.rating_band == band
            )
        )

    def peer_label(self, instrument_id: str) -> str:
        return "curve" if self._is_sovereign() else "rating band"

    def leg_bullish(self, leg: LegRef) -> int:
        return -1
