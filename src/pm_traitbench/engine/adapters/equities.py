"""Equities adapter: outright and same-sector pair ideas."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from pm_traitbench.engine.adapters.base import (
    common_fields,
    excluded_values,
    relative_move,
    target_reached,
    tracked_level,
)
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.engine.state import PmState, Position
from pm_traitbench.enums import AssetClass, Expression, InstrumentKind
from pm_traitbench.errors import EngineError
from pm_traitbench.market.levels import log_grid_step, nearest_level
from pm_traitbench.tables.schema import Instrument, Rule

_STOP_FIELD = "pnl_from_entry"


@dataclass(frozen=True)
class EquitiesAdapter:
    """Equities: single-name outrights and same-sector pair trades."""

    sub_style: str
    horizon_days: int
    asset_class: AssetClass = field(default=AssetClass.EQUITIES, init=False)

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "pnl_from_entry",
            "sector",
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

    def universe(
        self, instruments: Mapping[str, Instrument], rules: Sequence[Rule]
    ) -> tuple[str, ...]:
        excluded_sectors = excluded_values(rules, "sector")
        candidates = (
            instrument_id
            for instrument_id, instrument in instruments.items()
            if instrument.kind == InstrumentKind.EQUITY
            and instrument.sector not in excluded_sectors
        )
        return tuple(sorted(candidates))

    def forms(self, sub_style: str) -> tuple[Expression, ...]:
        return (Expression.OUTRIGHT, Expression.PAIR)

    def _pair_partner(
        self,
        instrument_id: str,
        view: MarketView,
        t: int,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> str | None:
        sector = view.instruments[instrument_id].sector
        candidates = [
            iid
            for iid in universe
            if iid != instrument_id and iid not in held and view.instruments[iid].sector == sector
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda iid: (
                view.trailing_move(self.outright_series(iid), t, self.horizon_days),
                iid,
            ),
        )

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
            return (LegRef(instrument_id, None, 1.0),)
        if form == Expression.PAIR:
            partner = self._pair_partner(instrument_id, view, t, universe, held)
            if partner is None:
                return None
            return (LegRef(instrument_id, None, 1.0), LegRef(partner, None, -1.0))
        raise EngineError(f"equities adapter cannot build legs for form '{form.value}'")

    def series(self, form: Expression, legs: tuple[LegRef, ...]) -> Series:
        return Series(legs=legs, bullish_sign=1, unit="pct")

    def outright_series(self, instrument_id: str) -> Series:
        return Series(legs=(LegRef(instrument_id, None, 1.0),), bullish_sign=1, unit="pct")

    def stop_distance(self, rule: Rule, series: Series) -> float:
        if rule.field != _STOP_FIELD:
            raise EngineError(f"equities adapter cannot evaluate stop rule field '{rule.field}'")
        return abs(float(rule.level))

    def position_fields(
        self, pos: Position, view: MarketView, t: int, state: PmState, pnl_unit: float
    ) -> dict[str, float | int | str | frozenset[str]]:
        level_now = tracked_level(pos, view, t)
        reached = target_reached(pos, level_now)
        fields: dict[str, float | int | str | frozenset[str]] = dict(
            common_fields(pos, state, t, pnl_unit, reached)
        )
        fields["pnl_from_entry"] = pnl_unit
        fields["sector"] = view.instruments[pos.instrument_id].sector
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
        return size_pct_book, book_size * size_pct_book / 100

    def leg_risk_amount(
        self, leg: LegRef, size: float, view: MarketView, risk_amount: float
    ) -> float:
        """Every leg of an equities idea carries the idea's full risk amount."""
        return risk_amount

    def leg_price(self, leg: LegRef, view: MarketView, t: int) -> float:
        raw = view.raw_level(leg.instrument_id, leg.tenor, t)
        return math.exp(raw / 100.0)

    def instrument_type(self, instrument_id: str, view: MarketView) -> InstrumentKind:
        return view.instruments[instrument_id].kind

    def round_step(self, series: Series, level: float) -> float:
        price = math.exp(level / 100.0)
        step = log_grid_step(price)
        return 100.0 * math.log(nearest_level(price, step))

    def peer_ids(
        self, instrument_id: str, instruments: Mapping[str, Instrument]
    ) -> tuple[str, ...]:
        sector = instruments[instrument_id].sector
        return tuple(
            sorted(
                iid
                for iid, instrument in instruments.items()
                if instrument.kind == InstrumentKind.EQUITY
                and instrument.sector == sector
                and iid != instrument_id
            )
        )

    def peer_label(self, instrument_id: str, instruments: Mapping[str, Instrument]) -> str:
        """The candidate's sector label, as signpost text names it."""
        return str(instruments[instrument_id].sector).replace("_", " ")

    def leg_bullish(self, leg: LegRef) -> int:
        return 1
