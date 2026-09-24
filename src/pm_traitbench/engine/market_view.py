"""In-memory market view the behaviour engine reads: one seed, built once.

`MarketView.build` turns the flat row tables (prices, curves, consensus,
calendar, regimes) into numpy arrays indexed by horizon day, so the engine
can query a day's level, volatility scale or event state in constant time
without re-scanning the row tables.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pm_traitbench.engine.constants import MIN_SD_DAYS, SD_FLOOR, TRAILING_SD_DAYS
from pm_traitbench.engine.series import Series
from pm_traitbench.enums import (
    SOVEREIGN_TENORS,
    EventType,
    InstrumentKind,
    Positioning,
    Regime,
    StreetView,
    Tenor,
)
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import (
    CalendarEvent,
    ConsensusRow,
    CurvePoint,
    Instrument,
    Price,
    RegimeSpan,
)

_EVENT_TYPES_EXCLUDED = frozenset(
    {EventType.CONTRACT_EXPIRY, EventType.POSITIONING_REPORT, EventType.CONSENSUS_FLIP}
)


def _valid_raw_level_key(instrument: Instrument, tenor: Tenor | None) -> bool:
    """Whether (instrument kind, tenor) is one of the engine's recognised level sources."""
    kind = instrument.kind
    if tenor is None:
        outright_kinds = (
            InstrumentKind.EQUITY,
            InstrumentKind.COMMODITY,
            InstrumentKind.CREDIT_ISSUER,
        )
        return kind in outright_kinds
    if kind == InstrumentKind.COMMODITY:
        return tenor.value.startswith("M")
    if kind == InstrumentKind.SOVEREIGN_CURVE:
        return tenor in SOVEREIGN_TENORS
    return False


@dataclass(frozen=True)
class MarketView:
    """One seed's prices, curves, consensus, calendar and regimes, keyed by horizon day."""

    seed: str
    dates: tuple[date, ...]
    instruments: dict[str, Instrument]
    _raw_levels: dict[tuple[str, Tenor | None], np.ndarray] = field(repr=False)
    _consensus: dict[str, dict[int, ConsensusRow]] = field(repr=False)
    _instrument_events: dict[tuple[str, int], frozenset[EventType]] = field(repr=False)
    _market_wide_events: dict[int, frozenset[EventType]] = field(repr=False)
    _event_types: dict[str, frozenset[EventType]] = field(repr=False)
    _expiry_days: dict[str, tuple[int, ...]] = field(repr=False)
    _regime_by_day: tuple[Regime | None, ...] = field(repr=False)
    _level_cache: dict[Series, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @classmethod
    def build(
        cls,
        seed: str,
        dates: Sequence[date],
        instruments: Sequence[Instrument],
        prices: Sequence[Price],
        curves: Sequence[CurvePoint],
        consensus: Sequence[ConsensusRow],
        calendar: Sequence[CalendarEvent],
        regimes: Sequence[RegimeSpan],
    ) -> "MarketView":
        """Build one seed's view; rows for other seeds are ignored.

        An instrument with no price or curve row on this seed (coverage can differ
        by seed, e.g. real versus synthetic) is dropped rather than kept unpriced.
        """
        dates_t = tuple(dates)
        date_index = {day: i for i, day in enumerate(dates_t)}
        n_days = len(dates_t)
        covered_ids = {row.instrument_id for row in prices if row.seed == seed}
        covered_ids |= {row.curve_id for row in curves if row.seed == seed}
        instruments_by_id = {
            i.instrument_id: i for i in instruments if i.instrument_id in covered_ids
        }

        raw_levels = cls._build_raw_levels(
            seed, dates_t, date_index, n_days, instruments_by_id, prices, curves
        )
        consensus_by_instrument = cls._build_consensus(seed, date_index, consensus)
        instrument_events, market_wide_events, event_types, expiry_days = cls._build_calendar(
            seed, date_index, calendar
        )
        regime_by_day = cls._build_regimes(seed, dates_t, regimes)

        return cls(
            seed=seed,
            dates=dates_t,
            instruments=instruments_by_id,
            _raw_levels=raw_levels,
            _consensus=consensus_by_instrument,
            _instrument_events=instrument_events,
            _market_wide_events=market_wide_events,
            _event_types=event_types,
            _expiry_days=expiry_days,
            _regime_by_day=regime_by_day,
        )

    @staticmethod
    def _build_raw_levels(
        seed: str,
        dates: tuple[date, ...],
        date_index: dict[date, int],
        n_days: int,
        instruments_by_id: dict[str, Instrument],
        prices: Sequence[Price],
        curves: Sequence[CurvePoint],
    ) -> dict[tuple[str, Tenor | None], np.ndarray]:
        raw_levels: dict[tuple[str, Tenor | None], np.ndarray] = {}

        by_instrument: dict[str, dict[int, Price]] = {}
        for row in prices:
            if row.seed != seed or row.date not in date_index:
                continue
            by_instrument.setdefault(row.instrument_id, {})[date_index[row.date]] = row

        for instrument_id, by_day in by_instrument.items():
            instrument = instruments_by_id.get(instrument_id)
            if instrument is None:
                continue
            missing = [dates[t] for t in range(n_days) if t not in by_day]
            if missing:
                raise EngineError(f"instrument '{instrument_id}' has no price on {missing[0]}")
            if instrument.kind == InstrumentKind.CREDIT_ISSUER:
                values = np.array([by_day[t].spread_bp for t in range(n_days)], dtype=float)
                raw_levels[(instrument_id, None)] = values
            else:
                values = np.array([by_day[t].price for t in range(n_days)], dtype=float)
                raw_levels[(instrument_id, None)] = 100.0 * np.log(values)

        by_curve: dict[tuple[str, Tenor], dict[int, float]] = {}
        for row in curves:
            if row.seed != seed or row.date not in date_index:
                continue
            by_curve.setdefault((row.curve_id, row.tenor), {})[date_index[row.date]] = row.level

        for (curve_id, tenor), by_day in by_curve.items():
            instrument = instruments_by_id.get(curve_id)
            if instrument is None:
                continue
            missing = [dates[t] for t in range(n_days) if t not in by_day]
            if missing:
                raise EngineError(
                    f"instrument '{curve_id}' has no curve level at tenor {tenor.value} on "
                    f"{missing[0]}"
                )
            values = np.array([by_day[t] for t in range(n_days)], dtype=float)
            if instrument.kind == InstrumentKind.SOVEREIGN_CURVE:
                raw_levels[(curve_id, tenor)] = 100.0 * values
            elif instrument.kind == InstrumentKind.COMMODITY:
                raw_levels[(curve_id, tenor)] = 100.0 * np.log(values)

        return raw_levels

    @staticmethod
    def _build_consensus(
        seed: str, date_index: dict[date, int], consensus: Sequence[ConsensusRow]
    ) -> dict[str, dict[int, ConsensusRow]]:
        by_instrument: dict[str, dict[int, ConsensusRow]] = {}
        for row in consensus:
            if row.seed != seed or row.date not in date_index:
                continue
            by_instrument.setdefault(row.instrument_id, {})[date_index[row.date]] = row
        return by_instrument

    @staticmethod
    def _build_calendar(
        seed: str, date_index: dict[date, int], calendar: Sequence[CalendarEvent]
    ) -> tuple[
        dict[tuple[str, int], frozenset[EventType]],
        dict[int, frozenset[EventType]],
        dict[str, frozenset[EventType]],
        dict[str, tuple[int, ...]],
    ]:
        instrument_day: dict[tuple[str, int], set[EventType]] = {}
        market_wide_day: dict[int, set[EventType]] = {}
        by_instrument: dict[str, set[EventType]] = {}
        expiry_days: dict[str, list[int]] = {}

        for row in calendar:
            if row.seed != seed or row.date not in date_index:
                continue
            t = date_index[row.date]
            if row.instrument_id is None:
                market_wide_day.setdefault(t, set()).add(row.event)
                continue
            instrument_day.setdefault((row.instrument_id, t), set()).add(row.event)
            if row.event not in _EVENT_TYPES_EXCLUDED:
                by_instrument.setdefault(row.instrument_id, set()).add(row.event)
            if row.event == EventType.CONTRACT_EXPIRY:
                expiry_days.setdefault(row.instrument_id, []).append(t)

        return (
            {key: frozenset(value) for key, value in instrument_day.items()},
            {key: frozenset(value) for key, value in market_wide_day.items()},
            {key: frozenset(value) for key, value in by_instrument.items()},
            {key: tuple(sorted(value)) for key, value in expiry_days.items()},
        )

    @staticmethod
    def _build_regimes(
        seed: str, dates: tuple[date, ...], regimes: Sequence[RegimeSpan]
    ) -> tuple[Regime | None, ...]:
        by_day: list[Regime | None] = [None] * len(dates)
        for row in regimes:
            if row.seed != seed:
                continue
            for t, day in enumerate(dates):
                if row.date_start <= day <= row.date_end:
                    if by_day[t] is not None:
                        raise EngineError(f"overlapping regime spans cover {day}")
                    by_day[t] = row.regime
        return tuple(by_day)

    def _raw_level_series(self, instrument_id: str, tenor: Tenor | None) -> np.ndarray:
        """The full-horizon raw level array for one instrument leg, looked up once per call site."""
        instrument = self.instruments.get(instrument_id)
        if instrument is None:
            raise EngineError(f"unknown instrument '{instrument_id}'")
        key = (instrument_id, tenor)
        if not _valid_raw_level_key(instrument, tenor) or key not in self._raw_levels:
            raise EngineError(
                f"no raw level for instrument '{instrument_id}' "
                f"(kind {instrument.kind.value}) at tenor {tenor}"
            )
        return self._raw_levels[key]

    def raw_level(self, instrument_id: str, tenor: Tenor | None, t: int) -> float:
        """The unit-normalised level for one instrument leg on day `t`.

        Equity or commodity outright: `100 * ln(price)`. Commodity futures
        tenor: `100 * ln(curve level)`. Sovereign curve tenor: `100 * curve
        level` (a percent yield times 100 is basis points). Credit issuer:
        `spread_bp`. Anything else is not a level the engine recognises.
        """
        return float(self._raw_level_series(instrument_id, tenor)[t])

    def _level_series(self, series: Series) -> np.ndarray:
        """The full-horizon level array for a series, cached per series (legs are hashable).

        The engine evaluates the same series many times (per idea, per day),
        so this is computed once and sliced thereafter rather than summing
        legs on every call.
        """
        cached = self._level_cache.get(series)
        if cached is not None:
            return cached
        total = np.zeros(self.n_days)
        for leg in series.legs:
            total = total + leg.coeff * self._raw_level_series(leg.instrument_id, leg.tenor)
        total.flags.writeable = False
        self._level_cache[series] = total
        return total

    def level(self, series: Series, t: int) -> float:
        """The series level on day `t`: the coefficient-weighted sum of its legs."""
        return float(self._level_series(series)[t])

    def daily_moves(self, series: Series, t: int) -> np.ndarray:
        """Day-over-day level changes over the trailing `TRAILING_SD_DAYS` window ending at `t`."""
        lo = max(0, t - TRAILING_SD_DAYS)
        return np.diff(self._level_series(series)[lo : t + 1])

    def sd_h(self, series: Series, t: int, h: int) -> float:
        """Realised volatility of the series over an `h`-day horizon, floored at `SD_FLOOR`.

        Normally the sample standard deviation (ddof=1) of `daily_moves`,
        scaled by `sqrt(h)`. Whenever that window holds fewer than
        `MIN_SD_DAYS` moves (early in the horizon), the engine instead looks
        ahead over `0 .. min(TRAILING_SD_DAYS, n_days - 1)` - a one-time
        look-ahead used only to give the earliest days a volatility scale,
        never for the level itself.
        """
        moves = self.daily_moves(series, t)
        if len(moves) < MIN_SD_DAYS:
            hi = min(TRAILING_SD_DAYS, self.n_days - 1)
            moves = np.diff(self._level_series(series)[: hi + 1])
        sd = float(np.std(moves, ddof=1)) if len(moves) >= 2 else 0.0
        return max(sd * math.sqrt(h), SD_FLOOR)

    def trailing_move(self, series: Series, t: int, h: int) -> float:
        """Change in level from `h` days before `t` (clamped at day 0) to `t`."""
        return self.level(series, t) - self.level(series, max(t - h, 0))

    def forward_move(self, series: Series, t: int, h: int) -> float:
        """Change in level from `t` to `h` days later (clamped at the last horizon day)."""
        return self.level(series, min(t + h, self.n_days - 1)) - self.level(series, t)

    def forward_days(self, t: int, h: int) -> int:
        """Sessions available between `t` and `t + h`, clamped at the horizon end."""
        return min(t + h, self.n_days - 1) - t

    def trailing_high(self, series: Series, t: int, days: int) -> float:
        """Highest level over the trailing `days` sessions up to and including `t`."""
        lo = max(0, t - days)
        return float(np.max(self._level_series(series)[lo : t + 1]))

    def trailing_low(self, series: Series, t: int, days: int) -> float:
        """Lowest level over the trailing `days` sessions up to and including `t`."""
        lo = max(0, t - days)
        return float(np.min(self._level_series(series)[lo : t + 1]))

    def street_view(self, instrument_id: str, t: int) -> StreetView | None:
        """The street's categorical view on day `t`, or None if the instrument has no consensus."""
        row = self._consensus_row(instrument_id, t)
        return row.street_view if row is not None else None

    def positioning(self, instrument_id: str, t: int) -> Positioning | None:
        """Street positioning crowding on day `t`, or None if the instrument has no consensus."""
        row = self._consensus_row(instrument_id, t)
        return row.positioning if row is not None else None

    def _consensus_row(self, instrument_id: str, t: int) -> ConsensusRow | None:
        by_day = self._consensus.get(instrument_id)
        if by_day is None:
            return None
        if t not in by_day:
            raise EngineError(f"instrument '{instrument_id}' has no consensus on {self.dates[t]}")
        return by_day[t]

    def events_on(self, instrument_id: str, t: int) -> frozenset[EventType]:
        """Event types on day `t` affecting this instrument, including market-wide events."""
        own = self._instrument_events.get((instrument_id, t), frozenset())
        market_wide = self._market_wide_events.get(t, frozenset())
        return own | market_wide

    def event_types(self, instrument_id: str) -> frozenset[EventType]:
        """Event types ever scheduled for this instrument over the seed.

        Excludes `contract_expiry`, `positioning_report` and `consensus_flip`,
        which are mechanical calendar rows rather than news the engine reacts to.
        """
        return self._event_types.get(instrument_id, frozenset())

    def regime(self, t: int) -> Regime:
        """The regime in effect on day `t`; raises if no span in the seed covers that date."""
        regime = self._regime_by_day[t]
        if regime is None:
            raise EngineError(f"no regime span covers {self.dates[t]}")
        return regime

    def days_to_expiry(self, instrument_id: str, t: int) -> int | None:
        """Sessions to the next contract expiry at or after `t`; None if there is none, or the
        instrument is not a commodity."""
        instrument = self.instruments.get(instrument_id)
        if instrument is None or instrument.kind != InstrumentKind.COMMODITY:
            return None
        for expiry_t in self._expiry_days.get(instrument_id, ()):
            if expiry_t >= t:
                return expiry_t - t
        return None

    def is_expiry_day(self, instrument_id: str, t: int) -> bool:
        """Whether `t` is a contract expiry day for this instrument."""
        return self.days_to_expiry(instrument_id, t) == 0

    def peer_move(
        self,
        instrument_ids: Sequence[str],
        series_for: Callable[[str], Series],
        t: int,
        h: int,
    ) -> float:
        """Mean trailing move of a peer group's outright series; 0.0 for an empty group."""
        if not instrument_ids:
            return 0.0
        moves = [self.trailing_move(series_for(iid), t, h) for iid in instrument_ids]
        return float(np.mean(moves))
