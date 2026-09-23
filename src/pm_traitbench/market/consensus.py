"""Street consensus and positioning: a trend-following EMA score with scheduled
revisions and random flips, and a slower EMA percentile for crowding.

Both series run on the full axis so burn-in gives them a warm state before the
first published day; only horizon days are emitted as rows.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pm_traitbench.config import MarketConfig
from pm_traitbench.enums import (
    HY_BANDS,
    EventType,
    Family,
    InstrumentKind,
    Positioning,
    StreetView,
    Tenor,
)
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import RngFor
from pm_traitbench.market.constants import ANNUALISATION_DAYS, FX_PAIRS, HORIZON_DAYS_PER_YEAR
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.tables.schema import CalendarEvent, ConsensusRow, Instrument


@dataclass(frozen=True)
class ConsensusResult:
    """Street consensus and positioning rows for one market seed, plus flip events."""

    rows: list[ConsensusRow]
    flips: list[CalendarEvent]
    drawn_flips: dict[str, int]
    street_score: dict[str, np.ndarray]
    positioning_pct: dict[str, np.ndarray]


def update_weight(gap: int, half_life: float) -> float:
    """Exponential-moving-average alpha closing the gap to target over `half_life`
    trading days, for an update that follows `gap` axis days after the last one.
    """
    return 1 - 2 ** (-gap / half_life)


def consensus_signal(instrument: Instrument, output: ProcessOutput) -> np.ndarray:
    """The street's tracked signal: log price, or minus the 10Y yield for curves."""
    if instrument.kind == InstrumentKind.SOVEREIGN_CURVE:
        return -output.curves[(instrument.instrument_id, Tenor.Y10)]
    return np.log(output.prices[instrument.instrument_id])


def signal_vol(instrument: Instrument, market: MarketConfig) -> float:
    """Annual volatility of `consensus_signal`, before any regime multiplier."""
    families = market.families
    if instrument.family == Family.EQUITIES:
        return float(
            np.sqrt(
                (instrument.beta * families.equity.market_vol) ** 2 + families.equity.idio_vol**2
            )
        )
    if instrument.family == Family.CREDIT:
        h = families.credit.hy_vol_multiplier if instrument.rating_band in HY_BANDS else 1.0
        base_spread = market.levels.credit_base_spread_bp[instrument.rating_band]
        return instrument.duration_years * base_spread * families.credit.factor_vol * h / 10000
    if instrument.kind == InstrumentKind.SOVEREIGN_CURVE:
        return families.rates.level_vol_bp[instrument.currency] / 100
    if instrument.family == Family.COMMODITIES:
        return families.commodity.group_vol[instrument.commodity_group]
    if instrument.family == Family.FX:
        pair = instrument.instrument_id.removeprefix("FX-")
        base, quote = FX_PAIRS[pair]
        vol = {ccy: 0.0 if ccy == "USD" else families.fx.currency_vol[ccy] for ccy in (base, quote)}
        return float(np.sqrt(vol[base] ** 2 + vol[quote] ** 2))
    raise ValueError(f"unhandled instrument for signal_vol: {instrument.instrument_id}")


def _trend(signal: np.ndarray, window: int, vol: float) -> np.ndarray:
    """Clipped `window`-day trailing change of `signal`, scaled by `vol` over the window."""
    lagged_index = np.maximum(0, np.arange(len(signal)) - window)
    scale = vol * np.sqrt(window / ANNUALISATION_DAYS)
    return np.clip((signal - signal[lagged_index]) / scale, -1, 1)


def _flip_days(rng: np.random.Generator, per_year: float, horizon_days: int) -> np.ndarray:
    """Poisson-counted flip day indices, capped at the horizon, drawn without replacement."""
    years = horizon_days / HORIZON_DAYS_PER_YEAR
    count = min(int(rng.poisson(per_year * years)), horizon_days)
    if count <= 0:
        return np.array([], dtype=int)
    return rng.choice(horizon_days, count, replace=False)


def _street_score(
    trend: np.ndarray,
    update_days: set[int],
    flip_days: set[int],
    half_life: float,
    threshold: float,
) -> np.ndarray:
    """Street score across the full axis: flip resets, updates ease toward trend
    by a weight set by the number of axis days since the last update or flip.
    """
    n = len(trend)
    score = np.empty(n)
    prev = 0.0
    last_update = 0
    for t in range(n):
        if t in flip_days:
            sign = 1.0 if prev >= 0 else -1.0
            score[t] = -sign * 2 * threshold
            last_update = t
        elif t in update_days:
            alpha = update_weight(t - last_update, half_life)
            score[t] = prev + alpha * (trend[t] - prev)
            last_update = t
        else:
            score[t] = prev
        prev = score[t]
    return score


def _positioning_pct(trend: np.ndarray, report_mask: np.ndarray, half_life: float) -> np.ndarray:
    """Positioning percentile across the full axis: eases toward its report-day
    target by a weight set by the number of axis days since the last report.
    """
    n = len(trend)
    pct = np.empty(n)
    prev = 50.0
    last_update = 0
    for t in range(n):
        if report_mask[t]:
            alpha = update_weight(t - last_update, half_life)
            target = 50 + 40 * np.tanh(2 * trend[t])
            pct[t] = np.clip(prev + alpha * (target - prev), 0, 100)
            last_update = t
        else:
            pct[t] = prev
        prev = pct[t]
    return pct


def build_consensus(
    instruments: Sequence[Instrument],
    axis: SimAxis,
    output: ProcessOutput,
    event_days: Mapping[str, set[int]],
    market: MarketConfig,
    rng_for: RngFor,
    seed: str,
) -> ConsensusResult:
    """Simulate street consensus and positioning for every instrument on the full axis."""
    cfg = market.consensus
    horizon_days = axis.n_days - axis.n_burn
    street_window = cfg.street_window_days
    positioning_window = cfg.positioning_window_multiple * street_window
    lo, hi = cfg.positioning_thresholds

    revision_mask = np.array([day.weekday() == cfg.revision_weekday for day in axis.dates])
    report_mask = np.array([day.weekday() == cfg.report_weekday for day in axis.dates])
    revision_days = {t for t in range(axis.n_days) if revision_mask[t]}

    rows: list[ConsensusRow] = []
    flips: list[CalendarEvent] = []
    drawn_flips: dict[str, int] = {}
    street_score: dict[str, np.ndarray] = {}
    positioning_pct: dict[str, np.ndarray] = {}

    for instrument in instruments:
        instrument_id = instrument.instrument_id
        signal = consensus_signal(instrument, output)
        vol = signal_vol(instrument, market)
        trend_street = _trend(signal, street_window, vol)
        trend_positioning = _trend(signal, positioning_window, vol)

        flip_rng = rng_for(seed, "flips", instrument_id)
        flip_horizon_idx = _flip_days(flip_rng, cfg.flips_per_instrument_year, horizon_days)
        flip_axis_idx = {int(idx) + axis.n_burn for idx in flip_horizon_idx}
        drawn_flips[instrument_id] = len(flip_axis_idx)
        for t in sorted(flip_axis_idx):
            flips.append(
                CalendarEvent(
                    seed=seed,
                    date=axis.dates[t],
                    instrument_id=instrument_id,
                    event=EventType.CONSENSUS_FLIP,
                    surprise=None,
                    affected=instrument.family.value,
                )
            )

        update_days = set(event_days.get(instrument_id, set())) | revision_days | flip_axis_idx
        score = _street_score(
            trend_street, update_days, flip_axis_idx, street_window, cfg.view_threshold
        )
        pct = _positioning_pct(trend_positioning, report_mask, positioning_window)
        street_score[instrument_id] = score
        positioning_pct[instrument_id] = pct

        for t in range(axis.n_burn, axis.n_days):
            s, p = float(score[t]), float(pct[t])
            if s > cfg.view_threshold:
                view = StreetView.OVERWEIGHT
            elif s < -cfg.view_threshold:
                view = StreetView.UNDERWEIGHT
            else:
                view = StreetView.NEUTRAL
            if p > hi:
                positioning = Positioning.CROWDED_LONG
            elif p < lo:
                positioning = Positioning.CROWDED_SHORT
            else:
                positioning = Positioning.NEUTRAL
            rows.append(
                ConsensusRow(
                    seed=seed,
                    date=axis.dates[t],
                    instrument_id=instrument_id,
                    street_score=s,
                    street_view=view,
                    positioning_pct=p,
                    positioning=positioning,
                )
            )

    flips.sort(key=lambda row: (row.date, row.instrument_id))
    rows.sort(key=lambda row: (row.date, row.instrument_id))
    return ConsensusResult(
        rows=rows,
        flips=flips,
        drawn_flips=drawn_flips,
        street_score=street_score,
        positioning_pct=positioning_pct,
    )
