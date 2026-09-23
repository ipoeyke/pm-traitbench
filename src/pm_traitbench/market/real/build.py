"""Real-market seed builder: turn a verified raw cache into one seed's full
simulated state, the same `SeedMarket` shape the synthetic builder returns.

Every price series keeps its real historical log changes; only its starting
level is rebased onto the configured simulated levels, so a real seed's
day-over-day moves come entirely from history while its scale matches the
rest of the pipeline.
"""

from collections.abc import Sequence
from datetime import date

import numpy as np

from pm_traitbench.config import Config, RealSeedSpec
from pm_traitbench.enums import FUTURES_TENORS, Family, InstrumentKind, Tenor
from pm_traitbench.errors import StageIOError
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import RngFor, event_day_indices, generated_rows, row_sort_key
from pm_traitbench.market.consensus import build_consensus
from pm_traitbench.market.constants import COMMODITIES, FX_PAIRS, USD_PAIR
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.real.align import aligned_series
from pm_traitbench.market.real.events import drawn_counts, real_event_days, surprise_rows
from pm_traitbench.market.real.fetch import RawCache
from pm_traitbench.market.real.sources import (
    CREDIT_DURATION_YEARS,
    CURVE_SERIES,
    REAL_INSTRUMENTS,
    REFERENCE_EQUITY,
)
from pm_traitbench.market.real.universe import real_axis_dates
from pm_traitbench.market.regimes import RegimeLookup, regime_path
from pm_traitbench.market.seed import SeedMarket, market_rng
from pm_traitbench.tables.schema import Instrument, RegimeSpan

_REGISTRY_BY_ID = {inst.instrument_id: inst for inst in REAL_INSTRUMENTS}
_COMMODITY_LOOKUP = {spec.name: spec for specs in COMMODITIES.values() for spec in specs}
# curve_start["USD"] holds 2Y/5Y/10Y/30Y; the 10Y level anchors the rebasing shift.
_Y10_CURVE_START_INDEX = 2


def real_schedule(
    spec: RealSeedSpec, axis: SimAxis, calendar_start: date, seed: str
) -> list[RegimeSpan]:
    """Map a real seed's regime start dates onto the axis, tiling the whole horizon.

    Each span ends on the last axis day before the next span's mapped start,
    or on the last axis day for the final span.
    """
    offset = spec.window_start - calendar_start
    starts = [regime_start - offset for _, regime_start in spec.regime_starts]
    spans = []
    for i, (regime, _) in enumerate(spec.regime_starts):
        if i + 1 < len(starts):
            date_end = axis.dates[axis.index(starts[i + 1]) - 1]
        else:
            date_end = axis.dates[-1]
        spans.append(RegimeSpan(seed=seed, regime=regime, date_start=starts[i], date_end=date_end))
    return spans


def _aligned(
    cache_values: dict[date, float], dates: list[date], name: str, fills: dict[str, int]
) -> np.ndarray:
    values, longest_run = aligned_series(cache_values, dates, name=name)
    fills[name] = longest_run
    return values


def _build_equities(
    instruments: Sequence[Instrument],
    cache: RawCache,
    dates: list[date],
    config: Config,
    fills: dict[str, int],
    prices: dict[str, np.ndarray],
    rng_for: RngFor,
) -> None:
    lo, hi = config.market.levels.equity_price_range
    for inst in instruments:
        if inst.family != Family.EQUITIES:
            continue
        reg = _REGISTRY_BY_ID[inst.instrument_id]
        raw = _aligned(cache.yahoo(reg.series), dates, inst.instrument_id, fills)
        rng = rng_for("real", "level", inst.instrument_id)
        start = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        prices[inst.instrument_id] = raw * start / raw[0]


def _build_commodities(
    instruments: Sequence[Instrument],
    cache: RawCache,
    dates: list[date],
    config: Config,
    fills: dict[str, int],
    prices: dict[str, np.ndarray],
    curves: dict[tuple[str, Tenor], np.ndarray],
) -> None:
    curve_slope = config.market.families.commodity.curve_slope
    for inst in instruments:
        if inst.family != Family.COMMODITIES:
            continue
        reg = _REGISTRY_BY_ID[inst.instrument_id]
        spec = _COMMODITY_LOOKUP[reg.name]
        raw = _aligned(cache.yahoo(reg.series), dates, inst.instrument_id, fills)
        m1 = raw * spec.start / raw[0]
        prices[inst.instrument_id] = m1
        slope = curve_slope[inst.commodity_group]
        for k, tenor in enumerate(FUTURES_TENORS, start=1):
            curves[(inst.instrument_id, tenor)] = m1 if k == 1 else m1 * (1 + slope * (k - 1) / 12)


def _build_curve(
    instruments: Sequence[Instrument],
    cache: RawCache,
    dates: list[date],
    config: Config,
    fills: dict[str, int],
    curves: dict[tuple[str, Tenor], np.ndarray],
) -> str:
    curve_inst = next(i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE)
    tenor_series = {
        tenor: _aligned(cache.fred(series), dates, series, fills)
        for tenor, series in CURVE_SERIES.items()
    }
    shift = (
        config.market.levels.curve_start["USD"][_Y10_CURVE_START_INDEX] - tenor_series[Tenor.Y10][0]
    )
    floor = config.market.levels.yield_floor_pct
    for tenor, y_real in tenor_series.items():
        curves[(curve_inst.instrument_id, tenor)] = np.maximum(y_real + shift, floor)
    return curve_inst.instrument_id


def _build_credit(
    instruments: Sequence[Instrument],
    cache: RawCache,
    dates: list[date],
    config: Config,
    fills: dict[str, int],
    prices: dict[str, np.ndarray],
    spreads: dict[str, np.ndarray],
    y5: np.ndarray,
    seed: str,
) -> None:
    dgs20 = _aligned(cache.fred("DGS20"), dates, "DGS20", fills)
    for inst in instruments:
        if inst.family != Family.CREDIT:
            continue
        reg = _REGISTRY_BY_ID[inst.instrument_id]
        yld = _aligned(cache.fred(reg.series), dates, inst.instrument_id, fills)
        s_real_bp = 100 * (yld - dgs20)
        if np.any(s_real_bp <= 0):
            raise StageIOError(
                f"real seed '{seed}': credit spread for '{reg.series}' is not positive on some day"
            )
        base = config.market.levels.credit_base_spread_bp[inst.rating_band]
        s = s_real_bp * base / s_real_bp[0]
        spreads[inst.instrument_id] = s

        n_days = len(dates)
        p = np.empty(n_days)
        p[0] = 100.0
        for t in range(1, n_days):
            move = (s[t] - s[t - 1]) + 100 * (y5[t] - y5[t - 1])
            p[t] = p[t - 1] * (1 - CREDIT_DURATION_YEARS * move / 10000)
        prices[inst.instrument_id] = p


def _build_fx(
    instruments: Sequence[Instrument],
    cache: RawCache,
    dates: list[date],
    config: Config,
    fills: dict[str, int],
    prices: dict[str, np.ndarray],
) -> None:
    fx_start = config.market.levels.fx_start
    n_days = len(dates)
    log_value: dict[str, np.ndarray] = {"USD": np.zeros(n_days)}
    for ccy, pair in USD_PAIR.items():
        base, _ = FX_PAIRS[pair]
        sign = 1.0 if base == ccy else -1.0
        reg = _REGISTRY_BY_ID[f"FX-{pair}"]
        raw = _aligned(cache.fred(reg.series), dates, f"FX-{pair}", fills)
        v_real = sign * np.log(raw)
        v_start = sign * np.log(fx_start[pair])
        log_value[ccy] = v_real - v_real[0] + v_start

    for inst in instruments:
        if inst.family != Family.FX:
            continue
        pair = inst.instrument_id.removeprefix("FX-")
        base, quote = FX_PAIRS[pair]
        prices[inst.instrument_id] = np.exp(log_value[base] - log_value[quote])


def build_seed(
    config: Config,
    seed: str,
    instruments: Sequence[Instrument],
    cache: RawCache,
    axis: SimAxis,
) -> SeedMarket:
    """Simulate one real market seed from its raw cache: every family's
    rebased series, the mapped regime schedule, the calendar and consensus.
    """
    instruments = tuple(instruments)
    spec = config.market.real.seeds[seed]
    calendar_start = config.calendar.start
    dates = real_axis_dates(spec, axis, calendar_start)
    rng_for = market_rng(config.seed.root)

    fills: dict[str, int] = {}
    prices: dict[str, np.ndarray] = {}
    spreads: dict[str, np.ndarray] = {}
    curves: dict[tuple[str, Tenor], np.ndarray] = {}

    curve_id = _build_curve(instruments, cache, dates, config, fills, curves)
    y5 = curves[(curve_id, Tenor.Y5)]

    _build_equities(instruments, cache, dates, config, fills, prices, rng_for)
    _build_commodities(instruments, cache, dates, config, fills, prices, curves)
    _build_credit(instruments, cache, dates, config, fills, prices, spreads, y5, seed)
    _build_fx(instruments, cache, dates, config, fills, prices)

    spy_raw = _aligned(cache.yahoo(REFERENCE_EQUITY), dates, REFERENCE_EQUITY, fills)
    spy_log_return = np.zeros(len(dates))
    spy_log_return[1:] = np.diff(np.log(spy_raw))
    z = spy_log_return / np.std(spy_log_return[1:])

    output = ProcessOutput(prices=prices, spreads=spreads, curves=curves)

    schedule = real_schedule(spec, axis, calendar_start, seed)
    lookup = RegimeLookup(schedule, spec.regime_starts[0][0])
    path = regime_path(axis, lookup, config)

    events = real_event_days(instruments, spec, axis, calendar_start, cache)
    y10_bp = curves[(curve_id, Tenor.Y10)] * 100
    event_rows = surprise_rows(events, output, spy_log_return, y10_bp, axis, seed)
    generated = generated_rows(instruments, axis, seed, config.market.consensus.report_weekday)

    event_days = event_day_indices(event_rows, axis)
    consensus = build_consensus(instruments, axis, output, event_days, config.market, rng_for, seed)

    calendar = sorted([*event_rows, *generated, *consensus.flips], key=row_sort_key)
    drawn_events = drawn_counts(event_rows)

    return SeedMarket(
        seed=seed,
        axis=axis,
        schedule=schedule,
        path=path,
        z=z,
        output=output,
        calendar=calendar,
        drawn_events=drawn_events,
        consensus=consensus,
        fills=fills,
    )
