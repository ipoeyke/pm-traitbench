"""Realised-moment check: verifies one generated market against what the config implies.

Every family collapses to one daily index (an equal-weight combination of its
instruments), which the model decomposes into a driver-loading and an
independent-noise term. Per regime span, the check compares the index's
realised annualised vol and its correlation with the driver against those two
model-implied moments, counts round-level crossings in the range span, and
checks that calendar row counts match what was drawn or is otherwise
deterministic.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pm_traitbench.config import Config, RegimeParams
from pm_traitbench.enums import (
    HY_BANDS,
    EventType,
    Family,
    InstrumentKind,
    RatingBand,
    Regime,
    Tenor,
)
from pm_traitbench.errors import MarketCheckError
from pm_traitbench.market.calendar import third_friday
from pm_traitbench.market.constants import ANNUALISATION_DAYS, FX_PAIRS, USD_PAIR
from pm_traitbench.market.generate import SeedMarket
from pm_traitbench.market.processes.common import ProcessOutput, log_grid_step, nearest_level
from pm_traitbench.tables.schema import Instrument

_IG_BANDS = set(RatingBand) - HY_BANDS
# macro_print and positioning_report are market-wide, with no single instrument
# family; reported under equities by convention.
_MARKET_WIDE_FAMILY = Family.EQUITIES
_EVENT_FAMILY: dict[EventType, Family] = {
    EventType.EARNINGS: Family.EQUITIES,
    EventType.RATING_DOWNGRADE: Family.CREDIT,
    EventType.RATING_UPGRADE: Family.CREDIT,
    EventType.CB_MEETING: Family.RATES,
    EventType.INVENTORY_REPORT: Family.COMMODITIES,
    EventType.CROP_REPORT: Family.COMMODITIES,
    EventType.MACRO_PRINT: _MARKET_WIDE_FAMILY,
}


@dataclass(frozen=True)
class CheckMetric:
    """One realised-versus-implied comparison, with the outcome already decided."""

    seed: str
    regime: Regime | None
    family: Family
    metric: str
    target: float
    realised: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class CheckReport:
    """Every metric checked for one market seed."""

    seed: str
    metrics: tuple[CheckMetric, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "metrics": [
                {
                    "seed": metric.seed,
                    "regime": metric.regime.value if metric.regime is not None else None,
                    "family": metric.family.value,
                    "metric": metric.metric,
                    "target": metric.target,
                    "realised": metric.realised,
                    "tolerance": metric.tolerance,
                    "passed": metric.passed,
                }
                for metric in self.metrics
            ],
        }


def _fx_currencies(instruments: Sequence[Instrument]) -> list[str]:
    """Non-USD currencies whose USD pair is configured, sorted for determinism."""
    pairs_present = {
        i.instrument_id.removeprefix("FX-") for i in instruments if i.family == Family.FX
    }
    currencies = sorted({ccy for code in pairs_present for ccy in FX_PAIRS[code] if ccy != "USD"})
    return [ccy for ccy in currencies if USD_PAIR.get(ccy) in pairs_present]


def _fx_currency_log_value(output: ProcessOutput, ccy: str) -> np.ndarray:
    """A currency's own log value against USD, signed by its anchor pair's quote convention."""
    anchor = USD_PAIR[ccy]
    base, quote = FX_PAIRS[anchor]
    price = output.prices[f"FX-{anchor}"]
    return np.log(price) if base == ccy else -np.log(price)


def _changes(level: np.ndarray) -> np.ndarray:
    """Day-over-day change series; position 0 is undefined, there is no prior day."""
    changes = np.full(level.shape, np.nan)
    changes[1:] = np.diff(level)
    return changes


def family_indices(
    market: SeedMarket, instruments: Sequence[Instrument]
) -> dict[Family, np.ndarray]:
    """Each family's daily index-change series over the whole axis: an equal-weight
    combination of its instruments, differenced day over day.
    """
    output = market.output

    equities = [i for i in instruments if i.family == Family.EQUITIES]
    equity_level = np.log(np.array([output.prices[e.instrument_id] for e in equities])).mean(axis=0)

    curves = [i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
    y10_level = np.array([output.curves[(c.instrument_id, Tenor.Y10)] for c in curves]).mean(axis=0)

    ig_issuers = [
        i for i in instruments if i.family == Family.CREDIT and i.rating_band in _IG_BANDS
    ]
    credit_level = np.log(
        np.array([output.spreads[i.instrument_id] for i in ig_issuers]).mean(axis=0)
    )

    commodities = [i for i in instruments if i.family == Family.COMMODITIES]
    commodity_level = np.log(np.array([output.prices[c.instrument_id] for c in commodities])).mean(
        axis=0
    )

    currencies = _fx_currencies(instruments)
    fx_level = np.array([_fx_currency_log_value(output, ccy) for ccy in currencies]).mean(axis=0)

    return {
        Family.EQUITIES: _changes(equity_level),
        Family.RATES: _changes(y10_level) * 100,
        Family.CREDIT: _changes(credit_level),
        Family.COMMODITIES: _changes(commodity_level),
        Family.FX: _changes(fx_level),
    }


def implied_moments(
    family: Family, params: RegimeParams, instruments: Sequence[Instrument], config: Config
) -> tuple[float, float]:
    """Model-implied annual vol and correlation with `z` for one family's index.

    Every index is a driver loading `B` plus independent noise of variance `I`
    (both annualised); vol is `m * sqrt(B^2 + I)` and correlation is
    `B / sqrt(B^2 + I)`, where `m` is the regime's vol multiplier.
    """
    families = config.market.families

    if family == Family.EQUITIES:
        equities = [i for i in instruments if i.family == Family.EQUITIES]
        cfg = families.equity
        loading = float(np.mean([e.beta for e in equities])) * cfg.market_vol
        independent = cfg.idio_vol**2 / len(equities)
    elif family == Family.RATES:
        curves = [i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
        n = len(curves)
        cfg = families.rates
        c = cfg.driver_corr
        loading = cfg.level_vol_bp * c
        independent = cfg.level_vol_bp**2 * (1 - c**2) / n + 0.15**2 * cfg.slope_vol_bp**2 / n
    elif family == Family.CREDIT:
        n_ig = sum(
            1 for i in instruments if i.family == Family.CREDIT and i.rating_band in _IG_BANDS
        )
        cfg = families.credit
        a, c = cfg.asymmetry, cfg.driver_corr
        k2 = (a**2 + a**-2) / 2 - (a - 1 / a) ** 2 / (2 * np.pi)
        loading = cfg.factor_vol * np.sqrt(k2) * c
        independent = cfg.factor_vol**2 * k2 * (1 - c**2) + cfg.issuer_vol**2 / n_ig
    elif family == Family.COMMODITIES:
        commodities = [i for i in instruments if i.family == Family.COMMODITIES]
        cfg = families.commodity
        share = cfg.group_share
        n_total = len(commodities)
        group_counts = Counter(c.commodity_group for c in commodities)
        loading = sum(
            (n_g / n_total) * cfg.group_vol[g] * cfg.driver_corr[g]
            for g, n_g in group_counts.items()
        )
        independent = sum(
            (n_g / n_total) ** 2 * cfg.group_vol[g] ** 2 * (1 - cfg.driver_corr[g] ** 2) * share
            for g, n_g in group_counts.items()
        ) + sum(
            n_g * cfg.group_vol[g] ** 2 * (1 - cfg.driver_corr[g] ** 2) * (1 - share) / n_total**2
            for g, n_g in group_counts.items()
        )
    elif family == Family.FX:
        currencies = _fx_currencies(instruments)
        n = len(currencies)
        cfg = families.fx
        loading = float(np.mean([cfg.currency_vol[c] * cfg.driver_corr[c] for c in currencies]))
        independent = (
            sum(cfg.currency_vol[c] ** 2 * (1 - cfg.driver_corr[c] ** 2) for c in currencies) / n**2
        )
    else:
        raise ValueError(f"unhandled family: {family}")

    variance = loading**2 + independent
    vol = params.vol_multiplier * np.sqrt(variance)
    rho = loading / np.sqrt(variance) if variance > 0 else 0.0
    return float(vol), float(rho)


def round_level_test_count(series: np.ndarray, step: float | None, band: float) -> int:
    """Count round-level tests along one instrument's close series: a step where
    the previous close is outside every grid level's band and the current close
    is inside a band, or where the two closes lie outside bands on opposite
    sides of a grid level. `step` is None for a price-dependent grid.
    """
    p0, p1 = series[:-1], series[1:]
    grid = log_grid_step(p1) if step is None else np.full_like(p1, step)

    raw0, raw1 = nearest_level(p0, grid), nearest_level(p1, grid)
    level0 = np.where(raw0 > 0, raw0, grid)
    level1 = np.where(raw1 > 0, raw1, grid)
    in0 = np.abs(p0 - level0) <= band * level0
    in1 = np.abs(p1 - level1) <= band * level1

    lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
    next_level = np.maximum(np.floor(lo / grid) + 1, 1) * grid
    crossing = next_level < hi

    return int(np.sum((~in0 & in1) | (~in0 & ~in1 & crossing)))


def _round_level_metrics(
    market: SeedMarket, instruments: Sequence[Instrument], config: Config
) -> list[CheckMetric]:
    """Family-mean round-level crossing counts, checked over the range span only."""
    range_span = next((span for span in market.schedule if span.regime == Regime.RANGE), None)
    assert range_span is not None, "every seed's schedule includes a range span"

    a = market.axis.index(range_span.date_start)
    b = market.axis.index(range_span.date_end)
    band = config.market.check.round_level_band
    min_tests = config.market.check.min_round_level_tests
    output = market.output

    def metric(family: Family, counts: list[int]) -> CheckMetric:
        mean_count = float(np.mean(counts))
        return CheckMetric(
            seed=market.seed,
            regime=Regime.RANGE,
            family=family,
            metric="round_level_tests",
            target=min_tests,
            realised=mean_count,
            tolerance=min_tests,
            passed=mean_count >= min_tests,
        )

    equities = [i for i in instruments if i.family == Family.EQUITIES]
    equity_counts = [
        round_level_test_count(output.prices[e.instrument_id][a - 1 : b + 1], None, band)
        for e in equities
    ]

    curves = [i for i in instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
    curve_counts = [
        round_level_test_count(
            output.curves[(c.instrument_id, Tenor.Y10)][a - 1 : b + 1], 0.25, band
        )
        for c in curves
    ]

    commodities = [i for i in instruments if i.family == Family.COMMODITIES]
    commodity_counts = [
        round_level_test_count(output.prices[c.instrument_id][a - 1 : b + 1], None, band)
        for c in commodities
    ]

    fx_pairs = [i for i in instruments if i.family == Family.FX]
    fx_counts = [
        round_level_test_count(
            output.prices[p.instrument_id][a - 1 : b + 1],
            1.0 if p.currency == "JPY" else 0.01,
            band,
        )
        for p in fx_pairs
    ]

    return [
        metric(Family.EQUITIES, equity_counts),
        metric(Family.RATES, curve_counts),
        metric(Family.COMMODITIES, commodity_counts),
        metric(Family.FX, fx_counts),
    ]


def _count_metric(seed: str, family: Family, name: str, target: int, realised: int) -> CheckMetric:
    return CheckMetric(
        seed=seed,
        regime=None,
        family=family,
        metric=f"count:{name}",
        target=float(target),
        realised=float(realised),
        tolerance=0.0,
        passed=target == realised,
    )


def _count_metrics(market: SeedMarket, instruments: Sequence[Instrument]) -> list[CheckMetric]:
    """One metric per calendar row count the pipeline can predict exactly."""
    calendar = market.calendar
    metrics: list[CheckMetric] = []

    event_counts = Counter(row.event for row in calendar)
    for event, expected in market.drawn_events.items():
        metrics.append(
            _count_metric(
                market.seed, _EVENT_FAMILY[event], event.value, expected, event_counts[event]
            )
        )

    flip_counts = Counter(
        row.instrument_id for row in calendar if row.event == EventType.CONSENSUS_FLIP
    )
    instrument_by_id = {i.instrument_id: i for i in instruments}
    for instrument_id, expected in market.consensus.drawn_flips.items():
        family = instrument_by_id[instrument_id].family
        metrics.append(
            _count_metric(
                market.seed, family, f"flip:{instrument_id}", expected, flip_counts[instrument_id]
            )
        )

    horizon_dates = market.axis.dates[market.axis.horizon]
    horizon_set = set(horizon_dates)
    months = {(day.year, day.month) for day in horizon_dates}
    expected_expiries = sum(1 for year, month in months if third_friday(year, month) in horizon_set)
    expiry_counts = Counter(
        row.instrument_id for row in calendar if row.event == EventType.CONTRACT_EXPIRY
    )
    for instrument in instruments:
        if instrument.family != Family.COMMODITIES:
            continue
        metrics.append(
            _count_metric(
                market.seed,
                Family.COMMODITIES,
                f"contract_expiry:{instrument.instrument_id}",
                expected_expiries,
                expiry_counts[instrument.instrument_id],
            )
        )

    expected_reports = sum(1 for day in horizon_dates if day.weekday() == 4)
    actual_reports = sum(1 for row in calendar if row.event == EventType.POSITIONING_REPORT)
    metrics.append(
        _count_metric(
            market.seed, _MARKET_WIDE_FAMILY, "positioning_report", expected_reports, actual_reports
        )
    )
    return metrics


def _check_error(misses: Sequence[CheckMetric]) -> MarketCheckError:
    first = misses[0]
    regime = first.regime.value if first.regime is not None else "none"
    return MarketCheckError(
        f"market check failed for seed {first.seed}, regime {regime}, family {first.family.value}, "
        f"metric {first.metric}: target {first.target:.4f}, realised {first.realised:.4f} "
        f"({len(misses)} misses)"
    )


def check_market(
    market: SeedMarket, instruments: Sequence[Instrument], config: Config
) -> CheckReport:
    """Verify one generated market's realised moments and calendar counts against
    what `config` implies. Raises `MarketCheckError` naming the first mismatch and
    the total miss count if any metric fails.
    """
    check_cfg = config.market.check
    indices = family_indices(market, instruments)
    metrics: list[CheckMetric] = []

    for span in market.schedule:
        params = config.market.regimes.params(span.regime)
        a = market.axis.index(span.date_start)
        b = market.axis.index(span.date_end)
        n_days = b - a + 1
        z_window = market.z[a : b + 1]

        for family in Family:
            changes = indices[family][a : b + 1]
            target_vol, rho = implied_moments(family, params, instruments, config)
            realised_vol = float(changes.std(ddof=1) * np.sqrt(ANNUALISATION_DAYS))
            metrics.append(
                CheckMetric(
                    seed=market.seed,
                    regime=span.regime,
                    family=family,
                    metric="vol",
                    target=target_vol,
                    realised=realised_vol,
                    tolerance=check_cfg.vol_tolerance,
                    passed=abs(realised_vol - target_vol) / target_vol <= check_cfg.vol_tolerance,
                )
            )

            if rho == 0.0:
                continue
            realised_corr = float(np.corrcoef(changes, z_window)[0, 1])
            corr_tolerance = float(check_cfg.corr_tolerance_se * (1 - rho**2) / np.sqrt(n_days))
            metrics.append(
                CheckMetric(
                    seed=market.seed,
                    regime=span.regime,
                    family=family,
                    metric="corr",
                    target=rho,
                    realised=realised_corr,
                    tolerance=corr_tolerance,
                    passed=abs(realised_corr - rho) <= corr_tolerance,
                )
            )

    metrics.extend(_round_level_metrics(market, instruments, config))
    metrics.extend(_count_metrics(market, instruments))

    misses = [metric for metric in metrics if not metric.passed]
    if misses:
        raise _check_error(misses)
    return CheckReport(seed=market.seed, metrics=tuple(metrics))
