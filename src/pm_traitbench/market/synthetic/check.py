"""Realised-moment check: verifies one generated market against what the config implies.

Every family collapses to one daily index (an equal-weight combination of its
instruments), which the model decomposes into a driver-loading and an
independent-noise term. Per regime span, the check compares the index's
realised annualised vol and its correlation with the driver against those two
model-implied moments, and counts round-level crossings in the range span.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from pm_traitbench.config import Config, RegimeParams
from pm_traitbench.enums import IG_BANDS, Family, InstrumentKind, Regime, Tenor
from pm_traitbench.market.check import (
    CheckMetric,
    CheckReport,
    check_error,
    count_metrics,
    family_indices,
    fx_currencies,
)
from pm_traitbench.market.constants import ANNUALISATION_DAYS
from pm_traitbench.market.seed import SeedMarket
from pm_traitbench.market.synthetic.processes.common import log_grid_step, nearest_level
from pm_traitbench.tables.schema import Instrument


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
        sigma_c = np.array([cfg.level_vol_bp[curve.currency] for curve in curves])
        loading = sigma_c.mean() * c
        independent = np.sum(sigma_c**2) * (1 - c**2) / n**2 + 0.15**2 * cfg.slope_vol_bp**2 / n
    elif family == Family.CREDIT:
        n_ig = sum(
            1 for i in instruments if i.family == Family.CREDIT and i.rating_band in IG_BANDS
        )
        cfg = families.credit
        a, c = cfg.asymmetry, cfg.driver_corr
        k2 = (a**2 + a**-2) / 2 - (a - 1 / a) ** 2 / (2 * np.pi)
        # The slope of the centred asymmetric shock on the driver is the mean of the
        # two scalings it takes on either side of zero.
        loading = cfg.factor_vol * c * (a + 1 / a) / 2
        independent = cfg.factor_vol**2 * k2 - loading**2 + cfg.issuer_vol**2 / n_ig
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
        currencies = fx_currencies(instruments)
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
            # Fisher z stabilises a correlation estimate's variance near +-1, where
            # the normal-theory band on the correlation itself is too tight.
            clip = 1 - 1e-12
            z_realised = np.arctanh(np.clip(realised_corr, -clip, clip))
            z_target = np.arctanh(np.clip(rho, -clip, clip))
            corr_tolerance = float(check_cfg.corr_tolerance_se / np.sqrt(n_days - 3))
            metrics.append(
                CheckMetric(
                    seed=market.seed,
                    regime=span.regime,
                    family=family,
                    metric="corr",
                    target=rho,
                    realised=realised_corr,
                    tolerance=corr_tolerance,
                    passed=bool(abs(z_realised - z_target) <= corr_tolerance),
                )
            )

    metrics.extend(_round_level_metrics(market, instruments, config))
    metrics.extend(count_metrics(market, instruments))

    misses = [metric for metric in metrics if not metric.passed]
    if misses:
        raise check_error(misses)
    return CheckReport(seed=market.seed, metrics=tuple(metrics))
