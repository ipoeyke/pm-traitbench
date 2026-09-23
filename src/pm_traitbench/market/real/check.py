"""Real-market data-quality check: a real seed has no model targets, so every
metric here reports a realised value rather than checking it against an
implied one, except for the fixed structural checks (finiteness, positivity,
flatness, fill runs and calendar counts) that must hold regardless of the
source data.
"""

from collections.abc import Sequence

import numpy as np

from pm_traitbench.config import REAL_FILL_LIMIT, Config
from pm_traitbench.enums import Family, InstrumentKind
from pm_traitbench.market.check import (
    CheckMetric,
    CheckReport,
    check_error,
    count_metrics,
    family_indices,
)
from pm_traitbench.market.constants import ANNUALISATION_DAYS
from pm_traitbench.market.real.sources import CURVE_SERIES, REFERENCE_EQUITY
from pm_traitbench.market.seed import SeedMarket
from pm_traitbench.tables.schema import Instrument


def _instrument_series(
    market: SeedMarket, instrument_by_id: dict[str, Instrument]
) -> list[tuple[str, Family, np.ndarray, bool]]:
    """Every price, spread and curve series, named for its metric and tagged
    `is_yield` where a floor applies instead of plain positivity (the
    sovereign curve's tenors; a commodity curve's tenors are futures prices).
    """
    output = market.output
    series: list[tuple[str, Family, np.ndarray, bool]] = []
    for instrument_id, values in output.prices.items():
        series.append((instrument_id, instrument_by_id[instrument_id].family, values, False))
    for instrument_id, values in output.spreads.items():
        family = instrument_by_id[instrument_id].family
        series.append((f"{instrument_id}:spread", family, values, False))
    for (curve_id, tenor), values in output.curves.items():
        inst = instrument_by_id[curve_id]
        is_yield = inst.kind == InstrumentKind.SOVEREIGN_CURVE
        series.append((f"{curve_id}:{tenor.value}", inst.family, values, is_yield))
    return series


def _finite_metrics(
    market: SeedMarket, instrument_by_id: dict[str, Instrument]
) -> list[CheckMetric]:
    metrics = []
    for name, family, values, _ in _instrument_series(market, instrument_by_id):
        realised = int(np.sum(~np.isfinite(values)))
        metrics.append(
            CheckMetric(
                seed=market.seed,
                regime=None,
                family=family,
                metric=f"finite:{name}",
                target=0.0,
                realised=float(realised),
                tolerance=0.0,
                passed=realised == 0,
            )
        )
    return metrics


def _positive_metrics(
    market: SeedMarket, instrument_by_id: dict[str, Instrument], config: Config
) -> list[CheckMetric]:
    floor = config.market.levels.yield_floor_pct
    metrics = []
    for name, family, values, is_yield in _instrument_series(market, instrument_by_id):
        realised = int(np.sum(values < floor)) if is_yield else int(np.sum(values <= 0))
        metrics.append(
            CheckMetric(
                seed=market.seed,
                regime=None,
                family=family,
                metric=f"positive:{name}",
                target=0.0,
                realised=float(realised),
                tolerance=0.0,
                passed=realised == 0,
            )
        )
    return metrics


def _floored_metrics(
    market: SeedMarket, instrument_by_id: dict[str, Instrument], config: Config
) -> list[CheckMetric]:
    """Reported only: how many days each yield curve tenor sits at the floor."""
    floor = config.market.levels.yield_floor_pct
    metrics = []
    for name, family, values, is_yield in _instrument_series(market, instrument_by_id):
        if not is_yield:
            continue
        realised = float(np.sum(values <= floor))
        metrics.append(
            CheckMetric(
                seed=market.seed,
                regime=None,
                family=family,
                metric=f"floored:{name}",
                target=realised,
                realised=realised,
                tolerance=0.0,
                passed=True,
            )
        )
    return metrics


def _fill_family(name: str, instrument_by_id: dict[str, Instrument]) -> Family:
    """A fill-run key's family: an instrument's own family where the key is
    one, else SPY (equities), DGS20 (the credit spread base) or a Treasury
    curve tenor id (rates).
    """
    if name in instrument_by_id:
        return instrument_by_id[name].family
    if name == REFERENCE_EQUITY:
        return Family.EQUITIES
    if name == "DGS20":
        return Family.CREDIT
    if name in CURVE_SERIES.values():
        return Family.RATES
    raise ValueError(f"fill-run key '{name}' is not a known instrument or reference series")


def _fill_run_metrics(
    market: SeedMarket, instrument_by_id: dict[str, Instrument]
) -> list[CheckMetric]:
    metrics = []
    for name, longest_run in market.fills.items():
        metrics.append(
            CheckMetric(
                seed=market.seed,
                regime=None,
                family=_fill_family(name, instrument_by_id),
                metric=f"fill_run:{name}",
                target=float(REAL_FILL_LIMIT),
                realised=float(longest_run),
                tolerance=0.0,
                passed=longest_run <= REAL_FILL_LIMIT,
            )
        )
    return metrics


def _realised_moment_metrics(
    market: SeedMarket, instruments: Sequence[Instrument]
) -> list[CheckMetric]:
    """Per regime span and family: `vol` and `corr` with `z`, reported for
    information only (target set equal to realised), and `flat` - a family
    index constant over a span is a data problem, checked against zero.
    """
    indices = family_indices(market, instruments)
    metrics = []
    for span in market.schedule:
        a = market.axis.index(span.date_start)
        b = market.axis.index(span.date_end)
        z_window = market.z[a : b + 1]
        for family in Family:
            changes = indices[family][a : b + 1]
            std = changes.std(ddof=1)
            vol = float(std * np.sqrt(ANNUALISATION_DAYS))
            metrics.append(
                CheckMetric(
                    seed=market.seed,
                    regime=span.regime,
                    family=family,
                    metric="vol",
                    target=vol,
                    realised=vol,
                    tolerance=0.0,
                    passed=True,
                )
            )
            metrics.append(
                CheckMetric(
                    seed=market.seed,
                    regime=span.regime,
                    family=family,
                    metric=f"flat:{family.value}",
                    target=0.0,
                    realised=float(std),
                    tolerance=0.0,
                    passed=bool(std > 0),
                )
            )
            # Guards only the divide; a flat index already fails via `flat` above.
            corr = float(np.corrcoef(changes, z_window)[0, 1]) if std > 0 else 0.0
            metrics.append(
                CheckMetric(
                    seed=market.seed,
                    regime=span.regime,
                    family=family,
                    metric="corr",
                    target=corr,
                    realised=corr,
                    tolerance=0.0,
                    passed=True,
                )
            )
    return metrics


def check_real_market(
    market: SeedMarket, instruments: Sequence[Instrument], config: Config
) -> CheckReport:
    """Verify one real seed's structural integrity, then report its realised
    moments. Structural metrics are checked first and raise on any miss
    before a realised moment is computed, since vol/corr/flat over bad data
    (a non-finite or non-positive series, say) cannot be trusted. Raises
    `MarketCheckError` naming the first mismatch and the total miss count.
    """
    instrument_by_id = {i.instrument_id: i for i in instruments}

    structural: list[CheckMetric] = []
    structural.extend(_finite_metrics(market, instrument_by_id))
    structural.extend(_positive_metrics(market, instrument_by_id, config))
    structural.extend(_floored_metrics(market, instrument_by_id, config))
    structural.extend(_fill_run_metrics(market, instrument_by_id))
    structural.extend(count_metrics(market, instruments))

    misses = [metric for metric in structural if not metric.passed]
    if misses:
        raise check_error(misses)

    moments = _realised_moment_metrics(market, instruments)
    misses = [metric for metric in moments if not metric.passed]
    if misses:
        raise check_error(misses)

    return CheckReport(seed=market.seed, metrics=tuple(structural) + tuple(moments))
