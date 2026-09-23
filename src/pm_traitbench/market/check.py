"""Shared realised-check building blocks: metric records, per-family indices and
calendar-count comparisons, used by both the synthetic and real market checks.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pm_traitbench.enums import (
    IG_BANDS,
    EventType,
    Family,
    InstrumentKind,
    Regime,
    Tenor,
)
from pm_traitbench.errors import MarketCheckError
from pm_traitbench.market.calendar import third_friday
from pm_traitbench.market.constants import FX_PAIRS, USD_PAIR
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.seed import SeedMarket
from pm_traitbench.tables.schema import Instrument

# macro_print has no single target instrument; its count metric carries no family.
_EVENT_FAMILY: dict[EventType, Family | None] = {
    EventType.EARNINGS: Family.EQUITIES,
    EventType.RATING_DOWNGRADE: Family.CREDIT,
    EventType.RATING_UPGRADE: Family.CREDIT,
    EventType.CB_MEETING: Family.RATES,
    EventType.INVENTORY_REPORT: Family.COMMODITIES,
    EventType.CROP_REPORT: Family.COMMODITIES,
    EventType.MACRO_PRINT: None,
}


@dataclass(frozen=True)
class CheckMetric:
    """One realised-versus-implied comparison, with the outcome already decided."""

    seed: str
    regime: Regime | None
    family: Family | None
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
                    "family": metric.family.value if metric.family is not None else None,
                    "metric": metric.metric,
                    "target": metric.target,
                    "realised": metric.realised,
                    "tolerance": metric.tolerance,
                    "passed": metric.passed,
                }
                for metric in self.metrics
            ],
        }


def fx_currencies(instruments: Sequence[Instrument]) -> list[str]:
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

    ig_issuers = [i for i in instruments if i.family == Family.CREDIT and i.rating_band in IG_BANDS]
    credit_level = np.log(
        np.array([output.spreads[i.instrument_id] for i in ig_issuers]).mean(axis=0)
    )

    commodities = [i for i in instruments if i.family == Family.COMMODITIES]
    commodity_level = np.log(np.array([output.prices[c.instrument_id] for c in commodities])).mean(
        axis=0
    )

    currencies = fx_currencies(instruments)
    fx_level = np.array([_fx_currency_log_value(output, ccy) for ccy in currencies]).mean(axis=0)

    return {
        Family.EQUITIES: _changes(equity_level),
        Family.RATES: _changes(y10_level) * 100,
        Family.CREDIT: _changes(credit_level),
        Family.COMMODITIES: _changes(commodity_level),
        Family.FX: _changes(fx_level),
    }


def _count_metric(
    seed: str, family: Family | None, name: str, target: int, realised: int
) -> CheckMetric:
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


def _date_mismatch_metric(
    seed: str, family: Family | None, name: str, expected: Sequence[date], actual: Sequence[date]
) -> CheckMetric:
    """A count metric that fails on any missing, duplicate or extra date: `target`
    is 0 mismatches, `realised` is the actual mismatch count between the expected
    and actual date multisets.
    """
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    mismatch = sum((actual_counts - expected_counts).values()) + sum(
        (expected_counts - actual_counts).values()
    )
    return CheckMetric(
        seed=seed,
        regime=None,
        family=family,
        metric=f"count:{name}",
        target=0.0,
        realised=float(mismatch),
        tolerance=0.0,
        passed=mismatch == 0,
    )


def count_metrics(market: SeedMarket, instruments: Sequence[Instrument]) -> list[CheckMetric]:
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
    month_expiries = {(year, month): third_friday(year, month) for year, month in months}
    expected_expiries = sorted(day for day in month_expiries.values() if day in horizon_set)
    expiry_dates: dict[str, list[date]] = {}
    for row in calendar:
        if row.event == EventType.CONTRACT_EXPIRY:
            expiry_dates.setdefault(row.instrument_id, []).append(row.date)
    for instrument in instruments:
        if instrument.family != Family.COMMODITIES:
            continue
        metrics.append(
            _date_mismatch_metric(
                market.seed,
                Family.COMMODITIES,
                f"contract_expiry:{instrument.instrument_id}",
                expected_expiries,
                expiry_dates.get(instrument.instrument_id, []),
            )
        )

    expected_reports = sorted(day for day in horizon_dates if day.weekday() == 4)
    actual_reports = [row.date for row in calendar if row.event == EventType.POSITIONING_REPORT]
    metrics.append(
        _date_mismatch_metric(
            market.seed, None, "positioning_report", expected_reports, actual_reports
        )
    )
    return metrics


def check_error(misses: Sequence[CheckMetric]) -> MarketCheckError:
    first = misses[0]
    regime = first.regime.value if first.regime is not None else "none"
    family = first.family.value if first.family is not None else "none"
    count = "1 miss" if len(misses) == 1 else f"{len(misses)} misses"
    return MarketCheckError(
        f"market check failed for seed {first.seed}, regime {regime}, family {family}, "
        f"metric {first.metric}: target {first.target:.4f}, realised {first.realised:.4f} "
        f"({count})"
    )
