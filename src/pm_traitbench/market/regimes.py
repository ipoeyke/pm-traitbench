"""Regime schedule: which of the three regimes applies on each simulated day.

Each market seed cycles range, risk-off and risk-on across three spans at
fixed week boundaries. Burn-in days before the published horizon carry the
seed's first regime, so processes warm up already in that state.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.enums import Regime
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.tables.schema import RegimeSpan
from pm_traitbench.timeline import Timeline


def build_schedule(config: Config, seed: str, timeline: Timeline) -> list[RegimeSpan]:
    """Build the three regime spans for one market seed's week boundaries."""
    order = config.market.seeds[seed]
    b1, b2 = config.market.boundary_weeks
    week_ranges = ((1, b1), (b1 + 1, b2), (b2 + 1, timeline.n_weeks))
    spans = []
    for regime, (first, last) in zip(order, week_ranges, strict=True):
        date_start = timeline.week_start(first)
        date_end = timeline.week_start(last) + timedelta(days=4)
        spans.append(RegimeSpan(seed=seed, regime=regime, date_start=date_start, date_end=date_end))
    return spans


class RegimeLookup:
    """Maps a date to its regime: the burn-in regime before the first span."""

    def __init__(self, spans: Sequence[RegimeSpan], burn_in_regime: Regime) -> None:
        self._spans = list(spans)
        self._burn_in_regime = burn_in_regime

    def regime(self, day: date) -> Regime:
        if not self._spans or day < self._spans[0].date_start:
            return self._burn_in_regime
        for span in self._spans:
            if span.date_start <= day <= span.date_end:
                return span.regime
        raise ValueError(f"date {day} is after the last regime span")


@dataclass(frozen=True)
class RegimePath:
    """Per-axis-day regime and the regime parameters that apply on it."""

    regimes: tuple[Regime, ...]
    driver_mean: np.ndarray
    vol_multiplier: np.ndarray
    kappa: np.ndarray


def _path_from_regimes(regimes: tuple[Regime, ...], config: Config) -> RegimePath:
    params_by_regime = {regime: config.market.regimes.params(regime) for regime in set(regimes)}
    driver_mean = np.array([params_by_regime[r].driver_mean for r in regimes])
    vol_multiplier = np.array([params_by_regime[r].vol_multiplier for r in regimes])
    kappa = np.array([params_by_regime[r].mean_reversion_kappa for r in regimes])
    return RegimePath(
        regimes=regimes, driver_mean=driver_mean, vol_multiplier=vol_multiplier, kappa=kappa
    )


def regime_path(axis: SimAxis, lookup: RegimeLookup, config: Config) -> RegimePath:
    """Build the per-day regime parameter arrays across the whole axis."""
    regimes = tuple(lookup.regime(day) for day in axis.dates)
    return _path_from_regimes(regimes, config)


def constant_path(regime: Regime, n_days: int, config: Config) -> RegimePath:
    """Build a path holding one regime for every day, for tests and long-axis checks."""
    return _path_from_regimes((regime,) * n_days, config)
