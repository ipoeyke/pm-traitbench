"""Sovereign curve process: Diebold-Li level and slope factors with a round-level pull.

Each currency's curve level and slope evolve as correlated random walks; a
fixed per-tenor offset reproduces the starting curve shape exactly, and every
published yield is floored at a configured minimum.
"""

import numpy as np

from pm_traitbench.enums import SOVEREIGN_TENORS, InstrumentKind, Tenor
from pm_traitbench.market.synthetic.drivers import daily_vol
from pm_traitbench.market.synthetic.processes.common import (
    ProcessInputs,
    ProcessOutput,
    nearest_level,
)

_WEIGHTS = np.array([-0.5, -0.15, 0.15, 0.5])
_Y10_INDEX = SOVEREIGN_TENORS.index(Tenor.Y10)


def simulate(inputs: ProcessInputs) -> ProcessOutput:
    """Simulate every sovereign curve's four tenor yields across the whole axis."""
    curves = [i for i in inputs.instruments if i.kind == InstrumentKind.SOVEREIGN_CURVE]
    if not curves:
        return ProcessOutput()

    market = inputs.market
    rates_cfg = market.families.rates
    n_days = inputs.axis.n_days
    n = len(curves)
    kappa = inputs.path.kappa

    starts = np.array([market.levels.curve_start[curve.currency] for curve in curves])
    level0 = starts.mean(axis=1)
    slope0 = ((starts - level0[:, None]) * _WEIGHTS).sum(axis=1) / (_WEIGHTS**2).sum()
    offsets = starts - (level0[:, None] + _WEIGHTS[None, :] * slope0[:, None])

    level_vol_bp = np.array([rates_cfg.level_vol_bp[curve.currency] for curve in curves])
    sigma_level = daily_vol(level_vol_bp[:, None] / 100, inputs.path)
    sigma_slope = daily_vol(rates_cfg.slope_vol_bp / 100, inputs.path)
    c = rates_cfg.driver_corr

    e_level = np.array(
        [
            inputs.rng_for("noise", curve.instrument_id, "level").standard_normal(n_days)
            for curve in curves
        ]
    )
    e_slope = np.array(
        [
            inputs.rng_for("noise", curve.instrument_id, "slope").standard_normal(n_days)
            for curve in curves
        ]
    )
    jump_bp = np.array([inputs.jumps.for_instrument(curve.instrument_id) for curve in curves])

    level = np.empty((n, n_days))
    slope = np.empty((n, n_days))
    level[:, 0] = level0
    slope[:, 0] = slope0

    for t in range(1, n_days):
        raw10_prev = level[:, t - 1] + 0.15 * slope[:, t - 1] + offsets[:, _Y10_INDEX]
        pull = kappa[t] * (raw10_prev - nearest_level(raw10_prev, 0.25))
        shock = c * inputs.z[t] + np.sqrt(1 - c**2) * e_level[:, t]
        level[:, t] = level[:, t - 1] + sigma_level[:, t] * shock - pull - jump_bp[:, t] / 100
        slope[:, t] = slope[:, t - 1] + sigma_slope[t] * e_slope[:, t]

    yields = np.maximum(
        level[:, None, :] + _WEIGHTS[None, :, None] * slope[:, None, :] + offsets[:, :, None],
        market.levels.yield_floor_pct,
    )

    curves_out = {
        (curve.instrument_id, tenor): yields[idx, tenor_idx]
        for idx, curve in enumerate(curves)
        for tenor_idx, tenor in enumerate(SOVEREIGN_TENORS)
    }
    return ProcessOutput(curves=curves_out)
