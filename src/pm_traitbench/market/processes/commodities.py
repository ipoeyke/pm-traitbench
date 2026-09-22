"""Commodity futures process: a group shock plus idiosyncratic noise on the spot.

Each commodity's front-month spot mean-reverts toward its deterministic round
level (spot carried forward by the group's roll slope), and the whole
M1-M12 curve marks off that same slope with independent per-tenor noise.
"""

import numpy as np

from pm_traitbench.enums import FUTURES_TENORS, CommodityGroup, Family
from pm_traitbench.market.constants import COMMODITIES
from pm_traitbench.market.drivers import daily_vol
from pm_traitbench.market.processes.common import (
    ProcessInputs,
    ProcessOutput,
    log_grid_step,
    round_log_gap,
    unit_student_t,
)

_SPEC_BY_CODE = {spec.code: spec for specs in COMMODITIES.values() for spec in specs}


def simulate(inputs: ProcessInputs) -> ProcessOutput:
    """Simulate every commodity's spot and M1-M12 curve across the whole axis."""
    commodities = [i for i in inputs.instruments if i.family == Family.COMMODITIES]
    if not commodities:
        return ProcessOutput()

    market = inputs.market
    commodity_cfg = market.families.commodity
    n_days = inputs.axis.n_days
    n = len(commodities)
    n_tenors = len(FUTURES_TENORS)
    kappa = inputs.path.kappa
    df = market.families.student_t_df
    share = commodity_cfg.group_share

    groups = [c.commodity_group for c in commodities]
    starts = np.array(
        [_SPEC_BY_CODE[c.instrument_id.removeprefix("CM-")].start for c in commodities]
    )
    corr = np.array([commodity_cfg.driver_corr[g] for g in groups])
    slope = np.array([commodity_cfg.curve_slope[g] for g in groups])

    sigma_by_group = {
        group: daily_vol(commodity_cfg.group_vol[group], inputs.path) for group in CommodityGroup
    }
    sigma = np.array([sigma_by_group[g] for g in groups])
    group_shock = np.array([inputs.group_shocks[g] for g in groups])

    t_noise = np.array(
        [
            unit_student_t(inputs.rng_for("noise", c.instrument_id, "idio"), df, n_days)
            for c in commodities
        ]
    )
    jump = np.array([inputs.jumps.for_instrument(c.instrument_id) for c in commodities])
    curve_noise = np.array(
        [
            inputs.rng_for("noise", c.instrument_id, "curve").standard_normal((n_days, n_tenors))
            for c in commodities
        ]
    )

    spot = np.empty((n, n_days))
    spot[:, 0] = starts

    for t in range(1, n_days):
        prev = spot[:, t - 1]
        front_det = prev * (1 + slope / 12)
        gap = round_log_gap(front_det, log_grid_step(front_det))
        non_driver = np.sqrt(share) * group_shock[:, t] + np.sqrt(1 - share) * t_noise[:, t]
        r_t = (
            sigma[:, t] * (corr * inputs.z[t] + np.sqrt(1 - corr**2) * non_driver)
            - kappa[t] * gap
            + jump[:, t]
        )
        spot[:, t] = prev * np.exp(r_t)

    k = np.arange(1, n_tenors + 1)
    curve = (
        spot[:, None, :]
        * (1 + slope[:, None, None] * k[None, :, None] / 12)
        * np.exp(0.002 * np.transpose(curve_noise, (0, 2, 1)))
    )

    prices = {c.instrument_id: curve[idx, 0] for idx, c in enumerate(commodities)}
    curves = {
        (c.instrument_id, tenor): curve[idx, tenor_idx]
        for idx, c in enumerate(commodities)
        for tenor_idx, tenor in enumerate(FUTURES_TENORS)
    }
    return ProcessOutput(prices=prices, curves=curves)
