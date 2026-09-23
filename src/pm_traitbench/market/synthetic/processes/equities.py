"""Equity price process: a market-beta driver plus idiosyncratic Student-t noise.

Each equity also mean-reverts toward the nearest round price level, pulled at
the regime's kappa, and carries its own sampled earnings jumps.
"""

import numpy as np

from pm_traitbench.enums import Family
from pm_traitbench.market.synthetic.drivers import daily_vol
from pm_traitbench.market.synthetic.processes.common import (
    ProcessInputs,
    ProcessOutput,
    log_grid_step,
    round_log_gap,
    unit_student_t,
)


def simulate(inputs: ProcessInputs) -> ProcessOutput:
    """Simulate every equity's price path across the whole axis."""
    equities = [i for i in inputs.instruments if i.family == Family.EQUITIES]
    if not equities:
        return ProcessOutput()

    market = inputs.market
    n_days = inputs.axis.n_days
    n = len(equities)

    sigma_m = daily_vol(market.families.equity.market_vol, inputs.path)
    sigma_i = daily_vol(market.families.equity.idio_vol, inputs.path)
    beta = np.array([equity.beta for equity in equities])
    kappa = inputs.path.kappa
    df = market.families.student_t_df

    log_lo, log_hi = (np.log(bound) for bound in market.levels.equity_price_range)
    t_noise = np.array(
        [
            unit_student_t(inputs.rng_for("noise", equity.instrument_id, "idio"), df, n_days)
            for equity in equities
        ]
    )
    jump = np.array([inputs.jumps.for_instrument(equity.instrument_id) for equity in equities])

    prices = np.empty((n, n_days))
    for idx, equity in enumerate(equities):
        rng = inputs.rng_for("level", equity.instrument_id)
        prices[idx, 0] = np.exp(rng.uniform(log_lo, log_hi))

    for t in range(1, n_days):
        prev = prices[:, t - 1]
        gap = round_log_gap(prev, log_grid_step(prev))
        r_t = (
            beta * sigma_m[t] * inputs.z[t]
            + sigma_i[t] * t_noise[:, t]
            - kappa[t] * gap
            + jump[:, t]
        )
        prices[:, t] = prev * np.exp(r_t)

    return ProcessOutput(prices={e.instrument_id: prices[idx] for idx, e in enumerate(equities)})
