"""Credit process: a common spread factor plus per-issuer AR(1) idiosyncratic noise.

Every issuer's spread mean-reverts toward the nearest 10bp round level, and
its bond price marks off both its own spread move and its currency's 5Y
sovereign yield move, scaled by duration.
"""

import numpy as np

from pm_traitbench.enums import HY_BANDS, Family, Tenor
from pm_traitbench.market.drivers import daily_vol
from pm_traitbench.market.processes.common import ProcessInputs, ProcessOutput, nearest_level

_PHI = 0.98  # AR(1) persistence: about a 34-trading-day half-life for issuer news


def simulate(inputs: ProcessInputs, rates: ProcessOutput) -> ProcessOutput:
    """Simulate every credit issuer's spread and bond price across the whole axis."""
    issuers = [i for i in inputs.instruments if i.family == Family.CREDIT]
    if not issuers:
        return ProcessOutput()

    market = inputs.market
    credit_cfg = market.families.credit
    n_days = inputs.axis.n_days
    n = len(issuers)
    kappa = inputs.path.kappa

    sigma_factor = daily_vol(credit_cfg.factor_vol, inputs.path)
    sigma_issuer = daily_vol(credit_cfg.issuer_vol, inputs.path)
    c = credit_cfg.driver_corr
    a = credit_cfg.asymmetry
    # Mean of the asymmetrically scaled standard-normal shock; subtracting it
    # keeps the skew from the asymmetry without adding a drift to the factor.
    m = (a - 1 / a) / np.sqrt(2 * np.pi)

    base_spread = np.array([market.levels.credit_base_spread_bp[i.rating_band] for i in issuers])
    hy_mult = np.array(
        [credit_cfg.hy_vol_multiplier if i.rating_band in HY_BANDS else 1.0 for i in issuers]
    )
    duration = np.array([i.duration_years for i in issuers])

    y5 = np.empty((n, n_days))
    for idx, issuer in enumerate(issuers):
        key = (f"RT-{issuer.currency}", Tenor.Y5)
        if key not in rates.curves:
            raise ValueError(f"no sovereign curve configured for currency '{issuer.currency}'")
        y5[idx] = rates.curves[key]

    e_factor = inputs.rng_for("noise", "credit_factor").standard_normal(n_days)
    e_issuer = np.array(
        [
            inputs.rng_for("noise", issuer.instrument_id, "issuer").standard_normal(n_days)
            for issuer in issuers
        ]
    )
    jump = np.array([inputs.jumps.for_instrument(issuer.instrument_id) for issuer in issuers])

    log_factor = np.zeros(n_days)
    noise = np.zeros((n, n_days))
    spreads = np.empty((n, n_days))
    prices = np.empty((n, n_days))
    spreads[:, 0] = base_spread * np.exp(hy_mult * log_factor[0] + noise[:, 0])
    prices[:, 0] = 100.0

    for t in range(1, n_days):
        shock = c * inputs.z[t] + np.sqrt(1 - c**2) * e_factor[t]
        widen = a if shock > 0 else 1 / a
        log_factor[t] = log_factor[t - 1] + sigma_factor[t] * (shock * widen - m)

        prev_spread = spreads[:, t - 1]
        level = np.maximum(nearest_level(prev_spread, 10.0), 10.0)
        pull = kappa[t] * (np.log(prev_spread) - np.log(level))
        noise[:, t] = _PHI * noise[:, t - 1] + sigma_issuer[t] * e_issuer[:, t] - pull - jump[:, t]

        spreads[:, t] = base_spread * np.exp(hy_mult * log_factor[t] + noise[:, t])
        prices[:, t] = prices[:, t - 1] * (
            1 - duration * ((spreads[:, t] - prev_spread) + 100 * (y5[:, t] - y5[:, t - 1])) / 10000
        )

    return ProcessOutput(
        prices={i.instrument_id: prices[idx] for idx, i in enumerate(issuers)},
        spreads={i.instrument_id: spreads[idx] for idx, i in enumerate(issuers)},
    )
