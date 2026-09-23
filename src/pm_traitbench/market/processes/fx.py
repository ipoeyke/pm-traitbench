"""FX process: each non-USD currency's log value against USD.

Every currency's value mean-reverts toward its nearest round level through
its own USD pair; a cross pair carries no pull of its own, only what its two
currencies' values carry, so cross rates stay arbitrage-consistent by
construction.
"""

import numpy as np

from pm_traitbench.enums import Family
from pm_traitbench.market.constants import FX_PAIRS, USD_PAIR
from pm_traitbench.market.drivers import daily_vol
from pm_traitbench.market.processes.common import ProcessInputs, ProcessOutput, round_log_gap

_JPY_GRID_STEP = 1.0
_MAJOR_GRID_STEP = 0.01


def simulate(inputs: ProcessInputs) -> ProcessOutput:
    """Simulate every configured FX pair's price across the whole axis."""
    pairs = [i for i in inputs.instruments if i.family == Family.FX]
    if not pairs:
        return ProcessOutput()

    market = inputs.market
    fx_cfg = market.families.fx
    n_days = inputs.axis.n_days

    pair_codes = [p.instrument_id.removeprefix("FX-") for p in pairs]
    currencies = sorted({ccy for code in pair_codes for ccy in FX_PAIRS[code] if ccy != "USD"})
    n = len(currencies)
    kappa = inputs.path.kappa

    anchor = {ccy: USD_PAIR[ccy] for ccy in currencies}
    anchor_base_is_ccy = {ccy: FX_PAIRS[anchor[ccy]][0] == ccy for ccy in currencies}
    sign = np.array([1.0 if anchor_base_is_ccy[ccy] else -1.0 for ccy in currencies])
    step = np.array(
        [
            _JPY_GRID_STEP if FX_PAIRS[anchor[ccy]][1] == "JPY" else _MAJOR_GRID_STEP
            for ccy in currencies
        ]
    )
    sigma = np.array([daily_vol(fx_cfg.currency_vol[ccy], inputs.path) for ccy in currencies])
    corr = np.array([fx_cfg.driver_corr[ccy] for ccy in currencies])
    e = np.array(
        [
            inputs.rng_for("noise", f"FX-{ccy}", "value").standard_normal(n_days)
            for ccy in currencies
        ]
    )

    v = np.empty((n, n_days))
    for idx, ccy in enumerate(currencies):
        start = market.levels.fx_start[anchor[ccy]]
        v[idx, 0] = np.log(start) if anchor_base_is_ccy[ccy] else -np.log(start)

    for t in range(1, n_days):
        prev = v[:, t - 1]
        p_anchor = np.where(sign > 0, np.exp(prev), np.exp(-prev))
        gap = round_log_gap(p_anchor, step)
        pull = -kappa[t] * gap * sign
        dv = sigma[:, t] * (corr * inputs.z[t] + np.sqrt(1 - corr**2) * e[:, t]) + pull
        v[:, t] = prev + dv

    log_value = {ccy: v[idx] for idx, ccy in enumerate(currencies)}
    log_value["USD"] = np.zeros(n_days)

    prices = {}
    for p, code in zip(pairs, pair_codes, strict=True):
        base, quote = FX_PAIRS[code]
        prices[p.instrument_id] = np.exp(log_value[base] - log_value[quote])

    return ProcessOutput(prices=prices)
