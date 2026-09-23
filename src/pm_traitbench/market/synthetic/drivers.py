"""Market driver shocks: the common risk factor and per-commodity-group shocks.

Every family's return process reads off the same common driver `z`; the
commodity groups additionally draw their own shock vector so commodities in
the same group co-move beyond what the common driver explains.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from pm_traitbench.enums import CommodityGroup
from pm_traitbench.market.constants import ANNUALISATION_DAYS
from pm_traitbench.market.regimes import RegimePath
from pm_traitbench.rng import stream


@dataclass(frozen=True)
class DriverShocks:
    """Standard-normal innovations: one common vector, one per commodity group."""

    common: np.ndarray
    groups: Mapping[CommodityGroup, np.ndarray]


def draw_shocks(root_seed: int, n_days: int) -> DriverShocks:
    """Draw the common and every commodity group's standard-normal shock vector.

    Seed-independent: drawn once per root seed regardless of the market seed,
    and every commodity group is drawn even if the config simulates none of
    its commodities.
    """
    common = stream(root_seed, "market", "driver", "common").standard_normal(n_days)
    groups = {
        group: stream(root_seed, "market", "driver", group.value).standard_normal(n_days)
        for group in CommodityGroup
    }
    return DriverShocks(common=common, groups=groups)


def seed_driver(shocks: DriverShocks, path: RegimePath, macro: np.ndarray) -> np.ndarray:
    """Combine the common shock with the regime's driver mean and the macro surprise sum."""
    return shocks.common + path.driver_mean + macro


def daily_vol(annual_vol: float, path: RegimePath) -> np.ndarray:
    """Scale an annual vol to daily, modulated by the regime's vol multiplier."""
    return annual_vol * path.vol_multiplier / np.sqrt(ANNUALISATION_DAYS)
