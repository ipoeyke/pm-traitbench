"""Shared types and helpers for per-family market price processes.

Every process reads the same `ProcessInputs` and returns a `ProcessOutput`
that the generator merges across families into the simulated tables.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from pm_traitbench.config import MarketConfig
from pm_traitbench.enums import CommodityGroup
from pm_traitbench.market.axis import SimAxis
from pm_traitbench.market.calendar import RngFor
from pm_traitbench.market.output import ProcessOutput
from pm_traitbench.market.regimes import RegimePath
from pm_traitbench.market.synthetic.events import EventJumps
from pm_traitbench.tables.schema import Instrument

__all__ = [
    "ProcessInputs",
    "ProcessOutput",
    "unit_student_t",
    "log_grid_step",
    "nearest_level",
    "round_log_gap",
]


@dataclass(frozen=True)
class ProcessInputs:
    """Everything one family's process needs to simulate the whole axis."""

    instruments: tuple[Instrument, ...]
    axis: SimAxis
    path: RegimePath
    z: np.ndarray
    group_shocks: Mapping[CommodityGroup, np.ndarray]
    jumps: EventJumps
    market: MarketConfig
    rng_for: RngFor


def unit_student_t(rng: np.random.Generator, df: int, size: int | tuple[int, ...]) -> np.ndarray:
    """Draw Student-t(df) innovations rescaled to unit variance."""
    return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2))


def log_grid_step(price: np.ndarray | float) -> np.ndarray | float:
    """Round-level grid spacing: half the decade below `price`."""
    return 10.0 ** np.floor(np.log10(price)) / 2


def nearest_level(value: np.ndarray | float, step: np.ndarray | float) -> np.ndarray | float:
    """Round `value` to the nearest multiple of `step`."""
    return np.round(value / step) * step


def round_log_gap(price: np.ndarray | float, step: np.ndarray | float) -> np.ndarray | float:
    """Log distance from `price` to its nearest round grid level."""
    return np.log(price) - np.log(nearest_level(price, step))
