"""The `Estimate` an estimator returns, and its callable and spec types."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from pm_traitbench.config import Gate1Config
from pm_traitbench.gates.gate1.inputs import PmInputs


@dataclass(frozen=True)
class Estimate:
    """An estimator's recovered statistic for one PM, parameter and split.

    `value` is null when the split had too little data to compute; `n` is the
    observation count behind it either way. `pairs` carries calibration points
    `(direction, trailing move / horizon sd)`, extrapolation only.
    """

    value: float | None
    n: int
    pairs: tuple[tuple[float, float], ...] = ()


Estimator = Callable[[PmInputs, frozenset[date], Gate1Config], Estimate]


@dataclass(frozen=True)
class EstimatorSpec:
    """An estimator paired with whether a higher recovered value means a stronger bias."""

    estimate: Estimator
    higher_is_stronger: bool
