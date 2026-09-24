"""Conviction-sizing bias: sizing rank drifts from stated conviction toward a random draw."""

import numpy as np

from pm_traitbench.engine.params import EffectiveParams


def size_rank(
    conviction: int, params: EffectiveParams, rng: np.random.Generator
) -> tuple[int, str | None]:
    """Sizing rank blended from `conviction` toward a random draw by the miscalibration weight."""
    m = params.value("conviction_size_miscalibration")
    u = rng.uniform(1, 5)
    rank = int(round((1 - m) * conviction + m * u))
    rank = min(max(rank, 1), 5)
    active = params.is_active("conviction_size_miscalibration")
    flag = "conviction:mis_sized" if active and rank != conviction else None
    return rank, flag
