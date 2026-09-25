"""Conviction-sizing bias: sizing rank is a mixture of stated conviction and a random rank.

With probability `m` the rank is drawn uniformly from 1-5 instead of matching
stated conviction, so `m` is the miscalibration between conviction and size.
"""

import numpy as np

from pm_traitbench.engine.params import EffectiveParams


def size_rank(
    conviction: int, params: EffectiveParams, rng: np.random.Generator
) -> tuple[int, str | None]:
    """Sizing rank: `conviction` with probability `1 - m`, else a uniform rank in 1-5."""
    m = params.value("conviction_size_miscalibration")
    u = rng.uniform()
    rank = int(rng.integers(1, 6)) if u < m else conviction
    active = params.is_active("conviction_size_miscalibration")
    flag = "conviction:mis_sized" if active and rank != conviction else None
    return rank, flag
