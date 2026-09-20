"""Shared helper for drawing a uniform index from a possibly-empty sequence."""

from numpy.random import Generator

from pm_traitbench.errors import SamplingError


def pick_index(rng: Generator, n: int, what: str) -> int:
    """Draw a uniform index in [0, n); raise SamplingError naming `what` if n <= 0."""
    if n <= 0:
        raise SamplingError(f"cannot pick from empty {what}")
    return int(rng.integers(n))
