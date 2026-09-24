"""Round-level grid helpers shared by market processes and the behaviour engine."""

import numpy as np

# Sovereign yield grid spacing the rates process rounds toward, in percent.
YIELD_STEP_PCT = 0.25
# Credit spread grid spacing the credit process rounds toward, in basis points.
SPREAD_STEP_BP = 10.0
# Curve slope grid spacing a curve position rounds toward, in basis points.
CURVE_STEP_BP = 25.0


def log_grid_step(price: np.ndarray | float) -> np.ndarray | float:
    """Round-level grid spacing: half the decade below `price`."""
    return 10.0 ** np.floor(np.log10(price)) / 2


def nearest_level(value: np.ndarray | float, step: np.ndarray | float) -> np.ndarray | float:
    """Round `value` to the nearest multiple of `step`."""
    return np.round(value / step) * step


def round_log_gap(price: np.ndarray | float, step: np.ndarray | float) -> np.ndarray | float:
    """Log distance from `price` to its nearest round grid level."""
    return np.log(price) - np.log(nearest_level(price, step))
