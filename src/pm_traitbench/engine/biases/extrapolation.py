"""Extrapolation bias: blends the PM's thesis move with the trailing market trend."""

from pm_traitbench.engine.params import EffectiveParams


def blend(thesis_move: float, trailing_move: float, params: EffectiveParams) -> float:
    """Forecast as a weighted blend of the thesis move and the trailing move."""
    theta = params.value("extrapolation_theta")
    return (1 - theta) * thesis_move + theta * trailing_move
