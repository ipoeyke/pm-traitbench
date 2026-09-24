"""Extrapolation bias: blends the PM's thesis move with the trailing market trend."""

from pm_traitbench.engine.params import EffectiveParams


def blend(thesis_move: float, trailing_move: float, params: EffectiveParams) -> float:
    """Forecast as a weighted blend of the thesis move and the trailing move."""
    theta = params.value("extrapolation_theta")
    return (1 - theta) * thesis_move + theta * trailing_move


def entered_after_run(trailing_move: float, sd_h: float, side_sign: int, bullish_sign: int) -> bool:
    """Whether the entry chases a trailing move that already ran beyond one sd of view."""
    return bullish_sign * side_sign * trailing_move > sd_h
