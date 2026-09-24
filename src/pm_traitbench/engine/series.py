"""Series and leg types: how the engine names a level it can measure and score.

A `Series` is the level the engine tracks for one trade idea - one leg for an
outright, several for a pair, curve or calendar spread.
"""

from dataclasses import dataclass

from pm_traitbench.enums import Tenor
from pm_traitbench.errors import EngineError

_VALID_UNITS = ("pct", "bp")
_VALID_BULLISH_SIGNS = (-1, 1)


@dataclass(frozen=True)
class LegRef:
    """One leg of a series: an instrument, its tenor (if any), and its signed weight."""

    instrument_id: str
    tenor: Tenor | None
    coeff: float


@dataclass(frozen=True)
class Series:
    """A weighted sum of leg levels, plus how a bullish view reads on it.

    `bullish_sign` is +1 when a higher level favours the idea's stated
    direction, -1 when a lower level does (e.g. a short outright). `unit` is
    the unit the resulting level is expressed in.
    """

    legs: tuple[LegRef, ...]
    bullish_sign: int
    unit: str

    def __post_init__(self) -> None:
        if self.bullish_sign not in _VALID_BULLISH_SIGNS:
            raise EngineError(
                f"bullish_sign must be one of {_VALID_BULLISH_SIGNS}, got {self.bullish_sign}"
            )
        if self.unit not in _VALID_UNITS:
            raise EngineError(f"unit must be one of {_VALID_UNITS}, got '{self.unit}'")
