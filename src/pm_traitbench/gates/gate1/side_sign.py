"""The directional sign of a trade idea's side: +1 for a buy, -1 for a sell.

Used by anchoring, to rebuild an exit level from `pnl_unit`, and by
extrapolation, to rebuild an entry's directional sign.
"""

from pm_traitbench.enums import Side

SIDE_SIGN: dict[Side, int] = {Side.BUY: 1, Side.SELL: -1}
