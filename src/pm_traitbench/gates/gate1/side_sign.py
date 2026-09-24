"""The directional sign of a trade idea's side: +1 for a buy, -1 for a sell.

Shared by every estimator that rebuilds a level or a move from a public
`pnl_unit`, `trailing_move` or `forward_move` in bullish units.
"""

from pm_traitbench.enums import Side

SIDE_SIGN: dict[Side, int] = {Side.BUY: 1, Side.SELL: -1}
