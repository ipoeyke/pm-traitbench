"""Series-level text: renders an engine level in the instrument's own terms.

Equity and commodity series run in `100 * ln(price)`, so an outright on them
shows as the price back in price space; every other series shows its level
with its unit. Shared by the engine, the plan stage and the dialogue stage, so
a thesis, a stance and the narrator's idea list all quote the same number.
"""

import math

from pm_traitbench.catalogues.loader import UNIT_DISPLAY
from pm_traitbench.enums import AssetClass, Expression

PRICE_QUOTED_OUTRIGHT_CLASSES = (AssetClass.EQUITIES, AssetClass.COMMODITIES)

# Display precision by series unit; a quoted price always shows 2 dp.
_LEVEL_DECIMALS: dict[str, int] = {"bp": 1, "pct": 2}
_PRICE_DECIMALS = 2
# Every adapter of an asset class builds its series in one unit.
_UNIT_BY_ASSET_CLASS: dict[AssetClass, str] = {
    AssetClass.EQUITIES: "pct",
    AssetClass.COMMODITIES: "pct",
    AssetClass.RATES_CREDIT: "bp",
}


def level_text(level: float, unit: str, *, price_quoted: bool) -> str:
    """A series level in the instrument's own terms: a bare price for a price-quoted
    outright (`exp(level / 100)`), else the rounded level with its unit.
    """
    if price_quoted:
        return f"{math.exp(level / 100.0):.{_PRICE_DECIMALS}f}"
    return f"{level:.{_LEVEL_DECIMALS[unit]}f}{UNIT_DISPLAY[unit]}"


def is_price_quoted(asset_class: AssetClass, expression: Expression) -> bool:
    """Whether an idea's levels are `100 * ln(price)`: an equity or commodity outright."""
    return expression == Expression.OUTRIGHT and asset_class in PRICE_QUOTED_OUTRIGHT_CLASSES


def idea_level_text(level: float, asset_class: AssetClass, expression: Expression) -> str:
    """An idea's entry, target, stop or round level as the PM would quote it."""
    return level_text(
        level,
        _UNIT_BY_ASSET_CLASS[asset_class],
        price_quoted=is_price_quoted(asset_class, expression),
    )
