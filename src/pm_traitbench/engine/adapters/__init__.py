"""Adapter registry: maps a PM's asset class and sub-style to its behaviour adapter.

Also holds the mapping from a stated expression preference to the trade form
it selects, so a PM's preference traits can steer how an idea is expressed.
"""

from collections.abc import Callable, Sequence

from pm_traitbench.engine.adapters.base import Adapter
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.engine.adapters.rates_credit import RatesCreditAdapter
from pm_traitbench.enums import AssetClass, Expression, Kind
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Trait

ADAPTERS: dict[AssetClass, Callable[[str, int], Adapter]] = {
    AssetClass.EQUITIES: EquitiesAdapter,
    AssetClass.RATES_CREDIT: RatesCreditAdapter,
}


def adapter_for(asset_class: AssetClass, sub_style: str, horizon_days: int) -> Adapter:
    """The adapter for a PM's asset class, sub-style and idea horizon."""
    if asset_class == AssetClass.MULTI_ASSET:
        raise EngineError("multi_asset has no single adapter; ideas route per leg asset class")
    factory = ADAPTERS.get(asset_class)
    if factory is None:
        raise EngineError(f"no adapter registered for asset class '{asset_class.value}'")
    return factory(sub_style, horizon_days)


FORM_FOR_PREFERENCE: dict[tuple[str, str], Expression | None] = {
    ("duration_expression", "steepeners over outright duration"): Expression.CURVE,
    ("duration_expression", "outright duration over curve trades"): Expression.OUTRIGHT,
    ("duration_expression", "butterflies over outright duration"): None,
    ("pair_vs_outright", "express the view as a pair trade"): Expression.PAIR,
    ("pair_vs_outright", "express the view as an outright position"): Expression.OUTRIGHT,
    ("pair_vs_outright", "express the view as a basket versus the index"): None,
    (
        "curve_trade_expression",
        "express curve views as calendar spreads",
    ): Expression.CALENDAR_SPREAD,
    ("curve_trade_expression", "express curve views as spread ratios"): None,
    ("curve_trade_expression", "express curve views as butterfly spreads"): None,
    ("hedge_instrument", "hedge with index futures"): None,
    ("hedge_instrument", "hedge with options"): None,
    ("hedge_instrument", "hedge with a basket of single names"): None,
    ("futures_vs_etf", "express the view with futures"): None,
    ("futures_vs_etf", "express the view with an ETF"): None,
    ("futures_vs_etf", "express the view with the underlying instrument"): None,
    ("fx_hedge_expression", "hedge fx exposure with forwards"): None,
    ("fx_hedge_expression", "hedge fx exposure with options"): None,
    ("fx_hedge_expression", "hedge fx exposure with currency futures"): None,
    ("credit_index_vs_single_name", "express the view with a credit index"): None,
    ("credit_index_vs_single_name", "express the view with single-name bonds"): None,
    ("credit_index_vs_single_name", "express the view with credit default swaps"): None,
}


def preferred_form(traits: Sequence[Trait]) -> Expression | None:
    """The first trait, in order, whose preference value maps to an expression form."""
    for trait in traits:
        if trait.kind != Kind.PREFERENCE:
            continue
        form = FORM_FOR_PREFERENCE.get((trait.param, trait.value))
        if form is not None:
            return form
    return None
