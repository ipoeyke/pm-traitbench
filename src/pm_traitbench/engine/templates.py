"""Catalogue-driven natural-language rendering: idea names, theses, outcomes and signposts.

Each render function picks one of the catalogue's candidate templates on the
given rng, then fills its slots. The catalogue's own consistency check
(`check_catalogue`) guarantees every template's slots are ones the loader
knows how to fill.
"""

import numpy as np

from pm_traitbench.catalogues.loader import render_signpost as _fill_signpost
from pm_traitbench.catalogues.loader import render_thesis as _fill_text
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef
from pm_traitbench.enums import FUTURES_TENORS, SOVEREIGN_TENORS, AssetClass, Expression, Tenor

# Within-family tenor order (short/front first); the two families never mix
# within one idea, so reused ranks across families never get compared.
_TENOR_RANK: dict[Tenor, int] = {t: i for i, t in enumerate(SOVEREIGN_TENORS)}
_TENOR_RANK.update({t: i for i, t in enumerate(FUTURES_TENORS)})


def idea_name(view: MarketView, legs: tuple[LegRef, ...], expression: Expression) -> str:
    """A human-readable name for a trade idea, from its legs and structural form."""
    if expression == Expression.OUTRIGHT:
        return view.instruments[legs[0].instrument_id].name
    if expression == Expression.PAIR:
        return (
            f"{view.instruments[legs[0].instrument_id].name} versus "
            f"{view.instruments[legs[1].instrument_id].name}"
        )
    if expression in (Expression.CURVE, Expression.CALENDAR_SPREAD):
        tenors = sorted((leg.tenor for leg in legs), key=lambda tenor: _TENOR_RANK[tenor])
        return f"{tenors[0].value} versus {tenors[1].value}"
    raise ValueError(f"idea_name: no naming rule for expression '{expression.value}'")


def render_thesis(
    catalogue: Catalogue,
    asset_class: AssetClass,
    expression: Expression,
    *,
    side: str,
    name: str,
    entry: float,
    target: float,
    move: float,
    unit: str | None,
    horizon: int,
    rng: np.random.Generator,
) -> str:
    """Render one of the catalogue's thesis templates for this idea's asset class and form."""
    templates = catalogue.theses.theses[asset_class][expression]
    template = templates[int(rng.integers(len(templates)))]
    return _fill_text(
        template,
        side=side,
        name=name,
        entry=entry,
        target=target,
        move=move,
        unit=unit,
        horizon=horizon,
    )


def render_outcome(
    catalogue: Catalogue,
    *,
    kind: str,
    pnl: float,
    unit: str,
    closer: str,
    rng: np.random.Generator,
) -> str:
    """Render one of the catalogue's outcome templates ('win', 'loss' or 'open')."""
    templates = catalogue.theses.outcomes[kind]
    template = templates[int(rng.integers(len(templates)))]
    return _fill_text(template, pnl=pnl, unit=unit, closer=closer)


def render_signpost_text(
    catalogue: Catalogue,
    asset_class: AssetClass,
    kind: str,
    *,
    level: float | str | None = None,
    unit: str | None = None,
    window: int | None = None,
    event: str | None = None,
    peer: str | None = None,
    rng: np.random.Generator,
) -> str:
    """Render one of the catalogue's signpost templates ('event', 'level' or 'relative')."""
    templates = getattr(catalogue.signposts[asset_class], kind)
    template = templates[int(rng.integers(len(templates)))]
    return _fill_signpost(template, level=level, unit=unit, window=window, event=event, peer=peer)
