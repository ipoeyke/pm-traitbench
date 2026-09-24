"""Catalogue-driven natural-language rendering: idea names, theses, outcomes and signposts.

Each render function picks one of the catalogue's candidate templates on the
given rng, then fills its slots. The catalogue's own consistency check
(`check_catalogue`) guarantees every template's slots are ones the loader
knows how to fill.
"""

import math

import numpy as np

from pm_traitbench.catalogues.loader import UNIT_DISPLAY
from pm_traitbench.catalogues.loader import render_signpost as _fill_signpost
from pm_traitbench.catalogues.loader import render_thesis as _fill_text
from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.series import LegRef
from pm_traitbench.enums import (
    FUTURES_TENORS,
    SOVEREIGN_TENORS,
    AssetClass,
    Expression,
    InstrumentKind,
    Tenor,
)

# Within-family tenor order (short/front first); the two families never mix
# within one idea, so reused ranks across families never get compared.
_TENOR_RANK: dict[Tenor, int] = {t: i for i, t in enumerate(SOVEREIGN_TENORS)}
_TENOR_RANK.update({t: i for i, t in enumerate(FUTURES_TENORS)})

# Display precision by series unit; a quoted price always shows 2 dp.
_LEVEL_DECIMALS: dict[str, int] = {"bp": 1, "pct": 2}
_PRICE_DECIMALS = 2


def level_text(level: float, unit: str, *, price_quoted: bool) -> str:
    """A series level in the instrument's own terms: a bare price for a price-quoted
    outright (`exp(level / 100)`), else the rounded level with its unit.
    """
    if price_quoted:
        return f"{math.exp(level / 100.0):.{_PRICE_DECIMALS}f}"
    return f"{level:.{_LEVEL_DECIMALS[unit]}f}{UNIT_DISPLAY[unit]}"


def move_text(move: float, unit: str) -> str:
    """A signed series move at the unit's fixed precision, without the unit."""
    return f"{move:+.{_LEVEL_DECIMALS[unit]}f}"


def _display_name(view: MarketView, instrument_id: str) -> str:
    return view.instruments[instrument_id].name.replace("_", " ")


def _curve_prefix(view: MarketView, instrument_id: str) -> str:
    """The currency for a sovereign curve (from its `RT-` id), else the instrument's name."""
    if view.instruments[instrument_id].kind == InstrumentKind.SOVEREIGN_CURVE:
        return instrument_id.removeprefix("RT-")
    return _display_name(view, instrument_id)


def idea_name(view: MarketView, legs: tuple[LegRef, ...], expression: Expression) -> str:
    """A human-readable name for a trade idea, from its legs and structural form."""
    first = legs[0].instrument_id
    if expression == Expression.OUTRIGHT:
        if view.instruments[first].kind == InstrumentKind.SOVEREIGN_CURVE:
            return f"{_curve_prefix(view, first)} {legs[0].tenor.value}"
        return _display_name(view, first)
    if expression == Expression.PAIR:
        return f"{_display_name(view, first)} versus {_display_name(view, legs[1].instrument_id)}"
    if expression in (Expression.CURVE, Expression.CALENDAR_SPREAD):
        tenors = sorted((leg.tenor for leg in legs), key=lambda tenor: _TENOR_RANK[tenor])
        return f"{_curve_prefix(view, first)} {tenors[0].value} versus {tenors[1].value}"
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
    unit: str,
    price_quoted: bool,
    horizon: int,
    rng: np.random.Generator,
) -> str:
    """Render one of the catalogue's thesis templates for this idea's asset class and form.

    `entry`, `target` and `move` are in series units; `move` is signed in the trade's
    favour and renders with `unit`, while entry and target go through `level_text`.
    """
    templates = catalogue.theses.theses[asset_class][expression]
    template = templates[int(rng.integers(len(templates)))]
    return _fill_text(
        template,
        side=side,
        name=name,
        entry=level_text(entry, unit, price_quoted=price_quoted),
        target=level_text(target, unit, price_quoted=price_quoted),
        move=move_text(move, unit),
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
    level: float | None = None,
    unit: str | None = None,
    price_quoted: bool = False,
    window: int | None = None,
    event: str | None = None,
    peer: str | None = None,
    rng: np.random.Generator,
) -> str:
    """Render one of the catalogue's signpost templates ('event', 'level' or 'relative').

    A numeric `level` (series units, with `unit`) renders through `level_text`.
    """
    templates = getattr(catalogue.signposts[asset_class], kind)
    template = templates[int(rng.integers(len(templates)))]
    rendered = None
    if level is not None:
        if unit is None:
            raise ValueError("render_signpost_text: a numeric level needs its unit")
        rendered = level_text(level, unit, price_quoted=price_quoted)
    return _fill_signpost(template, level=rendered, window=window, event=event, peer=peer)
