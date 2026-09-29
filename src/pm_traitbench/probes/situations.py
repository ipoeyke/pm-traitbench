"""Hypothetical market situations for trait probes, built on real instruments and levels."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.engine.adapters.base import Adapter
from pm_traitbench.engine.biases.anchoring import entry_anchor
from pm_traitbench.engine.constants import ANCHOR_FRACTION
from pm_traitbench.engine.ideas import PRICE_QUOTED_OUTRIGHT_CLASSES
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.templates import level_text
from pm_traitbench.enums import AssetClass, RuleScope, StreetView
from pm_traitbench.errors import ProbesError
from pm_traitbench.tables.schema import Rule


@dataclass(frozen=True)
class MarketEnv:
    """The market and adapter a PM's probes are built against."""

    view: MarketView
    adapter: Adapter
    universe: tuple[str, ...]
    asset_class: AssetClass


@dataclass(frozen=True)
class Situation:
    """A probe situation: series-space levels and the rendered text slots for the bank line."""

    param: str
    instrument_id: str
    levels: dict[str, float]
    slots: dict[str, str]


def day_index(view: MarketView, day: date) -> int:
    """Index of `day` in the market's dates."""
    try:
        return view.dates.index(day)
    except ValueError:
        raise ProbesError(f"{day} is not a market date") from None


def instrument_slots(env: MarketEnv, instrument_id: str, t: int) -> dict[str, str]:
    """The instrument name and its current outright level as text."""
    series = env.adapter.outright_series(instrument_id)
    current = env.view.level(series, t)
    price_quoted = env.asset_class in PRICE_QUOTED_OUTRIGHT_CLASSES
    return {
        "instrument": env.view.instruments[instrument_id].name.replace("_", " "),
        "level": level_text(current, series.unit, price_quoted=price_quoted),
    }


def _stop_rule(rules: Sequence[Rule]) -> Rule:
    for rule in rules:
        if rule.scope == RuleScope.PM and rule.param == "stop_loss":
            return rule
    raise ProbesError("PM has no stop-loss rule to size a probe situation")


def _levels(
    param: str, current: float, adverse: float, sd: float, rr: float, config: Config
) -> dict[str, float]:
    """Series-space levels for the param's geometry, for a long idea."""
    probes = config.probes
    if param == "loss_aversion_lambda":
        entry = current - adverse * probes.loss_depth * sd
        return {
            "current": current,
            "entry": entry,
            "stop": entry + adverse * sd,
            "target": entry - adverse * rr * sd,
        }
    if param == "disposition_ratio":
        entry = current + adverse * probes.disposition_progress * rr * sd
        return {"current": current, "entry": entry, "target": entry - adverse * rr * sd}
    if param == "anchoring_rho":
        entry = current + adverse * probes.anchor_approach * ANCHOR_FRACTION * rr * sd
        return {"current": current, "entry": entry, "target": entry - adverse * rr * sd}
    if param == "exit_deficiency":
        return {"current": current, "entry": current - adverse * sd, "stop": current}
    return {"current": current}


def situation_for(
    param: str,
    env: MarketEnv,
    t: int,
    pm_rules: Sequence[Rule],
    horizons: Mapping[str, int | None],
    config: Config,
    rng: np.random.Generator,
) -> Situation | None:
    """The first qualifying situation over a random instrument order, or None."""
    stop_rule = _stop_rule(pm_rules)
    rr = float(np.mean(config.engine.rr_range))
    order = rng.permutation(np.array(env.universe))[: config.probes.situation_attempts]
    for instrument_id in (str(i) for i in order):
        series = env.adapter.outright_series(instrument_id)
        current = env.view.level(series, t)
        sd = env.adapter.stop_distance(stop_rule, series)
        adverse = -float(series.bullish_sign)
        levels = _levels(param, current, adverse, sd, rr, config)
        street = None
        if param == "anchoring_rho":
            round_level = entry_anchor(env.adapter, series, levels["entry"], levels["target"])
            if round_level is None or (round_level - current) * (-adverse) <= 0:
                continue
            levels["round_level"] = round_level
        elif param == "herding_weight":
            street = env.view.street_view(instrument_id, t)
            if street not in (StreetView.OVERWEIGHT, StreetView.UNDERWEIGHT):
                continue
        price_quoted = env.asset_class in PRICE_QUOTED_OUTRIGHT_CLASSES
        slots = instrument_slots(env, instrument_id, t)
        for name, level in levels.items():
            if name != "current":
                slots[name] = level_text(level, series.unit, price_quoted=price_quoted)
        slots.update(_param_slots(param, street, horizons, config))
        return Situation(param, instrument_id, levels, slots)
    return None


def _param_slots(
    param: str, street: StreetView | None, horizons: Mapping[str, int | None], config: Config
) -> dict[str, str]:
    if param in ("loss_aversion_lambda", "disposition_ratio"):
        return {"horizon": str(horizons[param])}
    if param == "extrapolation_theta":
        return {
            "horizon": str(config.engine.horizon_days),
            "thesis_sd": f"{abs(config.probes.extrapolation_thesis_sd):.1f}",
            "trailing_sd": f"{abs(config.probes.extrapolation_trailing_sd):.1f}",
        }
    if param == "conviction_size_miscalibration":
        return {"rating": str(config.probes.conviction_rating)}
    if street is not None:
        return {
            "street": street.value,
            "own_side": "sell" if street == StreetView.OVERWEIGHT else "buy",
        }
    return {}
