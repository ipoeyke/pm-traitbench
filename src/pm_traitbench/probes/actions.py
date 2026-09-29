"""Closed-form MCQ actions per bias value, and option assembly."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pm_traitbench.config import Config
from pm_traitbench.engine.biases.disposition import sell_hazard
from pm_traitbench.engine.biases.extrapolation import blend
from pm_traitbench.engine.biases.overconfidence import size_factor
from pm_traitbench.engine.constants import (
    ENTRY_THRESHOLD,
    LOSS_ADD_CAP,
    LOSS_ADD_SLOPE,
    LOSS_CUT_HAZARD,
)
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.enums import OptionSource, PnlState

HAZARD_PARAMS: tuple[str, ...] = ("loss_aversion_lambda", "disposition_ratio")
_VALUE_PARAMS = frozenset({"anchoring_rho", "exit_deficiency", "herding_weight"})
_THRESHOLD_PARAMS = frozenset({"anchoring_rho", "herding_weight", "conviction_size_miscalibration"})
LETTERS = "ABCD"


@dataclass(frozen=True)
class PmFacts:
    """PM-level facts that shift which action is typical."""

    no_add_rule: bool
    lambda_active: bool
    exit_deficiency: float


NEUTRAL_FACTS = PmFacts(False, False, 0.0)


def _params(param: str, value: float) -> EffectiveParams:
    return EffectiveParams({param: value}, {param: True})


def loss_side_outcomes(value: float, facts: PmFacts, horizon: int) -> tuple[float, float, float]:
    """Probabilities that a losing position is added to first, held throughout, or cut first.

    Each of `horizon` loss-side sessions draws cut, add or hold from lambda-scaled hazards, as
    the engine does; the first cut or add ends the run.
    """
    p_cut = LOSS_CUT_HAZARD / value
    p_add = min(LOSS_ADD_CAP, LOSS_ADD_SLOPE * max(value - 1.0, 0.0))
    if facts.no_add_rule:
        # An add before any trigger needs a rule breach first.
        p_add *= facts.exit_deficiency
    total = p_cut + p_add
    if total == 0:
        return 0.0, 1.0, 0.0
    hold = max(1 - total, 0.0) ** horizon
    return p_add / total * (1 - hold), hold, p_cut / total * (1 - hold)


def typical_probability(
    param: str, value: float, facts: PmFacts, horizon: int | None, config: Config
) -> float:
    """Engine probability of the bias-typical outcome for a probability-driven param."""
    if param == "loss_aversion_lambda":
        if horizon is None:
            raise ValueError(f"{param} needs a horizon")
        return loss_side_outcomes(value, facts, horizon)[0]
    if param == "disposition_ratio":
        if horizon is None:
            raise ValueError(f"{param} needs a horizon")
        h = sell_hazard(
            config.probes.disposition_progress, PnlState.GAIN, _params(param, value), config
        )
        return 1 - (1 - h) ** horizon
    if param in _VALUE_PARAMS:
        return value
    if param == "conviction_size_miscalibration":
        # A uniform redraw lands on the stated rank 1 time in 5.
        return 0.8 * value
    raise ValueError(f"no closed-form probability for '{param}'")


def horizons(config: Config) -> dict[str, int | None]:
    """Horizon per hazard param, at the lower middle of the window separating active from neutral.

    The smallest horizon leaves about half of active PMs below 0.5, so the midpoint keeps the
    window's margin on both sides.
    """
    result: dict[str, int | None] = {}
    for param in HAZARD_PARAMS:
        spec = config.biases.params[param]
        active = spec.active.median_value()
        neutral = spec.neutral.median_value()
        window = [
            h
            for h in range(1, config.probes.max_horizon + 1)
            if typical_probability(param, active, NEUTRAL_FACTS, h, config) >= 0.5
            and typical_probability(param, neutral, NEUTRAL_FACTS, h, config) < 0.5
        ]
        result[param] = window[(len(window) - 1) // 2] if window else None
    return result


def action_index(
    param: str,
    value: float,
    facts: PmFacts,
    horizons: Mapping[str, int | None],
    config: Config,
) -> int:
    """Index into the bank's actions of the engine's most likely action at `value`.

    Loss aversion takes the most likely of add (0), hold (1) and cut (2) over its horizon, ties
    going to the lower index; disposition is binary on the sell probability.
    """
    if param in HAZARD_PARAMS:
        horizon = horizons[param]
        if horizon is None:
            raise ValueError(f"no horizon separates active from neutral for '{param}'")
        if param == "loss_aversion_lambda":
            outcomes = loss_side_outcomes(value, facts, horizon)
            return outcomes.index(max(outcomes))
        p = typical_probability(param, value, facts, horizon, config)
        return 0 if p >= 0.5 else 1
    if param in _THRESHOLD_PARAMS:
        return 0 if typical_probability(param, value, facts, None, config) >= 0.5 else 1
    if param == "exit_deficiency":
        if value >= 0.5:
            return 1 if facts.lambda_active else 0
        return 2
    if param == "extrapolation_theta":
        f = blend(
            config.probes.extrapolation_thesis_sd,
            config.probes.extrapolation_trailing_sd,
            _params(param, value),
        )
        if f >= ENTRY_THRESHOLD:
            return 0
        return 2 if f <= -ENTRY_THRESHOLD else 1
    if param == "overconfidence_coverage":
        factor = size_factor(_params(param, value))[0]
        lo, hi = config.probes.overconfidence_size_edges
        if factor >= hi:
            return 0
        return 1 if factor >= lo else 2
    raise ValueError(f"unknown bias parameter '{param}'")


@dataclass(frozen=True)
class OptionSet:
    """Shuffled option texts, their sources, and the letter of the current answer."""

    texts: tuple[str, ...]
    sources: tuple[OptionSource, ...]
    answer: str


def _shuffled(
    texts: Sequence[str], sources: Sequence[OptionSource], rng: np.random.Generator
) -> OptionSet:
    order = rng.permutation(len(texts))
    out_sources = tuple(sources[i] for i in order)
    return OptionSet(
        texts=tuple(texts[i] for i in order),
        sources=out_sources,
        answer=LETTERS[out_sources.index(OptionSource.CURRENT)],
    )


def assemble_action_options(
    actions: Sequence[str],
    sourced: Sequence[tuple[OptionSource, int]],
    rng: np.random.Generator,
) -> OptionSet:
    """Collapse sources sharing an action, fill unused actions with NONE, then shuffle."""
    if not sourced or sourced[0][0] != OptionSource.CURRENT:
        raise ValueError("the first sourced entry must be CURRENT")
    by_index: dict[int, OptionSource] = {}
    for source, idx in sourced:
        by_index.setdefault(idx, source)
    for idx in range(len(actions)):
        by_index.setdefault(idx, OptionSource.NONE)
    ordered = sorted(by_index)
    return _shuffled([actions[i] for i in ordered], [by_index[i] for i in ordered], rng)


def assemble_value_options(
    values: Sequence[str],
    current: str,
    pre_update: str | None,
    third_party: Collection[str],
    rng: np.random.Generator,
) -> OptionSet:
    """One option per catalogue value, tagged by where it came from, then shuffled."""
    if current not in values:
        raise ValueError(f"current value '{current}' is not among the options")

    def source(value: str) -> OptionSource:
        if value == current:
            return OptionSource.CURRENT
        if value == pre_update:
            return OptionSource.PRE_UPDATE
        if value in third_party:
            return OptionSource.THIRD_PARTY
        return OptionSource.NONE

    return _shuffled(list(values), [source(v) for v in values], rng)
