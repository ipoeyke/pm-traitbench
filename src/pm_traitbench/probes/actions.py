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
from pm_traitbench.enums import McqAction, OptionSource, PnlState

HAZARD_PARAMS: tuple[str, ...] = ("loss_aversion_lambda", "disposition_ratio")
_VALUE_PARAMS = frozenset({"anchoring_rho", "exit_deficiency", "herding_weight"})
LETTERS = "ABCD"
# Loss-side outcomes in `loss_side_outcomes` order, which is also the argmax tie order.
_LOSS_SIDE = (McqAction.ADD, McqAction.HOLD, McqAction.CUT)
# Bias-typical and default outcome per threshold param.
_THRESHOLD_ACTIONS: dict[str, tuple[McqAction, McqAction]] = {
    "anchoring_rho": (McqAction.EXIT_AT_ROUND_LEVEL, McqAction.HOLD_TO_TARGET),
    "herding_weight": (McqAction.FOLLOW_STREET, McqAction.OWN_READ),
    "conviction_size_miscalibration": (McqAction.SIZE_OFF_RATING, McqAction.SIZE_TO_RATING),
}


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


def action_for(
    param: str,
    value: float,
    facts: PmFacts,
    horizons: Mapping[str, int | None],
    config: Config,
) -> McqAction:
    """The engine's most likely outcome at `value`.

    Loss aversion takes the most likely of add, hold and cut over its horizon, ties going in
    that order; disposition is binary on the sell probability.
    """
    if param in HAZARD_PARAMS:
        horizon = horizons[param]
        if horizon is None:
            raise ValueError(f"no horizon separates active from neutral for '{param}'")
        if param == "loss_aversion_lambda":
            outcomes = loss_side_outcomes(value, facts, horizon)
            return _LOSS_SIDE[outcomes.index(max(outcomes))]
        p = typical_probability(param, value, facts, horizon, config)
        return McqAction.SELL_NOW if p >= 0.5 else McqAction.HOLD_TO_TARGET
    if param in _THRESHOLD_ACTIONS:
        typical, default = _THRESHOLD_ACTIONS[param]
        return typical if typical_probability(param, value, facts, None, config) >= 0.5 else default
    if param == "exit_deficiency":
        if value >= 0.5:
            return McqAction.ADD if facts.lambda_active else McqAction.LEAVE_ON
        return McqAction.EXIT_PER_STOP
    if param == "extrapolation_theta":
        f = blend(
            config.probes.extrapolation_thesis_sd,
            config.probes.extrapolation_trailing_sd,
            _params(param, value),
        )
        if f >= ENTRY_THRESHOLD:
            return McqAction.CHASE_RUN
        return McqAction.SELL_ON_THESIS if f <= -ENTRY_THRESHOLD else McqAction.STAND_ASIDE
    if param == "overconfidence_coverage":
        factor = size_factor(_params(param, value))[0]
        lo, hi = config.probes.overconfidence_size_edges
        if factor >= hi:
            return McqAction.SIZE_DOUBLE
        return McqAction.SIZE_ONE_AND_HALF if factor >= lo else McqAction.SIZE_STANDARD
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
    outcomes: Sequence[McqAction],
    texts: Mapping[McqAction, str],
    sourced: Sequence[tuple[OptionSource, McqAction]],
    rng: np.random.Generator,
) -> OptionSet:
    """Collapse sources sharing an outcome, fill unused outcomes with NONE, then shuffle.

    Options keep `outcomes` order before the shuffle, so free slots fill in that order.
    """
    if not sourced or sourced[0][0] != OptionSource.CURRENT:
        raise ValueError("the first sourced entry must be CURRENT")
    by_outcome: dict[McqAction, OptionSource] = {}
    for source, outcome in sourced:
        if outcome not in outcomes:
            raise ValueError(f"outcome '{outcome}' is not among the options")
        by_outcome.setdefault(outcome, source)
    for outcome in outcomes:
        by_outcome.setdefault(outcome, OptionSource.NONE)
    return _shuffled([texts[o] for o in outcomes], [by_outcome[o] for o in outcomes], rng)


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
