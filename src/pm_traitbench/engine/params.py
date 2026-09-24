"""Effective bias parameters for a simulation day: drift events, then regime multiplier."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from pm_traitbench.config import Config
from pm_traitbench.enums import DriftEventType, Kind, Regime
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import DriftEvent, Trait

# Params scored as probabilities/rates: multiplied on the logit scale so the
# result stays in (0, 1) regardless of how large the multiplier is.
LOGIT_PARAMS: frozenset[str] = frozenset(
    {
        "anchoring_rho",
        "extrapolation_theta",
        "herding_weight",
        "conviction_size_miscalibration",
        "exit_deficiency",
    }
)

_CLIP_LO = 1e-6
_CLIP_HI = 1.0 - 1e-6


def _clip01(x: float) -> float:
    return min(max(x, _CLIP_LO), _CLIP_HI)


def _logit(x: float) -> float:
    x = _clip01(x)
    return math.log(x / (1.0 - x))


def _sigmoid(z: float) -> float:
    return _clip01(1.0 / (1.0 + math.exp(-z)))


def _apply_multiplier(param: str, value: float, multiplier: float) -> float:
    if param in LOGIT_PARAMS:
        return _sigmoid(_logit(value) + math.log(multiplier))
    if param == "overconfidence_coverage":
        # Lower coverage is the stronger bias, so the multiplier acts on 1 - c.
        return 1.0 - _sigmoid(_logit(1.0 - value) + math.log(multiplier))
    return value * multiplier


@dataclass(frozen=True)
class EffectiveParams:
    """A PM's bias parameter values and activity for one simulation day."""

    values: dict[str, float]
    active: dict[str, bool]

    def value(self, param: str) -> float:
        if param not in self.values:
            raise EngineError(f"unknown bias parameter '{param}'")
        return self.values[param]

    def is_active(self, param: str) -> bool:
        if param not in self.active:
            raise EngineError(f"unknown bias parameter '{param}'")
        return self.active[param]


@dataclass(frozen=True)
class ParamSchedule:
    """A PM's base bias parameters plus their dated drift events.

    `for_day` applies every drift event dated on or before the given day, in
    date order, then the regime's multiplier.
    """

    base_values: dict[str, float]
    active: dict[str, bool]
    multipliers: dict[str, dict[Regime, float]]
    neutral_medians: dict[str, float]
    events: tuple[DriftEvent, ...]
    param_by_trait_id: dict[str, str]

    @staticmethod
    def build(
        traits: Sequence[Trait], drift_events: Sequence[DriftEvent], config: Config
    ) -> "ParamSchedule":
        # Trait ids repeat across PMs, so a table mixing PMs would silently
        # match an event to the wrong PM's trait.
        pm_ids = {t.pm_id for t in traits} | {e.pm_id for e in drift_events}
        if len(pm_ids) > 1:
            raise EngineError(
                f"ParamSchedule.build received rows for more than one PM: {sorted(pm_ids)}"
            )

        bias_traits = [t for t in traits if t.kind == Kind.BIAS]
        param_by_trait_id = {t.trait_id: t.param for t in bias_traits}

        base_values = {t.param: float(t.value) for t in bias_traits}
        active = {t.param: t.active for t in bias_traits}
        multipliers = {
            t.param: {
                Regime.RANGE: t.mult_range,
                Regime.RISK_OFF: t.mult_risk_off,
                Regime.RISK_ON: t.mult_risk_on,
            }
            for t in bias_traits
        }
        neutral_medians = {
            param: config.biases.params[param].neutral.median_value() for param in base_values
        }

        # Only bias-trait events apply (preference drift has no param here); a stable
        # sort by date keeps same-day events in the order they were given.
        relevant = [e for e in drift_events if e.trait_id in param_by_trait_id]
        ordered = sorted(enumerate(relevant), key=lambda pair: (pair[1].date, pair[0]))
        events = tuple(e for _, e in ordered)

        return ParamSchedule(
            base_values=base_values,
            active=active,
            multipliers=multipliers,
            neutral_medians=neutral_medians,
            events=events,
            param_by_trait_id=param_by_trait_id,
        )

    def for_day(self, day: date, regime: Regime) -> EffectiveParams:
        values = dict(self.base_values)
        pre_dormant: dict[str, float] = {}

        for event in self.events:
            if event.date > day:
                continue
            param = self.param_by_trait_id[event.trait_id]
            if event.event == DriftEventType.UPDATE:
                values[param] = float(event.to_value)
            elif event.event == DriftEventType.DORMANT:
                # A second dormant while already dormant must not clobber the
                # value that was live before the first one.
                if param not in pre_dormant:
                    pre_dormant[param] = values[param]
                values[param] = self.neutral_medians[param]
            elif event.event == DriftEventType.REVIVE:
                if param not in pre_dormant:
                    raise EngineError(
                        f"revive event for trait '{event.trait_id}' has no matching dormant event"
                    )
                values[param] = pre_dormant.pop(param)

        effective = {
            param: _apply_multiplier(param, value, self.multipliers[param][regime])
            for param, value in values.items()
        }
        return EffectiveParams(values=effective, active=dict(self.active))
