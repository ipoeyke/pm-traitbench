"""Date splits: the horizon windows a bias parameter is estimated over.

Beyond the full horizon, a bias parameter is also split by any regime its trait
is boosted in, and by before/after its drift event, so a bias that only shows
up in a regime or after it drifts is not diluted by the rest of the horizon.
"""

from datetime import date

from pm_traitbench.enums import Gate1Split, Regime
from pm_traitbench.gates.gate1.inputs import PmInputs
from pm_traitbench.tables.schema import multiplier_field

_REGIME_SPLIT: dict[Regime, Gate1Split] = {
    Regime.RANGE: Gate1Split.REGIME_RANGE,
    Regime.RISK_OFF: Gate1Split.REGIME_RISK_OFF,
    Regime.RISK_ON: Gate1Split.REGIME_RISK_ON,
}
_SPLIT_REGIME: dict[Gate1Split, Regime] = {split: regime for regime, split in _REGIME_SPLIT.items()}


def splits_for(inputs: PmInputs, param: str) -> tuple[Gate1Split, ...]:
    """Splits worth computing for `param`: `ALL`, a split per regime the trait is boosted in
    (a multiplier above 1.0, missing counts as 1.0), then `BEFORE` and `AFTER` when the
    trait drifted. Order: `ALL`, regimes in enum order, `BEFORE`, `AFTER`.
    """
    trait = inputs.traits[param]
    splits = [Gate1Split.ALL]
    for regime, split in _REGIME_SPLIT.items():
        mult = getattr(trait, multiplier_field(regime))
        if mult is not None and mult > 1.0:
            splits.append(split)
    if inputs.drift_dates[param]:
        splits.append(Gate1Split.BEFORE)
        splits.append(Gate1Split.AFTER)
    return tuple(splits)


def days_for(split: Gate1Split, inputs: PmInputs, param: str) -> frozenset[date]:
    """The horizon dates `split` covers for `param`.

    `BEFORE` is every date strictly before the first drift event; `AFTER` starts
    at that event and runs to the horizon end, or to the second event when one
    exists (the dormant window of a dormant/revive pair). Raises `ValueError`
    for `BEFORE` or `AFTER` on a param with no drift event.
    """
    if split == Gate1Split.ALL:
        return frozenset(inputs.view.dates)

    regime = _SPLIT_REGIME.get(split)
    if regime is not None:
        return frozenset(
            day for day in inputs.view.dates if inputs.view.regime(inputs.day_index[day]) == regime
        )

    events = inputs.drift_dates[param]
    if not events:
        raise ValueError(f"param '{param}' has no drift event for split '{split.value}'")
    if split == Gate1Split.BEFORE:
        return frozenset(day for day in inputs.view.dates if day < events[0])
    hi = events[1] if len(events) >= 2 else None
    return frozenset(
        day for day in inputs.view.dates if day >= events[0] and (hi is None or day < hi)
    )
