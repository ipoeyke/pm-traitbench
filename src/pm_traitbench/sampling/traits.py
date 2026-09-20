"""Trait row assembly: turns bias and preference draws into Trait rows."""

from pm_traitbench.enums import Kind
from pm_traitbench.sampling.biases import BiasDraw
from pm_traitbench.sampling.preferences import PreferenceDraw
from pm_traitbench.tables.schema import Trait, multiplier_field


def build_traits(
    pm_id: str, biases: list[BiasDraw], preferences: list[PreferenceDraw]
) -> list[Trait]:
    """Assemble a PM's bias and preference draws into Trait rows.

    Biases become t_01..t_08 in the given order; preferences follow from t_09.
    """
    traits: list[Trait] = []
    trait_index = 1

    for bias in biases:
        multiplier_kwargs = {
            multiplier_field(regime): value for regime, value in bias.multipliers.items()
        }
        traits.append(
            Trait(
                pm_id=pm_id,
                trait_id=f"t_{trait_index:02d}",
                kind=Kind.BIAS,
                param=bias.param,
                value=bias.value,
                active=bias.active,
                **multiplier_kwargs,
            )
        )
        trait_index += 1

    for preference in preferences:
        traits.append(
            Trait(
                pm_id=pm_id,
                trait_id=f"t_{trait_index:02d}",
                kind=Kind.PREFERENCE,
                param=preference.param,
                value=preference.value,
                active=True,
                mult_range=None,
                mult_risk_off=None,
                mult_risk_on=None,
            )
        )
        trait_index += 1

    return traits
