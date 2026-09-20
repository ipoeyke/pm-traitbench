"""Preference sampling: draws each PM's copilot-facing preferences."""

from dataclasses import dataclass

from numpy.random import Generator

from pm_traitbench.catalogues.models import Catalogue, PreferenceEntry, PreferenceGroup
from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass
from pm_traitbench.errors import SamplingError


@dataclass(frozen=True)
class PreferenceDraw:
    """One PM's sampled value for one preference parameter."""

    param: str
    value: str


def sample_preferences(
    asset_class: AssetClass, config: Config, catalogue: Catalogue, rng: Generator
) -> list[PreferenceDraw]:
    """Sample 4-8 preferences: one per group, then fill uniformly without replacement."""
    n = int(rng.integers(config.preferences.n_min, config.preferences.n_max + 1))
    applicable = catalogue.preferences_for(asset_class)
    if len(applicable) < n:
        raise SamplingError(
            f"asset class '{asset_class}' has only {len(applicable)} applicable preferences, "
            f"need {n}"
        )

    picked: list[PreferenceEntry] = []
    for group in PreferenceGroup:
        group_entries = [entry for entry in applicable if entry.group == group]
        picked.append(group_entries[rng.integers(len(group_entries))])

    pool = [entry for entry in applicable if entry not in picked]
    fill_count = n - len(picked)
    if fill_count > 0:
        fill_indices = rng.choice(len(pool), size=fill_count, replace=False)
        picked.extend(pool[i] for i in fill_indices)

    picked_params = {entry.param for entry in picked}
    ordered = [entry for entry in applicable if entry.param in picked_params]
    return [
        PreferenceDraw(param=entry.param, value=entry.values[rng.integers(len(entry.values))])
        for entry in ordered
    ]
