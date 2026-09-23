"""Population grid: enumerates which PMs exist, before any trait is sampled."""

from dataclasses import dataclass

from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, Split, Typicality

_TYPICALITIES = (Typicality.TYPICAL, Typicality.ANTI_TYPICAL)
_DRIFT_VALUES = (False, True)


@dataclass(frozen=True)
class PmSlot:
    """One PM's place in the population grid."""

    index: int
    pm_id: str
    split: Split
    asset_class: AssetClass
    market_seed: str
    typicality: Typicality
    drift: bool


def build_population(config: Config) -> list[PmSlot]:
    """Enumerate every PM slot: all pilot slots first, then all full slots.

    Both splits fill a grid of asset class, market seed, typicality and
    drift, so the pilot exercises drift as well as static behaviour. The
    pilot runs its own market seeds, kept separate from the full split's, so
    it can run on a different kind of market (a real one by default) while
    the full split stays fully synthetic.
    Within each split, replicate is the outermost loop so raising a
    per-cell count only appends slots and leaves existing indices unchanged.
    """
    pop = config.population
    cells: list[tuple[Split, AssetClass, str, Typicality, bool]] = []

    splits = (
        (Split.PILOT, pop.pilot_per_cell, pop.pilot_market_seeds),
        (Split.FULL, pop.full_per_cell, pop.market_seeds),
    )
    for split, per_cell, market_seeds in splits:
        for _ in range(per_cell):
            for asset_class in pop.asset_classes:
                for market_seed in market_seeds:
                    for typicality in _TYPICALITIES:
                        for drift in _DRIFT_VALUES:
                            cells.append((split, asset_class, market_seed, typicality, drift))

    width = max(3, len(str(len(cells))))
    return [
        PmSlot(
            index=index,
            pm_id=f"pm_{index:0{width}d}",
            split=split,
            asset_class=asset_class,
            market_seed=market_seed,
            typicality=typicality,
            drift=drift,
        )
        for index, (split, asset_class, market_seed, typicality, drift) in enumerate(cells, start=1)
    ]
