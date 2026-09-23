from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, Split, Typicality
from pm_traitbench.sampling.population import build_population


def _config(**population_overrides) -> Config:
    return Config.model_validate({"population": population_overrides})


def test_default_population_counts():
    slots = build_population(Config())
    pilot = [s for s in slots if s.split == Split.PILOT]
    full = [s for s in slots if s.split == Split.FULL]
    assert len(pilot) == 32
    assert len(full) == 144
    assert len(slots) == 176


def test_pm_ids_are_unique_and_zero_padded_to_three():
    slots = build_population(Config())
    ids = [s.pm_id for s in slots]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "pm_001"
    assert ids[-1] == "pm_176"


def test_pilot_slots_come_before_full_slots():
    slots = build_population(Config())
    splits = [s.split for s in slots]
    first_full = splits.index(Split.FULL)
    assert all(s == Split.PILOT for s in splits[:first_full])
    assert all(s == Split.FULL for s in splits[first_full:])


def _cell_counts(slots) -> dict[tuple[AssetClass, str, Typicality, bool], int]:
    counts: dict[tuple[AssetClass, str, Typicality, bool], int] = {}
    for s in slots:
        key = (s.asset_class, s.market_seed, s.typicality, s.drift)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _expected_cells(
    config: Config, market_seeds: tuple[str, ...]
) -> set[tuple[AssetClass, str, Typicality, bool]]:
    return {
        (asset_class, seed, typicality, drift)
        for asset_class in config.population.asset_classes
        for seed in market_seeds
        for typicality in (Typicality.TYPICAL, Typicality.ANTI_TYPICAL)
        for drift in (False, True)
    }


def test_every_pilot_cell_including_drift_has_exactly_pilot_per_cell_slots():
    config = Config()
    counts = _cell_counts(s for s in build_population(config) if s.split == Split.PILOT)
    assert set(counts) == _expected_cells(config, ("A",))
    assert all(count == config.population.pilot_per_cell for count in counts.values())


def test_every_full_cell_including_drift_has_exactly_full_per_cell_slots():
    config = Config()
    counts = _cell_counts(s for s in build_population(config) if s.split == Split.FULL)
    assert set(counts) == _expected_cells(config, ("A", "B", "C"))
    assert all(count == config.population.full_per_cell for count in counts.values())


def test_pilot_market_seed_count_widens_the_pilot_to_more_seeds():
    slots = build_population(_config(pilot_market_seed_count=2))
    pilot_seeds = {s.market_seed for s in slots if s.split == Split.PILOT}
    full_seeds = {s.market_seed for s in slots if s.split == Split.FULL}
    assert pilot_seeds == {"A", "B"}
    assert full_seeds == {"A", "B", "C"}


def test_half_of_the_slots_in_each_split_have_drift():
    slots = build_population(Config())
    for split in (Split.PILOT, Split.FULL):
        in_split = [s for s in slots if s.split == split]
        assert sum(1 for s in in_split if s.drift) == len(in_split) // 2


def test_drift_is_balanced_across_typicality_within_each_split():
    slots = build_population(Config())
    for split in (Split.PILOT, Split.FULL):
        for typicality in (Typicality.TYPICAL, Typicality.ANTI_TYPICAL):
            group = [s for s in slots if s.split == split and s.typicality == typicality]
            assert sum(1 for s in group if s.drift) == len(group) // 2


def test_raising_full_per_cell_leaves_existing_slot_indices_unchanged():
    default_slots = build_population(Config())
    grown_slots = build_population(_config(full_per_cell=4))
    assert grown_slots[: len(default_slots)] == default_slots


def test_restricted_config_gives_expected_counts():
    config = _config(
        asset_classes=[AssetClass.EQUITIES, AssetClass.RATES_CREDIT],
        market_seeds=["A"],
        pilot_per_cell=1,
        full_per_cell=1,
    )
    slots = build_population(config)
    pilot = [s for s in slots if s.split == Split.PILOT]
    full = [s for s in slots if s.split == Split.FULL]
    assert len(pilot) == 8
    assert len(full) == 8


def test_one_split_may_be_empty():
    slots = build_population(_config(pilot_per_cell=0))
    assert slots
    assert all(s.split == Split.FULL for s in slots)


def test_large_population_widens_ids_to_four_digits():
    config = Config.model_validate(
        {
            "population": {
                "asset_classes": list(AssetClass),
                "market_seeds": ["A", "B", "C", "D", "E"],
                "pilot_per_cell": 0,
                "full_per_cell": 13,
            },
            "market": {
                "seeds": {
                    "A": ["range", "risk_off", "risk_on"],
                    "B": ["risk_on", "range", "risk_off"],
                    "C": ["risk_off", "risk_on", "range"],
                    "D": ["range", "risk_on", "risk_off"],
                    "E": ["risk_off", "range", "risk_on"],
                }
            },
        }
    )
    slots = build_population(config)
    assert len(slots) >= 1000
    assert len(slots[0].pm_id) == len("pm_0000")
    assert slots[-1].pm_id == f"pm_{len(slots):04d}"
