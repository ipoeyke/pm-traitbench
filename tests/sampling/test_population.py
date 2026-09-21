from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, Split, Typicality
from pm_traitbench.sampling.population import build_population


def _config(**population_overrides) -> Config:
    return Config.model_validate({"population": population_overrides})


def test_default_population_counts():
    slots = build_population(Config())
    pilot = [s for s in slots if s.split == Split.PILOT]
    full = [s for s in slots if s.split == Split.FULL]
    assert len(pilot) == 24
    assert len(full) == 144
    assert len(slots) == 168


def test_pm_ids_are_unique_and_zero_padded_to_three():
    slots = build_population(Config())
    ids = [s.pm_id for s in slots]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "pm_001"
    assert ids[-1] == "pm_168"


def test_pilot_slots_come_before_full_slots():
    slots = build_population(Config())
    splits = [s.split for s in slots]
    first_full = splits.index(Split.FULL)
    assert all(s == Split.PILOT for s in splits[:first_full])
    assert all(s == Split.FULL for s in splits[first_full:])


def test_every_pilot_cell_has_exactly_one_slot():
    config = Config()
    slots = [s for s in build_population(config) if s.split == Split.PILOT]
    counts: dict[tuple[AssetClass, str, Typicality], int] = {}
    for s in slots:
        key = (s.asset_class, s.market_seed, s.typicality)
        counts[key] = counts.get(key, 0) + 1
    expected_keys = {
        (asset_class, seed, typicality)
        for asset_class in config.population.asset_classes
        for seed in config.population.market_seeds
        for typicality in (Typicality.TYPICAL, Typicality.ANTI_TYPICAL)
    }
    assert set(counts) == expected_keys
    assert all(count == 1 for count in counts.values())


def test_every_full_cell_including_drift_has_exactly_full_per_cell_slots():
    config = Config()
    slots = [s for s in build_population(config) if s.split == Split.FULL]
    counts: dict[tuple[AssetClass, str, Typicality, bool], int] = {}
    for s in slots:
        key = (s.asset_class, s.market_seed, s.typicality, s.drift)
        counts[key] = counts.get(key, 0) + 1
    assert all(count == config.population.full_per_cell for count in counts.values())


def test_half_of_full_slots_have_drift_and_no_pilot_slot_does():
    slots = build_population(Config())
    pilot = [s for s in slots if s.split == Split.PILOT]
    full = [s for s in slots if s.split == Split.FULL]
    assert not any(s.drift for s in pilot)
    assert sum(1 for s in full if s.drift) == len(full) // 2


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
    assert len(pilot) == 4
    assert len(full) == 8


def test_zero_counts_give_empty_population():
    config = _config(pilot_per_cell=0, full_per_cell=0)
    assert build_population(config) == []


def test_large_population_widens_ids_to_four_digits():
    config = _config(
        asset_classes=list(AssetClass),
        market_seeds=["A", "B", "C", "D", "E"],
        pilot_per_cell=0,
        full_per_cell=13,
    )
    slots = build_population(config)
    assert len(slots) >= 1000
    assert len(slots[0].pm_id) == len("pm_0000")
    assert slots[-1].pm_id == f"pm_{len(slots):04d}"
