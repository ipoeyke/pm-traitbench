"""Tests for the sample stage's pure sampling pass: sample_all."""

import pytest

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass, Kind, Typicality
from pm_traitbench.errors import ConfigError
from pm_traitbench.sampling.population import build_population
from pm_traitbench.sampling.stage import check_sampling_config, sample_all


def _config(**overrides) -> Config:
    population = {
        "asset_classes": [AssetClass.EQUITIES, AssetClass.RATES_CREDIT],
        "market_seeds": ["A"],
        "pilot_per_cell": 1,
        "full_per_cell": 1,
    }
    population.update(overrides.pop("population", {}))
    return Config.model_validate({"population": population, **overrides})


def test_persona_count_equals_population_size(fixture_catalogue: Catalogue):
    config = _config()
    result = sample_all(config, fixture_catalogue)
    assert len(result.personas) == len(build_population(config))


def test_every_persona_has_eight_bias_traits_and_four_to_eight_preference_traits(
    fixture_catalogue: Catalogue,
):
    config = _config()
    result = sample_all(config, fixture_catalogue)
    by_pm: dict[str, list] = {}
    for trait in result.traits:
        by_pm.setdefault(trait.pm_id, []).append(trait)
    assert set(by_pm) == {p.pm_id for p in result.personas}
    for traits in by_pm.values():
        biases = [t for t in traits if t.kind == Kind.BIAS]
        preferences = [t for t in traits if t.kind == Kind.PREFERENCE]
        assert len(biases) == 8
        assert 4 <= len(preferences) <= 8


def test_every_trait_rule_and_drift_pm_id_exists_in_personas(fixture_catalogue: Catalogue):
    config = _config()
    result = sample_all(config, fixture_catalogue)
    pm_ids = {p.pm_id for p in result.personas}
    assert all(t.pm_id in pm_ids for t in result.traits)
    assert all(r.pm_id in pm_ids for r in result.rules)
    assert all(d.pm_id in pm_ids for d in result.drift_events)


def test_every_drift_trait_id_exists_in_that_pms_traits(fixture_catalogue: Catalogue):
    config = _config()
    result = sample_all(config, fixture_catalogue)
    trait_ids_by_pm: dict[str, set[str]] = {}
    for trait in result.traits:
        trait_ids_by_pm.setdefault(trait.pm_id, set()).add(trait.trait_id)
    for event in result.drift_events:
        assert event.trait_id in trait_ids_by_pm[event.pm_id]


def test_drift_events_exist_only_for_drift_slots_and_each_has_at_least_one(
    fixture_catalogue: Catalogue,
):
    config = _config()
    slots = build_population(config)
    drift_pm_ids = {s.pm_id for s in slots if s.drift}
    no_drift_pm_ids = {s.pm_id for s in slots if not s.drift}
    result = sample_all(config, fixture_catalogue)
    pm_ids_with_events = {d.pm_id for d in result.drift_events}
    assert pm_ids_with_events == drift_pm_ids
    assert not (pm_ids_with_events & no_drift_pm_ids)


def test_typical_and_anti_typical_persona_counts_are_equal(fixture_catalogue: Catalogue):
    config = _config()
    result = sample_all(config, fixture_catalogue)
    typical = sum(1 for p in result.personas if p.typicality == Typicality.TYPICAL)
    anti_typical = sum(1 for p in result.personas if p.typicality == Typicality.ANTI_TYPICAL)
    assert typical == anti_typical


def test_two_calls_with_the_same_config_are_equal(fixture_catalogue: Catalogue):
    config = _config()
    first = sample_all(config, fixture_catalogue)
    second = sample_all(config, fixture_catalogue)
    assert first == second


def test_raising_full_per_cell_leaves_existing_rows_unchanged(fixture_catalogue: Catalogue):
    small = _config()
    grown = _config(population={"full_per_cell": 2})
    small_result = sample_all(small, fixture_catalogue)
    grown_result = sample_all(grown, fixture_catalogue)

    small_pm_ids = {p.pm_id for p in small_result.personas}
    grown_personas_by_id = {p.pm_id: p for p in grown_result.personas}
    for persona in small_result.personas:
        assert grown_personas_by_id[persona.pm_id] == persona

    def _by_pm_id(rows):
        grouped: dict[str, list] = {}
        for row in rows:
            grouped.setdefault(row.pm_id, []).append(row)
        return grouped

    small_traits, grown_traits = _by_pm_id(small_result.traits), _by_pm_id(grown_result.traits)
    small_rules, grown_rules = _by_pm_id(small_result.rules), _by_pm_id(grown_result.rules)
    small_drift, grown_drift = (
        _by_pm_id(small_result.drift_events),
        _by_pm_id(grown_result.drift_events),
    )
    for pm_id in small_pm_ids:
        assert grown_traits[pm_id] == small_traits[pm_id]
        assert grown_rules[pm_id] == small_rules[pm_id]
        assert grown_drift.get(pm_id, []) == small_drift.get(pm_id, [])


def test_changing_root_seed_changes_trait_values(fixture_catalogue: Catalogue):
    config_a = _config()
    config_b = _config(seed={"root": config_a.seed.root + 1})
    result_a = sample_all(config_a, fixture_catalogue)
    result_b = sample_all(config_b, fixture_catalogue)
    values_a = [t.value for t in result_a.traits]
    values_b = [t.value for t in result_b.traits]
    assert values_a != values_b


# --- check_sampling_config ---


def test_valid_default_config_passes(fixture_catalogue: Catalogue):
    check_sampling_config(Config(), fixture_catalogue)


def test_preferences_n_min_below_group_count_raises_naming_the_path(
    fixture_catalogue: Catalogue,
):
    config = Config.model_validate({"preferences": {"n_min": 2, "n_max": 8}})
    with pytest.raises(ConfigError, match="preferences.n_min"):
        check_sampling_config(config, fixture_catalogue)


def test_biases_min_active_below_two_raises_naming_the_path(fixture_catalogue: Catalogue):
    config = Config.model_validate({"biases": {"min_active": 1}})
    with pytest.raises(ConfigError, match="biases.min_active"):
        check_sampling_config(config, fixture_catalogue)


def test_rules_n_self_rules_max_too_small_raises_naming_the_path(fixture_catalogue: Catalogue):
    config = Config.model_validate({"rules": {"n_self_rules_min": 0, "n_self_rules_max": 0}})
    with pytest.raises(ConfigError, match="rules.n_self_rules_max"):
        check_sampling_config(config, fixture_catalogue)


def test_rules_n_self_rules_min_above_entry_count_raises_naming_the_path(
    fixture_catalogue: Catalogue,
):
    n_entries = len(fixture_catalogue.rules.entries)
    config = Config.model_validate(
        {"rules": {"n_self_rules_min": n_entries + 1, "n_self_rules_max": n_entries + 1}}
    )
    with pytest.raises(ConfigError, match="rules.n_self_rules_min"):
        check_sampling_config(config, fixture_catalogue)


def test_sample_all_runs_check_sampling_config_first(fixture_catalogue: Catalogue):
    config = _config(biases={"min_active": 1})
    with pytest.raises(ConfigError, match="biases.min_active"):
        sample_all(config, fixture_catalogue)
