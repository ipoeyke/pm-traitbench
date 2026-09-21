import pytest

from pm_traitbench.catalogues.models import Catalogue, PreferenceGroup
from pm_traitbench.config import Config
from pm_traitbench.enums import AssetClass
from pm_traitbench.errors import SamplingError
from pm_traitbench.rng import stream
from pm_traitbench.sampling.preferences import sample_preferences


def _draw(config: Config, catalogue: Catalogue, asset_class: AssetClass, i: int):
    rng = stream(1, "t", i, "preferences")
    return sample_preferences(asset_class, config, catalogue, rng)


def test_count_is_within_configured_bounds(fixture_catalogue: Catalogue):
    config = Config()
    for i in range(200):
        draws = _draw(config, fixture_catalogue, AssetClass.EQUITIES, i)
        assert config.preferences.n_min <= len(draws) <= config.preferences.n_max


def test_every_group_is_represented(fixture_catalogue: Catalogue):
    config = Config()
    by_param = {entry.param: entry for entry in fixture_catalogue.preferences}
    for i in range(200):
        draws = _draw(config, fixture_catalogue, AssetClass.EQUITIES, i)
        groups = {by_param[d.param].group for d in draws}
        assert groups == set(PreferenceGroup)


def test_params_are_unique(fixture_catalogue: Catalogue):
    config = Config()
    for i in range(200):
        draws = _draw(config, fixture_catalogue, AssetClass.EQUITIES, i)
        params = [d.param for d in draws]
        assert len(set(params)) == len(params)


def test_every_param_is_applicable_to_the_asset_class(fixture_catalogue: Catalogue):
    config = Config()
    for asset_class in AssetClass:
        applicable = {entry.param for entry in fixture_catalogue.preferences_for(asset_class)}
        for i in range(50):
            draws = _draw(config, fixture_catalogue, asset_class, i)
            assert all(d.param in applicable for d in draws)


def test_values_belong_to_their_entry(fixture_catalogue: Catalogue):
    config = Config()
    by_param = {entry.param: entry for entry in fixture_catalogue.preferences}
    for i in range(200):
        draws = _draw(config, fixture_catalogue, AssetClass.EQUITIES, i)
        for d in draws:
            assert d.value in by_param[d.param].values


def test_order_follows_catalogue_order(fixture_catalogue: Catalogue):
    config = Config()
    catalogue_order = [entry.param for entry in fixture_catalogue.preferences]
    for i in range(200):
        draws = _draw(config, fixture_catalogue, AssetClass.EQUITIES, i)
        indices = [catalogue_order.index(d.param) for d in draws]
        assert indices == sorted(indices)


def test_deterministic_for_a_given_stream(fixture_catalogue: Catalogue):
    config = Config()
    first = _draw(config, fixture_catalogue, AssetClass.EQUITIES, 7)
    second = _draw(config, fixture_catalogue, AssetClass.EQUITIES, 7)
    assert first == second


def test_over_many_draws_every_count_occurs(fixture_catalogue: Catalogue):
    config = Config()
    counts = set()
    for i in range(300):
        draws = _draw(config, fixture_catalogue, AssetClass.EQUITIES, i)
        counts.add(len(draws))
    assert counts == set(range(config.preferences.n_min, config.preferences.n_max + 1))


def test_drawn_count_below_group_count_raises_sampling_error(fixture_catalogue: Catalogue):
    # n_min/n_max are set below the number of preference groups directly on
    # the sampler, bypassing check_sampling_config, to exercise the sampler's
    # own defensive guard.
    config = Config.model_validate({"preferences": {"n_min": 1, "n_max": 1}})
    rng = stream(1, "t", 0, "preferences")
    with pytest.raises(SamplingError):
        sample_preferences(AssetClass.EQUITIES, config, fixture_catalogue, rng)


def test_catalogue_cut_below_n_raises_sampling_error(fixture_catalogue: Catalogue):
    one_per_group: list = []
    seen_groups = set()
    for entry in fixture_catalogue.preferences:
        if entry.group not in seen_groups:
            one_per_group.append(entry)
            seen_groups.add(entry.group)
    small_catalogue = fixture_catalogue.model_copy(update={"preferences": tuple(one_per_group)})
    config = Config.model_validate({"preferences": {"n_min": 5, "n_max": 5}})
    rng = stream(1, "t", 0, "preferences")
    try:
        sample_preferences(AssetClass.EQUITIES, config, small_catalogue, rng)
    except SamplingError:
        pass
    else:
        raise AssertionError("expected SamplingError")
