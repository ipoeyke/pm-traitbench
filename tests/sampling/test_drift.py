import pytest

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.enums import DriftEventType, Kind
from pm_traitbench.errors import SamplingError
from pm_traitbench.rng import stream
from pm_traitbench.sampling.drift import sample_drift
from pm_traitbench.tables.schema import Trait
from pm_traitbench.timeline import Timeline

_N = 300


def _bias_trait(index: int, param: str, value: float, active: bool = True) -> Trait:
    return Trait(
        pm_id="pm_001",
        trait_id=f"t_{index:02d}",
        kind=Kind.BIAS,
        param=param,
        value=value,
        active=active,
        mult_range=1.0,
        mult_risk_off=1.0,
        mult_risk_on=1.0,
    )


def _pref_trait(index: int, param: str, value: str) -> Trait:
    return Trait(
        pm_id="pm_001",
        trait_id=f"t_{index:02d}",
        kind=Kind.PREFERENCE,
        param=param,
        value=value,
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )


def _all_active_bias_traits(config: Config) -> list[Trait]:
    return [
        _bias_trait(i + 1, param, config.biases.params[param].active.median_value())
        for i, param in enumerate(BIAS_PARAMS)
    ]


def _full_traits(config: Config, catalogue: Catalogue) -> list[Trait]:
    bias_traits = _all_active_bias_traits(config)
    entries = catalogue.preferences[:4]
    pref_traits = [
        _pref_trait(len(bias_traits) + i + 1, entry.param, entry.values[0])
        for i, entry in enumerate(entries)
    ]
    return bias_traits + pref_traits


def _draw(config: Config, catalogue: Catalogue, timeline: Timeline, traits: list[Trait], i: int):
    rng = stream(1, "t", i, "drift")
    return sample_drift("pm_001", traits, config, catalogue, timeline, rng)


def test_exactly_one_bias_update_event_always(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    traits = _full_traits(config, catalogue)
    bias_trait_ids = {t.trait_id for t in traits if t.kind == Kind.BIAS}
    bias_update_dates = timeline.weekdays_in_weeks(*config.drift.bias_update_weeks)
    for i in range(_N):
        events = _draw(config, catalogue, timeline, traits, i)
        updates = [
            e for e in events if e.event == DriftEventType.UPDATE and e.trait_id in bias_trait_ids
        ]
        assert len(updates) == 1
        assert updates[0].date in bias_update_dates


@pytest.mark.parametrize("param", BIAS_PARAMS)
def test_bias_update_moves_the_value_toward_neutral_by_the_configured_fraction(
    param: str, catalogue: Catalogue
):
    config = Config()
    timeline = config.timeline()
    spec = config.biases.params[param]
    neutral = spec.neutral.median_value()
    from_value = spec.active.median_value()
    traits = [_bias_trait(1, param, from_value)]
    low, high = sorted((neutral, from_value))
    remaining_lo, remaining_hi = config.drift.bias_update_remaining
    for i in range(50):
        events = _draw(config, catalogue, timeline, traits, i)
        update = next(e for e in events if e.trait_id == "t_01")
        assert update.from_value == from_value
        to_value = update.to_value
        assert low < to_value < high
        ratio = abs(to_value - neutral) / abs(from_value - neutral)
        assert remaining_lo - 1e-4 <= ratio <= remaining_hi + 1e-4


def test_preference_update_present_in_roughly_half_of_seeds(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    traits = _full_traits(config, catalogue)
    by_id = {t.trait_id: t for t in traits}
    pref_trait_ids = {t.trait_id for t in traits if t.kind == Kind.PREFERENCE}
    by_param = {entry.param: entry for entry in catalogue.preferences}
    pref_update_dates = timeline.weekdays_in_weeks(*config.drift.preference_update_weeks)

    seen = 0
    for i in range(_N):
        events = _draw(config, catalogue, timeline, traits, i)
        pref_updates = [e for e in events if e.trait_id in pref_trait_ids]
        assert len(pref_updates) <= 1
        if not pref_updates:
            continue
        seen += 1
        event = pref_updates[0]
        assert event.event == DriftEventType.UPDATE
        assert event.date in pref_update_dates
        assert event.to_value != event.from_value
        assert event.from_value == by_id[event.trait_id].value
        entry = by_param[by_id[event.trait_id].param]
        assert event.to_value in entry.values

    share = seen / _N
    assert 0.4 <= share <= 0.6


def test_dormant_and_revive_appear_together_on_the_same_other_trait(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    traits = _full_traits(config, catalogue)
    bias_trait_ids = {t.trait_id for t in traits if t.kind == Kind.BIAS}
    dormant_dates = timeline.weekdays_in_weeks(*config.drift.dormant_weeks)
    revive_dates = timeline.weekdays_in_weeks(*config.drift.revive_weeks)

    seen_dormant = False
    for i in range(_N):
        events = _draw(config, catalogue, timeline, traits, i)
        dormant = [e for e in events if e.event == DriftEventType.DORMANT]
        revive = [e for e in events if e.event == DriftEventType.REVIVE]
        assert len(dormant) == len(revive)
        assert len(dormant) <= 1
        if not dormant:
            continue
        seen_dormant = True
        update = next(
            e for e in events if e.event == DriftEventType.UPDATE and e.trait_id in bias_trait_ids
        )
        assert dormant[0].trait_id == revive[0].trait_id
        assert dormant[0].trait_id in bias_trait_ids
        assert dormant[0].trait_id != update.trait_id
        assert dormant[0].from_value is None
        assert dormant[0].to_value is None
        assert revive[0].from_value is None
        assert revive[0].to_value is None
        assert dormant[0].date in dormant_dates
        assert revive[0].date in revive_dates

    assert seen_dormant


def test_single_active_bias_never_produces_a_dormant_event(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    param = BIAS_PARAMS[0]
    traits = [_bias_trait(1, param, config.biases.params[param].active.median_value())]
    for i in range(_N):
        events = _draw(config, catalogue, timeline, traits, i)
        assert not any(e.event in (DriftEventType.DORMANT, DriftEventType.REVIVE) for e in events)


def test_no_preference_traits_never_produces_a_preference_update(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    traits = _all_active_bias_traits(config)
    bias_trait_ids = {t.trait_id for t in traits}
    for i in range(100):
        events = _draw(config, catalogue, timeline, traits, i)
        assert all(e.trait_id in bias_trait_ids for e in events)


def test_preference_with_unknown_catalogue_param_skips_the_branch(catalogue: Catalogue):
    config = Config.model_validate({"drift": {"p_preference_update": 1.0}})
    timeline = config.timeline()
    bias_traits = _all_active_bias_traits(config)
    pref_trait = _pref_trait(len(bias_traits) + 1, "not_a_real_param", "only value")
    traits = bias_traits + [pref_trait]
    for i in range(50):
        events = _draw(config, catalogue, timeline, traits, i)
        assert all(e.trait_id != pref_trait.trait_id for e in events)


def test_preference_with_a_single_catalogue_value_skips_the_branch(catalogue: Catalogue):
    entry = catalogue.preferences[0]
    narrowed = entry.model_copy(update={"values": (entry.values[0],)})
    others = tuple(e for e in catalogue.preferences if e.param != entry.param)
    catalogue = catalogue.model_copy(update={"preferences": (narrowed, *others)})

    config = Config.model_validate({"drift": {"p_preference_update": 1.0}})
    timeline = config.timeline()
    bias_traits = _all_active_bias_traits(config)
    pref_trait = _pref_trait(len(bias_traits) + 1, entry.param, entry.values[0])
    traits = bias_traits + [pref_trait]
    for i in range(50):
        events = _draw(config, catalogue, timeline, traits, i)
        assert all(e.trait_id != pref_trait.trait_id for e in events)


def test_events_are_sorted_by_date_then_trait_id_then_event(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    traits = _full_traits(config, catalogue)
    for i in range(50):
        events = _draw(config, catalogue, timeline, traits, i)
        keys = [(e.date, e.trait_id, e.event.value) for e in events]
        assert keys == sorted(keys)


def test_deterministic_for_a_given_stream(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    traits = _full_traits(config, catalogue)
    first = _draw(config, catalogue, timeline, traits, 5)
    second = _draw(config, catalogue, timeline, traits, 5)
    assert first == second


def test_mixed_active_and_inactive_biases_only_touch_active_trait_ids(
    catalogue: Catalogue,
):
    config = Config()
    timeline = config.timeline()
    # Half the bias traits are inactive; every update/dormant/revive
    # trait_id must belong to the active subset, never the inactive one.
    traits = [
        _bias_trait(i + 1, param, config.biases.params[param].active.median_value(), active=i % 2)
        for i, param in enumerate(BIAS_PARAMS)
    ]
    active_ids = {t.trait_id for t in traits if t.active}
    inactive_ids = {t.trait_id for t in traits if not t.active}
    seen_dormant = False
    for i in range(_N):
        events = _draw(config, catalogue, timeline, traits, i)
        for event in events:
            assert event.trait_id in active_ids
            assert event.trait_id not in inactive_ids
            seen_dormant = seen_dormant or event.event == DriftEventType.DORMANT
    assert seen_dormant


def test_no_active_bias_trait_raises_sampling_error(catalogue: Catalogue):
    config = Config()
    timeline = config.timeline()
    param = BIAS_PARAMS[0]
    neutral = config.biases.params[param].neutral.median_value()
    traits = [_bias_trait(1, param, neutral, active=False)]
    rng = stream(1, "t", 0, "drift")
    with pytest.raises(SamplingError):
        sample_drift("pm_001", traits, config, catalogue, timeline, rng)
