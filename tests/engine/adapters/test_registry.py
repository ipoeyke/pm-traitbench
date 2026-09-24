"""Tests for the adapter registry: adapter_for, FORM_FOR_PREFERENCE, preferred_form."""

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import PreferenceGroup
from pm_traitbench.engine.adapters import FORM_FOR_PREFERENCE, adapter_for, preferred_form
from pm_traitbench.engine.adapters.equities import EquitiesAdapter
from pm_traitbench.enums import AssetClass, Expression, Kind
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Trait


def _preference_trait(param: str, value: str) -> Trait:
    return Trait(
        pm_id="pm_001",
        trait_id="t_01",
        kind=Kind.PREFERENCE,
        param=param,
        value=value,
        active=True,
        mult_range=None,
        mult_risk_off=None,
        mult_risk_on=None,
    )


def test_adapter_for_multi_asset_raises() -> None:
    with pytest.raises(EngineError):
        adapter_for(AssetClass.MULTI_ASSET, "global_macro")


def test_adapter_for_unregistered_class_raises() -> None:
    with pytest.raises(EngineError):
        adapter_for(AssetClass.RATES_CREDIT, "sovereign_rates")


def test_adapter_for_equities_accepts_any_sub_style() -> None:
    assert isinstance(adapter_for(AssetClass.EQUITIES, "value"), EquitiesAdapter)
    assert isinstance(adapter_for(AssetClass.EQUITIES, "made_up_sub_style"), EquitiesAdapter)


def test_form_for_preference_covers_every_catalogue_expression_value() -> None:
    catalogue = load_catalogue()
    for entry in catalogue.preferences:
        if entry.group != PreferenceGroup.EXPRESSION:
            continue
        for value in entry.values:
            assert (entry.param, value) in FORM_FOR_PREFERENCE


def test_preferred_form_returns_first_mapped_preference() -> None:
    traits = [_preference_trait("duration_expression", "steepeners over outright duration")]
    assert preferred_form(traits) == Expression.CURVE


def test_preferred_form_returns_none_when_nothing_maps() -> None:
    traits = [_preference_trait("hedge_instrument", "hedge with index futures")]
    assert preferred_form(traits) is None


def test_preferred_form_scans_traits_in_order() -> None:
    unmapped = _preference_trait(
        "pair_vs_outright", "express the view as a basket versus the index"
    )
    mapped = _preference_trait("pair_vs_outright", "express the view as a pair trade")
    assert preferred_form([unmapped, mapped]) == Expression.PAIR
