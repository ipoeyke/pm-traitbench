"""Tests for a PM's ground-truth traits as of a date."""

from datetime import date, timedelta

import pytest

from pm_traitbench.catalogues.models import PreferenceEntry, PreferenceGroup
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.enums import AssetClass, DriftEventType, Kind
from pm_traitbench.errors import CorpusError
from pm_traitbench.tables.schema import DriftEvent
from pm_traitbench.traits_truth import (
    bias_active_at,
    bias_value_at,
    compute_truth,
    latest_update,
    preference_value_at,
)
from tests.gates.gate2.fixtures import PM_A, trait

_ENTRIES = (
    PreferenceEntry(
        param="response_format",
        group=PreferenceGroup.COMMUNICATION,
        asset_classes=(AssetClass.EQUITIES,),
        values=("short bullets", "a table with columns"),
    ),
    PreferenceEntry(
        param="register",
        group=PreferenceGroup.COMMUNICATION,
        asset_classes=(AssetClass.EQUITIES,),
        values=("blunt", "formal"),
    ),
)
_START = date(2026, 1, 5)


def _week(n: int) -> date:
    return _START + timedelta(weeks=n)


def _event(
    trait_id: str, event: DriftEventType, when: date, from_value=None, to_value=None
) -> DriftEvent:
    return DriftEvent(
        pm_id=PM_A,
        date=when,
        event=event,
        trait_id=trait_id,
        from_value=from_value,
        to_value=to_value,
    )


def test_compute_truth_drift_rules() -> None:
    last_date = date(2026, 6, 1)
    updated = trait(PM_A, "t_01", BIAS_PARAMS[0], Kind.BIAS, 1.0)
    dormant_no_revive = trait(PM_A, "t_02", BIAS_PARAMS[1], Kind.BIAS, 1.0)
    dormant_revived = trait(PM_A, "t_03", BIAS_PARAMS[2], Kind.BIAS, 1.0)
    other_biases = [
        trait(PM_A, f"t_{i:02d}", param, Kind.BIAS, 1.0)
        for i, param in enumerate(BIAS_PARAMS[3:], start=4)
    ]
    held_pref = trait(PM_A, "t_09", "response_format", Kind.PREFERENCE, "short bullets")
    traits = [updated, dormant_no_revive, dormant_revived, *other_biases, held_pref]

    drift_events = [
        _event("t_01", DriftEventType.UPDATE, date(2026, 2, 1), 1.0, 1.5),
        _event("t_02", DriftEventType.DORMANT, date(2026, 2, 1)),
        _event("t_03", DriftEventType.DORMANT, date(2026, 2, 1)),
        _event("t_03", DriftEventType.REVIVE, date(2026, 3, 1)),
        _event(
            "t_09", DriftEventType.UPDATE, date(2026, 2, 1), "short bullets", "a table with columns"
        ),
    ]

    truth = compute_truth(traits, drift_events, last_date, _ENTRIES)

    assert truth[BIAS_PARAMS[0]].truth_active is True
    assert truth[BIAS_PARAMS[1]].truth_active is False
    assert truth[BIAS_PARAMS[2]].truth_active is True
    assert truth["response_format"].truth_value == "a table with columns"
    assert truth["response_format"].trait_id == held_pref.trait_id
    assert truth["register"].truth_value is None
    assert truth["register"].trait_id is None

    with pytest.raises(CorpusError):
        compute_truth(traits[:-2], drift_events, last_date, _ENTRIES)


def test_compute_truth_revive_without_dormant_leaves_an_inactive_bias_inactive() -> None:
    inactive = trait(PM_A, "t_01", BIAS_PARAMS[0], Kind.BIAS, 1.0, active=False)
    others = [
        trait(PM_A, f"t_{i:02d}", param, Kind.BIAS, 1.0)
        for i, param in enumerate(BIAS_PARAMS[1:], start=2)
    ]
    revive = _event("t_01", DriftEventType.REVIVE, date(2026, 2, 1))

    truth = compute_truth([inactive, *others], [revive], date(2026, 6, 1), ())

    assert truth[BIAS_PARAMS[0]].truth_active is False


def test_latest_update_picks_the_latest_on_or_before_the_day() -> None:
    first = _event("t_01", DriftEventType.UPDATE, _week(10), 1.0, 1.5)
    second = _event("t_01", DriftEventType.UPDATE, _week(20), 1.5, 2.0)
    other = _event("t_02", DriftEventType.UPDATE, _week(12), 1.0, 3.0)
    events = [second, other, first]

    assert latest_update("t_01", events, _week(15)) == first
    assert latest_update("t_01", events, _week(25)) == second
    assert latest_update("t_01", events, _week(5)) is None


def test_bias_value_at_follows_updates_and_ignores_dormancy() -> None:
    bias = trait(PM_A, "t_01", BIAS_PARAMS[0], Kind.BIAS, 2.0)
    events = [
        _event("t_01", DriftEventType.UPDATE, _week(20), 2.0, 1.5),
        _event("t_01", DriftEventType.DORMANT, _week(25)),
    ]

    assert bias_value_at(bias, events, _week(10)) == 2.0
    assert bias_value_at(bias, events, _week(30)) == 1.5
    assert bias_active_at(bias, events, _week(30)) is False


def test_preference_value_at_returns_the_update_to_value() -> None:
    pref = trait(PM_A, "t_09", "response_format", Kind.PREFERENCE, "short bullets")
    events = [
        _event("t_09", DriftEventType.UPDATE, _week(20), "short bullets", "a table with columns")
    ]

    assert preference_value_at(pref, events, _week(10)) == "short bullets"
    assert preference_value_at(pref, events, _week(30)) == "a table with columns"
