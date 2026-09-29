"""Tests for gate 2's recovery request, reply parser, ground truth and scorer."""

import asyncio
import json
from datetime import date

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import BiasDefinitions, PreferenceEntry, PreferenceGroup
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.dialogue.client import CachedClient
from pm_traitbench.enums import AssetClass, DriftEventType, Kind, Ownership
from pm_traitbench.errors import Gate2Error
from pm_traitbench.gates.gate2.recover import (
    BiasAnswer,
    PreferenceAnswer,
    RecoveryReply,
    TraitTruth,
    compute_truth,
    parse_recovery,
    pre_update_signal_ids,
    recovery_request,
    send_recovery,
    signal_rows,
    trait_rows,
)
from pm_traitbench.signals.assemble import session_id as build_session_id
from pm_traitbench.tables.schema import DriftEvent, Gate2TraitRow
from tests.dialogue.fixtures import FakeClient, fake_message, rule
from tests.gates.gate2.fixtures import PM_A, recovery_reply, signal, trait
from tests.signals.fixtures import persona as build_persona

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


def _good_biases() -> dict[str, tuple[bool, list[str]]]:
    return {param: (param == BIAS_PARAMS[0], []) for param in BIAS_PARAMS}


def test_recovery_request_carries_vocabulary_and_never_trait_data() -> None:
    catalogue = load_catalogue()
    # A synthetic definitions map, not the packaged catalogue: one packaged line uses the
    # ordinary English word "stated", which the leak check below must not trip on.
    definitions = BiasDefinitions(
        definitions={param: f"shows a documented tendency toward {param}" for param in BIAS_PARAMS}
    )
    entries = catalogue.preferences_for(AssetClass.EQUITIES)
    persona = build_persona(AssetClass.EQUITIES)
    pm_rules = [rule(rule_id="r_01", text="Exit after a 5 percent drawdown.")]
    config = Config().gate2

    request = recovery_request(
        persona, pm_rules, definitions, entries, "a plain transcript", config
    )

    assert set(request.keys()) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert request["output_config"]["effort"] == "high"

    system = request["system"]
    for param in BIAS_PARAMS:
        assert param in system
        assert definitions.definitions[param] in system
    for entry in entries:
        for value in entry.values:
            assert value in system
    assert persona.mandate.asset_class.value in system
    assert persona.mandate.sub_style in system
    assert persona.mandate.benchmark in system
    for pm_rule in pm_rules:
        assert pm_rule.text in system

    dump = json.dumps(request)
    assert "t_01" not in dump
    assert "2.7182818" not in dump
    assert "push to run this at twice the usual size" not in dump
    assert "stated" not in dump.lower()
    assert "revealed" not in dump.lower()


def test_parse_recovery_rejects_missing_duplicate_unknown_params_and_off_catalogue_values() -> None:
    good_preferences = {
        "response_format": (" Short Bullets ", []),
        "register": (None, []),
    }
    good = recovery_reply(_good_biases(), good_preferences)
    parsed = parse_recovery(good, _ENTRIES)
    assert parsed is not None
    resolved = next(a for a in parsed.preferences if a.param == "response_format")
    assert resolved.value == "short bullets"

    missing_bias = recovery_reply(
        {param: (False, []) for param in BIAS_PARAMS[:-1]}, good_preferences
    )
    assert parse_recovery(missing_bias, _ENTRIES) is None

    duplicate_payload = json.dumps(
        {
            "biases": [
                {"param": param, "present": False, "session_ids": []}
                for param in (*BIAS_PARAMS, BIAS_PARAMS[0])
            ],
            "preferences": [
                {"param": param, "value": None, "session_ids": []}
                for param in ("response_format", "register")
            ],
        }
    )
    duplicate_bias = fake_message([{"type": "text", "text": duplicate_payload}])
    assert parse_recovery(duplicate_bias, _ENTRIES) is None

    unknown_param = recovery_reply(
        _good_biases(), {"response_format": ("short bullets", []), "unknown_param": (None, [])}
    )
    assert parse_recovery(unknown_param, _ENTRIES) is None

    off_catalogue = recovery_reply(
        _good_biases(), {"response_format": ("loud and blunt", []), "register": (None, [])}
    )
    assert parse_recovery(off_catalogue, _ENTRIES) is None


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
        DriftEvent(
            pm_id=PM_A,
            date=date(2026, 2, 1),
            event=DriftEventType.UPDATE,
            trait_id=updated.trait_id,
            from_value=1.0,
            to_value=1.5,
        ),
        DriftEvent(
            pm_id=PM_A,
            date=date(2026, 2, 1),
            event=DriftEventType.DORMANT,
            trait_id=dormant_no_revive.trait_id,
            from_value=None,
            to_value=None,
        ),
        DriftEvent(
            pm_id=PM_A,
            date=date(2026, 2, 1),
            event=DriftEventType.DORMANT,
            trait_id=dormant_revived.trait_id,
            from_value=None,
            to_value=None,
        ),
        DriftEvent(
            pm_id=PM_A,
            date=date(2026, 3, 1),
            event=DriftEventType.REVIVE,
            trait_id=dormant_revived.trait_id,
            from_value=None,
            to_value=None,
        ),
        DriftEvent(
            pm_id=PM_A,
            date=date(2026, 2, 1),
            event=DriftEventType.UPDATE,
            trait_id=held_pref.trait_id,
            from_value="short bullets",
            to_value="a table with columns",
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

    with pytest.raises(Gate2Error):
        compute_truth(traits[:-2], drift_events, last_date, _ENTRIES)


def test_compute_truth_revive_without_dormant_leaves_an_inactive_bias_inactive() -> None:
    inactive = trait(PM_A, "t_01", BIAS_PARAMS[0], Kind.BIAS, 1.0, active=False)
    others = [
        trait(PM_A, f"t_{i:02d}", param, Kind.BIAS, 1.0)
        for i, param in enumerate(BIAS_PARAMS[1:], start=2)
    ]
    revive = DriftEvent(
        pm_id=PM_A,
        date=date(2026, 2, 1),
        event=DriftEventType.REVIVE,
        trait_id=inactive.trait_id,
        from_value=None,
        to_value=None,
    )

    truth = compute_truth([inactive, *others], [revive], date(2026, 6, 1), ())

    assert truth[BIAS_PARAMS[0]].truth_active is False


def test_trait_rows_scores_and_verifies_citations() -> None:
    session_known = build_session_id(PM_A, date(2026, 1, 5), 0)
    session_unknown = build_session_id(PM_A, date(2026, 1, 6), 0)

    truth = {
        "loss_aversion_lambda": TraitTruth(
            kind=Kind.BIAS, trait_id="t_01", truth_active=True, truth_value=None
        ),
        "response_format": TraitTruth(
            kind=Kind.PREFERENCE, trait_id=None, truth_active=None, truth_value=None
        ),
    }
    reply = RecoveryReply(
        biases=(
            BiasAnswer(
                param="loss_aversion_lambda",
                present=True,
                session_ids=(session_known, session_unknown),
            ),
        ),
        preferences=(PreferenceAnswer(param="response_format", value=None, session_ids=()),),
    )
    traits = [trait(PM_A, "t_01", "loss_aversion_lambda", Kind.BIAS, 1.0)]
    signals = [
        signal(
            PM_A,
            session_known,
            date(2026, 1, 5),
            "t_01",
            ownership=Ownership.COLLEAGUE,
            signal_id="sg_101",
        )
    ]

    rows, warnings = trait_rows(PM_A, reply, truth, signals, traits, {session_known})

    bias_row = next(r for r in rows if r.param == "loss_aversion_lambda")
    assert bias_row.correct is True
    assert bias_row.cited_session_ids == (session_known,)
    assert bias_row.false_attribution_ids == (session_known,)
    assert warnings == (
        f"pm {PM_A}: recovery cited unknown session '{session_unknown}' for loss_aversion_lambda",
    )

    pref_row = next(r for r in rows if r.param == "response_format")
    assert pref_row.predicted_value is None
    assert pref_row.truth_value is None
    assert pref_row.correct is True


def test_signal_rows_join_citations_and_classification() -> None:
    session_a = build_session_id(PM_A, date(2026, 1, 5), 0)
    session_b = build_session_id(PM_A, date(2026, 1, 6), 0)
    session_c = build_session_id(PM_A, date(2026, 1, 7), 0)
    session_dropped = build_session_id(PM_A, date(2026, 1, 8), 0)

    bias_trait = trait(PM_A, "t_01", "loss_aversion_lambda", Kind.BIAS, 1.0)
    pref_trait = trait(PM_A, "t_02", "response_format", Kind.PREFERENCE, "short bullets")
    traits = [bias_trait, pref_trait]

    trait_row_bias = Gate2TraitRow(
        pm_id=PM_A,
        param="loss_aversion_lambda",
        trait_id="t_01",
        kind=Kind.BIAS,
        truth_active=True,
        truth_value=None,
        predicted_active=True,
        predicted_value=None,
        correct=True,
        cited_session_ids=(session_a,),
        false_attribution_ids=(),
    )
    trait_row_pref = Gate2TraitRow(
        pm_id=PM_A,
        param="response_format",
        trait_id=pref_trait.trait_id,
        kind=Kind.PREFERENCE,
        truth_active=None,
        truth_value="short bullets",
        predicted_active=None,
        predicted_value="a table with columns",
        correct=False,
        cited_session_ids=(session_b,),
        false_attribution_ids=(),
    )

    sig_recovered = signal(PM_A, session_a, date(2026, 1, 5), "t_01", signal_id="sg_001")
    sig_dropped = signal(PM_A, session_dropped, date(2026, 1, 8), "t_01", signal_id="sg_002")
    sig_cited_wrong = signal(PM_A, session_b, date(2026, 1, 6), "t_02", signal_id="sg_003")
    sig_uncited = signal(PM_A, session_c, date(2026, 1, 7), "t_02", signal_id="sg_004")
    signals = [sig_recovered, sig_dropped, sig_cited_wrong, sig_uncited]

    pre_update = frozenset({sig_cited_wrong.signal_id})
    classification = {
        sig_recovered.signal_id: (Kind.BIAS, True),
        sig_uncited.signal_id: (Kind.BIAS, False),
    }
    session_ids = {session_a, session_b, session_c}

    rows = signal_rows(
        PM_A,
        signals,
        traits,
        [trait_row_bias, trait_row_pref],
        pre_update,
        classification,
        session_ids,
    )

    ids = {row.signal_id for row in rows}
    assert sig_dropped.signal_id not in ids
    assert len(rows) == 3

    row_recovered = next(r for r in rows if r.signal_id == sig_recovered.signal_id)
    assert row_recovered.cited is True
    assert row_recovered.recovered is True
    assert row_recovered.classified is True
    assert row_recovered.kind_predicted == Kind.BIAS
    assert row_recovered.kind_ok is True

    row_cited_wrong = next(r for r in rows if r.signal_id == sig_cited_wrong.signal_id)
    assert row_cited_wrong.cited is True
    assert row_cited_wrong.recovered is False
    assert row_cited_wrong.pre_update is True
    assert row_cited_wrong.classified is False
    assert row_cited_wrong.kind_predicted is None
    assert row_cited_wrong.kind_ok is None

    row_uncited = next(r for r in rows if r.signal_id == sig_uncited.signal_id)
    assert row_uncited.cited is False
    assert row_uncited.recovered is False
    assert row_uncited.classified is True
    assert row_uncited.kind_predicted == Kind.BIAS
    assert row_uncited.kind_ok is False


def test_pre_update_signal_ids_only_flags_earlier_preference_signals() -> None:
    session_before = build_session_id(PM_A, date(2026, 1, 1), 0)
    session_after = build_session_id(PM_A, date(2026, 3, 1), 0)
    pref_trait = trait(PM_A, "t_02", "response_format", Kind.PREFERENCE, "short bullets")
    bias_trait = trait(PM_A, "t_01", "loss_aversion_lambda", Kind.BIAS, 1.0)

    before = signal(PM_A, session_before, date(2026, 1, 1), "t_02", signal_id="sg_010")
    after = signal(PM_A, session_after, date(2026, 3, 1), "t_02", signal_id="sg_011")
    bias_signal = signal(PM_A, session_before, date(2026, 1, 1), "t_01", signal_id="sg_012")

    drift_events = [
        DriftEvent(
            pm_id=PM_A,
            date=date(2026, 2, 1),
            event=DriftEventType.UPDATE,
            trait_id="t_02",
            from_value="short bullets",
            to_value="a table with columns",
        )
    ]

    ids = pre_update_signal_ids(
        [before, after, bias_signal], [pref_trait, bias_trait], drift_events
    )

    assert ids == frozenset({before.signal_id})


def test_send_recovery_labels_failure_by_pm(tmp_path) -> None:
    unparsable = fake_message([{"type": "text", "text": "not json"}])
    fake = FakeClient(lambda _request: unparsable)
    client = CachedClient(lambda: fake, tmp_path, token_budget=None)
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(Gate2Error, match=r"^pm pm_001: "):
        asyncio.run(send_recovery(client, request, (), "pm_001", max_retries=0))
