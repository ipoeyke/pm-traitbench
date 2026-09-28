"""Gate 2 recovery: the per-PM request, its reply parser, ground truth and the scorer that
turns a parsed reply into trait and signal rows.

The request never carries a trait id, a bias trait's numeric value, a stance line, a
signal mode or which sessions carry a signal: only the bias vocabulary, the candidate
preference params and their catalogue values, the mandate and the PM-scope rule texts.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pm_traitbench.catalogues.models import BiasDefinitions, PreferenceEntry
from pm_traitbench.config import BIAS_PARAMS, Gate2Config
from pm_traitbench.dialogue.client import CachedClient, last_text_json, send_until_accepted
from pm_traitbench.dialogue.prompts import base_request
from pm_traitbench.enums import DriftEventType, Kind, Ownership, RuleScope, Valence
from pm_traitbench.errors import Gate2Error
from pm_traitbench.tables.schema import (
    DriftEvent,
    Gate2SignalRow,
    Gate2TraitRow,
    Mandate,
    Persona,
    Rule,
    Signal,
    Trait,
)

ROLE_LINE = (
    "You are reviewing a year of conversations between a portfolio manager and their "
    "advisor. Decide what this PM does and wants, using only the transcript."
)

_CITATION_INSTRUCTION = (
    "For every bias you mark present and every preference you mark held, list the "
    "session ids that show it. Most PMs show a minority of the eight biases."
)

RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "gate2_recovery",
    "properties": {
        "biases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                    "present": {"type": "boolean"},
                    "session_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["param", "present", "session_ids"],
                "additionalProperties": False,
            },
        },
        "preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                    "value": {"type": ["string", "null"]},
                    "session_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["param", "value", "session_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["biases", "preferences"],
    "additionalProperties": False,
}


class BiasAnswer(BaseModel):
    """One bias verdict: whether the param is present and the sessions cited for it."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    param: str
    present: bool
    session_ids: tuple[str, ...] = Field(strict=False)


class PreferenceAnswer(BaseModel):
    """One preference verdict: the held catalogue value, or null, and its citations."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    param: str
    value: str | None
    session_ids: tuple[str, ...] = Field(strict=False)


class RecoveryReply(BaseModel):
    """A recovery model's full reply: one verdict per bias param and per candidate preference."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    biases: tuple[BiasAnswer, ...] = Field(strict=False)
    preferences: tuple[PreferenceAnswer, ...] = Field(strict=False)


@dataclass(frozen=True)
class TraitTruth:
    """One param's ground truth: a bias's true activity, or a preference's true held value."""

    kind: Kind
    trait_id: str | None
    truth_active: bool | None
    truth_value: str | None


def _normalized(value: str) -> str:
    """Case- and whitespace-insensitive form used to match a reply value to a catalogue one."""
    return value.strip().lower()


def _mandate_line(mandate: Mandate) -> str:
    return (
        f"Mandate: asset class {mandate.asset_class.value}, sub-style {mandate.sub_style}, "
        f"book size {mandate.book_size}, risk unit {mandate.risk_unit}, "
        f"benchmark {mandate.benchmark}."
    )


def _rules_section(pm_rules: Sequence[Rule]) -> str:
    ordered = sorted(
        (rule for rule in pm_rules if rule.scope == RuleScope.PM), key=lambda rule: rule.rule_id
    )
    lines = "\n".join(rule.text for rule in ordered)
    return f"Rules the PM is held to:\n{lines}"


def _biases_section(definitions: BiasDefinitions) -> str:
    lines = "\n".join(f"- {param}: {definitions.definitions[param]}" for param in BIAS_PARAMS)
    return f'Tendencies to decide on, one entry each in "biases":\n{lines}'


def _preferences_section(entries: Sequence[PreferenceEntry]) -> str:
    lines = "\n".join(f"- {entry.param}: {'; '.join(entry.values)}" for entry in entries)
    return (
        'Preferences to decide on, one entry each in "preferences", value one of the listed '
        f"options or null when the transcript gives no evidence:\n{lines}"
    )


def recovery_request(
    persona: Persona,
    pm_rules: Sequence[Rule],
    definitions: BiasDefinitions,
    entries: Sequence[PreferenceEntry],
    transcript: str,
    config: Gate2Config,
) -> dict[str, Any]:
    """The recovery model's Messages API request body: only the allowed keys."""
    system = "\n\n".join(
        [
            ROLE_LINE,
            _mandate_line(persona.mandate),
            _rules_section(pm_rules),
            _biases_section(definitions),
            _preferences_section(entries),
            _CITATION_INSTRUCTION,
        ]
    )
    return base_request(
        config.model,
        config.recovery_max_output_tokens,
        config.effort,
        system,
        [{"role": "user", "content": transcript}],
        RECOVERY_SCHEMA,
    )


def parse_recovery(
    response: Mapping[str, Any], entries: Sequence[PreferenceEntry]
) -> RecoveryReply | None:
    """A validated `RecoveryReply`, or `None` for anything schema-invalid or off-vocabulary.

    Requires exactly one entry per bias param (no duplicates) and per `entries` param,
    and every non-null preference value to match a catalogue value once normalised; the
    returned reply carries the catalogue's own spelling of that value.
    """
    try:
        reply = RecoveryReply.model_validate(last_text_json(response))
    except ValidationError:
        return None

    bias_params = [answer.param for answer in reply.biases]
    if len(bias_params) != len(BIAS_PARAMS) or set(bias_params) != set(BIAS_PARAMS):
        return None

    entry_by_param = {entry.param: entry for entry in entries}
    pref_params = [answer.param for answer in reply.preferences]
    if len(pref_params) != len(entry_by_param) or set(pref_params) != set(entry_by_param):
        return None

    resolved: list[PreferenceAnswer] = []
    for answer in reply.preferences:
        if answer.value is None:
            resolved.append(answer)
            continue
        entry = entry_by_param[answer.param]
        target = _normalized(answer.value)
        match = next((v for v in entry.values if _normalized(v) == target), None)
        if match is None:
            return None
        resolved.append(answer.model_copy(update={"value": match}))
    return reply.model_copy(update={"preferences": tuple(resolved)})


async def send_recovery(
    client: CachedClient,
    request: Mapping[str, Any],
    entries: Sequence[PreferenceEntry],
    pm_id: str,
    max_retries: int,
) -> tuple[RecoveryReply, int]:
    """Send the recovery request, retrying an unparsable reply; raises `Gate2Error` at the cap."""

    def classify(response: Mapping[str, Any]) -> tuple[RecoveryReply | None, str]:
        reason = "the recovery reply was unparsable or schema-invalid"
        return parse_recovery(response, entries), reason

    _, parsed, rejected = await send_until_accepted(
        client,
        request,
        classify,
        scope=pm_id,
        max_retries=max_retries,
        error_type=Gate2Error,
        label="pm",
    )
    return parsed, rejected


def _bias_truth_active(trait: Trait, drift_events: Sequence[DriftEvent], last_date: date) -> bool:
    active = trait.active
    own_events = sorted(
        (e for e in drift_events if e.trait_id == trait.trait_id and e.date <= last_date),
        key=lambda e: e.date,
    )
    for event in own_events:
        if event.event == DriftEventType.DORMANT:
            active = False
        elif event.event == DriftEventType.REVIVE:
            active = True
    return active


def _preference_truth_value(
    trait: Trait, drift_events: Sequence[DriftEvent], last_date: date
) -> str:
    updates = [
        e
        for e in drift_events
        if e.trait_id == trait.trait_id and e.event == DriftEventType.UPDATE and e.date <= last_date
    ]
    if not updates:
        return trait.value
    latest = max(updates, key=lambda e: e.date)
    return latest.to_value


def compute_truth(
    traits: Sequence[Trait],
    drift_events: Sequence[DriftEvent],
    last_date: date,
    entries: Sequence[PreferenceEntry],
) -> dict[str, TraitTruth]:
    """The ground truth per candidate param: the eight biases, then `entries`' preferences.

    Raises `Gate2Error` when `traits` does not hold all eight bias params. A bias's
    truth stays whatever `active` was set to by the latest dormant-or-revive event on
    or before `last_date`; an update never changes it. A preference's truth is the
    trait's value, replaced by the latest update's `to` value on or before `last_date`;
    `None` when the PM does not hold the param.
    """
    bias_by_param = {trait.param: trait for trait in traits if trait.kind == Kind.BIAS}
    missing = set(BIAS_PARAMS) - set(bias_by_param)
    if missing:
        pm_id = traits[0].pm_id if traits else "?"
        raise Gate2Error(f"pm {pm_id}: missing bias trait(s) {sorted(missing)}")

    truth: dict[str, TraitTruth] = {}
    for param in BIAS_PARAMS:
        trait = bias_by_param[param]
        truth[param] = TraitTruth(
            kind=Kind.BIAS,
            trait_id=trait.trait_id,
            truth_active=_bias_truth_active(trait, drift_events, last_date),
            truth_value=None,
        )

    pref_by_param = {trait.param: trait for trait in traits if trait.kind == Kind.PREFERENCE}
    for entry in entries:
        held = pref_by_param.get(entry.param)
        truth_value = (
            _preference_truth_value(held, drift_events, last_date) if held is not None else None
        )
        truth[entry.param] = TraitTruth(
            kind=Kind.PREFERENCE, trait_id=None, truth_active=None, truth_value=truth_value
        )
    return truth


def trait_rows(
    pm_id: str,
    reply: RecoveryReply,
    truth: Mapping[str, TraitTruth],
    signals: Sequence[Signal],
    traits: Sequence[Trait],
    session_ids: Collection[str],
) -> tuple[tuple[Gate2TraitRow, ...], tuple[str, ...]]:
    """One `Gate2TraitRow` per candidate param, scored against `truth`, plus citation warnings.

    A cited session id outside `session_ids` is dropped and reported as a warning. A
    cited session carrying a retracted or non-self signal of the same param's trait is
    a false attribution.
    """
    trait_id_by_param = {trait.param: trait.trait_id for trait in traits}
    signals_by_session: dict[str, list[Signal]] = {}
    for signal in signals:
        signals_by_session.setdefault(signal.session_id, []).append(signal)

    bias_by_param = {answer.param: answer for answer in reply.biases}
    pref_by_param = {answer.param: answer for answer in reply.preferences}

    rows: list[Gate2TraitRow] = []
    warnings: list[str] = []
    for param, truth_row in truth.items():
        if truth_row.kind == Kind.BIAS:
            answer = bias_by_param[param]
            cited_raw: tuple[str, ...] = answer.session_ids
            predicted_active: bool | None = answer.present
            predicted_value: str | None = None
            correct = predicted_active == truth_row.truth_active
        else:
            answer = pref_by_param[param]
            cited_raw = answer.session_ids
            predicted_active = None
            predicted_value = answer.value
            correct = predicted_value == truth_row.truth_value

        cited_known = sorted({session_id for session_id in cited_raw if session_id in session_ids})
        for session_id in cited_raw:
            if session_id not in session_ids:
                warnings.append(
                    f"pm {pm_id}: recovery cited unknown session '{session_id}' for {param}"
                )

        param_trait_id = trait_id_by_param.get(param)
        false_attribution = sorted(
            session_id
            for session_id in cited_known
            if any(
                signal.trait_id == param_trait_id
                and (signal.valence == Valence.RETRACTED or signal.ownership != Ownership.SELF)
                for signal in signals_by_session.get(session_id, ())
            )
        )

        rows.append(
            Gate2TraitRow(
                pm_id=pm_id,
                param=param,
                trait_id=truth_row.trait_id,
                kind=truth_row.kind,
                truth_active=truth_row.truth_active,
                truth_value=truth_row.truth_value,
                predicted_active=predicted_active,
                predicted_value=predicted_value,
                correct=correct,
                cited_session_ids=tuple(cited_known),
                false_attribution_ids=tuple(false_attribution),
            )
        )
    return tuple(rows), tuple(warnings)


def pre_update_signal_ids(
    signals: Sequence[Signal], traits: Sequence[Trait], drift_events: Sequence[DriftEvent]
) -> frozenset[str]:
    """Ids of preference signals dated strictly before the earliest update on their trait."""
    trait_by_id = {trait.trait_id: trait for trait in traits}
    earliest_update: dict[str, date] = {}
    for event in drift_events:
        if event.event != DriftEventType.UPDATE:
            continue
        current = earliest_update.get(event.trait_id)
        if current is None or event.date < current:
            earliest_update[event.trait_id] = event.date

    ids: set[str] = set()
    for signal in signals:
        trait = trait_by_id.get(signal.trait_id)
        if trait is None or trait.kind != Kind.PREFERENCE:
            continue
        earliest = earliest_update.get(signal.trait_id)
        if earliest is not None and signal.date < earliest:
            ids.add(signal.signal_id)
    return frozenset(ids)


def signal_rows(
    pm_id: str,
    signals: Sequence[Signal],
    traits: Sequence[Trait],
    trait_rows: Sequence[Gate2TraitRow],
    pre_update: Collection[str],
    classification: Mapping[str, tuple[Kind | None, bool]],
    session_ids: Collection[str],
) -> tuple[Gate2SignalRow, ...]:
    """One `Gate2SignalRow` per signal whose session survived, joined to its trait row."""
    trait_by_id = {trait.trait_id: trait for trait in traits}
    row_by_param = {row.param: row for row in trait_rows}

    rows: list[Gate2SignalRow] = []
    for signal in signals:
        if signal.session_id not in session_ids:
            continue
        trait = trait_by_id[signal.trait_id]
        row = row_by_param[trait.param]
        cited = signal.session_id in row.cited_session_ids
        classified = signal.signal_id in classification
        kind_predicted, kind_ok = classification[signal.signal_id] if classified else (None, None)
        rows.append(
            Gate2SignalRow(
                pm_id=pm_id,
                signal_id=signal.signal_id,
                session_id=signal.session_id,
                trait_id=signal.trait_id,
                param=trait.param,
                kind=trait.kind,
                mode=signal.mode,
                valence=signal.valence,
                ownership=signal.ownership,
                pre_update=signal.signal_id in pre_update,
                cited=cited,
                recovered=cited and row.correct,
                classified=classified,
                kind_predicted=kind_predicted,
                kind_ok=kind_ok,
            )
        )
    return tuple(rows)
