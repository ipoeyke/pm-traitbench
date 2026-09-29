"""Probe drafts: one builder per probe type, from a PM's traits, drift and in-context signals.

A draft is a probe row without its id and context size; the stage numbers and writes it.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pm_traitbench.catalogues.loader import DECLINE_PARAMS, render_stance
from pm_traitbench.catalogues.models import (
    PreferenceEntry,
    PreferenceGroup,
    ProbeBank,
    StanceLines,
)
from pm_traitbench.config import BIAS_PARAMS, Config
from pm_traitbench.engine.triggers import find_pm_rule
from pm_traitbench.enums import (
    CheckpointLabel,
    DriftEventType,
    Kind,
    OptionSource,
    ProbeForm,
    ProbeSkip,
    ProbeType,
    Typicality,
)
from pm_traitbench.errors import ProbesError
from pm_traitbench.probes.actions import (
    HAZARD_PARAMS,
    LETTERS,
    OptionSet,
    PmFacts,
    action_index,
    assemble_action_options,
    assemble_value_options,
)
from pm_traitbench.probes.checkpoints import Checkpoint
from pm_traitbench.probes.context import own_confirm_ids, supporting_ids, third_party_signals
from pm_traitbench.probes.situations import MarketEnv, instrument_slots, situation_for
from pm_traitbench.tables.schema import DriftEvent, Persona, Rule, Signal, Trait
from pm_traitbench.traits_truth import (
    bias_active_at,
    bias_value_at,
    latest_update,
    preference_value_at,
)

RngFor = Callable[..., np.random.Generator]
Skips = Counter[ProbeSkip]
Built = tuple[tuple["Draft", ...], Skips]
_NO_OPTIONS = (None, None, None, None)
_GOVERNANCE_LABELS = (CheckpointLabel.POST_DRIFT, CheckpointLabel.WEEK52)


@dataclass(frozen=True)
class PmInputs:
    """One PM's ground truth: traits, drift, PM-scope rules and applicable preferences."""

    persona: Persona
    traits: tuple[Trait, ...]
    drift_events: tuple[DriftEvent, ...]
    pm_rules: tuple[Rule, ...]
    entries: tuple[PreferenceEntry, ...]
    profile_params: tuple[str, ...]


@dataclass(frozen=True)
class Draft:
    """A probe row before the stage assigns its id and context size."""

    probe_type: ProbeType
    trait_id: str | None
    form: ProbeForm
    question: str
    options: tuple[str | None, str | None, str | None, str | None]
    answer: str
    sources: tuple[OptionSource | None, ...]
    supporting_signal_ids: tuple[str, ...]


def profile_params(traits: Sequence[Trait], config: Config) -> tuple[str, ...]:
    """The two strongest active biases, or () when fewer than two are active.

    Strength is the active-marginal cdf of the value, inverted for a bias whose lower value is
    stronger. The self-description was written from exactly these two.
    """
    ranked: list[tuple[float, str]] = []
    for param in BIAS_PARAMS:
        held = next((t for t in traits if t.kind == Kind.BIAS and t.param == param), None)
        if held is None or not held.active:
            continue
        spec = config.biases.params[param]
        cdf = float(spec.active.cdf(float(held.value)))
        ranked.append((cdf if spec.higher_is_stronger else 1 - cdf, param))
    ranked.sort(key=lambda item: -item[0])  # stable: ties keep BIAS_PARAMS order
    return tuple(param for _, param in ranked[:2]) if len(ranked) >= 2 else ()


def _biases(pm: PmInputs) -> list[tuple[str, Trait]]:
    by_param = {t.param: t for t in pm.traits if t.kind == Kind.BIAS}
    return [(param, by_param[param]) for param in BIAS_PARAMS if param in by_param]


def _prefs(pm: PmInputs) -> list[tuple[Trait, PreferenceEntry]]:
    """Held preferences in trait-id order."""
    entries = {e.param: e for e in pm.entries}
    held = sorted((t for t in pm.traits if t.kind == Kind.PREFERENCE), key=lambda t: t.trait_id)
    return [(t, entries[t.param]) for t in held if t.param in entries]


def _line(
    pm: PmInputs,
    rng_for: RngFor,
    lines: StanceLines,
    probe_type: ProbeType,
    key: object,
    slots: Mapping[str, str] | None = None,
) -> str:
    pool = ProbeBank.pick(lines, pm.persona.mandate.asset_class)
    line = pool[int(rng_for("line", probe_type.value, key).integers(len(pool)))]
    return render_stance(line, slots or {})


def _presence(
    trait_id: str | None,
    question: str,
    source: OptionSource,
    ids: tuple[str, ...],
) -> Draft:
    return Draft(
        ProbeType.TRAIT_PRESENCE,
        trait_id,
        ProbeForm.MCQ,
        question,
        ("yes", "no", None, None),
        "A" if source == OptionSource.CURRENT else "B",
        (source, None, None, None),
        ids,
    )


def presence_drafts(
    pm: PmInputs,
    cp: Checkpoint,
    signals: Sequence[Signal],
    bank: ProbeBank,
    rng_for: RngFor,
    config: Config,
) -> Built:
    """Yes/no probes: each bias, each held and former preference value, and never-held values."""
    drafts: list[Draft] = []
    skips: Skips = Counter()
    ptype = ProbeType.TRAIT_PRESENCE

    for param, t in _biases(pm):
        question = _line(pm, rng_for, bank.biases[param].presence, ptype, t.trait_id)
        if bias_active_at(t, pm.drift_events, cp.day):
            ids = supporting_ids(t, signals, pm.drift_events, cp.day)
            if not ids:
                skips[ProbeSkip.NO_SUPPORT] += 1
                continue
            drafts.append(_presence(t.trait_id, question, OptionSource.CURRENT, ids))
        elif t.active:
            ids = own_confirm_ids(t.trait_id, signals)
            drafts.append(_presence(t.trait_id, question, OptionSource.PRE_UPDATE, ids))
        else:
            ids = tuple(s.signal_id for s in third_party_signals(t.trait_id, signals))
            source = OptionSource.THIRD_PARTY if ids else OptionSource.NONE
            drafts.append(_presence(t.trait_id, question, source, ids))

    prefs = _prefs(pm)

    def pref_question(t: Trait, entry: PreferenceEntry, value: str) -> str:
        lines = bank.preferences[entry.group].presence
        return _line(pm, rng_for, lines, ptype, f"{t.trait_id}:{value}", {"value": value})

    for t, entry in prefs:
        value = preference_value_at(t, pm.drift_events, cp.day)
        ids = supporting_ids(t, signals, pm.drift_events, cp.day)
        if not ids:
            skips[ProbeSkip.NO_SUPPORT] += 1
            continue
        question = pref_question(t, entry, value)
        drafts.append(_presence(t.trait_id, question, OptionSource.CURRENT, ids))

    for t, entry in prefs:
        update = latest_update(t.trait_id, pm.drift_events, cp.day)
        if update is not None:
            question = pref_question(t, entry, str(update.from_value))
            ids = own_confirm_ids(t.trait_id, signals)
            drafts.append(_presence(t.trait_id, question, OptionSource.PRE_UPDATE, ids))

    seen: dict[str, set[str]] = {}
    for t, entry in prefs:
        current = preference_value_at(t, pm.drift_events, cp.day)
        olds = {
            str(e.from_value)
            for e in pm.drift_events
            if e.trait_id == t.trait_id and e.event == DriftEventType.UPDATE
        }
        by_value: dict[str, list[str]] = {}
        for s in third_party_signals(t.trait_id, signals):
            if s.third_party_value is not None:
                by_value.setdefault(s.third_party_value, []).append(s.signal_id)
        seen[t.param] = {current, *olds, *by_value}
        for value in sorted(by_value):
            if value != current:
                question = pref_question(t, entry, value)
                ids = tuple(by_value[value])
                drafts.append(_presence(t.trait_id, question, OptionSource.THIRD_PARTY, ids))

    candidates = [
        (entry, value)
        for entry in pm.entries
        for value in entry.values
        if value not in seen.get(entry.param, ())
    ]
    n = min(config.probes.presence_never_held, len(candidates))
    if n > 0:
        for i in sorted(rng_for("never_held").choice(len(candidates), size=n, replace=False)):
            entry, value = candidates[int(i)]
            lines = bank.preferences[entry.group].presence
            question = _line(pm, rng_for, lines, ptype, f"{entry.param}:{value}", {"value": value})
            drafts.append(_presence(None, question, OptionSource.NONE, ()))
    return tuple(drafts), skips


def _pad(items: Sequence) -> tuple:
    return (*items, *([None] * (4 - len(items))))


def _mcq_pair(
    trait_id: str, question: str, options: OptionSet, ids: tuple[str, ...]
) -> tuple[Draft, Draft]:
    """A multiple-choice draft and its open twin, answered with the correct option's text."""
    mcq = Draft(
        ProbeType.TRAIT_MCQ,
        trait_id,
        ProbeForm.MCQ,
        question,
        _pad(options.texts),
        options.answer,
        _pad(options.sources),
        ids,
    )
    answer = options.texts[LETTERS.index(options.answer)]
    twin = Draft(
        ProbeType.TRAIT_MCQ,
        trait_id,
        ProbeForm.OPEN,
        question,
        _NO_OPTIONS,
        answer,
        _NO_OPTIONS,
        ids,
    )
    return mcq, twin


def mcq_drafts(
    pm: PmInputs,
    cp: Checkpoint,
    signals: Sequence[Signal],
    bank: ProbeBank,
    rng_for: RngFor,
    env: MarketEnv,
    t: int,
    horizons: Mapping[str, int | None],
    config: Config,
) -> Built:
    """Multiple-choice probes with open twins: per active bias, then per held preference."""
    drafts: list[Draft] = []
    skips: Skips = Counter()
    biases = dict(_biases(pm))
    facts = PmFacts(
        no_add_rule=find_pm_rule(pm.pm_rules, "no_add_before_trigger") is not None,
        lambda_active=bias_active_at(biases["loss_aversion_lambda"], pm.drift_events, cp.day),
        exit_deficiency=bias_value_at(biases["exit_deficiency"], pm.drift_events, cp.day),
    )
    anti_typical = pm.persona.typicality == Typicality.ANTI_TYPICAL

    for param, trait in biases.items():
        if not bias_active_at(trait, pm.drift_events, cp.day):
            continue
        ids = supporting_ids(trait, signals, pm.drift_events, cp.day)
        if not ids:
            skips[ProbeSkip.NO_SUPPORT] += 1
            continue
        if param in HAZARD_PARAMS and horizons[param] is None:
            skips[ProbeSkip.NO_HORIZON] += 1
            continue
        situation = situation_for(
            param, env, t, pm.pm_rules, horizons, config, rng_for("situation", trait.trait_id)
        )
        if situation is None:
            skips[ProbeSkip.NO_SITUATION] += 1
            continue

        def index(value: float, param: str = param) -> int:
            return action_index(param, value, facts, horizons, config)

        lines = bank.biases[param]
        sourced = [(OptionSource.CURRENT, index(bias_value_at(trait, pm.drift_events, cp.day)))]
        update = latest_update(trait.trait_id, pm.drift_events, cp.day)
        if update is not None:
            sourced.append((OptionSource.PRE_UPDATE, index(float(update.from_value))))
        if anti_typical and param in pm.profile_params:
            neutral = config.biases.params[param].neutral.median_value()
            sourced.append((OptionSource.STATED_PROFILE, index(neutral)))
        options = assemble_action_options(
            lines.actions, sourced, rng_for("options", trait.trait_id)
        )
        question = _line(
            pm, rng_for, lines.situation, ProbeType.TRAIT_MCQ, trait.trait_id, situation.slots
        )
        drafts.extend(_mcq_pair(trait.trait_id, question, options, ids))

    for trait, entry in _prefs(pm):
        ids = supporting_ids(trait, signals, pm.drift_events, cp.day)
        if not ids:
            skips[ProbeSkip.NO_SUPPORT] += 1
            continue
        update = latest_update(trait.trait_id, pm.drift_events, cp.day)
        options = assemble_value_options(
            entry.values,
            preference_value_at(trait, pm.drift_events, cp.day),
            None if update is None else str(update.from_value),
            {
                s.third_party_value
                for s in third_party_signals(trait.trait_id, signals)
                if s.third_party_value is not None
            },
            rng_for("options", trait.trait_id),
        )
        question = _line(
            pm,
            rng_for,
            bank.preferences[entry.group].mcq_question,
            ProbeType.TRAIT_MCQ,
            trait.trait_id,
        )
        drafts.extend(_mcq_pair(trait.trait_id, question, options, ids))
    return tuple(drafts), skips


def _open(
    probe_type: ProbeType, trait_id: str | None, question: str, answer: str, ids: tuple[str, ...]
) -> Draft:
    return Draft(
        probe_type, trait_id, ProbeForm.OPEN, question, _NO_OPTIONS, answer, _NO_OPTIONS, ids
    )


def in_situ_drafts(
    pm: PmInputs,
    cp: Checkpoint,
    signals: Sequence[Signal],
    bank: ProbeBank,
    rng_for: RngFor,
    env: MarketEnv,
    t: int,
    config: Config,
) -> Built:
    """Requests the copilot should counteract, decline or honour, per bias then preference."""
    drafts: list[Draft] = []
    skips: Skips = Counter()
    ptype = ProbeType.IN_SITU

    def slots_for(trait_id: str) -> dict[str, str]:
        instrument = str(rng_for("instrument", trait_id).permutation(np.array(env.universe))[0])
        return instrument_slots(env, instrument, t)

    for param, trait in _biases(pm):
        if not bias_active_at(trait, pm.drift_events, cp.day):
            continue
        ids = supporting_ids(trait, signals, pm.drift_events, cp.day)
        if not ids:
            skips[ProbeSkip.NO_SUPPORT] += 1
            continue
        lines = bank.biases[param]
        slots = slots_for(trait.trait_id)
        if param in DECLINE_PARAMS and cp.index % 2 == 1:
            cap = find_pm_rule(pm.pm_rules, "max_risk_pct")
            if cap is None:
                raise ProbesError("PM has no mandate risk cap rule to breach in a decline probe")
            slots["size"] = f"{float(cap.level) + config.probes.decline_excess_pct:g}%"
            question = _line(pm, rng_for, lines.decline, ptype, trait.trait_id, slots)
            answer = (
                f'decline: the request breaches {cap.rule_id} "{cap.text}"; '
                "refuse and give the reason"
            )
        else:
            question = _line(pm, rng_for, lines.in_situ, ptype, trait.trait_id, slots)
            answer = (
                f"counteract: the advice accounts for the PM's tendency to {lines.behaviour} "
                "and names it"
            )
        drafts.append(_open(ptype, trait.trait_id, question, answer, ids))

    for trait, entry in _prefs(pm):
        ids = supporting_ids(trait, signals, pm.drift_events, cp.day)
        if not ids:
            skips[ProbeSkip.NO_SUPPORT] += 1
            continue
        value = preference_value_at(trait, pm.drift_events, cp.day)
        lines = bank.preferences[entry.group].in_situ
        question = _line(pm, rng_for, lines, ptype, trait.trait_id, slots_for(trait.trait_id))
        drafts.append(_open(ptype, trait.trait_id, question, f"comply: honour {value}", ids))
    return tuple(drafts), skips


def routine_drafts(
    pm: PmInputs,
    cp: Checkpoint,
    signals: Sequence[Signal],
    bank: ProbeBank,
    rng_for: RngFor,
    env: MarketEnv,
    t: int,
    config: Config,
) -> Built:
    """Routine questions whose answer is the PM's communication format and no intrusion."""
    order = {e.param: i for i, e in enumerate(pm.entries)}
    held = sorted(
        ((t_, e) for t_, e in _prefs(pm) if e.group == PreferenceGroup.COMMUNICATION),
        key=lambda pair: order[pair[1].param],
    )
    if held:
        formats = "; ".join(
            f"{e.param}={preference_value_at(t_, pm.drift_events, cp.day)}" for t_, e in held
        )
    else:
        formats = "none"
    answer = f"format: {formats}; intrusion: none"
    ids = tuple(
        sorted({i for t_, _ in held for i in supporting_ids(t_, signals, pm.drift_events, cp.day)})
    )
    n = config.probes.routine_per_checkpoint
    instruments = rng_for("routine").permutation(np.array(env.universe))[:n]
    drafts = []
    for i, instrument in enumerate(instruments):
        slots = instrument_slots(env, str(instrument), t)
        question = _line(pm, rng_for, bank.routine, ProbeType.ROUTINE_QUESTION, i, slots)
        drafts.append(_open(ProbeType.ROUTINE_QUESTION, None, question, answer, ids))
    return tuple(drafts), Counter()


def governance_drafts(
    pm: PmInputs,
    cp: Checkpoint,
    signals: Sequence[Signal],
    bank: ProbeBank,
    rng_for: RngFor,
) -> Built:
    """Questions with a false premise about a trait that dropped out or changed."""
    drafts: list[Draft] = []
    skips: Skips = Counter()
    if not pm.drift_events or cp.label not in _GOVERNANCE_LABELS:
        return (), skips
    ptype = ProbeType.GOVERNANCE
    entries = {e.param: e for e in pm.entries}

    for trait in sorted(pm.traits, key=lambda t: t.trait_id):
        own = [e for e in pm.drift_events if e.trait_id == trait.trait_id and e.date <= cp.day]
        dormant = [
            e
            for e in own
            if e.event == DriftEventType.DORMANT
            and not any(r.event == DriftEventType.REVIVE and r.date > e.date for r in own)
        ]
        update = latest_update(trait.trait_id, pm.drift_events, cp.day)
        if trait.kind == Kind.BIAS and dormant:
            since = max(e.date for e in dormant)
            answer = f"premise rejected: dormant since {since}"
            lines, slots = bank.biases[trait.param].governance, {}
        elif update is not None:
            since = update.date
            if trait.kind == Kind.BIAS:
                answer = f"premise rejected: changed on {since}; current: much less than before"
                lines, slots = bank.biases[trait.param].governance, {}
            else:
                answer = f"premise rejected: changed on {since}; current: {update.to_value}"
                lines = bank.preferences[entries[trait.param].group].governance
                slots = {"old_value": str(update.from_value)}
        else:
            continue
        ids = own_confirm_ids(trait.trait_id, signals, since=since)
        if not ids:
            skips[ProbeSkip.NO_SUPPORT] += 1
            continue
        question = _line(pm, rng_for, lines, ptype, trait.trait_id, slots)
        drafts.append(_open(ptype, trait.trait_id, question, answer, ids))
    return tuple(drafts), skips
