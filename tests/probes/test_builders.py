import zlib
from collections import Counter

import numpy as np
import pytest

from pm_traitbench.catalogues.loader import banned_words_in, load_catalogue, matched_params
from pm_traitbench.catalogues.models import PreferenceEntry, PreferenceGroup
from pm_traitbench.config import BIAS_PARAMS
from pm_traitbench.engine.adapters import adapter_for
from pm_traitbench.enums import (
    Action,
    AssetClass,
    CheckpointLabel,
    DriftEventType,
    Kind,
    OptionSource,
    ProbeForm,
    ProbeSkip,
    ProbeType,
    RuleScope,
    StreetView,
    Typicality,
)
from pm_traitbench.probes.actions import LETTERS, horizons
from pm_traitbench.probes.builders import (
    Draft,
    PmInputs,
    governance_drafts,
    in_situ_drafts,
    mcq_drafts,
    presence_drafts,
    profile_params,
    routine_drafts,
)
from pm_traitbench.probes.checkpoints import Checkpoint
from pm_traitbench.probes.situations import MarketEnv
from pm_traitbench.signals.assemble import session_id
from pm_traitbench.tables.schema import ProbeRow, probe_id
from tests.engine.fixtures import stage_config
from tests.probes.fixtures import PM_A, drift, signal, third_party, trait

CONFIG = stage_config()
CATALOGUE = load_catalogue()
BANK = CATALOGUE.probes
ENTRIES = CATALOGUE.preferences_for(AssetClass.EQUITIES)
HORIZONS = horizons(CONFIG)
T = 25
LOSS, DISP = "loss_aversion_lambda", "disposition_ratio"
UPDATE_T, SIGNAL_T = 10, 12


def rng_for(*keys):
    return np.random.default_rng(zlib.crc32(repr(keys).encode()))


def entry_of(group, index=0):
    return [e for e in ENTRIES if e.group == group][index]


def active_value(param):
    return CONFIG.biases.params[param].active.median_value()


def neutral_value(param):
    return CONFIG.biases.params[param].neutral.median_value()


@pytest.fixture
def env_rules(fixture_view, neutral_pm):
    persona, _, rules = neutral_pm(AssetClass.EQUITIES, "equity_long_short")
    adapter = adapter_for(AssetClass.EQUITIES, "equity_long_short", CONFIG.engine.horizon_days)
    universe = adapter.universe(fixture_view.instruments, rules)
    env = MarketEnv(fixture_view, adapter, universe, AssetClass.EQUITIES)
    return env, persona, [r for r in rules if r.scope == RuleScope.PM]


class Corpus:
    """A PM's inputs and in-context signals, assembled by hand."""

    def __init__(self, env, persona, rules, dates):
        self.env, self.persona, self.rules, self.dates = env, persona, rules, dates
        self.day = dates[T]
        self.signals = []
        self.drift_events = []
        self.traits = []
        self.entries = ENTRIES
        self.profile = ()

    def bias(self, param, active=True, value=None):
        i = BIAS_PARAMS.index(param) + 1
        v = value if value is not None else (active_value if active else neutral_value)(param)
        self.traits.append(trait(PM_A, f"t_{i:02d}", param, Kind.BIAS, v, active=active))
        return f"t_{i:02d}"

    def pref(self, param, value):
        tid = f"t_{9 + sum(t.kind == Kind.PREFERENCE for t in self.traits):02d}"
        self.traits.append(trait(PM_A, tid, param, Kind.PREFERENCE, value))
        return tid

    def sig(self, trait_id, t=SIGNAL_T, colleague=None):
        day = self.dates[t]
        n = len(self.signals) + 1
        sid = session_id(PM_A, day, 0)
        if colleague is None:
            s = signal(PM_A, sid, day, trait_id, signal_id=f"sg_{n:03d}")
        else:
            s = third_party(PM_A, sid, day, trait_id, colleague, f"sg_{n:03d}")
        self.signals.append(s)
        return s.signal_id

    def event(self, kind, trait_id, t=UPDATE_T, old=None, new=None):
        self.drift_events.append(drift(PM_A, self.dates[t], kind, trait_id, old, new))

    def inputs(self):
        for param in BIAS_PARAMS:
            if not any(t.param == param for t in self.traits):
                self.bias(param, active=False)
        return PmInputs(
            persona=self.persona,
            traits=tuple(sorted(self.traits, key=lambda t: t.trait_id)),
            drift_events=tuple(self.drift_events),
            pm_rules=tuple(self.rules),
            entries=tuple(self.entries),
            profile_params=self.profile,
        )

    def cp(self, label=CheckpointLabel.WEEK13, index=0, covers=None):
        return Checkpoint(label, self.day, index, frozenset(covers or {label}))


@pytest.fixture
def corpus(fixture_view, env_rules):
    env, persona, rules = env_rules
    return Corpus(env, persona, rules, fixture_view.dates)


def presence(c, config=CONFIG, cp=None):
    return presence_drafts(c.inputs(), cp or c.cp(), c.signals, BANK, rng_for, config)


def mcq(c, cp=None, hz=HORIZONS, t=T):
    return mcq_drafts(c.inputs(), cp or c.cp(), c.signals, BANK, rng_for, c.env, t, hz, CONFIG)


def in_situ(c, cp=None):
    return in_situ_drafts(c.inputs(), cp or c.cp(), c.signals, BANK, rng_for, c.env, T, CONFIG)


def routine(c):
    return routine_drafts(c.inputs(), c.cp(), c.signals, BANK, rng_for, c.env, T, CONFIG)


def governance(c, cp=None):
    cp = cp or c.cp(CheckpointLabel.POST_DRIFT)
    return governance_drafts(c.inputs(), cp, c.signals, BANK, rng_for)


def of_trait(drafts, trait_id):
    return [d for d in drafts if d.trait_id == trait_id]


# presence


def test_active_bias_with_support_answers_yes_on_current(corpus):
    tid = corpus.bias(LOSS)
    sid = corpus.sig(tid)
    drafts, skips = presence(corpus)
    (d,) = of_trait(drafts, tid)
    assert (d.options, d.answer, d.sources) == (
        ("yes", "no", None, None),
        "A",
        (OptionSource.CURRENT, None, None, None),
    )
    assert d.supporting_signal_ids == (sid,)
    assert d.probe_type == ProbeType.TRAIT_PRESENCE
    assert not skips


def test_active_bias_without_support_is_skipped_and_counted(corpus):
    tid = corpus.bias(LOSS)
    drafts, skips = presence(corpus)
    assert of_trait(drafts, tid) == []
    assert skips == Counter({ProbeSkip.NO_SUPPORT: 1})


def test_dormant_bias_answers_no_pre_update(corpus):
    tid = corpus.bias(LOSS)
    sid = corpus.sig(tid, t=3)
    corpus.event(DriftEventType.DORMANT, tid)
    (d,) = of_trait(presence(corpus)[0], tid)
    assert (d.answer, d.sources[0], d.supporting_signal_ids) == (
        "B",
        OptionSource.PRE_UPDATE,
        (sid,),
    )


def test_inactive_bias_cites_colleague_signal_else_none(corpus):
    other = corpus.bias(DISP, active=False)
    quiet = corpus.bias(LOSS, active=False)
    sid = corpus.sig(other, colleague="always")
    drafts, _ = presence(corpus)
    (d,) = of_trait(drafts, other)
    assert (d.answer, d.sources[0], d.supporting_signal_ids) == (
        "B",
        OptionSource.THIRD_PARTY,
        (sid,),
    )
    (q,) = of_trait(drafts, quiet)
    assert (q.sources[0], q.supporting_signal_ids) == (OptionSource.NONE, ())


def test_bias_rows_follow_bias_param_order(corpus):
    for param in reversed(BIAS_PARAMS):
        corpus.bias(param, active=False)
    ids = [d.trait_id for d in presence(corpus)[0] if d.trait_id]
    assert ids == sorted(ids)


def test_updated_preference_gives_yes_on_new_and_no_on_old(corpus):
    entry = entry_of(PreferenceGroup.COMMUNICATION)
    old, new = entry.values[0], entry.values[1]
    tid = corpus.pref(entry.param, old)
    corpus.event(DriftEventType.UPDATE, tid, old=old, new=new)
    before = corpus.sig(tid, t=3)
    after = corpus.sig(tid, t=SIGNAL_T)
    drafts, _ = presence(corpus)
    yes, no = of_trait(drafts, tid)
    assert (yes.answer, yes.sources[0], yes.supporting_signal_ids) == (
        "A",
        OptionSource.CURRENT,
        (after,),
    )
    assert new in yes.question and old not in yes.question
    assert (no.answer, no.sources[0], no.supporting_signal_ids) == (
        "B",
        OptionSource.PRE_UPDATE,
        (before, after),
    )
    assert old in no.question


def test_third_party_preference_value_gives_no_and_skips_the_current_value(corpus):
    entry = entry_of(PreferenceGroup.INFORMATION)
    current, other = entry.values[0], entry.values[1]
    tid = corpus.pref(entry.param, current)
    corpus.sig(tid)
    same = corpus.sig(tid, colleague=current)
    diff = corpus.sig(tid, colleague=other)
    drafts, _ = presence(corpus)
    negatives = [d for d in of_trait(drafts, tid) if d.sources[0] == OptionSource.THIRD_PARTY]
    (d,) = negatives
    assert d.supporting_signal_ids == (diff,) and same not in d.supporting_signal_ids
    assert other in d.question


def never_held(drafts):
    return [d for d in drafts if d.trait_id is None]


def lone_entry_corpus(corpus, n_never):
    entry = PreferenceEntry(
        param="probe_pref",
        group=PreferenceGroup.COMMUNICATION,
        asset_classes=(AssetClass.EQUITIES,),
        values=("vcur", "vold", "vtpa", "vfree1", "vfree2"),
    )
    corpus.entries = (entry,)
    tid = corpus.pref(entry.param, "vold")
    corpus.event(DriftEventType.UPDATE, tid, old="vold", new="vcur")
    corpus.sig(tid)
    corpus.sig(tid, colleague="vtpa")
    config = CONFIG.model_copy(
        update={"probes": CONFIG.probes.model_copy(update={"presence_never_held": n_never})}
    )
    return config


def test_never_held_values_avoid_current_old_and_third_party(corpus):
    config = lone_entry_corpus(corpus, 10)
    drafts, _ = presence(corpus, config)
    held = never_held(drafts)
    assert len(held) == 2
    for d in held:
        assert d.answer == "B" and d.sources[0] == OptionSource.NONE
        assert d.supporting_signal_ids == ()
        assert not any(v in d.question for v in ("vcur", "vold", "vtpa"))
    assert sorted("vfree1" in d.question for d in held) == [False, True]


def test_never_held_count_is_capped_by_config(corpus):
    config = lone_entry_corpus(corpus, 1)
    assert len(never_held(presence(corpus, config)[0])) == 1


def test_never_held_zero_gives_none(corpus):
    config = lone_entry_corpus(corpus, 0)
    assert never_held(presence(corpus, config)[0]) == []


# mcq


def bias_mcqs(drafts):
    return [d for d in drafts if d.form == ProbeForm.MCQ]


def test_typical_static_bias_mcq_has_one_current_and_rest_none(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    drafts, skips = mcq(corpus)
    (d,) = bias_mcqs(drafts)
    assert Counter(d.sources) == Counter({OptionSource.CURRENT: 1, OptionSource.NONE: 3})
    assert d.options[LETTERS.index(d.answer)] == d.options[d.sources.index(OptionSource.CURRENT)]
    assert not skips


def test_anti_typical_pm_gets_stated_profile_option(corpus):
    tid = corpus.bias(DISP)
    corpus.sig(tid)
    corpus.persona = corpus.persona.model_copy(update={"typicality": Typicality.ANTI_TYPICAL})
    corpus.profile = (DISP, LOSS)
    (d,) = bias_mcqs(mcq(corpus)[0])
    assert OptionSource.STATED_PROFILE in d.sources


def test_stated_profile_needs_anti_typical_and_a_profile_param(corpus):
    tid = corpus.bias(DISP)
    corpus.sig(tid)
    corpus.profile = (DISP, LOSS)
    (typical,) = bias_mcqs(mcq(corpus)[0])
    assert OptionSource.STATED_PROFILE not in typical.sources
    corpus.persona = corpus.persona.model_copy(update={"typicality": Typicality.ANTI_TYPICAL})
    corpus.profile = (LOSS, "anchoring_rho")
    (outside,) = bias_mcqs(mcq(corpus)[0])
    assert OptionSource.STATED_PROFILE not in outside.sources


def test_stated_profile_collapsing_onto_current_is_dropped(corpus):
    tid = corpus.bias(DISP, value=neutral_value(DISP))
    corpus.sig(tid)
    corpus.persona = corpus.persona.model_copy(update={"typicality": Typicality.ANTI_TYPICAL})
    corpus.profile = (DISP, LOSS)
    (d,) = bias_mcqs(mcq(corpus)[0])
    assert OptionSource.STATED_PROFILE not in d.sources


def test_updated_bias_mcq_carries_pre_update(corpus):
    tid = corpus.bias(DISP)
    corpus.sig(tid)
    corpus.event(DriftEventType.UPDATE, tid, old=active_value(DISP), new=neutral_value(DISP))
    (d,) = bias_mcqs(mcq(corpus)[0])
    assert Counter(d.sources)[OptionSource.PRE_UPDATE] == 1
    assert Counter(d.sources)[OptionSource.CURRENT] == 1


def test_dormant_exit_deficiency_uses_neutral_value_for_no_add_answer(corpus):
    no_add = corpus.rules[0].model_copy(
        update={"rule_id": "r_90", "param": "no_add_before_trigger", "action": Action.NO_ADD}
    )
    corpus.rules = [*corpus.rules, no_add]
    loss = corpus.bias(LOSS)
    corpus.sig(loss)
    exit_id = corpus.bias("exit_deficiency", value=0.9)
    (d,) = bias_mcqs(mcq(corpus)[0])
    actions = BANK.biases[LOSS].actions
    assert d.options[LETTERS.index(d.answer)] == actions[0]
    corpus.event(DriftEventType.DORMANT, exit_id, t=UPDATE_T)
    (d,) = bias_mcqs(mcq(corpus)[0])
    assert d.options[LETTERS.index(d.answer)] == actions[2]


def test_exit_deficiency_add_reads_the_planted_loss_aversion_flag(corpus):
    exit_id = corpus.bias("exit_deficiency", value=0.9)
    corpus.sig(exit_id)
    loss = corpus.bias(LOSS)
    corpus.event(DriftEventType.DORMANT, loss, t=UPDATE_T)
    (d,) = [m for m in bias_mcqs(mcq(corpus)[0]) if m.trait_id == exit_id]
    actions = BANK.biases["exit_deficiency"].actions
    assert d.options[LETTERS.index(d.answer)] == actions[1]


def test_each_mcq_is_followed_by_its_open_twin(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    entry = entry_of(PreferenceGroup.WORKFLOW)
    corpus.sig(corpus.pref(entry.param, entry.values[0]))
    drafts, _ = mcq(corpus)
    assert len(drafts) == 4
    for first, twin in zip(drafts[::2], drafts[1::2], strict=True):
        assert first.form == ProbeForm.MCQ and twin.form == ProbeForm.OPEN
        assert twin.question == first.question and twin.trait_id == first.trait_id
        assert twin.options == (None,) * 4 and twin.sources == (None,) * 4
        assert twin.answer == first.options[LETTERS.index(first.answer)]
        assert twin.supporting_signal_ids == first.supporting_signal_ids
        assert twin.probe_type == ProbeType.TRAIT_MCQ


def test_preference_mcq_lists_every_catalogue_value_once(corpus):
    entry = entry_of(PreferenceGroup.COMMUNICATION, 3)
    tid = corpus.pref(entry.param, entry.values[1])
    corpus.sig(tid)
    (d,) = bias_mcqs(mcq(corpus)[0])
    assert sorted(t for t in d.options if t) == sorted(entry.values)
    assert d.options[LETTERS.index(d.answer)] == entry.values[1]
    assert Counter(d.sources)[OptionSource.CURRENT] == 1


def test_preference_mcq_tags_old_and_third_party_values(corpus):
    entry = entry_of(PreferenceGroup.COMMUNICATION, 1)
    old, new, other = entry.values[:3]
    tid = corpus.pref(entry.param, old)
    corpus.event(DriftEventType.UPDATE, tid, old=old, new=new)
    corpus.sig(tid)
    corpus.sig(tid, colleague=other)
    (d,) = bias_mcqs(mcq(corpus)[0])
    by_text = dict(zip(d.options, d.sources, strict=False))
    assert by_text[new] == OptionSource.CURRENT
    assert by_text[old] == OptionSource.PRE_UPDATE
    assert by_text[other] == OptionSource.THIRD_PARTY


def test_mcq_skips_bias_without_support_and_without_horizon(corpus):
    corpus.bias(LOSS)
    drafts, skips = mcq(corpus)
    assert drafts == () and skips == Counter({ProbeSkip.NO_SUPPORT: 1})
    corpus.sig("t_01")
    drafts, skips = mcq(corpus, hz={**HORIZONS, LOSS: None})
    assert drafts == () and skips == Counter({ProbeSkip.NO_HORIZON: 1})


def test_mcq_skips_when_no_situation(corpus):
    tid = corpus.bias("herding_weight")
    corpus.sig(tid)
    view = corpus.env.view
    neutral_t = next(
        t
        for t in range(len(view.dates))
        if all(view.street_view(i, t) == StreetView.NEUTRAL for i in corpus.env.universe)
    )
    drafts, skips = mcq(corpus, t=neutral_t)
    assert drafts == () and skips == Counter({ProbeSkip.NO_SITUATION: 1})


# in situ


def test_loss_aversion_alternates_counteract_and_decline(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    even, _ = in_situ(corpus, corpus.cp(index=0))
    odd, _ = in_situ(corpus, corpus.cp(index=1))
    (e,), (o,) = even, odd
    assert e.answer.startswith("counteract: ") and BANK.biases[LOSS].behaviour in e.answer
    assert o.answer.startswith("decline: the request breaches r_")
    cap_rule = next(r for r in corpus.rules if r.param == "max_risk_pct")
    assert f'{cap_rule.rule_id} "{cap_rule.text}"' in o.answer
    size = f"{float(cap_rule.level) + CONFIG.probes.decline_excess_pct:g}%"
    assert size in o.question
    assert o.form == ProbeForm.OPEN and o.options == (None,) * 4
    assert o.probe_type == ProbeType.IN_SITU and o.trait_id == tid


def test_only_decline_params_decline_at_odd_index(corpus):
    tid = corpus.bias(DISP)
    corpus.sig(tid)
    (d,), _ = in_situ(corpus, corpus.cp(index=1))
    assert d.answer.startswith("counteract: ")


def test_preference_request_never_contains_its_value(corpus):
    for group in PreferenceGroup:
        entry = entry_of(group)
        corpus.sig(corpus.pref(entry.param, entry.values[0]))
    drafts, _ = in_situ(corpus)
    assert len(drafts) == 4
    for d, group in zip(drafts, PreferenceGroup, strict=True):
        value = entry_of(group).values[0]
        assert d.answer == f"comply: honour {value}"
        assert value not in d.question


def test_in_situ_skips_trait_without_support(corpus):
    corpus.bias(LOSS)
    drafts, skips = in_situ(corpus)
    assert drafts == () and skips == Counter({ProbeSkip.NO_SUPPORT: 1})


# routine


def test_routine_rows_have_null_trait_and_list_communication_values_in_catalogue_order(corpus):
    first, second = (
        entry_of(PreferenceGroup.COMMUNICATION, 0),
        entry_of(PreferenceGroup.COMMUNICATION, 1),
    )
    t_second = corpus.pref(second.param, second.values[0])
    t_first = corpus.pref(first.param, first.values[1])
    other = entry_of(PreferenceGroup.WORKFLOW)
    corpus.pref(other.param, other.values[0])
    ids = sorted([corpus.sig(t_second), corpus.sig(t_first)])
    drafts, skips = routine(corpus)
    assert len(drafts) == CONFIG.probes.routine_per_checkpoint == 2
    expected = (
        f"format: {first.param}={first.values[1]}; {second.param}={second.values[0]}"
        "; intrusion: none"
    )
    for d in drafts:
        assert d.trait_id is None and d.form == ProbeForm.OPEN
        assert d.probe_type == ProbeType.ROUTINE_QUESTION
        assert d.answer == expected
        assert d.supporting_signal_ids == tuple(ids)
    assert not skips


def test_routine_omits_communication_preference_without_signal_in_context(corpus):
    shown, hidden = (
        entry_of(PreferenceGroup.COMMUNICATION, 0),
        entry_of(PreferenceGroup.COMMUNICATION, 1),
    )
    t_shown = corpus.pref(shown.param, shown.values[0])
    t_hidden = corpus.pref(hidden.param, hidden.values[0])
    sid = corpus.sig(t_shown)
    late = corpus.sig(t_hidden, t=T + 1)
    seen = [s for s in corpus.signals if s.date <= corpus.day]
    drafts, _ = routine_drafts(
        corpus.inputs(), corpus.cp(), seen, BANK, rng_for, corpus.env, T, CONFIG
    )
    for d in drafts:
        assert d.answer == f"format: {shown.param}={shown.values[0]}; intrusion: none"
        assert d.supporting_signal_ids == (sid,) and late not in d.supporting_signal_ids


def test_routine_with_no_supported_communication_preference_gets_format_none(corpus):
    entry = entry_of(PreferenceGroup.COMMUNICATION)
    corpus.pref(entry.param, entry.values[0])
    drafts, _ = routine(corpus)
    assert {d.answer for d in drafts} == {"format: none; intrusion: none"}


def test_routine_without_communication_preferences_gets_format_none(corpus):
    drafts, _ = routine(corpus)
    assert {d.answer for d in drafts} == {"format: none; intrusion: none"}
    assert all(d.supporting_signal_ids == () for d in drafts)


# governance


def test_static_pm_gets_no_governance_probes(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    assert governance(corpus) == ((), Counter())


def test_governance_only_at_drift_or_final_checkpoints(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    corpus.event(DriftEventType.UPDATE, tid, old=active_value(LOSS), new=neutral_value(LOSS))
    assert governance(corpus, corpus.cp(CheckpointLabel.WEEK13)) == ((), Counter())
    assert governance(corpus, corpus.cp(CheckpointLabel.PRE_DRIFT)) == ((), Counter())
    assert len(governance(corpus, corpus.cp(CheckpointLabel.WEEK52))[0]) == 1


def test_governance_follows_the_covered_labels_not_the_winning_label(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    corpus.event(DriftEventType.UPDATE, tid, old=active_value(LOSS), new=neutral_value(LOSS))
    final = corpus.cp(
        CheckpointLabel.REGIME_SHIFT, covers={CheckpointLabel.REGIME_SHIFT, CheckpointLabel.WEEK52}
    )
    assert len(governance(corpus, final)[0]) == 1
    shared = corpus.cp(
        CheckpointLabel.PRE_DRIFT, covers={CheckpointLabel.PRE_DRIFT, CheckpointLabel.POST_DRIFT}
    )
    assert len(governance(corpus, shared)[0]) == 1


def test_drift_pm_gets_one_row_per_updated_or_dormant_trait(corpus):
    loss = corpus.bias(LOSS)
    disp = corpus.bias(DISP)
    calm = corpus.bias("herding_weight")
    entry = entry_of(PreferenceGroup.EXPRESSION)
    old, new = entry.values[:2]
    pref = corpus.pref(entry.param, old)
    corpus.event(DriftEventType.UPDATE, loss, t=8, old=active_value(LOSS), new=1.0)
    corpus.event(DriftEventType.DORMANT, disp, t=9)
    corpus.event(DriftEventType.UPDATE, pref, t=10, old=old, new=new)
    ids = {t: corpus.sig(t, t=SIGNAL_T) for t in (loss, disp, calm, pref)}
    drafts, skips = governance(corpus)
    assert [d.trait_id for d in drafts] == [loss, disp, pref]
    by = {d.trait_id: d for d in drafts}
    day = corpus.dates
    assert (
        by[loss].answer == f"premise rejected: changed on {day[8]}; current: much less than before"
    )
    assert by[disp].answer == f"premise rejected: dormant since {day[9]}"
    assert by[pref].answer == f"premise rejected: changed on {day[10]}; current: {new}"
    assert old in by[pref].question
    for tid, d in by.items():
        assert d.supporting_signal_ids == (ids[tid],)
        assert d.form == ProbeForm.OPEN and d.probe_type == ProbeType.GOVERNANCE
    assert not skips


def test_governance_uses_signals_since_the_event(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid, t=3)
    corpus.event(DriftEventType.UPDATE, tid, t=8, old=active_value(LOSS), new=1.0)
    drafts, skips = governance(corpus)
    assert drafts == () and skips == Counter({ProbeSkip.NO_SUPPORT: 1})


def test_dormant_bias_revived_before_the_day_gets_no_governance_probe(corpus):
    tid = corpus.bias(LOSS)
    corpus.sig(tid)
    corpus.event(DriftEventType.DORMANT, tid, t=8)
    corpus.event(DriftEventType.REVIVE, tid, t=15)
    assert governance(corpus) == ((), Counter())


# profile_params


def test_profile_params_orders_by_active_strength_and_inverts_coverage():
    specs = CONFIG.biases.params

    def bias(param, u):
        i = BIAS_PARAMS.index(param) + 1
        value = float(specs[param].active.ppf(u))
        return trait(PM_A, f"t_{i:02d}", param, Kind.BIAS, value, active=True)

    traits = [
        bias(LOSS, 0.5),
        bias(DISP, 0.7),
        bias("overconfidence_coverage", 0.05),
        bias("exit_deficiency", 0.6),
    ]
    assert profile_params(traits, CONFIG) == ("overconfidence_coverage", DISP)


def test_profile_params_is_empty_below_two_active_biases():
    one = [trait(PM_A, "t_01", LOSS, Kind.BIAS, active_value(LOSS), active=True)]
    idle = [trait(PM_A, "t_02", DISP, Kind.BIAS, neutral_value(DISP), active=False)]
    assert profile_params(one + idle, CONFIG) == ()
    assert profile_params([], CONFIG) == ()


# every draft


def full_corpus(corpus):
    """Every bias active but one, a preference per group, drift and third-party signals."""
    for param in BIAS_PARAMS:
        active = param != "herding_weight"
        tid = corpus.bias(param, active=active)
        if active:
            corpus.sig(tid)
    corpus.sig("t_05", colleague="always")
    corpus.event(DriftEventType.UPDATE, "t_01", old=active_value(LOSS), new=neutral_value(LOSS))
    corpus.event(DriftEventType.DORMANT, "t_02")
    for group in PreferenceGroup:
        entry = entry_of(group)
        tid = corpus.pref(entry.param, entry.values[0])
        corpus.sig(tid)
        corpus.sig(tid, colleague=entry.values[1])
    p_entry = entry_of(PreferenceGroup.COMMUNICATION)
    corpus.event(DriftEventType.UPDATE, "t_09", old=p_entry.values[0], new=p_entry.values[2])
    corpus.persona = corpus.persona.model_copy(update={"typicality": Typicality.ANTI_TYPICAL})
    corpus.profile = (LOSS, DISP)
    return corpus


def all_drafts(c, index):
    cp = c.cp(CheckpointLabel.POST_DRIFT, index)
    parts = (
        presence(c, cp=cp),
        mcq(c, cp=cp),
        in_situ(c, cp=cp),
        routine(c),
        governance(c, cp),
    )
    return [d for drafts, _ in parts for d in drafts]


@pytest.mark.parametrize("index", [0, 1])
def test_every_draft_builds_a_valid_probe_row(corpus, index):
    drafts = all_drafts(full_corpus(corpus), index)
    assert {d.probe_type for d in drafts} == set(ProbeType)
    for n, d in enumerate(drafts, start=1):
        assert isinstance(d, Draft)
        assert len(d.options) == 4 and len(d.sources) == 4
        ProbeRow(
            probe_id=probe_id(PM_A, n),
            pm_id=PM_A,
            checkpoint_date=corpus.day,
            checkpoint_label=CheckpointLabel.POST_DRIFT,
            probe_type=d.probe_type,
            trait_id=d.trait_id,
            form=d.form,
            question=d.question,
            option_a=d.options[0],
            option_b=d.options[1],
            option_c=d.options[2],
            option_d=d.options[3],
            answer=d.answer,
            source_a=d.sources[0],
            source_b=d.sources[1],
            source_c=d.sources[2],
            source_d=d.sources[3],
            supporting_signal_ids=d.supporting_signal_ids,
            context_chars=0,
        )


@pytest.mark.parametrize("index", [0, 1])
def test_no_question_names_the_trait(corpus, index):
    for d in all_drafts(full_corpus(corpus), index):
        assert banned_words_in(d.question) == ()
        assert matched_params(d.question, BIAS_PARAMS) == ()
        assert "—" not in d.question
