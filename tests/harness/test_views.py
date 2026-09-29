import json
from datetime import date

import pytest

from pm_traitbench.enums import ProbeForm, ProbeType, RuleScope
from pm_traitbench.errors import HarnessError
from pm_traitbench.gates.gate2.transcript import render_pm
from pm_traitbench.harness.protocol import PublicProbe, PublicProfile, PublicSession
from pm_traitbench.harness.views import (
    pm_replays,
    public_probe,
    public_profile,
    public_session,
)
from tests.harness.fixtures import persona_row, probe_row, rule_row, session_row

PM = "pm_001"
D1, D2, D3 = date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19)


def _open_probe(n: int, day: date):
    return probe_row(
        PM,
        n,
        day,
        form=ProbeForm.OPEN,
        probe_type=ProbeType.ROUTINE_QUESTION,
        options=(),
        answer="reference",
        trait_id=None,
    )


def test_public_model_field_sets_are_exact():
    assert set(PublicProfile.model_fields) == {"pm_id", "mandate", "self_description", "rules"}
    assert set(PublicSession.model_fields) == {
        "session_id",
        "date",
        "kind",
        "turns",
        "idea_rules",
    }
    assert set(PublicProbe.model_fields) == {"probe_id", "form", "question", "options"}


def test_no_hidden_name_reaches_public_objects():
    rules = [
        rule_row(PM, "r_01", RuleScope.PM),
        rule_row(PM, "r_02", RuleScope.IDEA, "ti_001"),
    ]
    probe = probe_row(PM, 1, D1, trait_id="t_07")
    open_probe = _open_probe(2, D1)
    session = session_row(PM, "s_pm001_2026-01-05_a", D1, ("ti_001",))
    dumps = [
        public_profile(persona_row(PM), rules).model_dump_json(),
        public_session(session, rules).model_dump_json(),
        public_probe(probe).model_dump_json(),
        public_probe(open_probe).model_dump_json(),
    ]
    forbidden_keys = {
        "typicality",
        "anti_typical",
        "market_seed",
        "bias_flag",
        "source_a",
        "supporting_signal_ids",
        "trait_id",
        "checkpoint_label",
        "answer",
        "probe_type",
        "context_chars",
        "split",
    }

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    for dump in dumps:
        assert not forbidden_keys & set(keys(json.loads(dump)))
        for name in ("typicality", "market_seed", "bias_flag", "source_", "supporting_signal"):
            assert name not in dump
        assert "t_07" not in dump
        assert "reference" not in dump


def test_profile_keeps_pm_scope_rules_of_its_pm_only():
    rules = [
        rule_row("pm_002", "r_01", RuleScope.PM),
        rule_row(PM, "r_03", RuleScope.PM),
        rule_row(PM, "r_02", RuleScope.IDEA, "ti_001"),
        rule_row(PM, "r_01", RuleScope.PM),
    ]
    profile = public_profile(persona_row(PM), rules)
    assert [r.rule_id for r in profile.rules] == ["r_01", "r_03"]
    assert all(r.pm_id == PM and r.scope == RuleScope.PM for r in profile.rules)
    assert profile.self_description == "I run a disciplined, rules-based book."


def test_session_carries_rules_of_discussed_ideas_only():
    rules = [
        rule_row(PM, "r_03", RuleScope.IDEA, "ti_003"),
        rule_row(PM, "r_02", RuleScope.IDEA, "ti_002"),
        rule_row(PM, "r_01", RuleScope.IDEA, "ti_001"),
        rule_row(PM, "r_04", RuleScope.PM),
    ]
    session = session_row(PM, "s_pm001_2026-01-05_a", D1, ("ti_001", "ti_003"))
    view = public_session(session, rules)
    assert [r.rule_id for r in view.idea_rules] == ["r_01", "r_03"]
    assert view.turns == session.turns


def test_public_probe_drops_null_options():
    assert public_probe(probe_row(PM, 1, D1, options=("x", "y", "z"))).options == ("x", "y", "z")
    assert public_probe(_open_probe(2, D1)).options == ()


def test_pm_replays_groups_probes_by_checkpoint_and_sorts():
    probes = [
        probe_row(PM, 3, D2),
        probe_row(PM, 2, D1),
        probe_row(PM, 1, D1),
    ]
    sessions = [
        session_row(PM, "s_pm001_2026-01-12_a", D2),
        session_row(PM, "s_pm001_2026-01-05_b", D1),
        session_row(PM, "s_pm001_2026-01-05_a", D1),
    ]
    replays = pm_replays([persona_row(PM)], [], sessions, probes)
    replay = replays[PM]
    assert [c.day for c in replay.checkpoints] == [D1, D2]
    assert [[p.probe_id for p in c.probes] for c in replay.checkpoints] == [
        ["p_pm001_0001", "p_pm001_0002"],
        ["p_pm001_0003"],
    ]
    assert [s.session_id for s in replay.sessions] == [
        "s_pm001_2026-01-05_a",
        "s_pm001_2026-01-05_b",
        "s_pm001_2026-01-12_a",
    ]


def test_pm_replays_raises_on_missing_persona():
    with pytest.raises(HarnessError):
        pm_replays([persona_row("pm_002")], [], [], [probe_row(PM, 1, D1)])


def test_render_pm_accepts_public_sessions():
    sessions = [
        session_row(PM, "s_pm001_2026-01-12_a", D2),
        session_row(PM, "s_pm001_2026-01-05_a", D1),
    ]
    views = [public_session(s, []) for s in sessions]
    assert render_pm(views) == render_pm(sessions)
