"""Tests for the no-memory and full-context baselines: request content, caching and lifecycle."""

import json
from datetime import date

import pytest

from pm_traitbench.dialogue.prompts import read_advisor_prompt
from pm_traitbench.enums import ProbeForm, RuleScope
from pm_traitbench.errors import DialogueBudgetError, HarnessError
from pm_traitbench.gates.gate2.prompt_parts import mandate_line, pm_rules_section
from pm_traitbench.gates.gate2.transcript import render_pm
from pm_traitbench.harness.baselines import (
    MCQ_INSTRUCTION,
    NO_TOOLS_NOTE,
    OPEN_INSTRUCTION,
    FullContext,
    NoMemory,
    baseline_factory,
    render_probe,
)
from pm_traitbench.harness.protocol import PublicProbe, PublicProfile, PublicSession
from pm_traitbench.harness.runner import run_dir, run_sut
from pm_traitbench.tables.specs import PROBES, RESPONSES
from tests.dialogue.fixtures import FakeClient, fake_message
from tests.engine.fixtures import stage_config
from tests.gates.gate2.fixtures import session_of
from tests.harness.fixtures import persona_row, rule_row, validated_corpus_with_probes

PM = "pm_001"
AS_OF = date(2026, 2, 3)


def _responder(request):
    return fake_message([{"type": "text", "text": json.dumps({"answer": "B"})}])


def _plain_responder(request):
    return fake_message([{"type": "text", "text": "B"}])


def _profile() -> PublicProfile:
    persona = persona_row(PM)
    return PublicProfile(
        pm_id=PM,
        mandate=persona.mandate,
        self_description=persona.stated_profile.self_description,
        rules=(rule_row(PM, "r_01", RuleScope.PM, text="Cut losers at 5 percent."),),
    )


def _session(day: date, letter: int, text: str, idea_rules=()) -> PublicSession:
    row = session_of(PM, day, [text], letter=letter)
    return PublicSession(
        session_id=row.session_id, date=row.date, turns=row.turns, idea_rules=tuple(idea_rules)
    )


def _sessions() -> list[PublicSession]:
    idea_rule = rule_row(
        PM, "r_02", RuleScope.IDEA, trade_idea_id="ti_001", text="Trim at 10 percent."
    )
    return [
        _session(date(2026, 1, 6), 0, "first note", [idea_rule]),
        _session(date(2026, 1, 20), 0, "second note", [idea_rule]),
    ]


def _probe(form: ProbeForm = ProbeForm.MCQ) -> PublicProbe:
    options = ("x", "y", "z") if form == ProbeForm.MCQ else ()
    return PublicProbe(probe_id="p_pm_001_01", form=form, question="Which?", options=options)


def _make(cls, tmp_path, responder=_responder, config=None):
    config = config or stage_config()
    clients: list[FakeClient] = []

    def client_factory(_config):
        client = FakeClient(responder)
        clients.append(client)
        return client

    sut = cls(config, _profile(), tmp_path, "r1", client_factory)
    return sut, clients, config


def test_render_probe_mcq_and_open() -> None:
    assert render_probe(_probe()) == f"Which?\nA. x\nB. y\nC. z\n\n{MCQ_INSTRUCTION}"
    assert render_probe(_probe(ProbeForm.OPEN)) == f"Which?\n\n{OPEN_INSTRUCTION}"


def test_full_context_sends_observed_transcript(tmp_path) -> None:
    sut, clients, config = _make(FullContext, tmp_path)
    sessions = _sessions()
    for session in sessions:
        sut.observe(session)

    assert sut.answer(AS_OF, _probe()) == "B"

    (request,) = clients[0].requests
    memory, tail = request["messages"][0]["content"]
    assert memory["text"].startswith(render_pm(sessions))
    assert "Idea rules discussed:\nTrim at 10 percent." in memory["text"]
    assert memory["text"].count("Trim at 10 percent.") == 1
    assert tail["text"] == f"Today is {AS_OF.isoformat()}.\n\n{render_probe(_probe())}"
    (system_block,) = request["system"]
    system = system_block["text"]
    profile = _profile()
    assert read_advisor_prompt(config.dialogue.advisor_prompt_path) in system
    assert NO_TOOLS_NOTE in system
    assert "tools" not in request
    assert mandate_line(profile.mandate) in system
    assert pm_rules_section(profile.rules) in system
    sut.close()


def test_full_context_includes_only_observed_sessions(tmp_path) -> None:
    sut, clients, _ = _make(FullContext, tmp_path)
    first, second = _sessions()
    sut.observe(first)

    sut.answer(AS_OF, _probe())

    text = json.dumps(clients[0].requests[0])
    assert first.session_id in text
    assert second.session_id not in text
    sut.close()


def test_no_memory_ignores_sessions(tmp_path) -> None:
    sut, clients, _ = _make(NoMemory, tmp_path)
    for session in _sessions():
        sut.observe(session)

    sut.answer(AS_OF, _probe())

    request = clients[0].requests[0]
    (tail,) = request["messages"][0]["content"]
    assert "Session " not in tail["text"]
    assert "Idea rules discussed" not in tail["text"]
    assert "Cut losers at 5 percent." in request["system"][0]["text"]
    sut.close()


def test_request_uses_harness_config(tmp_path) -> None:
    sut, clients, config = _make(FullContext, tmp_path)

    sut.answer(AS_OF, _probe())

    request = clients[0].requests[0]
    assert request["model"] == config.harness.model
    assert request["max_tokens"] == config.harness.max_answer_tokens
    assert request["output_config"]["effort"] == config.harness.effort.value
    sut.close()


def test_cache_breakpoints_end_the_shared_prefixes(tmp_path) -> None:
    """System and memory are marked so a PM's probes read them; the probe block is not."""
    cached = {"type": "ephemeral"}
    sut, clients, _ = _make(FullContext, tmp_path)
    for session in _sessions():
        sut.observe(session)
    sut.answer(AS_OF, _probe())
    request = clients[0].requests[0]
    assert "cache_control" not in request
    assert request["system"][0]["cache_control"] == cached
    memory, tail = request["messages"][0]["content"]
    assert memory["cache_control"] == cached
    assert "cache_control" not in tail
    sut.close()

    sut, clients, _ = _make(NoMemory, tmp_path)
    sut.answer(AS_OF, _probe())
    request = clients[0].requests[0]
    assert "cache_control" not in request
    assert request["system"][0]["cache_control"] == cached
    (tail,) = request["messages"][0]["content"]
    assert "cache_control" not in tail
    sut.close()


def test_unparsable_reply_retries_then_returns_empty(tmp_path) -> None:
    sut, clients, config = _make(FullContext, tmp_path, responder=_plain_responder)

    assert sut.answer(AS_OF, _probe()) == ""

    assert len(clients[0].requests) == config.dialogue.max_retries + 1
    sut.close()


def test_spent_token_budget_still_raises(tmp_path) -> None:
    base = stage_config()
    config = base.model_copy(
        update={"harness": base.harness.model_copy(update={"pm_token_budget": 1})}
    )
    sut, _, _ = _make(FullContext, tmp_path, config=config)
    assert sut.answer(AS_OF, _probe()) == "B"

    other = PublicProbe(probe_id="q_other", form=ProbeForm.OPEN, question="Else?", options=())
    with pytest.raises(DialogueBudgetError):
        sut.answer(AS_OF, other)
    sut.close()


def test_second_answer_hits_cache(tmp_path) -> None:
    first, _, _ = _make(FullContext, tmp_path)
    for session in _sessions():
        first.observe(session)
    first.answer(AS_OF, _probe())
    first.close()

    second, clients, _ = _make(FullContext, tmp_path)
    for session in _sessions():
        second.observe(session)

    assert second.answer(AS_OF, _probe()) == "B"
    assert all(not client.requests for client in clients)
    second.close()


def test_close_closes_client(tmp_path) -> None:
    sut, clients, _ = _make(NoMemory, tmp_path)
    sut.answer(AS_OF, _probe())

    sut.close()

    assert clients[0].closed is True


def test_close_without_answer_is_safe(tmp_path) -> None:
    sut, clients, _ = _make(NoMemory, tmp_path)

    sut.close()

    assert clients == []


def test_unknown_baseline_raises(tmp_path) -> None:
    with pytest.raises(HarnessError, match="unknown baseline"):
        baseline_factory("nope", stage_config(), tmp_path, "r1")


def test_baseline_runs_through_runner(tmp_path, fixture_market, neutral_pm, monkeypatch) -> None:
    config, store = validated_corpus_with_probes(tmp_path, fixture_market, neutral_pm, monkeypatch)
    factory = baseline_factory(
        "full-context",
        config,
        run_dir(store.data_dir, "r1"),
        "r1",
        client_factory=lambda c: FakeClient(_responder),
    )

    result = run_sut(config, store, factory, sut_name="full-context", run_name="r1", workers=2)

    assert result.failed == {}
    responses = result.run_store.read(RESPONSES)
    assert sorted(r.probe_id for r in responses) == sorted(p.probe_id for p in store.read(PROBES))


def test_default_client_factory_resolves_at_call_time(tmp_path, monkeypatch) -> None:
    from pm_traitbench.harness import baselines

    made: list[FakeClient] = []

    def fake_anthropic(max_concurrency, max_retries):
        client = FakeClient(_responder)
        made.append(client)
        return client

    monkeypatch.setattr(baselines, "AnthropicClient", fake_anthropic)
    sut = baseline_factory("no-memory", stage_config(), tmp_path, "r1")(_profile())

    assert sut.answer(AS_OF, _probe()) == "B"
    sut.close()
    assert len(made) == 1
