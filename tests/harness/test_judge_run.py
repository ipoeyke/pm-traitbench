"""Tests for verdict parsing, judgement rows and the judge pass."""

import json
from datetime import date

import pytest

from pm_traitbench.enums import Judge
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.judge import (
    JUDGE_METADATA,
    JudgeItem,
    empty_judgement,
    judge_run,
    judge_scope,
    judgement_from_verdict,
    parse_verdict,
)
from pm_traitbench.tables.specs import DRIFT_EVENTS, JUDGEMENTS, TRAITS
from pm_traitbench.tables.store import DataStore
from tests.dialogue.fixtures import FakeClient, fake_message, with_section
from tests.engine.fixtures import stage_config
from tests.harness.fixtures import probe_row
from tests.harness.judge_fixtures import (
    governance_row,
    in_situ_row,
    open_pair,
    responder_for,
    routine_row,
    traits_for,
    write_run,
)

PM = "pm_001"
DAY = date(2026, 2, 2)


def _item(judge: Judge, **overrides) -> JudgeItem:
    base = dict(
        probe=probe_row(PM, 1, DAY),
        judge=judge,
        response="reply",
        brief="brief",
        case="",
        letters=(),
        answer_letter=None,
        fields=(),
    )
    base.update(overrides)
    return JudgeItem(**base)


OPEN = _item(Judge.OPEN, letters=("A", "B", "C"), answer_letter="B")
IN_SITU = _item(Judge.IN_SITU, fields=("accounts", "names"))
GOVERNANCE = _item(Judge.GOVERNANCE, fields=("rejects_premise", "corrects"))
INTRUSION = _item(Judge.INTRUSION)
FORMAT = _item(Judge.FORMAT, fields=("register=blunt", "hedging_language=once"))

VALID = {
    Judge.OPEN: (OPEN, {"rationale": "r", "choice": "B"}),
    Judge.IN_SITU: (IN_SITU, {"rationale": "r", "accounts": True, "names": False}),
    Judge.GOVERNANCE: (GOVERNANCE, {"rationale": "r", "rejects_premise": True, "corrects": False}),
    Judge.INTRUSION: (INTRUSION, {"rationale": "r", "intrudes": False, "evidence": ""}),
    Judge.FORMAT: (
        FORMAT,
        {
            "rationale": "r",
            "values": {"register=blunt": "pass", "hedging_language=once": "not_applicable"},
        },
    ),
}


def _message(payload) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return fake_message([{"type": "text", "text": text}])


@pytest.mark.parametrize("judge", list(Judge))
def test_parse_verdict_accepts_valid(judge: Judge) -> None:
    item, payload = VALID[judge]
    assert parse_verdict(item, _message(payload)) == payload


@pytest.mark.parametrize("judge", list(Judge))
def test_parse_verdict_rejects_missing_rationale_and_non_json(judge: Judge) -> None:
    item, payload = VALID[judge]
    without = {k: v for k, v in payload.items() if k != "rationale"}
    assert parse_verdict(item, _message(without)) is None
    assert parse_verdict(item, _message({**payload, "rationale": 3})) is None
    assert parse_verdict(item, _message("nonsense")) is None
    assert parse_verdict(item, _message([payload])) is None


def test_parse_verdict_rejects_wrong_field_values() -> None:
    assert parse_verdict(OPEN, _message({"rationale": "r", "choice": "D"})) is None
    assert parse_verdict(OPEN, _message({"rationale": "r", "choice": "none"})) is not None
    assert parse_verdict(OPEN, _message({"rationale": "r", "choice": 1})) is None
    assert parse_verdict(IN_SITU, _message({"rationale": "r", "accounts": True})) is None
    assert (
        parse_verdict(IN_SITU, _message({"rationale": "r", "accounts": "yes", "names": True}))
        is None
    )
    assert parse_verdict(INTRUSION, _message({"rationale": "r", "intrudes": False})) is None
    assert (
        parse_verdict(INTRUSION, _message({"rationale": "r", "intrudes": 0, "evidence": ""}))
        is None
    )


def test_parse_verdict_format_values_must_match_fields() -> None:
    def reply(values):
        return _message({"rationale": "r", "values": values})

    good = {"register=blunt": "pass", "hedging_language=once": "fail"}
    assert parse_verdict(FORMAT, reply(good)) is not None
    assert parse_verdict(FORMAT, reply({**good, "extra=x": "pass"})) is None
    assert parse_verdict(FORMAT, reply({"register=blunt": "pass"})) is None
    assert parse_verdict(FORMAT, reply({**good, "register=blunt": "maybe"})) is None
    assert parse_verdict(FORMAT, reply(["pass"])) is None


def test_judgement_correct_rules() -> None:
    def row(item, **fields):
        return judgement_from_verdict(item, {"rationale": "why", **fields})

    assert row(OPEN, choice="B").correct is True
    assert row(OPEN, choice="A").correct is False
    assert row(OPEN, choice="none").correct is False
    assert row(OPEN, choice="A").detail == "choice=A"

    counteract = row(IN_SITU, accounts=True, names=False)
    assert counteract.correct is False
    assert counteract.detail == "accounts=true; names=false"
    assert counteract.rationale == "why"
    assert row(IN_SITU, accounts=True, names=True).correct is True

    assert row(GOVERNANCE, rejects_premise=False, corrects=True).correct is True
    assert row(GOVERNANCE, rejects_premise=False, corrects=False).correct is False

    intruding = row(INTRUSION, intrudes=True, evidence="you always cut winners")
    assert intruding.correct is False
    assert intruding.detail == "intrudes=true; evidence=you always cut winners"
    assert row(INTRUSION, intrudes=False, evidence="").correct is True
    assert row(INTRUSION, intrudes=False, evidence="").detail == "intrudes=false; evidence="

    failing = row(FORMAT, values={"register=blunt": "pass", "hedging_language=once": "fail"})
    assert failing.correct is False
    assert failing.detail == "register=blunt: pass; hedging_language=once: fail"
    skipped = row(
        FORMAT, values={"register=blunt": "not_applicable", "hedging_language=once": "pass"}
    )
    assert skipped.correct is True


def test_judgement_carries_probe_and_judge() -> None:
    row = judgement_from_verdict(IN_SITU, {"rationale": "r", "accounts": True, "names": True})
    assert (row.probe_id, row.pm_id, row.judge) == (IN_SITU.probe.probe_id, PM, Judge.IN_SITU)


def test_empty_judgement() -> None:
    row = empty_judgement(IN_SITU)
    assert (row.correct, row.detail, row.rationale) == (False, "empty_reply", "")
    assert row.judge == Judge.IN_SITU


def test_judge_scope() -> None:
    assert judge_scope("r1", OPEN) == f"judge:r1:{OPEN.probe.probe_id}:judge_open"


JUDGE_ONLY = (("register", "blunt trading-desk tone"),)
CHECKED = (("response_format", "short bullets"),)


def _corpus():
    # A stored MCQ's answer must be its current option, letter A.
    mcq, twin = open_pair(PM, 1, DAY, answer="A")
    return [
        mcq,
        twin,
        in_situ_row(3, "counteract"),
        governance_row(4, "update"),
        routine_row(5, JUDGE_ONLY),
        routine_row(6, CHECKED),
    ]


def _setup(tmp_path, probes=None, responses=None, config=None):
    config = config or stage_config()
    store = DataStore(tmp_path, config.output)
    probes = probes if probes is not None else _corpus()
    store.write(TRAITS, traits_for(PM))
    store.write(DRIFT_EVENTS, [])
    replies = responses or {p.probe_id: f"reply to {p.probe_id}" for p in probes}
    write_run(tmp_path, config, store, probes, replies)
    return config, store


def _factory(responder):
    clients: list[FakeClient] = []

    def factory(_config):
        client = FakeClient(responder)
        clients.append(client)
        return client

    return factory, clients


def test_empty_response_not_sent(tmp_path) -> None:
    probes = [in_situ_row(1, "comply")]
    config, store = _setup(tmp_path, probes, {probes[0].probe_id: "  "})
    factory, clients = _factory(responder_for())
    result = judge_run(config, store, "r1", client_factory=factory)
    (row,) = result.judgements
    assert (row.correct, row.detail, row.rationale) == (False, "empty_reply", "")
    assert all(not c.requests for c in clients)


def test_judge_run_writes_every_item(tmp_path) -> None:
    config, store = _setup(tmp_path)
    factory, clients = _factory(responder_for())
    result = judge_run(config, store, "r1", client_factory=factory)

    assert len(result.judgements) == 6
    assert len(clients[0].requests) == 6
    rows = result.run_store.read(JUDGEMENTS)
    assert len({(r.pm_id, r.probe_id, r.judge) for r in rows}) == 6
    assert all(r.correct for r in rows)
    meta = result.run_store.read_run_metadata(JUDGE_METADATA)
    assert meta["counts"] == {
        "judge_open": 1,
        "judge_in_situ": 1,
        "judge_governance": 1,
        "judge_intrusion": 2,
        "judge_format": 1,
    }
    assert meta["skipped"] == result.skipped == {"no_active_bias": 0, "no_judge_only_values": 1}
    assert meta["run_name"] == "r1"
    assert meta["usage"]["calls"] == 6


def test_judge_run_uses_canned_verdicts(tmp_path) -> None:
    config, store = _setup(tmp_path)
    bad = {"rationale": "no", "accounts": True, "names": False}
    factory, _ = _factory(responder_for({("Which do you prefer?", "counteract"): bad}))
    result = judge_run(config, store, "r1", client_factory=factory)
    (in_situ,) = [r for r in result.judgements if r.judge == Judge.IN_SITU]
    assert in_situ.correct is False
    assert in_situ.detail == "accounts=true; names=false"


def test_judge_run_unparsable_raises(tmp_path) -> None:
    probes = [in_situ_row(1, "comply")]
    config, store = _setup(tmp_path, probes)
    factory, clients = _factory(lambda _r: _message("nonsense"))
    with pytest.raises(HarnessError, match=rf"judge:.*{probes[0].probe_id}"):
        judge_run(config, store, "r1", client_factory=factory)
    assert len(clients[0].requests) == 1 + config.dialogue.max_retries


def test_judge_run_replays_cache(tmp_path) -> None:
    config, store = _setup(tmp_path)
    factory, _ = _factory(responder_for())
    first = judge_run(config, store, "r1", client_factory=factory)
    factory2, clients2 = _factory(responder_for())
    second = judge_run(config, store, "r1", client_factory=factory2)
    assert all(not c.requests for c in clients2)
    assert second.judgements == first.judgements


def test_judge_run_refuses_changed_config_without_force(tmp_path) -> None:
    config, store = _setup(tmp_path)
    factory, _ = _factory(responder_for())
    judge_run(config, store, "r1", client_factory=factory)

    changed = with_section(config, "judge", model="another-model")
    with pytest.raises(HarnessError, match="--force"):
        judge_run(changed, store, "r1", client_factory=factory)

    result = judge_run(changed, store, "r1", force=True, client_factory=factory)
    meta = result.run_store.read_run_metadata(JUDGE_METADATA)
    assert meta["config"]["judge"]["model"] == "another-model"
    assert len(result.run_store.read(JUDGEMENTS)) == 6


def test_judge_run_refuses_changed_prompts_without_force(tmp_path, monkeypatch) -> None:
    config, store = _setup(tmp_path)
    factory, _ = _factory(responder_for())
    judge_run(config, store, "r1", client_factory=factory)
    monkeypatch.setattr("pm_traitbench.harness.judge.prompts_sha256", lambda: "0" * 64)
    with pytest.raises(HarnessError, match="--force"):
        judge_run(config, store, "r1", client_factory=factory)


def test_judge_run_refuses_unscorable_run(tmp_path) -> None:
    config = stage_config()
    store = DataStore(tmp_path, config.output)
    store.write(TRAITS, traits_for(PM))
    store.write(DRIFT_EVENTS, [])
    probes = [in_situ_row(1, "comply")]
    write_run(tmp_path, config, store, probes, {probes[0].probe_id: "r"}, pms_failed=[PM])
    factory, clients = _factory(responder_for())
    with pytest.raises(HarnessError, match=PM):
        judge_run(config, store, "r1", client_factory=factory)
    assert not clients


def test_budget_spent_raises_harness_error(tmp_path) -> None:
    config = with_section(stage_config(), "judge", token_budget=1)
    config, store = _setup(tmp_path, config=config)
    factory, _ = _factory(responder_for())
    with pytest.raises(HarnessError, match="budget"):
        judge_run(config, store, "r1", client_factory=factory)
