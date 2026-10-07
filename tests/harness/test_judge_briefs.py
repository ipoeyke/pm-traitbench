"""Tests for judge item selection, briefs, schemas and requests."""

from datetime import date

import pytest

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.config import JudgeConfig
from pm_traitbench.enums import GovernanceKind, InSituCase, Judge
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness import judge as judge_module
from pm_traitbench.harness.checks import load_check_map
from pm_traitbench.harness.judge import (
    INSTRUCTIONS,
    SYSTEM_PROMPT,
    JudgeInputs,
    active_bias_phrases,
    build_request,
    governance_kind,
    in_situ_case,
    judge_only_values,
    mcq_index,
    prompts_sha256,
    schema_for,
    select_items,
    sibling_mcq,
)
from pm_traitbench.tables.schema import probe_id
from tests.harness.judge_fixtures import (
    GOVERNANCE_ANSWERS,
    IN_SITU_ANSWERS,
    bank,
    dormant_event,
    governance_row,
    in_situ_row,
    open_pair,
    presence_row,
    revive_event,
    routine_row,
    traits_for,
)

PM = "pm_001"
DAY = date(2026, 2, 2)
JUDGE_ONLY = (("register", "blunt trading-desk tone"),)
HEDGE = (("hedging_language", "flag uncertainty once, then commit to a view"),)
CHECKED = (("response_format", "short bullets"),)


def _check_map():
    return load_check_map(load_catalogue())


def _inputs(probes, drift_events=(), traits=None):
    responses = {p.probe_id: f"reply to {p.probe_id}" for p in probes}
    return JudgeInputs(
        probes=probes,
        responses=responses,
        traits=traits if traits is not None else traits_for(PM),
        drift_events=drift_events,
        bank=bank(),
        check_map=_check_map(),
    )


def test_in_situ_case_and_governance_kind() -> None:
    for case, answer in IN_SITU_ANSWERS.items():
        assert in_situ_case(answer) == InSituCase(case)
    for kind, answer in GOVERNANCE_ANSWERS.items():
        assert governance_kind(answer) == GovernanceKind(kind)
    with pytest.raises(HarnessError, match="bogus: x"):
        in_situ_case("bogus: x")
    with pytest.raises(HarnessError, match="premise rejected: whatever"):
        governance_kind("premise rejected: whatever")


def test_judge_only_values() -> None:
    row = routine_row(1, (*JUDGE_ONLY, *HEDGE, *CHECKED))
    assert judge_only_values(row, _check_map()) == (*JUDGE_ONLY, *HEDGE)
    with pytest.raises(HarnessError, match="no check"):
        judge_only_values(routine_row(2, (("response_format", "made up"),)), _check_map())


def test_sibling_mcq_found_and_errors() -> None:
    mcq, twin = open_pair(PM, 1, DAY)
    assert sibling_mcq(twin, mcq_index([mcq, twin])) == mcq
    with pytest.raises(HarnessError, match=twin.probe_id):
        sibling_mcq(twin, mcq_index([twin]))
    other = mcq.model_copy(update={"probe_id": probe_id(PM, 9)})
    with pytest.raises(HarnessError, match=twin.probe_id):
        sibling_mcq(twin, mcq_index([mcq, other, twin]))
    wrong_answer = twin.model_copy(update={"answer": "not an option"})
    with pytest.raises(HarnessError, match="not the correct option"):
        sibling_mcq(wrong_answer, mcq_index([mcq, twin]))


def test_active_bias_phrases_respects_dormancy() -> None:
    traits = traits_for(PM)
    lines = bank().biases
    both = (lines["loss_aversion_lambda"].behaviour, lines["disposition_ratio"].behaviour)
    assert active_bias_phrases(PM, DAY, traits, [], bank()) == both
    dormant = dormant_event(PM, "t_01", DAY)
    assert active_bias_phrases(PM, DAY, traits, [dormant], bank()) == both[1:]
    revive = revive_event(PM, "t_01", date(2026, 2, 3))
    assert active_bias_phrases(PM, date(2026, 2, 3), traits, [dormant, revive], bank()) == both


def test_active_bias_phrases_ignores_other_pms_events() -> None:
    traits = traits_for(PM)
    other = dormant_event("pm_002", "t_01", DAY)
    both = active_bias_phrases(PM, DAY, traits, [], bank())
    assert active_bias_phrases(PM, DAY, traits, [other], bank()) == both


def test_active_bias_phrases_unknown_param_is_harness_error() -> None:
    bad = bank().model_copy(update={"biases": {}})
    with pytest.raises(HarnessError):
        active_bias_phrases(PM, DAY, traits_for(PM), [], bad)


def _corpus():
    mcq, twin = open_pair(PM, 1, DAY)
    return [
        mcq,
        twin,
        in_situ_row(3, "counteract"),
        governance_row(4, "update"),
        routine_row(5, JUDGE_ONLY),
        routine_row(6, CHECKED),
        presence_row(7),
    ]


def test_select_items_per_type() -> None:
    selection = select_items(_inputs(_corpus()))
    judges = [item.judge for item in selection.items]
    assert sorted(judges) == sorted(
        [
            Judge.OPEN,
            Judge.IN_SITU,
            Judge.GOVERNANCE,
            Judge.INTRUSION,
            Judge.INTRUSION,
            Judge.FORMAT,
        ]
    )
    by_judge = {item.judge: item for item in selection.items}
    open_item = by_judge[Judge.OPEN]
    assert open_item.letters == ("A", "B", "C")
    assert open_item.answer_letter == "B"
    assert "A. sell now" in open_item.brief
    assert by_judge[Judge.IN_SITU].case == InSituCase.COUNTERACT
    assert by_judge[Judge.IN_SITU].fields == ("accounts", "names")
    assert by_judge[Judge.GOVERNANCE].case == GovernanceKind.UPDATE
    assert by_judge[Judge.GOVERNANCE].fields == ("rejects_premise", "corrects")
    assert by_judge[Judge.FORMAT].fields == ("register=blunt trading-desk tone",)
    assert by_judge[Judge.FORMAT].brief == "- register=blunt trading-desk tone"
    assert all(item.probe.probe_id != probe_id(PM, 7) for item in selection.items)
    assert selection.skipped == {"no_active_bias": 0, "no_judge_only_values": 1}


def test_select_items_field_names_per_case() -> None:
    probes = [in_situ_row(1, "comply"), in_situ_row(2, "decline")]
    items = select_items(_inputs(probes)).items
    assert [i.fields for i in items] == [("honours",), ("refuses", "gives_reason")]


def test_select_items_in_probe_id_order() -> None:
    probes = [in_situ_row(3, "comply"), in_situ_row(1, "comply")]
    items = select_items(_inputs(probes)).items
    assert [i.probe.probe_id for i in items] == [probe_id(PM, 1), probe_id(PM, 3)]


def test_select_items_skips_intrusion_without_active_bias() -> None:
    traits = [t.model_copy(update={"active": False}) for t in traits_for(PM)]
    selection = select_items(_inputs([routine_row(1, CHECKED)], traits=traits))
    assert selection.items == ()
    assert selection.skipped == {"no_active_bias": 1, "no_judge_only_values": 1}


def test_select_items_names_probe_in_rubric_errors() -> None:
    bad_in_situ = in_situ_row(1, "comply").model_copy(update={"answer": "bogus: x"})
    with pytest.raises(HarnessError, match=f"^probe {probe_id(PM, 1)}: unknown in-situ rubric"):
        select_items(_inputs([bad_in_situ]))
    bad_gov = governance_row(2, "update").model_copy(update={"answer": "premise rejected: x"})
    with pytest.raises(HarnessError, match=f"^probe {probe_id(PM, 2)}: unknown governance rubric"):
        select_items(_inputs([bad_gov]))


def test_select_items_missing_response_raises() -> None:
    inputs = _inputs([in_situ_row(1, "comply")])
    empty = JudgeInputs(**{**inputs.__dict__, "responses": {}})
    with pytest.raises(HarnessError, match=probe_id(PM, 1)):
        select_items(empty)


def test_select_items_keeps_empty_response() -> None:
    inputs = _inputs([in_situ_row(1, "comply")])
    blank = JudgeInputs(**{**inputs.__dict__, "responses": {probe_id(PM, 1): ""}})
    assert len(select_items(blank).items) == 1


def test_build_request_content() -> None:
    config = JudgeConfig(model="m", max_tokens=512)
    items = {i.judge: i for i in select_items(_inputs(_corpus())).items}
    assert set(items) == set(Judge)
    for judge, item in items.items():
        request = build_request(item, config)
        user = request["messages"][0]["content"]
        assert item.probe.question in user
        assert item.response in user
        assert item.brief in user
        assert user.startswith("Question put to the copilot:\n")
        assert INSTRUCTIONS[judge] in request["system"]
        assert SYSTEM_PROMPT in request["system"]
        assert "cache_control" not in request
        assert request["model"] == "m"
        assert request["max_tokens"] == 512
        assert request["output_config"]["effort"] == config.effort.value
        schema = request["output_config"]["format"]["schema"]
        assert schema == schema_for(item)
        assert list(schema["properties"])[0] == "rationale"
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
    open_schema = schema_for(items[Judge.OPEN])
    assert open_schema["properties"]["choice"]["enum"] == ["A", "B", "C", "none"]
    fmt = schema_for(items[Judge.FORMAT])["properties"]["values"]
    assert fmt["required"] == list(items[Judge.FORMAT].fields)
    assert fmt["additionalProperties"] is False
    intrusion = schema_for(items[Judge.INTRUSION])["properties"]
    assert intrusion["intrudes"] == {"type": "boolean"}
    assert intrusion["evidence"] == {"type": "string"}


def test_prompts_sha256_stable_and_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    before = prompts_sha256()
    assert prompts_sha256() == before
    monkeypatch.setattr(judge_module, "INSTRUCTIONS", {**INSTRUCTIONS, Judge.OPEN: "changed"})
    assert prompts_sha256() != before
