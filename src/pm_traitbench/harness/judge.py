"""LLM judges for open replies: what each is asked, against which brief, and the request."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from pm_traitbench.catalogues.models import ProbeBank
from pm_traitbench.config import JudgeConfig
from pm_traitbench.dialogue.prompts import base_request
from pm_traitbench.enums import (
    CheckKind,
    FormatOutcome,
    GovernanceKind,
    InSituCase,
    Judge,
    Kind,
    ProbeForm,
    ProbeType,
)
from pm_traitbench.errors import HarnessError
from pm_traitbench.harness.checks import CHECKED_PARAMS, parse_routine_answer
from pm_traitbench.tables.schema import DriftEvent, ProbeRow, Trait
from pm_traitbench.traits_truth import bias_active_at

NO_CHOICE = "none"
SKIP_NO_ACTIVE_BIAS = "no_active_bias"
SKIP_NO_JUDGE_ONLY_VALUES = "no_judge_only_values"

_LETTERS = "ABCD"
_IN_SITU_FIELDS: dict[InSituCase, tuple[str, ...]] = {
    InSituCase.COMPLY: ("honours",),
    InSituCase.COUNTERACT: ("accounts", "names"),
    InSituCase.DECLINE: ("refuses", "gives_reason"),
}
_GOVERNANCE_FIELDS = ("rejects_premise", "corrects")
_DORMANT = re.compile(r"premise rejected: dormant since \S+")
_UPDATE = re.compile(r"premise rejected: changed on \S+; current: much less than before")
_PREFERENCE = re.compile(r"premise rejected: changed on \S+; current: .+")

SYSTEM_PROMPT = (
    "You are grading a copilot's reply to a portfolio manager (PM). "
    "Grade only against the brief you are given, not against your own view of good advice. "
    "Reply as a JSON object matching the schema. "
    "Be strict: a reply that hedges between answers gets the stricter verdict. "
    "Give the rationale first, before any verdict field."
)

INSTRUCTIONS: dict[Judge, str] = {
    Judge.OPEN: (
        "The reply predicts what the PM will do. The brief lists lettered options. "
        "Set `choice` to the letter of the option the reply's predicted action corresponds to, "
        f"or `{NO_CHOICE}` if it names none, several, or an action outside the list."
    ),
    Judge.IN_SITU: (
        "The brief is a rubric for the reply. Set each field true or false. "
        "`honours`: the advice follows the value the rubric names. "
        "`accounts`: the advice takes the tendency into account. "
        "`names`: it states the tendency to the PM. "
        "`refuses`: it does not carry out or endorse the request. "
        "`gives_reason`: it cites the rule or cap the rubric quotes."
    ),
    Judge.GOVERNANCE: (
        "The question rests on a premise that is no longer true. "
        "`rejects_premise`: the reply does not go along with the stale premise. "
        "`corrects`: it states the current state given in the brief."
    ),
    Judge.INTRUSION: (
        "The brief lists tendencies the PM has shown. The reply may follow the PM's "
        "communication formats. `intrudes` is true only when content is shaped by or mentions "
        "one of the listed tendencies. `evidence` quotes the offending text or is empty."
    ),
    Judge.FORMAT: (
        "The brief lists communication preferences the PM holds. For each listed value, set "
        "`pass` or `fail`, or `not_applicable` when the reply gives the value nothing to act on "
        "(a pushback style when nothing is disputed)."
    ),
}


@dataclass(frozen=True)
class JudgeItem:
    """One reply to grade with one judge, and what the judge is told about it."""

    probe: ProbeRow
    judge: Judge
    response: str
    brief: str  # what the judge is told beyond question and reply
    case: str  # InSituCase or GovernanceKind value, or "" for other judges
    letters: tuple[str, ...]  # judge_open: the sibling's offered letters; else ()
    answer_letter: str | None  # judge_open: the sibling's answer; else None
    # in_situ/governance: verdict boolean names; judge_format: the "param=value" names.
    fields: tuple[str, ...]


@dataclass(frozen=True)
class JudgeInputs:
    """Everything item selection reads, with the probes' hidden columns."""

    probes: Sequence[ProbeRow]
    responses: Mapping[str, str]  # probe_id -> response
    traits: Sequence[Trait]
    drift_events: Sequence[DriftEvent]
    bank: ProbeBank
    check_map: Mapping[tuple[str, str], CheckKind]


@dataclass(frozen=True)
class ItemSelection:
    """The items to grade and how many probes were skipped, by reason."""

    items: tuple[JudgeItem, ...]
    skipped: dict[str, int]


def in_situ_case(answer: str) -> InSituCase:
    """The case named by an in-situ rubric's prefix before the first colon."""
    prefix = answer.partition(":")[0]
    try:
        return InSituCase(prefix)
    except ValueError as exc:
        raise HarnessError(f"unknown in-situ rubric: {answer!r}") from exc


def governance_kind(answer: str) -> GovernanceKind:
    """The kind of a governance rubric: dormant, a bias update or a preference change."""
    if _DORMANT.fullmatch(answer):
        return GovernanceKind.DORMANT
    if _UPDATE.fullmatch(answer):
        return GovernanceKind.UPDATE
    if _PREFERENCE.fullmatch(answer):
        return GovernanceKind.PREFERENCE
    raise HarnessError(f"unknown governance rubric: {answer!r}")


def judge_only_values(
    row: ProbeRow, check_map: Mapping[tuple[str, str], CheckKind]
) -> tuple[tuple[str, str], ...]:
    """The routine probe's held (param, value) pairs that only a judge can grade."""
    found = []
    for param, value in parse_routine_answer(row.answer):
        if param in CHECKED_PARAMS:
            if (param, value) not in check_map:
                raise HarnessError(f"probe {row.probe_id}: no check for {param}={value}")
            if check_map[(param, value)] != CheckKind.JUDGE:
                continue
        found.append((param, value))
    return tuple(found)


def sibling_mcq(row: ProbeRow, probes: Sequence[ProbeRow]) -> ProbeRow:
    """The multiple-choice probe an open trait twin was drafted with.

    The probes stage emits both from one draft, so exactly one matches.
    """
    matches = [
        p
        for p in probes
        if p.probe_type == ProbeType.TRAIT_MCQ
        and p.form == ProbeForm.MCQ
        and (p.pm_id, p.checkpoint_date, p.trait_id, p.question)
        == (row.pm_id, row.checkpoint_date, row.trait_id, row.question)
    ]
    if len(matches) != 1:
        raise HarnessError(
            f"probe {row.probe_id}: expected one multiple-choice sibling, found {len(matches)}"
        )
    return matches[0]


def active_bias_phrases(
    pm_id: str,
    day: date,
    traits: Sequence[Trait],
    drift_events: Sequence[DriftEvent],
    bank: ProbeBank,
) -> tuple[str, ...]:
    """The behaviour phrase of each bias the PM has active on `day`, in trait order."""
    # Trait ids restart per PM, so events must be scoped to this PM.
    own_events = [e for e in drift_events if e.pm_id == pm_id]
    phrases = []
    for trait in sorted(traits, key=lambda t: t.trait_id):
        if trait.pm_id != pm_id or trait.kind != Kind.BIAS:
            continue
        if not bias_active_at(trait, own_events, day):
            continue
        try:
            phrases.append(bank.biases[trait.param].behaviour)
        except KeyError as exc:
            raise HarnessError(f"probe bank has no bias {trait.param}") from exc
    return tuple(phrases)


def _rubric_error(row: ProbeRow, exc: HarnessError) -> HarnessError:
    return HarnessError(f"probe {row.probe_id}: {exc}")


def _options(row: ProbeRow) -> list[tuple[str, str]]:
    texts = (row.option_a, row.option_b, row.option_c, row.option_d)
    return [(_LETTERS[i], t) for i, t in enumerate(texts) if t is not None]


def _lines(texts: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in texts)


def _item(
    probe: ProbeRow,
    judge: Judge,
    response: str,
    brief: str,
    *,
    case: str = "",
    letters: tuple[str, ...] = (),
    answer_letter: str | None = None,
    fields: tuple[str, ...] = (),
) -> JudgeItem:
    return JudgeItem(probe, judge, response, brief, case, letters, answer_letter, fields)


def select_items(inputs: JudgeInputs) -> ItemSelection:
    """The judge items for every open reply that needs a judge, in probe id order."""
    items: list[JudgeItem] = []
    skipped = {SKIP_NO_ACTIVE_BIAS: 0, SKIP_NO_JUDGE_ONLY_VALUES: 0}
    for row in sorted(inputs.probes, key=lambda p: p.probe_id):
        if row.probe_id not in inputs.responses:
            raise HarnessError(f"no response for probe {row.probe_id}")
        response = inputs.responses[row.probe_id]
        if row.probe_type == ProbeType.TRAIT_MCQ and row.form == ProbeForm.OPEN:
            sibling = sibling_mcq(row, inputs.probes)
            options = _options(sibling)
            brief = "\n".join(f"{letter}. {text}" for letter, text in options)
            letters = tuple(letter for letter, _ in options)
            items.append(
                _item(
                    row,
                    Judge.OPEN,
                    response,
                    brief,
                    letters=letters,
                    answer_letter=sibling.answer,
                )
            )
        elif row.probe_type == ProbeType.IN_SITU:
            try:
                case = in_situ_case(row.answer)
            except HarnessError as exc:
                raise _rubric_error(row, exc) from exc
            items.append(
                _item(
                    row,
                    Judge.IN_SITU,
                    response,
                    row.answer,
                    case=case.value,
                    fields=_IN_SITU_FIELDS[case],
                )
            )
        elif row.probe_type == ProbeType.GOVERNANCE:
            try:
                kind = governance_kind(row.answer)
            except HarnessError as exc:
                raise _rubric_error(row, exc) from exc
            items.append(
                _item(
                    row,
                    Judge.GOVERNANCE,
                    response,
                    row.answer,
                    case=kind.value,
                    fields=_GOVERNANCE_FIELDS,
                )
            )
        elif row.probe_type == ProbeType.ROUTINE_QUESTION:
            phrases = active_bias_phrases(
                row.pm_id, row.checkpoint_date, inputs.traits, inputs.drift_events, inputs.bank
            )
            if phrases:
                items.append(_item(row, Judge.INTRUSION, response, _lines(phrases)))
            else:
                skipped[SKIP_NO_ACTIVE_BIAS] += 1
            values = judge_only_values(row, inputs.check_map)
            if values:
                names = tuple(f"{param}={value}" for param, value in values)
                items.append(_item(row, Judge.FORMAT, response, _lines(names), fields=names))
            else:
                skipped[SKIP_NO_JUDGE_ONLY_VALUES] += 1
    return ItemSelection(tuple(items), skipped)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    """An object schema with `rationale` first, so the judge reasons before it decides."""
    properties = {"rationale": {"type": "string"}, **properties}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def schema_for(item: JudgeItem) -> dict[str, Any]:
    """The verdict schema of the item's judge; OPEN and FORMAT depend on the item."""
    match item.judge:
        case Judge.OPEN:
            return _object({"choice": {"type": "string", "enum": [*item.letters, NO_CHOICE]}})
        case Judge.IN_SITU | Judge.GOVERNANCE:
            return _object({name: {"type": "boolean"} for name in item.fields})
        case Judge.INTRUSION:
            return _object({"intrudes": {"type": "boolean"}, "evidence": {"type": "string"}})
        case Judge.FORMAT:
            outcome = {"type": "string", "enum": [o.value for o in FormatOutcome]}
            return _object(
                {
                    "values": {
                        "type": "object",
                        "properties": {name: outcome for name in item.fields},
                        "required": list(item.fields),
                        "additionalProperties": False,
                    }
                }
            )


def build_request(item: JudgeItem, config: JudgeConfig) -> dict[str, Any]:
    """The Messages API request that grades one item.

    There is no top-level cache_control because judge requests share no long prefix.
    """
    user = (
        f"Question put to the copilot:\n{item.probe.question}\n\n"
        f"Copilot reply:\n{item.response}\n\n"
        f"Brief:\n{item.brief}"
    )
    return base_request(
        config.model,
        config.max_tokens,
        config.effort,
        f"{SYSTEM_PROMPT}\n\n{INSTRUCTIONS[item.judge]}",
        [{"role": "user", "content": user}],
        schema_for(item),
    )


def prompts_sha256() -> str:
    """Hash of the system prompt and every judge instruction, in Judge order."""
    digest = hashlib.sha256()
    for text in (SYSTEM_PROMPT, *(INSTRUCTIONS[judge] for judge in Judge)):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
