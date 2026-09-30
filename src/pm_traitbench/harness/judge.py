"""LLM judges for open replies: what each is asked, against which brief, and the request."""

import asyncio
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Any

from pm_traitbench.catalogues.loader import load_catalogue
from pm_traitbench.catalogues.models import ProbeBank
from pm_traitbench.config import Config, JudgeConfig
from pm_traitbench.dialogue.client import (
    AnthropicClient,
    CachedClient,
    LlmClient,
    last_text_json,
    send_parsed,
)
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
from pm_traitbench.errors import DialogueBudgetError, HarnessError
from pm_traitbench.harness.checks import CHECKED_PARAMS, load_check_map, parse_routine_answer
from pm_traitbench.harness.runner import (
    check_run_name,
    check_scorable,
    probes_sha256,
    run_dir,
)
from pm_traitbench.tables.schema import DriftEvent, JudgementRow, ProbeRow, Trait
from pm_traitbench.tables.specs import DRIFT_EVENTS, JUDGEMENTS, PROBES, RESPONSES, TRAITS
from pm_traitbench.tables.store import DataStore
from pm_traitbench.traits_truth import bias_active_at

JUDGE_METADATA = "eval-judge"
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


def parse_verdict(item: JudgeItem, response: Mapping[str, Any]) -> dict[str, Any] | None:
    """The reply's verdict object when it fits the item's schema, else None."""
    payload = last_text_json(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("rationale"), str):
        return None
    match item.judge:
        case Judge.OPEN:
            valid = payload.get("choice") in (*item.letters, NO_CHOICE)
        case Judge.IN_SITU | Judge.GOVERNANCE:
            valid = all(isinstance(payload.get(name), bool) for name in item.fields)
        case Judge.INTRUSION:
            valid = isinstance(payload.get("intrudes"), bool) and isinstance(
                payload.get("evidence"), str
            )
        case Judge.FORMAT:
            values = payload.get("values")
            outcomes = {o.value for o in FormatOutcome}
            valid = (
                isinstance(values, dict)
                and set(values) == set(item.fields)
                and all(isinstance(v, str) and v in outcomes for v in values.values())
            )
    return payload if valid else None


def _flag(value: bool) -> str:
    return "true" if value else "false"


def judgement_from_verdict(item: JudgeItem, verdict: Mapping[str, Any]) -> JudgementRow:
    """The judgement row a parsed verdict gives: correctness, detail and rationale."""
    match item.judge:
        case Judge.OPEN:
            correct = verdict["choice"] == item.answer_letter
            detail = f"choice={verdict['choice']}"
        case Judge.IN_SITU | Judge.GOVERNANCE:
            flags = [verdict[name] for name in item.fields]
            correct = all(flags) if item.judge == Judge.IN_SITU else any(flags)
            detail = "; ".join(f"{n}={_flag(f)}" for n, f in zip(item.fields, flags, strict=True))
        case Judge.INTRUSION:
            correct = not verdict["intrudes"]
            detail = f"intrudes={_flag(verdict['intrudes'])}; evidence={verdict['evidence']}"
        case Judge.FORMAT:
            outcomes = [verdict["values"][name] for name in item.fields]
            correct = FormatOutcome.FAIL not in outcomes
            detail = "; ".join(f"{n}: {o}" for n, o in zip(item.fields, outcomes, strict=True))
    return JudgementRow(
        probe_id=item.probe.probe_id,
        pm_id=item.probe.pm_id,
        judge=item.judge,
        correct=correct,
        detail=detail,
        rationale=verdict["rationale"],
    )


def split_detail(detail: str, judge: Judge) -> dict[str, str]:
    """The fields of a judgement's `detail`, as `judgement_from_verdict` wrote them.

    Pairs are separated by `; ` and split on the first `=`. INTRUSION takes everything
    after the first `; evidence=` as the quote, which may itself hold `=` or `;`.
    FORMAT keys are `param=value` and hold `=`, so each pair splits on its last `: `.
    """
    if judge == Judge.INTRUSION:
        head, sep, evidence = detail.partition("; evidence=")
        key, eq, flag = head.partition("=")
        if not sep or not eq:
            raise HarnessError(f"malformed intrusion detail: {detail!r}")
        return {key: flag, "evidence": evidence}
    fields = {}
    for pair in detail.split("; "):
        if judge == Judge.FORMAT:
            key, sep, value = pair.rpartition(": ")
        else:
            key, sep, value = pair.partition("=")
        if not sep:
            raise HarnessError(f"malformed {judge.value} detail: {detail!r}")
        fields[key] = value
    return fields


def empty_judgement(item: JudgeItem) -> JudgementRow:
    """The wrong judgement of an empty reply, which is never sent to a judge."""
    return JudgementRow(
        probe_id=item.probe.probe_id,
        pm_id=item.probe.pm_id,
        judge=item.judge,
        correct=False,
        detail="empty_reply",
        rationale="",
    )


def judge_scope(run_name: str, item: JudgeItem) -> str:
    """The cache scope of one (probe, judge) call in a run."""
    return f"judge:{run_name}:{item.probe.probe_id}:{item.judge}"


@dataclass(frozen=True)
class JudgeResult:
    """The judgements a pass wrote, the probes it skipped by reason, and the run's store."""

    judgements: tuple[JudgementRow, ...]
    skipped: dict[str, int]
    run_store: DataStore


def _default_client(config: Config) -> LlmClient:
    return AnthropicClient(config.judge.max_concurrency, config.dialogue.api_max_retries)


def _prepare_pass(config: Config, run_store: DataStore, force: bool) -> None:
    """Clear a previous pass on `force`, else refuse to mix judgements of different settings."""
    metadata_path = run_store.data_dir / "run_metadata" / f"{JUDGE_METADATA}.json"
    if force:
        run_store.path(JUDGEMENTS).unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return
    previous = run_store.read_run_metadata(JUDGE_METADATA)
    if previous is None:
        return
    same_config = previous["config"]["judge"] == config.judge.model_dump(mode="json")
    if not same_config or previous.get("prompts_sha256") != prompts_sha256():
        raise HarnessError(
            "judgements exist from other judge settings or prompts; "
            "rerun with --force to replace it"
        )


async def _judge_items(
    client: CachedClient, config: Config, run_name: str, items: Sequence[JudgeItem]
) -> list[JudgementRow]:
    async def one(item: JudgeItem) -> JudgementRow:
        if item.response.strip() == "":
            return empty_judgement(item)
        verdict, _ = await send_parsed(
            client,
            build_request(item, config.judge),
            partial(parse_verdict, item),
            scope=judge_scope(run_name, item),
            max_retries=config.dialogue.max_retries,
            error_type=HarnessError,
            label="judge",
            reason="reply is not a verdict object",
        )
        return judgement_from_verdict(item, verdict)

    try:
        # Let siblings finish and commit so a failure never discards replies already paid for.
        results = await asyncio.gather(*(one(item) for item in items), return_exceptions=True)
    finally:
        await client.aclose()
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return list(results)


def load_judge_inputs(store: DataStore, run_store: DataStore) -> JudgeInputs:
    """The inputs item selection reads, from the corpus and a run's responses."""
    catalogue = load_catalogue()
    return JudgeInputs(
        probes=store.read(PROBES),
        responses={r.probe_id: r.response for r in run_store.read(RESPONSES)},
        traits=store.read(TRAITS),
        drift_events=store.read(DRIFT_EVENTS),
        bank=catalogue.probes,
        check_map=load_check_map(catalogue),
    )


def judge_run(
    config: Config,
    store: DataStore,
    run_name: str,
    *,
    force: bool = False,
    client_factory: Callable[[Config], LlmClient] | None = None,
) -> JudgeResult:
    """Grade a finished run's open replies with the judges and write its judgements table.

    Finished calls are cached under the run, so a rerun after a spent budget continues.
    """
    check_run_name(run_name)
    rd = run_dir(store.data_dir, run_name)
    run_store = DataStore(rd, config.output)
    check_scorable(store, run_store, run_name)
    _prepare_pass(config, run_store, force)

    inputs = load_judge_inputs(store, run_store)
    selection = select_items(inputs)

    factory = client_factory or _default_client
    client = CachedClient(lambda: factory(config), rd / "cache", config.judge.token_budget)
    try:
        with asyncio.Runner() as runner:
            rows = runner.run(_judge_items(client, config, run_name, selection.items))
    except DialogueBudgetError as exc:
        raise HarnessError(
            "judge token budget spent; finished items are cached, rerun to continue"
        ) from exc
    rows.sort(key=lambda r: (r.pm_id, r.probe_id, r.judge))

    run_store.write(JUDGEMENTS, rows)
    counts = {judge.value: 0 for judge in Judge}
    for item in selection.items:
        counts[item.judge.value] += 1
    run_store.write_run_metadata(
        JUDGE_METADATA,
        config,
        {
            "run_name": run_name,
            "probes_sha256": probes_sha256(store),
            "prompts_sha256": prompts_sha256(),
            "counts": counts,
            "skipped": selection.skipped,
            "usage": client.totals.as_metadata(),
        },
    )
    return JudgeResult(tuple(rows), selection.skipped, run_store)
