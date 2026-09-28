"""Gate 2 stated-signal classification: a per-session request, its reply parser and scorer.

The request carries no trait vocabulary: only the materiality definitions, mandate
facts, rule texts, the session's ledger rows, its transcript and N.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pm_traitbench.config import Gate2Config
from pm_traitbench.dialogue.client import CachedClient, last_text_json, send_until_accepted
from pm_traitbench.dialogue.prompts import base_request
from pm_traitbench.enums import (
    Kind,
    Ownership,
    RuleScope,
    SignalMode,
    StanceEntry,
    TurnRole,
    Valence,
)
from pm_traitbench.errors import Gate2Error
from pm_traitbench.gates.gate2.transcript import render_session
from pm_traitbench.tables.schema import (
    DialogueLog,
    LedgerRow,
    Mandate,
    Persona,
    Rule,
    Session,
    Signal,
    Skeleton,
)

_MATERIALITY = (
    "A bias is a habit whose honouring or breaching would move the PM's expected P&L or "
    "risk. A preference is about how the PM wants to be advised or informed, and would "
    "not move either."
)

_QUOTE_INSTRUCTION = "Quote each statement verbatim from a PM turn in the transcript below."

_IDEA_RULE_PARAMS = ("stop", "target")

CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "gate2_classification",
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "kind": {"type": "string", "enum": ["bias", "preference"]},
                },
                "required": ["quote", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["statements"],
    "additionalProperties": False,
}


def is_classifiable(signal: Signal) -> bool:
    """Whether `signal` is a self-owned, confirmed, stated one - the only kind classified."""
    return (
        signal.mode == SignalMode.STATED
        and signal.ownership == Ownership.SELF
        and signal.valence == Valence.CONFIRM
    )


def signal_turn_index(skeleton: Skeleton, log: DialogueLog, signal_id: str) -> int:
    """The index into `log.turns` of the PM turn that rendered `signal_id`'s stance.

    Raises `Gate2Error` when the skeleton lacks exactly one non-claim stance for
    `signal_id`, or that stance's text matches zero or more than one PM turn's directive.
    """
    stances = tuple(
        stance
        for stance in skeleton.stances
        if stance.signal_id == signal_id and stance.entry != StanceEntry.CLAIM
    )
    if len(stances) != 1:
        raise Gate2Error(
            f"signal '{signal_id}': matches {len(stances)} non-claim stances in skeleton "
            f"'{skeleton.session_id}'"
        )
    stance_text = stances[0].stance
    matches = [
        index
        for index, turn in enumerate(log.turns)
        if turn.role == TurnRole.PM and turn.directive == stance_text
    ]
    if len(matches) != 1:
        raise Gate2Error(
            f"signal '{signal_id}': stance matches {len(matches)} turns in session "
            f"'{log.session_id}'"
        )
    return matches[0]


@dataclass(frozen=True)
class ClassifyUnit:
    """One session's classifiable signals, ready for a classification request and its scorer."""

    session: Session
    log: DialogueLog
    skeleton: Skeleton
    signals: tuple[Signal, ...]
    turn_index_by_signal: Mapping[str, int]


def classify_units(
    sessions_by_id: Mapping[str, Session],
    logs_by_id: Mapping[str, DialogueLog],
    skeletons_by_id: Mapping[str, Skeleton],
    signals: Sequence[Signal],
) -> tuple[ClassifyUnit, ...]:
    """One `ClassifyUnit` per surviving session carrying at least one classifiable signal.

    A session dropped from `sessions_by_id` contributes no unit. Units are sorted by
    session id; a unit's signals are sorted by signal id.
    """
    by_session: dict[str, list[Signal]] = {}
    for sig in signals:
        if is_classifiable(sig):
            by_session.setdefault(sig.session_id, []).append(sig)

    units: list[ClassifyUnit] = []
    for session_id in sorted(by_session):
        session = sessions_by_id.get(session_id)
        if session is None:
            continue
        session_signals = tuple(sorted(by_session[session_id], key=lambda sig: sig.signal_id))
        log = logs_by_id.get(session_id)
        skeleton = skeletons_by_id.get(session_id)
        if log is None or skeleton is None:
            raise Gate2Error(f"session '{session_id}': survived but has no log or skeleton")
        turn_index_by_signal = {
            sig.signal_id: signal_turn_index(skeleton, log, sig.signal_id)
            for sig in session_signals
        }
        units.append(
            ClassifyUnit(
                session=session,
                log=log,
                skeleton=skeleton,
                signals=session_signals,
                turn_index_by_signal=turn_index_by_signal,
            )
        )
    return tuple(units)


def _role_line(n: int) -> str:
    return (
        "You are reviewing one session between a portfolio manager and their advisor. "
        f"Find up to {n} statements where the PM describes a habit or a preference about "
        "how they invest, and classify each one."
    )


def _mandate_line(mandate: Mandate) -> str:
    return (
        f"Mandate: asset class {mandate.asset_class.value}, sub-style {mandate.sub_style}, "
        f"book size {mandate.book_size}, risk unit {mandate.risk_unit}, "
        f"benchmark {mandate.benchmark}."
    )


def _pm_rules_section(pm_rules: Sequence[Rule]) -> str:
    ordered = sorted(
        (rule for rule in pm_rules if rule.scope == RuleScope.PM), key=lambda rule: rule.rule_id
    )
    lines = "\n".join(rule.text for rule in ordered)
    return f"Rules the PM is held to:\n{lines}"


def _idea_rules_section(idea_rules: Sequence[Rule], trade_idea_ids: Sequence[str]) -> str:
    ids = set(trade_idea_ids)
    ordered = sorted(
        (
            rule
            for rule in idea_rules
            if rule.scope == RuleScope.IDEA
            and rule.param in _IDEA_RULE_PARAMS
            and rule.trade_idea_id in ids
        ),
        key=lambda rule: rule.rule_id,
    )
    lines = "\n".join(rule.text for rule in ordered)
    return f"Stops and targets on the ideas discussed:\n{lines}"


def _ledger_section(ledger_rows: Sequence[LedgerRow], session: Session) -> str:
    ids = set(session.trade_idea_ids)
    ordered = sorted(
        (row for row in ledger_rows if row.trade_idea_id in ids and row.date <= session.date),
        key=lambda row: (row.date, row.trade_idea_id),
    )
    body = (
        "\n".join(
            f"{row.date.isoformat()} {row.trade_idea_id} {row.instrument_id} "
            f"{row.side.value} size {row.size} at {row.price_or_yield}"
            for row in ordered
        )
        if ordered
        else "none"
    )
    return f"Trades on the ideas discussed, on or before this session:\n{body}"


def classify_request(
    persona: Persona,
    pm_rules: Sequence[Rule],
    idea_rules: Sequence[Rule],
    ledger_rows: Sequence[LedgerRow],
    session: Session,
    n: int,
    config: Gate2Config,
) -> dict[str, Any]:
    """The classification model's Messages API request body: only the allowed keys.

    Carries no bias param name and no catalogue value list - only the materiality
    definitions, mandate facts, rule texts, the session's ledger rows, its transcript
    and `n`.
    """
    system = "\n\n".join(
        [
            _role_line(n),
            _MATERIALITY,
            _mandate_line(persona.mandate),
            _pm_rules_section(pm_rules),
            _idea_rules_section(idea_rules, session.trade_idea_ids),
            _QUOTE_INSTRUCTION,
        ]
    )
    user = "\n\n".join(
        [_ledger_section(ledger_rows, session), render_session(session), f"Find {n} statements."]
    )
    return base_request(
        config.model,
        config.classify_max_output_tokens,
        config.effort,
        system,
        [{"role": "user", "content": user}],
        CLASSIFY_SCHEMA,
    )


class Statement(BaseModel):
    """One quoted PM statement and whether it shows a bias or a preference."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    quote: str
    kind: Kind = Field(strict=False)


class _ClassificationReply(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    statements: tuple[Statement, ...] = Field(strict=False)


def parse_classification(response: Mapping[str, Any], n: int) -> tuple[Statement, ...] | None:
    """A validated 1-to-`n` statements, or `None` for anything schema-invalid or out of bounds."""
    try:
        reply = _ClassificationReply.model_validate(last_text_json(response))
    except ValidationError:
        return None
    if not 1 <= len(reply.statements) <= n:
        return None
    return reply.statements


def normalise(text: str) -> str:
    """Case- and whitespace-insensitive form used to match a quote to a PM turn's text."""
    return " ".join(text.lower().split())


def score_classification(
    unit: ClassifyUnit, statements: Sequence[Statement], kind_by_trait: Mapping[str, Kind]
) -> tuple[dict[str, tuple[Kind | None, bool]], tuple[str, ...]]:
    """Per-signal kind prediction and correctness, plus warnings for unlocated quotes.

    A statement survives when its quote, once normalised, is non-empty and lies inside
    some PM turn's text; one that does not is dropped and warned about instead. Each
    signal is scored against the first surviving statement whose quote lies inside its
    own turn only, so a statement quoting a different turn is never credited to it.
    """
    all_turns = [normalise(turn.text) for turn in unit.log.turns]
    pm_turns = [normalise(turn.text) for turn in unit.log.turns if turn.role == TurnRole.PM]

    surviving: list[Statement] = []
    warnings: list[str] = []
    for statement in statements:
        needle = normalise(statement.quote)
        if not needle:
            warnings.append(f"session {unit.session.session_id}: classification quote is empty")
        elif any(needle in haystack for haystack in pm_turns):
            surviving.append(statement)
        else:
            warnings.append(
                f"session {unit.session.session_id}: classification quote not in a PM "
                f'turn: "{statement.quote}"'
            )

    scores: dict[str, tuple[Kind | None, bool]] = {}
    for signal in unit.signals:
        own_turn = all_turns[unit.turn_index_by_signal[signal.signal_id]]
        predicted = next((s.kind for s in surviving if normalise(s.quote) in own_turn), None)
        scores[signal.signal_id] = (predicted, predicted == kind_by_trait[signal.trait_id])
    return scores, tuple(warnings)


async def send_classification(
    client: CachedClient,
    request: Mapping[str, Any],
    n: int,
    session_id: str,
    max_retries: int,
) -> tuple[tuple[Statement, ...], int]:
    """Send the classification request through `send_until_accepted`; raises `Gate2Error`
    at the retry cap.
    """

    def classify(response: Mapping[str, Any]) -> tuple[tuple[Statement, ...] | None, str]:
        return (
            parse_classification(response, n),
            "the classification reply was unparsable, schema-invalid or out of bounds",
        )

    _, statements, rejected = await send_until_accepted(
        client, request, classify, scope=session_id, max_retries=max_retries, error_type=Gate2Error
    )
    return statements, rejected
