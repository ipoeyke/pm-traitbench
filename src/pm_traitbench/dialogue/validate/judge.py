"""The leakage, forbidden-trait and stance judges of the validate stage: request builders
and reply parsers.

The leak and forbidden judges see only transcript text and, for the forbidden
judge, the session's rendered avoid lines - never a trait id, bias param, trait
value, stance line, signal mode or the PM's persona. The stance judge sees one
PM turn and the stance line it was told to carry out, and checks adherence only.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pm_traitbench.config import BIAS_PARAMS, ValidateConfig
from pm_traitbench.dialogue.client import CachedClient, last_text_json, send_until_accepted
from pm_traitbench.dialogue.prompts import base_request
from pm_traitbench.enums import TurnRole
from pm_traitbench.errors import ValidateError
from pm_traitbench.tables.schema import DialogueLog

LEAK_SYSTEM = (
    "Read the PM side of a transcript between a portfolio manager and an advisor. Decide "
    "whether the PM explicitly names or self-labels a psychological or trading tendency of "
    "their own, either by using a term for it or by stating it as a general habit such as "
    '"I always", "I tend to", "my weakness is". Most transcripts contain no such label: '
    "describing one decision on its merits, or discussing the market, is not explicit, and "
    "then explicit is false with a null label. When it is explicit, name the tendency with "
    'the one label from the list that fits it, or "other" when none does, and give the '
    "shortest verbatim quote from a PM turn that shows it."
)

# The labels a leak verdict may carry: every bias param, or "other" for any other habit.
LEAK_LABELS: tuple[str, ...] = (*BIAS_PARAMS, "other")

# Framed as verification with the null result first: a judge asked to "list every item"
# over-reports on ordinary desk talk, and the avoid lines describe ordinary decisions.
FORBIDDEN_SYSTEM = (
    "Here is a transcript and a numbered list of things the PM must not do. For each item, "
    "decide whether a PM turn shows the PM doing that thing. Most transcripts show none of "
    "them, and the expected answer is an empty list. Report an item only when a verbatim "
    "quote from a PM turn shows the PM doing it, not discussing the topic, asking about it, "
    "or taking one decision that their own rules allow. Give that quote for each reported item."
)

# Separate from the leak and forbidden judges, which must never see a stance line.
STANCE_SYSTEM = (
    "Here is one message a portfolio manager (PM) wrote to their advisor and the instruction "
    "that message was written to carry out. Decide whether the message carries the instruction "
    "out. Different wording, a paraphrase, or first acknowledging a point the advisor made all "
    "still count, as long as the message ends up doing what the instruction says. It fails "
    "only when it drops, reverses or waters down the instruction, for example by giving in to "
    "the advisor's pushback. Give a one-sentence reason."
)

LEAK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "leak_verdict",
    "properties": {
        "explicit": {"type": "boolean"},
        # A nullable enum must be an anyOf: the API rejects an enum under a type list.
        "label": {"anyOf": [{"type": "string", "enum": list(LEAK_LABELS)}, {"type": "null"}]},
        "quote": {"type": "string"},
    },
    "required": ["explicit", "label", "quote"],
    "additionalProperties": False,
}

FORBIDDEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "forbidden_verdict",
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["index", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["violations"],
    "additionalProperties": False,
}


STANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "stance_verdict",
    "properties": {
        "carried_out": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["carried_out", "reason"],
    "additionalProperties": False,
}


def transcript_text(log: DialogueLog) -> str:
    """One line per turn, `PM: {text}` or `ADVISOR: {text}`, joined by blank lines."""
    lines = (
        f"{'PM' if turn.role == TurnRole.PM else 'ADVISOR'}: {turn.text}" for turn in log.turns
    )
    return "\n\n".join(lines)


def _judge_request(
    system: str, content: str, schema: Mapping[str, Any], config: ValidateConfig
) -> dict[str, Any]:
    """A judge's Messages API request body: only the allowed keys."""
    return base_request(
        config.judge_model,
        config.max_output_tokens,
        config.effort,
        system,
        [{"role": "user", "content": content}],
        schema,
    )


def leak_request(
    log: DialogueLog, config: ValidateConfig, transcript: str | None = None
) -> dict[str, Any]:
    """The leakage judge's request body; `transcript` reuses an already rendered `log`."""
    content = transcript if transcript is not None else transcript_text(log)
    return _judge_request(LEAK_SYSTEM, content, LEAK_SCHEMA, config)


def forbidden_request(
    log: DialogueLog,
    avoid_lines: Sequence[str],
    config: ValidateConfig,
    transcript: str | None = None,
) -> dict[str, Any]:
    """The forbidden-trait judge's request body; `transcript` reuses an already rendered `log`."""
    rendered = transcript if transcript is not None else transcript_text(log)
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(avoid_lines, 1))
    content = f"{rendered}\n\nThe PM must not:\n{numbered}"
    return _judge_request(FORBIDDEN_SYSTEM, content, FORBIDDEN_SCHEMA, config)


def stance_request(pm_text: str, stance: str, config: ValidateConfig) -> dict[str, Any]:
    """The stance judge's request body for one PM turn and the stance line it carried."""
    # XML-tagged and message-first to minimise API reasoning-extraction refusals
    content = f"<pm_message>{pm_text}</pm_message>\n\n<instruction>{stance}</instruction>"
    return _judge_request(STANCE_SYSTEM, content, STANCE_SCHEMA, config)


class LeakVerdict(BaseModel):
    """One leakage judge verdict: whether the PM named a tendency, its label and quote.

    `label` is a bias param, "other" for a habit that is no planted bias, or null.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    explicit: bool
    label: str | None
    quote: str

    @field_validator("label")
    @classmethod
    def _label_in_list(cls, value: str | None) -> str | None:
        if value is not None and value not in LEAK_LABELS:
            raise ValueError(f"label must be one of {LEAK_LABELS} or null")
        return value


class StanceVerdict(BaseModel):
    """One stance judge verdict: whether the PM turn carried its stance out, and why."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    carried_out: bool
    reason: str


class Violation(BaseModel):
    """One forbidden-trait violation: the avoid line's number and the quote that shows it."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    index: int
    quote: str


class _ForbiddenVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    # Lax only here so the JSON array arrives as a tuple; each item stays strict.
    violations: tuple[Violation, ...] = Field(strict=False)


def parse_leak(response: Mapping[str, Any]) -> LeakVerdict | None:
    """The leak verdict from a judge reply's last text block, validated against `LEAK_SCHEMA`."""
    try:
        return LeakVerdict.model_validate(last_text_json(response))
    except ValidationError:
        return None


def parse_stance(response: Mapping[str, Any]) -> StanceVerdict | None:
    """The stance verdict from a judge reply's last text block, validated against its schema."""
    try:
        return StanceVerdict.model_validate(last_text_json(response))
    except ValidationError:
        return None


def parse_forbidden(response: Mapping[str, Any]) -> tuple[Violation, ...] | None:
    """The violations from a judge reply's last text block, validated against `FORBIDDEN_SCHEMA`."""
    try:
        return _ForbiddenVerdict.model_validate(last_text_json(response)).violations
    except ValidationError:
        return None


_UNPARSABLE_REASON = "the judge reply was unparsable or schema-invalid"
_REFUSED_REASON = "the judge reply was refused"


class _Refused:
    """Marks a judge reply the model refused, so the request is re-sent to the fallback model."""


_REFUSED = _Refused()


async def send_judged[T](
    client: CachedClient,
    request: Mapping[str, Any],
    parse: Callable[[Mapping[str, Any]], T | None],
    session_id: str,
    max_retries: int,
    fallback_model: str | None = None,
) -> tuple[T, int, bool]:
    """Send `request` until `parse` accepts a reply; raises `ValidateError` once retries are spent.

    With `fallback_model`, a refusal is committed and the request is re-sent once to that model
    with its own retries. Returns the parsed reply, the rejected count summed across both models,
    and whether the fallback model answered.
    """

    def classify(response: Mapping[str, Any]) -> tuple[T | _Refused | None, str]:
        if response.get("stop_reason") == "refusal":
            return (_REFUSED if fallback_model is not None else None), _REFUSED_REASON
        return parse(response), _UNPARSABLE_REASON

    _, parsed, rejected = await send_until_accepted(
        client,
        request,
        classify,
        scope=session_id,
        max_retries=max_retries,
        error_type=ValidateError,
    )
    if not isinstance(parsed, _Refused):
        return parsed, rejected, False
    assert fallback_model is not None
    fallback_parsed, fallback_rejected, _ = await send_judged(
        client, {**request, "model": fallback_model}, parse, session_id, max_retries
    )
    return fallback_parsed, rejected + 1 + fallback_rejected, True
