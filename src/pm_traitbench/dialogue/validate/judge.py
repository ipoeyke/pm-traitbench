"""The leakage and forbidden-trait judges of the validate stage: request builders, reply
parsers and the leak label to bias param mapper.

Each judge sees only transcript text and, for the forbidden judge, the
session's rendered avoid lines - never a trait id, bias param, trait value,
stance line, signal mode or the PM's persona.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pm_traitbench.catalogues.models import BiasLabels
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
    "then explicit is false with a null label. When it is explicit, return the label the PM "
    "used and the shortest verbatim quote from a PM turn that shows it."
)

# Framed as verification with the null result first: a judge asked to "list every item"
# over-reports on ordinary desk talk, and the avoid lines describe ordinary decisions.
FORBIDDEN_SYSTEM = (
    "Here is a transcript and a numbered list of things the PM must not do. For each item, "
    "decide whether a PM turn shows the PM doing that thing. Most transcripts show none of "
    "them, and the expected answer is an empty list. Report an item only when a verbatim "
    "quote from a PM turn shows the PM doing it, not discussing the topic, asking about it, "
    "or taking one decision that their own rules allow. Give that quote for each reported item."
)

LEAK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "leak_verdict",
    "properties": {
        "explicit": {"type": "boolean"},
        "label": {"type": ["string", "null"]},
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


class LeakVerdict(BaseModel):
    """One leakage judge verdict: whether the PM named a tendency, its label and quote."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    explicit: bool
    label: str | None
    quote: str


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


def parse_forbidden(response: Mapping[str, Any]) -> tuple[Violation, ...] | None:
    """The violations from a judge reply's last text block, validated against `FORBIDDEN_SCHEMA`."""
    try:
        return _ForbiddenVerdict.model_validate(last_text_json(response)).violations
    except ValidationError:
        return None


def map_label(label: str | None, labels: BiasLabels) -> str | None:
    """First `BIAS_PARAMS` param with a catalogue phrase in `label`; `None` for a blank label."""
    if label is None:
        return None
    normalized = label.strip().lower()
    if not normalized:
        return None
    for param in BIAS_PARAMS:
        if any(phrase in normalized for phrase in labels.labels.get(param, ())):
            return param
    return None


async def send_judged[T](
    client: CachedClient,
    request: Mapping[str, Any],
    parse: Callable[[Mapping[str, Any]], T | None],
    session_id: str,
    max_retries: int,
) -> tuple[T, int]:
    """Send `request` through `send_until_accepted`, retrying an unparsable reply.

    Raises `ValidateError` once `max_retries` retries are spent.
    """

    def classify(response: Mapping[str, Any]) -> tuple[T | None, str]:
        return parse(response), "the judge reply was unparsable or schema-invalid"

    _, parsed, rejected = await send_until_accepted(
        client,
        request,
        classify,
        scope=session_id,
        max_retries=max_retries,
        error_type=ValidateError,
    )
    return parsed, rejected
