"""The leakage and forbidden-trait judges of the validate stage: request builders, reply
parsers and the leak label to bias param mapper.

Each judge sees only transcript text and, for the forbidden judge, the
session's rendered avoid lines - never a trait id, bias param, trait value,
stance line, signal mode or the PM's persona.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pm_traitbench.catalogues.models import BiasLabels
from pm_traitbench.config import BIAS_PARAMS, ValidateConfig
from pm_traitbench.dialogue.client import CachedClient
from pm_traitbench.enums import TurnRole
from pm_traitbench.errors import ValidateError
from pm_traitbench.tables.schema import DialogueLog

LEAK_SYSTEM = (
    "Read the PM side of a transcript between a portfolio manager and an advisor. Decide "
    "whether the PM explicitly names or self-labels a psychological or trading tendency of "
    "their own, either by using a term for it or by stating it as a general habit such as "
    '"I always", "I tend to", "my weakness is". Describing one decision on its merits is not '
    "explicit. Return the label the PM used and the shortest quote that shows it."
)

FORBIDDEN_SYSTEM = (
    "Here is a transcript and a numbered list of things the PM must not do. List every item "
    "the PM does, with the shortest quote that shows it. An empty list means none."
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


def leak_request(log: DialogueLog, config: ValidateConfig) -> dict[str, Any]:
    """The leakage judge's Messages API request body: only the allowed keys."""
    return {
        "model": config.judge_model,
        "max_tokens": config.max_output_tokens,
        "system": LEAK_SYSTEM,
        "messages": [{"role": "user", "content": transcript_text(log)}],
        "output_config": {
            "effort": config.effort.value,
            "format": {"type": "json_schema", "schema": LEAK_SCHEMA},
        },
    }


def forbidden_request(
    log: DialogueLog, avoid_lines: Sequence[str], config: ValidateConfig
) -> dict[str, Any]:
    """The forbidden-trait judge's Messages API request body: only the allowed keys."""
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(avoid_lines, 1))
    content = f"{transcript_text(log)}\n\nThe PM must not:\n{numbered}"
    return {
        "model": config.judge_model,
        "max_tokens": config.max_output_tokens,
        "system": FORBIDDEN_SYSTEM,
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": config.effort.value,
            "format": {"type": "json_schema", "schema": FORBIDDEN_SCHEMA},
        },
    }


@dataclass(frozen=True)
class LeakVerdict:
    """One leakage judge verdict: whether the PM named a tendency, its label and quote."""

    explicit: bool
    label: str | None
    quote: str


@dataclass(frozen=True)
class Violation:
    """One forbidden-trait violation: the avoid line's number and the quote that shows it."""

    index: int
    quote: str


def _last_text_payload(response: Mapping[str, Any]) -> Any:
    """The last text block's JSON payload, or a sentinel-free `None` if it cannot be read.

    `None` when `stop_reason` is not `end_turn`, there is no text block, or
    the text is not valid JSON.
    """
    if response.get("stop_reason") != "end_turn":
        return None
    text_block: Mapping[str, Any] | None = None
    for block in response.get("content") or ():
        if block.get("type") == "text":
            text_block = block
    if text_block is None:
        return None
    try:
        return json.loads(text_block["text"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def parse_leak(response: Mapping[str, Any]) -> LeakVerdict | None:
    """The leak verdict from a judge reply's last text block, validated against `LEAK_SCHEMA`."""
    payload = _last_text_payload(response)
    if not isinstance(payload, dict):
        return None
    try:
        explicit = payload["explicit"]
        label = payload["label"]
        quote = payload["quote"]
    except KeyError:
        return None
    if not isinstance(explicit, bool):
        return None
    if label is not None and not isinstance(label, str):
        return None
    if not isinstance(quote, str):
        return None
    return LeakVerdict(explicit=explicit, label=label, quote=quote)


def parse_forbidden(response: Mapping[str, Any]) -> tuple[Violation, ...] | None:
    """The violations from a judge reply's last text block, validated against `FORBIDDEN_SCHEMA`."""
    payload = _last_text_payload(response)
    if not isinstance(payload, dict):
        return None
    raw_violations = payload.get("violations")
    if not isinstance(raw_violations, list):
        return None
    violations: list[Violation] = []
    for item in raw_violations:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        quote = item.get("quote")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if not isinstance(quote, str):
            return None
        violations.append(Violation(index=index, quote=quote))
    return tuple(violations)


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
    """Send `request`, retrying an unparsable reply up to `max_retries` times.

    Mirrors `session._send_accepted`: a rejected reply that came from the
    cache makes every later attempt bypass the cache, and an accepted reply
    is committed before it is returned. Raises `ValidateError` at the cap.
    """
    refresh = False
    rejected = 0
    for _ in range(1 + max_retries):
        reply = await client.send(request, scope=session_id, refresh=refresh)
        parsed = parse(reply.response)
        if parsed is not None:
            client.commit(reply)
            return parsed, rejected
        refresh = refresh or reply.cached
        rejected += 1
    raise ValidateError(f"session {session_id}: the judge reply was unparsable or schema-invalid")
