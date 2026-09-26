"""The async driver that narrates one session: alternates the PM narrator and the simulated
advisor through the cached client, validates and retries replies, runs the advisor's tool
loop, and returns the session's public row and hidden log.

Every request passes through `CachedClient`, which only caches a response once
it is validated here, so a refusal or a schema-invalid reply is never
replayed on a rerun.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pm_traitbench.config import DialogueConfig
from pm_traitbench.dialogue.client import CachedClient, Reply
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.prompts import (
    NARRATOR_OPENING_MESSAGE,
    _opening_line,
    advisor_request,
    advisor_system,
    narrator_directive,
    narrator_request,
)
from pm_traitbench.dialogue.tools import run_tool
from pm_traitbench.enums import AdvisorTool, TurnRole
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import (
    CallUsage,
    DialogueLog,
    Mention,
    Session,
    ToolCall,
    Turn,
    TurnLog,
)

_ADVISOR_TOOL_NAMES = frozenset(tool.value for tool in AdvisorTool)
_ZERO_USAGE = CallUsage(
    input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
)


@dataclass(frozen=True)
class TurnOutput:
    """One accepted `{text, mentions}` reply, already validated into row-model mentions."""

    text: str
    mentions: tuple[Mention, ...]


@dataclass(frozen=True)
class SessionResult:
    """One narrated session: its public row, hidden log, and any tool-round-cap warnings."""

    session: Session
    log: DialogueLog
    warnings: tuple[str, ...]


def parse_turn(response: Mapping[str, Any]) -> TurnOutput | None:
    """The last text block parsed as JSON and validated into a `TurnOutput`.

    None when `stop_reason` is not `end_turn`, no text block exists, or the
    JSON or its mentions fail validation.
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
        payload = json.loads(text_block["text"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        text = payload["text"]
        if not isinstance(text, str):
            return None
        mentions = tuple(Mention(**item) for item in payload.get("mentions") or ())
    except (TypeError, ValueError, KeyError):
        return None
    return TurnOutput(text=text, mentions=mentions)


@dataclass(frozen=True)
class _Accepted:
    """One accepted reply: either a validated final turn, or the tool_use blocks to run."""

    output: TurnOutput | None
    tool_blocks: tuple[Mapping[str, Any], ...]


def _classify(response: Mapping[str, Any], allow_tool_use: bool) -> _Accepted | None:
    """Accept a tool-use round (every block names an `AdvisorTool`) or a final parsed turn."""
    if allow_tool_use and response.get("stop_reason") == "tool_use":
        blocks = tuple(
            block for block in response.get("content") or () if block.get("type") == "tool_use"
        )
        if blocks and all(block.get("name") in _ADVISOR_TOOL_NAMES for block in blocks):
            return _Accepted(output=None, tool_blocks=blocks)
        return None
    output = parse_turn(response)
    if output is None:
        return None
    return _Accepted(output=output, tool_blocks=())


def _reject_reason(response: Mapping[str, Any]) -> str:
    stop_reason = response.get("stop_reason")
    if stop_reason in ("refusal", "max_tokens"):
        return f"reply rejected ({stop_reason})"
    if stop_reason == "tool_use":
        return "tool_use reply named an unknown tool"
    return "reply was unparsable or schema-invalid"


async def _send_accepted(
    client: CachedClient,
    request: Mapping[str, Any],
    config: DialogueConfig,
    session_id: str,
    *,
    allow_tool_use: bool,
) -> tuple[Reply, _Accepted]:
    """Send `request`, retrying the same body on a rejected reply up to `max_retries` times."""
    reason = "no attempt made"
    for _ in range(1 + config.max_retries):
        reply = await client.send(request)
        accepted = _classify(reply.response, allow_tool_use)
        if accepted is not None:
            client.commit(reply)
            return reply, accepted
        reason = _reject_reason(reply.response)
    raise DialogueError(f"session {session_id}: {reason}")


def _usage_of(response: Mapping[str, Any]) -> CallUsage:
    usage = response.get("usage") or {}
    return CallUsage(
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_input_tokens=usage.get("cache_read_input_tokens") or 0,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens") or 0,
    )


def _sum_usage(totals: CallUsage, response: Mapping[str, Any]) -> CallUsage:
    added = _usage_of(response)
    return CallUsage(
        input_tokens=totals.input_tokens + added.input_tokens,
        output_tokens=totals.output_tokens + added.output_tokens,
        cache_read_input_tokens=totals.cache_read_input_tokens + added.cache_read_input_tokens,
        cache_creation_input_tokens=(
            totals.cache_creation_input_tokens + added.cache_creation_input_tokens
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pm_directive_text(ctx: SessionContext, pm_index: int) -> str | None:
    """The stance line if the turn carries a stance, else turn 0's opening line, else None."""
    directive = ctx.turn_plan.pm_directives[pm_index]
    if directive.stance is not None:
        return directive.stance.stance
    if pm_index == 0 and directive.opening is not None:
        return _opening_line(ctx, directive.opening)
    return None


async def narrate_session(
    ctx: SessionContext,
    client: CachedClient,
    config: DialogueConfig,
    advisor_prompt: str,
    *,
    feedback: str | None = None,
) -> SessionResult:
    """Narrate one session: alternate PM and advisor turns and return the row plus its log."""
    session_id = ctx.skeleton.session_id
    n_pm = len(ctx.turn_plan.pm_directives)
    system_advisor = advisor_system(advisor_prompt, ctx.skeleton.date)

    narrator_messages: list[dict[str, Any]] = [
        {"role": "user", "content": NARRATOR_OPENING_MESSAGE}
    ]
    advisor_messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    turn_logs: list[TurnLog] = []

    for i in range(n_pm):
        narrator_messages.append({"role": "system", "content": narrator_directive(ctx, i)})
        pm_request = narrator_request(ctx, narrator_messages, config, feedback)
        pm_reply, pm_accepted = await _send_accepted(
            client, pm_request, config, session_id, allow_tool_use=False
        )
        narrator_messages.append({"role": "assistant", "content": pm_reply.response["content"]})
        pm_output = pm_accepted.output
        assert pm_output is not None
        turn_logs.append(
            TurnLog(
                role=TurnRole.PM,
                text=pm_output.text,
                mentions=pm_output.mentions,
                directive=_pm_directive_text(ctx, i),
                scripted_violation=False,
                tool_calls=(),
                model=pm_request["model"],
                request_hashes=(pm_reply.key,),
                usage=_usage_of(pm_reply.response),
            )
        )

        advisor_messages.append({"role": "user", "content": pm_output.text})
        is_violation = i == ctx.turn_plan.violation_advisor_index
        advisor_directive: str | None = None
        if is_violation:
            advisor_directive = ctx.skeleton.advisor_violation
            advisor_messages.append({"role": "system", "content": advisor_directive})

        tool_calls: list[ToolCall] = []
        request_hashes: list[str] = []
        usage = _ZERO_USAGE
        rounds = 0
        tools_disabled = False
        model_name = config.advisor_model
        while True:
            advisor_body = advisor_request(
                system_advisor, advisor_messages, config, tools_disabled=tools_disabled
            )
            model_name = advisor_body["model"]
            reply, accepted = await _send_accepted(
                client, advisor_body, config, session_id, allow_tool_use=not tools_disabled
            )
            request_hashes.append(reply.key)
            usage = _sum_usage(usage, reply.response)

            if accepted.tool_blocks:
                advisor_messages.append({"role": "assistant", "content": reply.response["content"]})
                rounds += 1
                tool_result_blocks = []
                for block in accepted.tool_blocks:
                    tool_input = block.get("input") or {}
                    outcome = run_tool(ctx.lookup, block["name"], tool_input, ctx.skeleton.date)
                    tool_calls.append(
                        ToolCall(
                            name=AdvisorTool(block["name"]),
                            input_json=_canonical_json(tool_input),
                            result_json=_canonical_json(outcome.result),
                            is_error=outcome.is_error,
                        )
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": json.dumps(outcome.result, sort_keys=True),
                            "is_error": outcome.is_error,
                        }
                    )
                advisor_messages.append({"role": "user", "content": tool_result_blocks})
                if rounds >= config.max_tool_rounds and not tools_disabled:
                    tools_disabled = True
                    warnings.append(f"{session_id}: advisor reply {i} hit the tool-round cap")
                continue

            advisor_messages.append({"role": "assistant", "content": reply.response["content"]})
            advisor_output = accepted.output
            assert advisor_output is not None
            turn_logs.append(
                TurnLog(
                    role=TurnRole.ADVISOR,
                    text=advisor_output.text,
                    mentions=advisor_output.mentions,
                    directive=advisor_directive,
                    scripted_violation=is_violation,
                    tool_calls=tuple(tool_calls),
                    model=model_name,
                    request_hashes=tuple(request_hashes),
                    usage=usage,
                )
            )
            if i < n_pm - 1:
                narrator_messages.append({"role": "user", "content": advisor_output.text})
            break

    turns = tuple(Turn(role=log.role, text=log.text) for log in turn_logs)
    session = Session(
        session_id=ctx.skeleton.session_id,
        pm_id=ctx.skeleton.pm_id,
        date=ctx.skeleton.date,
        kind=ctx.skeleton.kind,
        trade_idea_ids=ctx.skeleton.trade_idea_ids,
        turns=turns,
    )
    log = DialogueLog(
        session_id=ctx.skeleton.session_id,
        pm_id=ctx.skeleton.pm_id,
        voice_id=ctx.voice.voice_id,
        turns=tuple(turn_logs),
    )
    return SessionResult(session=session, log=log, warnings=tuple(warnings))
