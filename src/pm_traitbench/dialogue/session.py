"""The async driver that narrates one session: alternates the PM narrator and the simulated
advisor through the cached client, validates and retries replies, runs the advisor's tool
loop, and returns the session's public row and hidden log.

Every request passes through `CachedClient`, which only caches a response once
it is validated here, so a refusal or a schema-invalid reply is never
replayed on a rerun.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pm_traitbench.config import DialogueConfig
from pm_traitbench.dialogue.client import (
    CachedClient,
    Reply,
    last_text_json,
    send_until_accepted,
    session_prefix,
)
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.prompts import (
    NARRATOR_OPENING_MESSAGE,
    advisor_instruments,
    advisor_request,
    advisor_system,
    narrator_directive,
    narrator_request,
    opening_line,
)
from pm_traitbench.dialogue.tools import run_tool
from pm_traitbench.dialogue.usage import ZERO_USAGE, usage_of
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
    canonical_json,
)

_ADVISOR_TOOL_NAMES = frozenset(tool.value for tool in AdvisorTool)


@dataclass(frozen=True)
class TurnOutput:
    """One accepted `{text, mentions}` reply, already validated into row-model mentions."""

    text: str
    mentions: tuple[Mention, ...]


@dataclass(frozen=True)
class SessionResult:
    """One narrated session: its public row, hidden log, any tool-round-cap warnings, and
    how many rejected replies it took to produce them.
    """

    session: Session
    log: DialogueLog
    warnings: tuple[str, ...]
    rejected_replies: int


# Sent as an operator message after the last tool results once the round cap is hit,
# so the advisor answers in text rather than returning nothing when its lookups failed.
_TOOLS_OFF_NOTE = (
    "No more lookups are available for this reply; later replies may use the tools again. "
    "Answer the PM in text now, briefly. Where a lookup failed or returned no data, say so "
    "plainly and state no figure for it; a short reply that names what you could not check "
    "is the right answer here."
)


def parse_turn(response: Mapping[str, Any]) -> TurnOutput | None:
    """The last text block parsed as JSON and validated into a `TurnOutput`.

    None when `stop_reason` is not `end_turn`, no text block exists, the text
    is blank, or the JSON or its mentions fail validation. A blank `text`
    would otherwise pass the turn schema (which sets no `minLength`) and later
    fail `Turn`'s own `min_length=1` after the reply is already cached, so it
    is rejected here instead.
    """
    payload = last_text_json(response)
    if not isinstance(payload, dict):
        return None
    try:
        text = payload["text"]
        if not isinstance(text, str) or not text.strip():
            return None
        mentions = tuple(Mention(**item) for item in payload.get("mentions") or ())
    except (TypeError, ValueError, KeyError):
        return None
    return TurnOutput(text=text, mentions=mentions)


@dataclass(frozen=True)
class _Accepted:
    """One accepted reply: a validated final turn, the tool_use blocks to run, or a refusal
    accepted only so it is cached and the fallback model can answer instead."""

    output: TurnOutput | None
    tool_blocks: tuple[Mapping[str, Any], ...]
    refused: bool = False


def _refusal_reason(response: Mapping[str, Any]) -> str:
    """ "the reply was refused", with the API's category and explanation when it gives them."""
    details = response.get("stop_details")
    if not isinstance(details, dict):
        return "the reply was refused"
    category = details.get("category") or "uncategorised"
    explanation = details.get("explanation")
    return f"the reply was refused ({category}{f': {explanation}' if explanation else ''})"


def _classify(
    response: Mapping[str, Any], allow_tool_use: bool, fallback_available: bool
) -> tuple[_Accepted | None, str]:
    """Accept a tool-use round or a final parsed turn; on rejection, name the reason why.

    With a fallback model available, a refusal is accepted as such rather than retried,
    so `_send_accepted` can commit it and send the request to that model instead.
    """
    stop_reason = response.get("stop_reason")
    if stop_reason == "refusal":
        if fallback_available:
            return _Accepted(output=None, tool_blocks=(), refused=True), ""
        return None, _refusal_reason(response)
    if stop_reason == "max_tokens":
        return None, "the reply hit max_tokens"
    if stop_reason == "tool_use":
        if not allow_tool_use:
            return None, "the reply used a tool where tools are not available"
        blocks = tuple(
            block for block in response.get("content") or () if block.get("type") == "tool_use"
        )
        if not blocks:
            return None, "the reply set stop_reason tool_use but named no tool"
        names = {str(block.get("name")) for block in blocks}
        unknown = sorted(names - _ADVISOR_TOOL_NAMES)
        if unknown:
            return None, f"the reply named an unknown tool: {', '.join(unknown)}"
        return _Accepted(output=None, tool_blocks=blocks), ""
    output = parse_turn(response)
    if output is None:
        return None, "the reply was unparsable or schema-invalid"
    return _Accepted(output=output, tool_blocks=()), ""


async def _send_accepted(
    client: CachedClient,
    request: Mapping[str, Any],
    config: DialogueConfig,
    session_id: str,
    *,
    allow_tool_use: bool,
) -> tuple[Reply, _Accepted, int, str]:
    """Send `request` until `_classify` accepts a reply; returns it with the model that answered.

    A refusal from the request's model is committed and the request re-sent once to
    `config.refusal_fallback_model`, with its own retries, when one is configured. The
    rejected count sums both models' attempts, the refusal included.
    """
    fallback = config.refusal_fallback_model

    async def send(
        body: Mapping[str, Any], fallback_available: bool
    ) -> tuple[Reply, _Accepted, int]:
        return await send_until_accepted(
            client,
            body,
            lambda response: _classify(response, allow_tool_use, fallback_available),
            scope=session_id,
            max_retries=config.max_retries,
            error_type=DialogueError,
        )

    reply, accepted, rejected = await send(request, fallback is not None)
    if not accepted.refused:
        return reply, accepted, rejected, request["model"]
    assert fallback is not None
    reply, accepted, fallback_rejected = await send({**request, "model": fallback}, False)
    return reply, accepted, rejected + 1 + fallback_rejected, fallback


def _sum_usage(totals: CallUsage, response: Mapping[str, Any]) -> CallUsage:
    added = usage_of(response)
    return CallUsage(
        input_tokens=totals.input_tokens + added.input_tokens,
        output_tokens=totals.output_tokens + added.output_tokens,
        cache_read_input_tokens=totals.cache_read_input_tokens + added.cache_read_input_tokens,
        cache_creation_input_tokens=(
            totals.cache_creation_input_tokens + added.cache_creation_input_tokens
        ),
    )


def _pm_directive_text(ctx: SessionContext, pm_index: int) -> str | None:
    """The stance line if the turn carries a stance, else turn 0's opening line, else None."""
    directive = ctx.turn_plan.pm_directives[pm_index]
    if directive.stance is not None:
        return directive.stance.stance
    if pm_index == 0 and directive.opening is not None:
        return opening_line(ctx, directive.opening)
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
    system_advisor = advisor_system(advisor_prompt, ctx.skeleton.date, advisor_instruments(ctx))

    narrator_messages: list[dict[str, Any]] = [
        {"role": "user", "content": NARRATOR_OPENING_MESSAGE}
    ]
    advisor_messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    turn_logs: list[TurnLog] = []
    rejected_replies = 0

    for i in range(n_pm):
        narrator_messages.append({"role": "system", "content": narrator_directive(ctx, i)})
        pm_request = narrator_request(ctx, i, narrator_messages, config, feedback)
        pm_reply, pm_accepted, pm_rejected, pm_model = await _send_accepted(
            client, pm_request, config, session_id, allow_tool_use=False
        )
        rejected_replies += pm_rejected
        narrator_messages.append({"role": "assistant", "content": pm_reply.response["content"]})
        pm_output = pm_accepted.output
        if pm_output is None:
            raise DialogueError(
                f"{session_prefix(session_id)}accepted narrator reply carried no output"
            )
        turn_logs.append(
            TurnLog(
                role=TurnRole.PM,
                text=pm_output.text,
                mentions=pm_output.mentions,
                directive=_pm_directive_text(ctx, i),
                scripted_violation=False,
                tool_calls=(),
                model=pm_model,
                request_hashes=(pm_reply.key,),
                usage=usage_of(pm_reply.response),
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
        usage = ZERO_USAGE
        rounds = 0
        tools_disabled = False
        while True:
            advisor_body = advisor_request(
                system_advisor, advisor_messages, config, tools_disabled=tools_disabled
            )
            reply, accepted, turn_rejected, advisor_model = await _send_accepted(
                client, advisor_body, config, session_id, allow_tool_use=not tools_disabled
            )
            rejected_replies += turn_rejected
            request_hashes.append(reply.key)
            usage = _sum_usage(usage, reply.response)

            if accepted.tool_blocks:
                advisor_messages.append({"role": "assistant", "content": reply.response["content"]})
                rounds += 1
                tool_result_blocks = []
                for block in accepted.tool_blocks:
                    tool_input = block.get("input") or {}
                    outcome = run_tool(
                        ctx.lookup,
                        block["name"],
                        tool_input,
                        ctx.skeleton.date,
                        ctx.instrument_names,
                    )
                    result_json = canonical_json(outcome.result)
                    tool_calls.append(
                        ToolCall(
                            name=AdvisorTool(block["name"]),
                            input_json=canonical_json(tool_input),
                            result_json=result_json,
                            is_error=outcome.is_error,
                        )
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": result_json,
                            "is_error": outcome.is_error,
                        }
                    )
                advisor_messages.append({"role": "user", "content": tool_result_blocks})
                if rounds >= config.max_tool_rounds and not tools_disabled:
                    tools_disabled = True
                    advisor_messages.append({"role": "system", "content": _TOOLS_OFF_NOTE})
                    # The last round's error count splits heavy research from failed guessing.
                    failed = sum(1 for block in tool_result_blocks if block.get("is_error"))
                    warnings.append(
                        f"{session_prefix(session_id)}advisor reply {i} used every tool round "
                        f"({failed} of {len(accepted.tool_blocks)} lookups in the last round "
                        "failed)"
                    )
                continue

            advisor_messages.append({"role": "assistant", "content": reply.response["content"]})
            advisor_output = accepted.output
            if advisor_output is None:
                raise DialogueError(
                    f"{session_prefix(session_id)}accepted advisor reply carried no output"
                )
            turn_logs.append(
                TurnLog(
                    role=TurnRole.ADVISOR,
                    text=advisor_output.text,
                    mentions=advisor_output.mentions,
                    directive=advisor_directive,
                    scripted_violation=is_violation,
                    tool_calls=tuple(tool_calls),
                    model=advisor_model,
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
    return SessionResult(
        session=session, log=log, warnings=tuple(warnings), rejected_replies=rejected_replies
    )
