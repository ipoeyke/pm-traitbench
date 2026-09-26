"""Narrator and advisor system prompts and Messages API request builders.

The narrator's inputs never contain a trait id, bias param name, trait
value, signal mode or the word "bias"; the advisor's inputs never contain
persona, rules, ideas, ledger, traits or skeleton text. Every request is
built from a `SessionContext` and a plain messages sequence with no
non-deterministic step, so identical inputs give identical cache keys.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from pm_traitbench.config import DialogueConfig
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.tools import TOOL_DEFINITIONS
from pm_traitbench.dialogue.turns import Opening
from pm_traitbench.enums import Side, Tenor
from pm_traitbench.errors import DialogueError
from pm_traitbench.tables.schema import LedgerRow

NARRATOR_OPENING_MESSAGE = "The advisor is ready for your first message."

_MENTION_REQUIRED_KEYS = [
    "kind",
    "instrument_id",
    "trade_idea_id",
    "tenor",
    "side",
    "size",
    "field",
    "value",
]
_TENOR_ENUM: list[str | None] = [tenor.value for tenor in Tenor] + [None]
_SIDE_ENUM: list[str] = [side.value for side in Side]
_NULL = {"type": "null"}

_TRADE_MENTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"const": "trade"},
        "instrument_id": {"type": "string"},
        "trade_idea_id": {"type": "string"},
        "tenor": {"enum": _TENOR_ENUM},
        "side": {"enum": _SIDE_ENUM},
        "size": {"type": "number"},
        "field": _NULL,
        "value": _NULL,
    },
    "required": _MENTION_REQUIRED_KEYS,
    "additionalProperties": False,
}
_LEVEL_MENTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"const": "level"},
        "instrument_id": {"type": "string"},
        "trade_idea_id": _NULL,
        "tenor": {"enum": _TENOR_ENUM},
        "side": _NULL,
        "size": _NULL,
        "field": {"type": "string"},
        "value": {"type": "number"},
    },
    "required": _MENTION_REQUIRED_KEYS,
    "additionalProperties": False,
}
TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "mentions": {
            "type": "array",
            "items": {"anyOf": [_TRADE_MENTION_SCHEMA, _LEVEL_MENTION_SCHEMA]},
        },
    },
    "required": ["text", "mentions"],
    "additionalProperties": False,
}
ADVISOR_TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        # The advisor never sees a trade idea id, so it never offers the trade branch.
        "mentions": {"type": "array", "items": _LEVEL_MENTION_SCHEMA},
    },
    "required": ["text", "mentions"],
    "additionalProperties": False,
}

_ADVISOR_MENTIONS_INSTRUCTION = (
    "Return your reply as `text` and `mentions`. For every market number you state, add "
    'a mention with kind "level", using the instrument_id and field name a tool '
    'returned (a curve point\'s field is "level"), the tenor a tool gave or null, and '
    "the value you stated."
)


def read_advisor_prompt(path: Path | None) -> str:
    """The advisor system prompt text: the packaged file when `path` is None.

    Raises `DialogueError` if the file cannot be read or its content is blank.
    """
    if path is None:
        traversable = resources.files("pm_traitbench.catalogues").joinpath("advisor_prompt.md")
        try:
            text = traversable.read_text(encoding="utf-8")
        except OSError as e:
            raise DialogueError(f"packaged advisor prompt is missing: {e}") from e
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise DialogueError(f"advisor prompt file '{path}' is missing: {e}") from e
    if not text.strip():
        raise DialogueError("advisor prompt is blank")
    return text


def narrator_system(ctx: SessionContext, feedback: str | None) -> str:
    """The narrator's stable system prompt: role, mandate, ideas, voice and forbidden list.

    Never contains a trait id, bias param name, trait value, signal mode or
    the word "bias", so the narrator cannot recover which trait a stance carries.
    """
    persona = ctx.persona
    mandate = persona.mandate
    sections = [
        "Write only the PM's side of a conversation with their investment copilot, in "
        "first person, in the voice below. Never name a psychological trait, tendency or "
        "preference. Mention a trade only when a turn's directive lists it, and never "
        "invent one. Fill `mentions` for every trade and every market level you state, "
        "using the ids given.",
        f"Today is {ctx.skeleton.date.isoformat()}.",
        f"Asset class: {mandate.asset_class.value}. Sub-style: {mandate.sub_style}. "
        f"Book size: {mandate.book_size}. Risk unit: {mandate.risk_unit}. "
        f"Benchmark: {mandate.benchmark}.",
        persona.stated_profile.self_description,
    ]
    sections.extend(rule.text for rule in ctx.pm_rules)
    for idea in ctx.ideas:
        name = ctx.instrument_names[idea.instrument_id]
        sections.append(
            f"Idea {idea.trade_idea_id}: {name} ({idea.instrument_id}), {idea.side.value}, "
            f"entry {idea.entry_level}, target {idea.target_level}, stop {idea.stop_level}. "
            f"Thesis: {idea.thesis}"
        )
        sections.extend(
            rule.text for rule in ctx.idea_rules if rule.trade_idea_id == idea.trade_idea_id
        )
    sections.append(ctx.voice.line)
    if ctx.avoid_lines:
        sections.append(
            "Never do any of the following:\n" + "\n".join(f"- {line}" for line in ctx.avoid_lines)
        )
    if feedback:
        sections.append(f"Correction for this session:\n{feedback}")
    return "\n\n".join(sections)


def _trade_line(ctx: SessionContext, trade: LedgerRow) -> str:
    name = ctx.instrument_names[trade.instrument_id]
    tenor = f", tenor {trade.tenor.value}" if trade.tenor is not None else ""
    return (
        f"{trade.trade_idea_id}: {name} ({trade.instrument_id}){tenor}, {trade.side.value} "
        f"{trade.size} at {trade.price_or_yield}"
    )


_NO_OPEN_POSITIONS = (
    "Open with a routine check-in. You hold no open positions today, so ask the advisor "
    "about the markets you trade without naming a position."
)


def opening_line(ctx: SessionContext, opening: Opening) -> str:
    """The free-text instruction for turn 0's opening, one line per `Opening` kind."""
    if opening == Opening.SESSION_IDEAS:
        return "Open the conversation about today's decision on your ideas listed above."
    if opening == Opening.OPEN_POSITIONS:
        if not ctx.open_positions:
            return _NO_OPEN_POSITIONS
        names = "; ".join(
            f"{ctx.instrument_names[p.instrument_id]} ({p.side.value})" for p in ctx.open_positions
        )
        return f"Open with a routine check on your open positions: {names}"
    instrument = ctx.question_instrument
    if instrument is None:
        raise DialogueError("a market_question opening requires a question_instrument")
    name = ctx.instrument_names[instrument.instrument_id]
    return (
        f"Ask the advisor one purely factual market question about {name} "
        f"({instrument.instrument_id}). Do not talk about your own positions, rules or habits."
    )


def narrator_directive(ctx: SessionContext, pm_index: int) -> str:
    """One PM turn's mid-conversation directive text.

    Never includes the turn's stance's trait id, mode or entry: only its free-text line.
    """
    directive = ctx.turn_plan.pm_directives[pm_index]
    lines: list[str] = []
    if pm_index == 0 and directive.opening is not None:
        lines.append(opening_line(ctx, directive.opening))
        if directive.trades:
            trades = "; ".join(_trade_line(ctx, trade) for trade in directive.trades)
            lines.append(f"Mention each of these trades: {trades}")
    if directive.stance is not None:
        lines.append(f"In this message: {directive.stance.stance}")
    if not lines:
        lines.append("Continue the conversation naturally in one short message.")
    return "\n".join(lines)


def advisor_system(advisor_prompt: str, day: date) -> str:
    """The advisor prompt, the mentions instruction, then the session date.

    The mentions instruction is appended here rather than living in the
    authored prompt file, so swapping that file can never drop it.
    """
    return f"{advisor_prompt}\n\n{_ADVISOR_MENTIONS_INSTRUCTION}\n\nToday is {day.isoformat()}."


def narrator_request(
    ctx: SessionContext,
    messages: Sequence[Mapping[str, Any]],
    config: DialogueConfig,
    feedback: str | None,
) -> dict[str, Any]:
    """The narrator's Messages API request body.

    Only `model`, `max_tokens`, `system`, `messages`, `output_config` and
    `cache_control`; never `thinking` or a sampling param.
    """
    return {
        "model": config.narrator_model,
        "max_tokens": config.max_output_tokens,
        "system": narrator_system(ctx, feedback),
        "messages": list(messages),
        "output_config": {
            "effort": config.effort.value,
            "format": {"type": "json_schema", "schema": TURN_SCHEMA},
        },
        "cache_control": {"type": "ephemeral"},
    }


def advisor_request(
    system: str,
    messages: Sequence[Mapping[str, Any]],
    config: DialogueConfig,
    *,
    tools_disabled: bool = False,
) -> dict[str, Any]:
    """The advisor's Messages API request body.

    The narrator's keys plus `tools`, and `tool_choice` only when tools are
    disabled; never `thinking` or a sampling param. Its schema is
    `ADVISOR_TURN_SCHEMA`, not `TURN_SCHEMA`, since the advisor never sees a
    trade idea id and so can never fill a trade mention.
    """
    request: dict[str, Any] = {
        "model": config.advisor_model,
        "max_tokens": config.max_output_tokens,
        "system": system,
        "messages": list(messages),
        "output_config": {
            "effort": config.effort.value,
            "format": {"type": "json_schema", "schema": ADVISOR_TURN_SCHEMA},
        },
        "cache_control": {"type": "ephemeral"},
        "tools": list(TOOL_DEFINITIONS),
    }
    if tools_disabled:
        request["tool_choice"] = {"type": "none"}
    return request
