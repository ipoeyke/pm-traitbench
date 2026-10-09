"""Narrator and advisor system prompts and Messages API request builders.

The narrator's inputs never contain a trait id, bias param name, trait
value, signal mode or the word "bias"; the advisor's inputs never contain
persona, rules, ideas, ledger, traits or skeleton text. Every request is
built from a `SessionContext` and a plain messages sequence with no
non-deterministic step, so identical inputs give identical cache keys.
"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from pm_traitbench.config import DialogueConfig
from pm_traitbench.dialogue.context import SessionContext, trade_key
from pm_traitbench.dialogue.tools import TOOL_DEFINITIONS
from pm_traitbench.dialogue.turns import Opening
from pm_traitbench.enums import Effort, Family, Side, Tenor
from pm_traitbench.errors import DialogueError
from pm_traitbench.levels import idea_level_text
from pm_traitbench.tables.schema import Instrument, LedgerRow

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


def _trade_mention_schema(trade_idea_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"const": "trade"},
            "instrument_id": {"type": "string"},
            # Structured output then rejects any id outside the turn's listed trades.
            "trade_idea_id": {"enum": list(trade_idea_ids)},
            "tenor": {"enum": _TENOR_ENUM},
            "side": {"enum": _SIDE_ENUM},
            "size": {"type": "number"},
            "field": _NULL,
            "value": _NULL,
        },
        "required": _MENTION_REQUIRED_KEYS,
        "additionalProperties": False,
    }


def _turn_schema(mention_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "mentions": {"type": "array", "items": mention_schema},
        },
        "required": ["text", "mentions"],
        "additionalProperties": False,
    }


def narrator_turn_schema(trade_idea_ids: Sequence[str]) -> dict[str, Any]:
    """A PM turn's reply schema, offering the trade branch only when the turn lists trades.

    Its `trade_idea_id` is an enum of exactly those ids, so neither an invented id nor
    a trade mention on a turn without listed trades can be generated.
    """
    if not trade_idea_ids:
        return _turn_schema(_LEVEL_MENTION_SCHEMA)
    return _turn_schema({"anyOf": [_trade_mention_schema(trade_idea_ids), _LEVEL_MENTION_SCHEMA]})


# The advisor never sees a trade idea id, so it never offers the trade branch.
ADVISOR_TURN_SCHEMA: dict[str, Any] = _turn_schema(_LEVEL_MENTION_SCHEMA)

_ADVISOR_MENTIONS_INSTRUCTION = (
    "Return your reply as `text` and `mentions`. For every market number you state, add "
    'a mention with kind "level", using the instrument_id and field name a tool '
    'returned (a curve point\'s field is "level"), the tenor a tool gave or null, and '
    "the value you stated. Summarise a price history (its latest level, range and trend) "
    "rather than listing every close."
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


def prompt_sha256(text: str) -> str:
    """sha256 hex of a prompt's UTF-8 text, recorded so a later stage can detect an edit."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_STANCE_HOLDS = (
    "A turn's \"In this message\" line is the PM's own decision, and the advisor's replies "
    "never change it: the PM may acknowledge the advisor's point, but carries the decision "
    "out anyway. When that line has the PM state a preference, the PM states only that "
    "preference and adds no other request about how the advisor should answer, such as its "
    "length, order, format, units or how hard it pushes back."
)


def _position_instrument_ids(ctx: SessionContext) -> tuple[str, ...]:
    """Every leg instrument of the open positions, then of the ideas, first occurrence kept."""
    ids = (leg.instrument_id for idea in (*ctx.open_positions, *ctx.ideas) for leg in idea.legs)
    return tuple(dict.fromkeys(ids))


def market_levels_section(ctx: SessionContext) -> str:
    """The latest levels on or before the session date for every position and idea instrument.

    Each level is named by the field a level mention uses, so the narrator can cite it.
    """
    today = ctx.skeleton.date
    lines: list[str] = []
    for instrument_id in _position_instrument_ids(ctx):
        name = ctx.lookup.instruments[instrument_id].name
        price = ctx.lookup.latest_price(instrument_id, today)
        if price is not None:
            spread = "" if price.spread_bp is None else f", spread_bp {price.spread_bp:.6g}"
            lines.append(
                f"- {name} ({instrument_id}): price {price.price:.6g}{spread} "
                f"(close {price.date.isoformat()})"
            )
        curve = ctx.lookup.curve_on_or_before(instrument_id, today)
        if curve is not None:
            curve_date, levels = curve
            tenors = ", ".join(
                f"{tenor.value} {levels[tenor]:.6g}" for tenor in Tenor if tenor in levels
            )
            lines.append(
                f"- {name} ({instrument_id}) curve, field level: {tenors} "
                f"(close {curve_date.isoformat()})"
            )
    if not lines:
        return "Latest market levels: none are available to you, so state no market level."
    return (
        "Latest market levels, the only market numbers you may state (in a level mention, "
        "use the field named here, with the tenor for a curve level):\n" + "\n".join(lines)
    )


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
        "preference. Add a trade mention only for a trade a turn's \"Mention each of these "
        'trades" directive lists, using its ti_ id and size exactly as given, and never '
        "invent one. Refer to open positions and the ideas below by instrument, or with a "
        "level mention for a market number you state, never with a trade mention. Fill "
        "`mentions` for every such trade and every market level you state, using the ids "
        "given. Describe each listed trade as its note says: a trade noted as closing the "
        "whole position is a full exit, not a trim, and the note names what drove it. "
        "State only the market levels listed under the latest market levels below.",
        _STANCE_HOLDS,
        f"Today is {ctx.skeleton.date.isoformat()}.",
        f"Asset class: {mandate.asset_class.value}. Sub-style: {mandate.sub_style}. "
        f"Book size: {mandate.book_size}. Risk unit: {mandate.risk_unit}. "
        f"Benchmark: {mandate.benchmark}.",
        persona.stated_profile.self_description,
    ]
    sections.extend(rule.text for rule in ctx.pm_rules)
    asset_class = mandate.asset_class
    for idea in ctx.ideas:
        name = ctx.instrument_names[idea.instrument_id]
        entry, target, stop = (
            idea_level_text(level, asset_class, idea.expression)
            for level in (idea.entry_level, idea.target_level, idea.stop_level)
        )
        sections.append(
            f"Idea {idea.trade_idea_id}: {name} ({idea.instrument_id}), {idea.side.value}, "
            f"entry {entry}, target {target}, stop {stop}. Thesis: {idea.thesis}"
        )
        sections.extend(
            rule.text for rule in ctx.idea_rules if rule.trade_idea_id == idea.trade_idea_id
        )
    sections.append(market_levels_section(ctx))
    sections.append(ctx.voice.line)
    if ctx.avoid_lines:
        sections.append(
            "Never do any of the following:\n" + "\n".join(f"- {line}" for line in ctx.avoid_lines)
        )
    if feedback:
        sections.append(f"Correction for this session:\n{feedback}")
    return "\n\n".join(sections)


def _trade_line(ctx: SessionContext, trade: LedgerRow) -> str:
    """One listed trade: its legs and size, then what it does to the position and why."""
    name = ctx.instrument_names[trade.instrument_id]
    tenor = f", tenor {trade.tenor.value}" if trade.tenor is not None else ""
    note = ctx.trade_notes[trade_key(trade)]
    what = note.kind if note.trigger is None else f"{note.kind}, {note.trigger}"
    return (
        f"{trade.trade_idea_id}: {name} ({trade.instrument_id}){tenor}, {trade.side.value} "
        f"{trade.size} at {trade.price_or_yield} ({what})"
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


def listed_trades(ctx: SessionContext, pm_index: int) -> tuple[LedgerRow, ...]:
    """The trades a PM turn's directive tells it to mention: turn 0's day trades, else none."""
    directive = ctx.turn_plan.pm_directives[pm_index]
    if pm_index == 0 and directive.opening is not None:
        return directive.trades
    return ()


def narrator_directive(ctx: SessionContext, pm_index: int) -> str:
    """One PM turn's mid-conversation directive text.

    Never includes the turn's stance's trait id, mode or entry: only its free-text line.
    """
    directive = ctx.turn_plan.pm_directives[pm_index]
    lines: list[str] = []
    if pm_index == 0 and directive.opening is not None:
        lines.append(opening_line(ctx, directive.opening))
        if trades_to_mention := listed_trades(ctx, pm_index):
            trades = "; ".join(_trade_line(ctx, trade) for trade in trades_to_mention)
            lines.append(f"Mention each of these trades: {trades}")
    if directive.stance is not None:
        lines.append(
            f"In this message: {directive.stance.stance}. This is your decision, and the "
            "advisor's replies never change it: you may acknowledge the advisor's point, but "
            "carry this out anyway. If it states a preference, state only that one and ask "
            "nothing else about how the advisor should answer."
        )
    if not lines:
        lines.append("Continue the conversation naturally in one short message.")
    return "\n".join(lines)


def instrument_universe_section(instruments: Iterable[Instrument]) -> str:
    """Every instrument the tools resolve, one line per family as `id: name` pairs."""
    by_family: dict[Family, list[str]] = {}
    for inst in sorted(instruments, key=lambda i: i.instrument_id):
        by_family.setdefault(inst.family, []).append(f"{inst.instrument_id}: {inst.name}")
    lines = [f"- {family.value}: {'; '.join(entries)}" for family, entries in by_family.items()]
    return "Market universe, the instruments a lookup accepts, by id or name:\n" + "\n".join(lines)


def advisor_system(advisor_prompt: str, day: date, instruments: Iterable[Instrument]) -> str:
    """The advisor prompt, the mentions instruction, the universe, then the session date.

    The mentions instruction and the universe are appended here rather than living
    in the authored prompt file, so swapping that file can never drop them.
    """
    return (
        f"{advisor_prompt}\n\n{_ADVISOR_MENTIONS_INSTRUCTION}\n\n"
        f"{instrument_universe_section(instruments)}\n\nToday is {day.isoformat()}."
    )


def base_request(
    model: str,
    max_tokens: int,
    effort: Effort,
    system: str,
    messages: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """A Messages API request body with structured output: the keys every request shares."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": list(messages),
        "output_config": {
            "effort": effort.value,
            "format": {"type": "json_schema", "schema": schema},
        },
    }


def narrator_request(
    ctx: SessionContext,
    pm_index: int,
    messages: Sequence[Mapping[str, Any]],
    config: DialogueConfig,
    feedback: str | None,
) -> dict[str, Any]:
    """The narrator's Messages API request body for PM turn `pm_index`.

    Only `model`, `max_tokens`, `system`, `messages`, `output_config` and
    `cache_control`; never `thinking` or a sampling param. Its schema is
    `narrator_turn_schema` over the turn's listed trades.
    """
    trade_idea_ids = sorted({trade.trade_idea_id for trade in listed_trades(ctx, pm_index)})
    return {
        **base_request(
            config.narrator_model,
            config.max_output_tokens,
            config.effort,
            narrator_system(ctx, feedback),
            messages,
            narrator_turn_schema(trade_idea_ids),
        ),
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
    `ADVISOR_TURN_SCHEMA`, not a narrator schema, since the advisor never sees a
    trade idea id and so can never fill a trade mention.
    """
    request = {
        **base_request(
            config.advisor_model,
            config.max_output_tokens,
            config.effort,
            system,
            messages,
            ADVISOR_TURN_SCHEMA,
        ),
        "cache_control": {"type": "ephemeral"},
        "tools": list(TOOL_DEFINITIONS),
    }
    if tools_disabled:
        request["tool_choice"] = {"type": "none"}
    return request
