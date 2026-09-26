"""Shared dialogue-test fixtures: fake message builders, an in-memory `LlmClient`, and a
`MarketLookup` built on the shared fixture market.

Consumed by client, tools, prompt and session tests, so a canned response's
shape only has to match `Message.to_dict()` in one place.
"""

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from pm_traitbench.dialogue.tools import MarketLookup
from tests.engine.conftest import fixture_market  # noqa: F401


def fake_message(
    content: list[dict],
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict:
    """A dict shaped like `Message.to_dict()`, for a fake client's canned responses."""
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def turn_text(text: str, mentions: list[dict] | None = None) -> dict:
    """A text content block whose text is a turn's `{text, mentions}` JSON payload."""
    return {"type": "text", "text": json.dumps({"text": text, "mentions": mentions})}


def tool_use(name: str, tool_input: dict, tool_use_id: str) -> dict:
    """A tool-use content block."""
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}


class FakeClient:
    """An in-memory `LlmClient` scripted by a responder function; records every request."""

    def __init__(self, responder: Callable[[Mapping[str, Any]], dict]) -> None:
        self._responder = responder
        self.requests: list[Mapping[str, Any]] = []

    async def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return self._responder(request)


def default_responder(request: Mapping[str, Any]) -> dict:
    """Canned replies: advisor requests carry `tools`, narrator requests do not."""
    if "tools" in request:
        return fake_message([turn_text("Sounds reasonable, tell me more.")])
    return fake_message([turn_text("Feeling good about the book today.")])


@pytest.fixture(scope="module")
def market_lookup(fixture_market: dict) -> MarketLookup:
    """A `MarketLookup` built from the shared fixture market, seed 'T'."""
    return MarketLookup.build(
        seed="T",
        instruments=fixture_market["instruments"],
        prices=fixture_market["prices"],
        curves=fixture_market["curves"],
        consensus=fixture_market["consensus"],
        calendar=fixture_market["calendar"],
    )
