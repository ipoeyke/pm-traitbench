"""Tests for the cached LLM client: the request key, the disk cache and the fake backend."""

import asyncio
import json
from pathlib import Path

import pytest

from pm_traitbench.dialogue.client import AnthropicClient, CachedClient, request_key
from pm_traitbench.errors import DialogueBudgetError, DialogueError
from tests.dialogue.conftest import FakeClient, default_responder, fake_message, turn_text

_REQUEST = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}

_CREDENTIAL_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)


def _failing_factory() -> FakeClient:
    raise AssertionError("inner_factory must not be called")


def test_request_key_ignores_dict_order() -> None:
    a = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
    b = {"messages": [{"content": "hi", "role": "user"}], "model": "claude-opus-5-5"}
    assert request_key(a) == request_key(b)


def test_request_key_changes_with_model() -> None:
    a = {**_REQUEST, "model": "claude-opus-5-5"}
    b = {**_REQUEST, "model": "claude-sonnet-5"}
    assert request_key(a) != request_key(b)


def test_miss_calls_inner_and_commit_makes_the_next_send_a_hit(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        first = await cached.send(_REQUEST)
        assert first.cached is False
        cached.commit(first)
        second = await cached.send(_REQUEST)
        assert second.cached is True
        assert second.response == first.response

    asyncio.run(scenario())
    assert len(inner.requests) == 1
    assert cached.totals.cache_hits == 1
    assert cached.totals.calls == 2


def test_uncommitted_reply_is_not_cached(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        first = await cached.send(_REQUEST)
        second = await cached.send(_REQUEST)
        assert first.cached is False
        assert second.cached is False

    asyncio.run(scenario())
    assert len(inner.requests) == 2


def test_cache_file_is_written_atomically_under_a_two_char_shard(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        reply = await cached.send(_REQUEST)
        cached.commit(reply)
        return reply

    reply = asyncio.run(scenario())
    key = request_key(_REQUEST)
    shard_dir = tmp_path / key[:2]
    expected_file = shard_dir / f"{key}.json"
    assert expected_file.exists()
    assert [p.name for p in shard_dir.iterdir()] == [f"{key}.json"]
    stored = json.loads(expected_file.read_text(encoding="utf-8"))
    assert stored["key"] == key
    assert stored["response"] == reply.response


def test_inner_factory_is_not_called_when_every_request_hits(tmp_path: Path) -> None:
    warm = CachedClient(lambda: FakeClient(default_responder), tmp_path, token_budget=None)

    async def prime() -> None:
        reply = await warm.send(_REQUEST)
        warm.commit(reply)

    asyncio.run(prime())

    cached = CachedClient(_failing_factory, tmp_path, token_budget=None)

    async def scenario() -> None:
        reply = await cached.send(_REQUEST)
        assert reply.cached is True

    asyncio.run(scenario())


def test_budget_error_before_a_fresh_call_once_spent(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=150)
    other_request = {**_REQUEST, "messages": [{"role": "user", "content": "bye"}]}

    async def scenario() -> None:
        first = await cached.send(_REQUEST)
        cached.commit(first)
        with pytest.raises(DialogueBudgetError):
            await cached.send(other_request)
        # A cached send still succeeds once the budget is spent.
        replay = await cached.send(_REQUEST)
        assert replay.cached is True

    asyncio.run(scenario())


def test_fresh_totals_exclude_cache_hits(tmp_path: Path) -> None:
    inner = FakeClient(
        lambda request: fake_message([turn_text("ok")], input_tokens=7, output_tokens=3)
    )
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        first = await cached.send(_REQUEST)
        cached.commit(first)
        await cached.send(_REQUEST)

    asyncio.run(scenario())
    assert cached.totals.input_tokens == 7
    assert cached.totals.output_tokens == 3
    assert cached.totals.cache_hits == 1


def test_anthropic_client_defers_construction_until_the_first_send() -> None:
    client = AnthropicClient(max_concurrency=2)
    assert client._client is None


def test_anthropic_client_reports_missing_credentials_as_a_dialogue_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No credentials, no profile pointer file discoverable: the SDK resolves
    # this at request time (a bare TypeError), never by reaching the network.
    for name in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    client = AnthropicClient(max_concurrency=1)
    minimal_request = {
        "model": "claude-opus-5-5",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }

    with pytest.raises(DialogueError, match="ant auth login"):
        asyncio.run(client.send(minimal_request))
