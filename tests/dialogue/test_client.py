"""Tests for the cached LLM client: the request key, the disk cache and the fake backend."""

import asyncio
import json
import os
from pathlib import Path

import anthropic
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
    # The blocked send never reached the inner client.
    assert len(inner.requests) == 1


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


def test_fresh_totals_accumulate_cache_read_tokens(tmp_path: Path) -> None:
    response = fake_message([turn_text("ok")], input_tokens=7, output_tokens=3)
    response["usage"]["cache_read_input_tokens"] = 40
    inner = FakeClient(lambda request: response)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    asyncio.run(cached.send(_REQUEST))
    assert cached.totals.cache_read_tokens == 40


def test_null_usage_fields_count_as_zero(tmp_path: Path) -> None:
    # `to_dict()` can carry explicit nulls for the cache usage fields; a null
    # must not blow up token accounting with a `+= None` TypeError.
    response = fake_message([turn_text("ok")], input_tokens=5, output_tokens=2)
    response["usage"]["cache_read_input_tokens"] = None
    response["usage"]["cache_creation_input_tokens"] = None
    inner = FakeClient(lambda request: response)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    reply = asyncio.run(cached.send(_REQUEST))

    assert reply.cached is False
    assert cached.totals.input_tokens == 5
    assert cached.totals.output_tokens == 2
    assert cached.totals.cache_read_tokens == 0


def test_unreadable_cache_entry_is_treated_as_a_miss_and_rewritten(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(_REQUEST)
    shard_dir = tmp_path / key[:2]
    shard_dir.mkdir(parents=True)
    (shard_dir / f"{key}.json").write_text("{not valid json", encoding="utf-8")

    async def scenario() -> None:
        reply = await cached.send(_REQUEST)
        assert reply.cached is False
        cached.commit(reply)
        return reply

    reply = asyncio.run(scenario())
    assert len(inner.requests) == 1
    stored = json.loads((shard_dir / f"{key}.json").read_text(encoding="utf-8"))
    assert stored["response"] == reply.response


def test_cache_entry_missing_the_response_key_is_treated_as_a_miss(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(_REQUEST)
    shard_dir = tmp_path / key[:2]
    shard_dir.mkdir(parents=True)
    (shard_dir / f"{key}.json").write_text(json.dumps({"key": key}), encoding="utf-8")

    reply = asyncio.run(cached.send(_REQUEST))
    assert reply.cached is False
    assert len(inner.requests) == 1


def test_temp_file_is_removed_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    reply = asyncio.run(cached.send(_REQUEST))

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(OSError):
        cached.commit(reply)

    key = request_key(_REQUEST)
    shard_dir = tmp_path / key[:2]
    assert list(shard_dir.iterdir()) == []


def test_anthropic_client_builds_the_sdk_client_lazily_on_first_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_count = 0

    class _StubMessage:
        def to_dict(self) -> dict:
            return fake_message([turn_text("ok")])

    class _StubMessages:
        async def create(self, **request: object) -> _StubMessage:
            return _StubMessage()

    class _StubClient:
        def __init__(self) -> None:
            nonlocal build_count
            build_count += 1
            self.messages = _StubMessages()

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _StubClient)

    client = AnthropicClient(max_concurrency=1)
    assert build_count == 0  # not built until the first send

    asyncio.run(client.send(_REQUEST))
    assert build_count == 1

    asyncio.run(client.send(_REQUEST))
    assert build_count == 1  # reused, not rebuilt


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
