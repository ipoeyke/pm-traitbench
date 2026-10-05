"""Tests for the cached LLM client: the request key, the disk cache and the fake backend."""

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import pytest

from pm_traitbench.config import Config
from pm_traitbench.dialogue import client as client_module
from pm_traitbench.dialogue.client import (
    AnthropicClient,
    CachedClient,
    request_key,
    send_until_accepted,
)
from pm_traitbench.errors import DialogueBudgetError, DialogueError
from tests.dialogue.fixtures import (
    SESSION_ID,
    FakeClient,
    default_responder,
    fake_message,
    turn_text,
)

_API_URL = "https://api.anthropic.com/v1/messages"


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=httpx2.Request("POST", _API_URL))
    return cls("boom", response=response, body=None)


def _connection_error(
    cls: type[anthropic.APIConnectionError],
) -> anthropic.APIConnectionError:
    return cls(request=httpx2.Request("POST", _API_URL))


_REQUEST = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}
_SCOPE = SESSION_ID

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
    assert request_key(a, _SCOPE) == request_key(b, _SCOPE)


def test_request_key_changes_with_model() -> None:
    a = {**_REQUEST, "model": "claude-opus-5-5"}
    b = {**_REQUEST, "model": "claude-sonnet-5"}
    assert request_key(a, _SCOPE) != request_key(b, _SCOPE)


def test_request_key_changes_with_scope() -> None:
    """Two sessions that build the byte-identical request must never share a cache entry."""
    assert request_key(_REQUEST, "session_a") != request_key(_REQUEST, "session_b")


def test_byte_identical_requests_under_different_scopes_use_distinct_cache_files(
    tmp_path: Path,
) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        first = await cached.send(_REQUEST, scope="session_a")
        cached.commit(first)
        second = await cached.send(_REQUEST, scope="session_b")
        cached.commit(second)
        return first, second

    first, second = asyncio.run(scenario())
    assert first.key != second.key
    assert (tmp_path / first.key[:2] / f"{first.key}.json").exists()
    assert (tmp_path / second.key[:2] / f"{second.key}.json").exists()
    # Neither send was a hit: the second one must not have reused the first's reply.
    assert first.cached is False
    assert second.cached is False
    assert len(inner.requests) == 2


def test_miss_calls_inner_and_commit_makes_the_next_send_a_hit(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        first = await cached.send(_REQUEST, scope=_SCOPE)
        assert first.cached is False
        cached.commit(first)
        second = await cached.send(_REQUEST, scope=_SCOPE)
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
        first = await cached.send(_REQUEST, scope=_SCOPE)
        second = await cached.send(_REQUEST, scope=_SCOPE)
        assert first.cached is False
        assert second.cached is False

    asyncio.run(scenario())
    assert len(inner.requests) == 2


def test_cache_file_is_written_atomically_under_a_two_char_shard(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> None:
        reply = await cached.send(_REQUEST, scope=_SCOPE)
        cached.commit(reply)
        return reply

    reply = asyncio.run(scenario())
    key = request_key(_REQUEST, _SCOPE)
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
        reply = await warm.send(_REQUEST, scope=_SCOPE)
        warm.commit(reply)

    asyncio.run(prime())

    cached = CachedClient(_failing_factory, tmp_path, token_budget=None)

    async def scenario() -> None:
        reply = await cached.send(_REQUEST, scope=_SCOPE)
        assert reply.cached is True

    asyncio.run(scenario())


def test_budget_error_before_a_fresh_call_once_spent(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=150)
    other_request = {**_REQUEST, "messages": [{"role": "user", "content": "bye"}]}

    async def scenario() -> None:
        first = await cached.send(_REQUEST, scope=_SCOPE)
        cached.commit(first)
        with pytest.raises(DialogueBudgetError):
            await cached.send(other_request, scope=_SCOPE)
        # A cached send still succeeds once the budget is spent.
        replay = await cached.send(_REQUEST, scope=_SCOPE)
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
        first = await cached.send(_REQUEST, scope=_SCOPE)
        cached.commit(first)
        await cached.send(_REQUEST, scope=_SCOPE)

    asyncio.run(scenario())
    assert cached.totals.input_tokens == 7
    assert cached.totals.output_tokens == 3
    assert cached.totals.cache_hits == 1


def test_fresh_totals_accumulate_cache_read_tokens(tmp_path: Path) -> None:
    response = fake_message([turn_text("ok")], input_tokens=7, output_tokens=3)
    response["usage"]["cache_read_input_tokens"] = 40
    inner = FakeClient(lambda request: response)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    asyncio.run(cached.send(_REQUEST, scope=_SCOPE))
    assert cached.totals.cache_read_tokens == 40


def test_totals_split_usage_by_model_and_cost_it(tmp_path: Path) -> None:
    def respond(request):
        response = fake_message([turn_text("ok")], input_tokens=1_000_000, output_tokens=100_000)
        response["usage"]["cache_read_input_tokens"] = 500_000
        response["usage"]["cache_creation_input_tokens"] = 200_000
        return response

    cached = CachedClient(lambda: FakeClient(respond), tmp_path, token_budget=None)
    asyncio.run(cached.send(_REQUEST, scope=_SCOPE))
    asyncio.run(cached.send({**_REQUEST, "model": "claude-sonnet-5"}, scope=_SCOPE))
    asyncio.run(cached.send({**_REQUEST, "model": "claude-sonnet-5"}, scope="other"))
    asyncio.run(cached.send({**_REQUEST, "model": "claude-mystery-1"}, scope=_SCOPE))

    prices = Config().prices.models
    metadata = cached.totals.as_metadata(prices)

    assert cached.totals.cache_creation_tokens == 800_000
    opus = metadata["usage_by_model"]["claude-opus-5-5"]
    assert opus["input_tokens"] == 1_000_000 and opus["cache_creation_input_tokens"] == 200_000
    # 1M input at $4, 100K output at $20, 500K cache reads at $0.20, 200K writes at $5.
    assert opus["cost_usd"] == 4.0 + 2.0 + 0.1 + 1.0
    sonnet = metadata["usage_by_model"]["claude-sonnet-5"]
    assert sonnet["input_tokens"] == 2_000_000
    assert sonnet["cost_usd"] == 2 * (2.0 + 1.0 + 0.1 + 0.5)
    assert "cost_usd" not in metadata["usage_by_model"]["claude-mystery-1"]
    assert metadata["unpriced_models"] == ["claude-mystery-1"]
    assert metadata["cost_usd"] == round(7.1 + 7.2, 4)


def test_cache_hits_add_no_usage_or_cost(tmp_path: Path) -> None:
    response = fake_message([turn_text("ok")], input_tokens=7, output_tokens=3)
    cached = CachedClient(lambda: FakeClient(lambda _r: response), tmp_path, token_budget=None)
    reply = asyncio.run(cached.send(_REQUEST, scope=_SCOPE))
    cached.commit(reply)
    asyncio.run(cached.send(_REQUEST, scope=_SCOPE))

    metadata = cached.totals.as_metadata(Config().prices.models)

    assert metadata["cache_hits"] == 1
    assert metadata["usage_by_model"]["claude-opus-5-5"]["input_tokens"] == 7
    assert metadata["cost_usd"] == round(7 * 4e-6 + 3 * 20e-6, 4)


def test_null_usage_fields_count_as_zero(tmp_path: Path) -> None:
    # `to_dict()` can carry explicit nulls for the cache usage fields; a null
    # must not blow up token accounting with a `+= None` TypeError.
    response = fake_message([turn_text("ok")], input_tokens=5, output_tokens=2)
    response["usage"]["cache_read_input_tokens"] = None
    response["usage"]["cache_creation_input_tokens"] = None
    inner = FakeClient(lambda request: response)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    reply = asyncio.run(cached.send(_REQUEST, scope=_SCOPE))

    assert reply.cached is False
    assert cached.totals.input_tokens == 5
    assert cached.totals.output_tokens == 2
    assert cached.totals.cache_read_tokens == 0


def test_unreadable_cache_entry_is_treated_as_a_miss_and_rewritten(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(_REQUEST, _SCOPE)
    shard_dir = tmp_path / key[:2]
    shard_dir.mkdir(parents=True)
    (shard_dir / f"{key}.json").write_text("{not valid json", encoding="utf-8")

    async def scenario() -> None:
        reply = await cached.send(_REQUEST, scope=_SCOPE)
        assert reply.cached is False
        cached.commit(reply)
        return reply

    reply = asyncio.run(scenario())
    assert len(inner.requests) == 1
    stored = json.loads((shard_dir / f"{key}.json").read_text(encoding="utf-8"))
    assert stored["response"] == reply.response


def test_cache_entry_with_invalid_utf8_bytes_is_treated_as_a_miss(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(_REQUEST, _SCOPE)
    shard_dir = tmp_path / key[:2]
    shard_dir.mkdir(parents=True)
    (shard_dir / f"{key}.json").write_bytes(b"\xff\xfe not valid utf-8")

    reply = asyncio.run(cached.send(_REQUEST, scope=_SCOPE))
    assert reply.cached is False
    assert len(inner.requests) == 1


def test_cache_entry_that_is_not_a_json_object_is_treated_as_a_miss(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(_REQUEST, _SCOPE)
    shard_dir = tmp_path / key[:2]
    shard_dir.mkdir(parents=True)
    (shard_dir / f"{key}.json").write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    reply = asyncio.run(cached.send(_REQUEST, scope=_SCOPE))
    assert reply.cached is False
    assert len(inner.requests) == 1


def test_cache_entry_missing_the_response_key_is_treated_as_a_miss(tmp_path: Path) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    key = request_key(_REQUEST, _SCOPE)
    shard_dir = tmp_path / key[:2]
    shard_dir.mkdir(parents=True)
    (shard_dir / f"{key}.json").write_text(json.dumps({"key": key}), encoding="utf-8")

    reply = asyncio.run(cached.send(_REQUEST, scope=_SCOPE))
    assert reply.cached is False
    assert len(inner.requests) == 1


def test_temp_file_is_removed_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = FakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    reply = asyncio.run(cached.send(_REQUEST, scope=_SCOPE))

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(OSError):
        cached.commit(reply)

    key = request_key(_REQUEST, _SCOPE)
    shard_dir = tmp_path / key[:2]
    assert list(shard_dir.iterdir()) == []


def test_refresh_bypasses_the_cache_and_the_commit_overwrites_the_entry(tmp_path: Path) -> None:
    calls = {"n": 0}

    def responder(request: dict) -> dict:
        calls["n"] += 1
        return fake_message([turn_text(f"reply {calls['n']}")])

    inner = FakeClient(responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)

    async def scenario() -> tuple:
        first = await cached.send(_REQUEST, scope=_SCOPE)
        cached.commit(first)
        second = await cached.send(_REQUEST, scope=_SCOPE, refresh=True)
        cached.commit(second)
        return first, second

    first, second = asyncio.run(scenario())
    assert second.cached is False
    assert second.response != first.response
    assert len(inner.requests) == 2
    stored = json.loads((tmp_path / second.key[:2] / f"{second.key}.json").read_text())
    assert stored["response"] == second.response


class _StubMessage:
    def to_dict(self) -> dict:
        return fake_message([turn_text("ok")])


class _StubStream:
    """Stands in for the SDK's stream manager; raises `on_enter` on open, `on_read` on read."""

    def __init__(self, on_enter: Exception | None = None, on_read: Exception | None = None):
        self._on_enter = on_enter
        self._on_read = on_read

    async def __aenter__(self) -> "_StubStream":
        if self._on_enter is not None:
            raise self._on_enter
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get_final_message(self) -> _StubMessage:
        if self._on_read is not None:
            raise self._on_read
        return _StubMessage()


class _StubMessages:
    def __init__(self, stream: Callable[[], _StubStream] = _StubStream) -> None:
        self._stream = stream
        self.requests: list[dict] = []

    def stream(self, **request: object) -> _StubStream:
        self.requests.append(request)
        return self._stream()


def _stub_client_factory(build_count: dict[str, int], captured_kwargs: dict[str, object]) -> type:
    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            build_count["n"] += 1
            captured_kwargs.update(kwargs)
            self.messages = _StubMessages()

    return _StubClient


def test_anthropic_client_builds_the_sdk_client_lazily_on_first_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_count = {"n": 0}
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _stub_client_factory(build_count, {}))

    client = AnthropicClient(max_concurrency=1)
    assert build_count["n"] == 0  # not built until the first send

    asyncio.run(client.send(_REQUEST))
    assert build_count["n"] == 1

    asyncio.run(client.send(_REQUEST))
    assert build_count["n"] == 1  # reused, not rebuilt


def test_anthropic_client_passes_max_retries_to_the_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", _stub_client_factory({"n": 0}, captured_kwargs)
    )

    client = AnthropicClient(max_concurrency=1, max_retries=7)
    asyncio.run(client.send(_REQUEST))

    assert captured_kwargs["max_retries"] == 7


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

    with pytest.raises(DialogueError, match="request could not be built"):
        asyncio.run(client.send(minimal_request))


@pytest.mark.parametrize(
    ("build_error", "expected_status"),
    [
        (lambda: _status_error(anthropic.RateLimitError, 429), 429),
        (lambda: _status_error(anthropic.InternalServerError, 529), 529),
        (lambda: _status_error(anthropic.NotFoundError, 404), 404),
        (lambda: _connection_error(anthropic.APIConnectionError), None),
        (lambda: _connection_error(anthropic.APITimeoutError), None),
    ],
)
@pytest.mark.parametrize("phase", ["on_enter", "on_read"])
def test_transient_sdk_errors_map_to_a_dialogue_error(
    monkeypatch: pytest.MonkeyPatch, build_error, expected_status: int | None, phase: str
) -> None:
    """Errors map the same whether raised opening the stream or reading it."""
    error = build_error()

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            self.messages = _StubMessages(lambda: _StubStream(**{phase: error}))

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _StubClient)
    client = AnthropicClient(max_concurrency=1)

    with pytest.raises(DialogueError) as excinfo:
        asyncio.run(client.send(_REQUEST))

    message = str(excinfo.value)
    assert type(error).__name__ in message
    if expected_status is not None:
        assert str(expected_status) in message


def test_anthropic_client_aclose_closes_the_sdk_client_if_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = {"n": 0}

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            self.messages = _StubMessages()

        async def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _StubClient)
    client = AnthropicClient(max_concurrency=1)
    asyncio.run(client.send(_REQUEST))

    asyncio.run(client.aclose())

    assert closed["n"] == 1


def test_anthropic_client_streams_the_whole_request(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _StubMessages()

    class _StubClient:
        def __init__(self, **kwargs: object) -> None:
            self.messages = messages

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _StubClient)
    client = AnthropicClient(max_concurrency=1)

    response = asyncio.run(client.send(_REQUEST))

    assert messages.requests == [_REQUEST]
    assert response == _StubMessage().to_dict()


_REAL_ASYNC_ANTHROPIC = anthropic.AsyncAnthropic

_MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 0},
    },
}
_TEXT_EVENTS = [
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "he"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "llo"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
]


def _sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


class _SseBody(httpx2.AsyncByteStream):
    """An SSE body that yields `events`, then raises `then_raise` if given."""

    def __init__(self, events: list[dict], then_raise: Exception | None = None) -> None:
        self._events = events
        self._then_raise = then_raise

    async def __aiter__(self):
        for event in self._events:
            yield _sse(event)
        if self._then_raise is not None:
            raise self._then_raise


def _serve_streams(monkeypatch: pytest.MonkeyPatch, bodies: list[_SseBody]) -> list[int]:
    """Back `AnthropicClient` with the real SDK over a mock transport serving `bodies` in order.

    Returns a one-item list counting the HTTP requests the transport received.
    """
    served = [0]

    def handler(_request: httpx2.Request) -> httpx2.Response:
        body = bodies[served[0]]
        served[0] += 1
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    def factory(**kwargs: object) -> anthropic.AsyncAnthropic:
        transport = httpx2.MockTransport(handler)
        return _REAL_ASYNC_ANTHROPIC(
            api_key="test-key", http_client=httpx2.AsyncClient(transport=transport), **kwargs
        )

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    return served


_STREAM_REQUEST = {**_REQUEST, "max_tokens": 10}


def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the client's retry sleep with a recorder of the delays it asked for."""
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(client_module, "_sleep", record)
    return slept


def test_real_sdk_stream_is_assembled_into_the_final_message_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_streams(monkeypatch, [_SseBody([_MESSAGE_START, *_TEXT_EVENTS])])
    client = AnthropicClient(max_concurrency=1)

    response = asyncio.run(client.send(_STREAM_REQUEST))

    assert response["content"] == [{"type": "text", "text": "hello"}]
    assert response["stop_reason"] == "end_turn"
    assert response["usage"]["input_tokens"] == 12
    assert response["usage"]["output_tokens"] == 7


def test_real_sdk_error_event_mid_stream_maps_to_a_dialogue_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overloaded = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    _serve_streams(monkeypatch, [_SseBody([_MESSAGE_START, overloaded])])
    client = AnthropicClient(max_concurrency=1)

    with pytest.raises(DialogueError, match="Overloaded"):
        asyncio.run(client.send(_STREAM_REQUEST))


@pytest.mark.parametrize(
    "drop",
    [httpx2.ReadError("connection reset"), httpx2.RemoteProtocolError("peer closed")],
    ids=["read_error", "remote_protocol_error"],
)
def test_real_sdk_connection_lost_mid_stream_is_retried_then_maps_to_a_dialogue_error(
    monkeypatch: pytest.MonkeyPatch, drop: Exception
) -> None:
    """A transport error after the stream opens is not wrapped or retried by the SDK, so
    the client retries it itself, `max_retries` times, before giving up.
    """
    served = _serve_streams(
        monkeypatch, [_SseBody([_MESSAGE_START], then_raise=drop) for _ in range(3)]
    )
    slept = _patch_sleep(monkeypatch)
    client = AnthropicClient(max_concurrency=1, max_retries=2)

    with pytest.raises(
        DialogueError,
        match=f"{type(drop).__name__} while streaming the reply.*gave up after 2 retries",
    ):
        asyncio.run(client.send(_STREAM_REQUEST))

    assert served == [3]
    assert slept == [0.5, 1.0]


def test_a_dropped_stream_is_retried_and_the_retry_can_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _serve_streams(
        monkeypatch,
        [
            _SseBody([_MESSAGE_START], then_raise=httpx2.RemoteProtocolError("peer closed")),
            _SseBody([_MESSAGE_START, *_TEXT_EVENTS]),
        ],
    )
    slept = _patch_sleep(monkeypatch)
    client = AnthropicClient(max_concurrency=1, max_retries=4)

    response = asyncio.run(client.send(_STREAM_REQUEST))

    assert response["content"] == [{"type": "text", "text": "hello"}]
    assert served == [2]
    assert slept == [0.5]


def test_a_failed_stream_releases_its_concurrency_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_streams(
        monkeypatch,
        [
            _SseBody([_MESSAGE_START], then_raise=httpx2.ReadError("connection reset")),
            _SseBody([_MESSAGE_START, *_TEXT_EVENTS]),
        ],
    )
    client = AnthropicClient(max_concurrency=1, max_retries=0)

    async def scenario() -> dict:
        with pytest.raises(DialogueError):
            await client.send(_STREAM_REQUEST)
        return await asyncio.wait_for(client.send(_STREAM_REQUEST), timeout=5)

    assert asyncio.run(scenario())["stop_reason"] == "end_turn"


def test_anthropic_client_aclose_is_a_noop_when_never_built() -> None:
    client = AnthropicClient(max_concurrency=1)

    asyncio.run(client.aclose())  # must not raise


class _NoAcloseClient:
    """A minimal `LlmClient` with no `aclose`, to check `CachedClient.aclose` tolerates that."""

    async def send(self, request: dict) -> dict:
        return default_responder(request)


def test_cached_client_aclose_closes_the_inner_client_if_it_was_built_and_supports_it(
    tmp_path: Path,
) -> None:
    class _ClosingFakeClient(FakeClient):
        def __init__(self, responder) -> None:
            super().__init__(responder)
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    inner = _ClosingFakeClient(default_responder)
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    asyncio.run(cached.send(_REQUEST, scope=_SCOPE))

    asyncio.run(cached.aclose())

    assert inner.closed is True


def test_cached_client_aclose_is_a_noop_when_the_inner_client_was_never_built(
    tmp_path: Path,
) -> None:
    cached = CachedClient(_failing_factory, tmp_path, token_budget=None)

    asyncio.run(cached.aclose())  # must not call the factory or raise


def test_cached_client_aclose_is_a_noop_when_the_inner_client_has_no_aclose(
    tmp_path: Path,
) -> None:
    inner = _NoAcloseClient()
    cached = CachedClient(lambda: inner, tmp_path, token_budget=None)
    asyncio.run(cached.send(_REQUEST, scope=_SCOPE))

    asyncio.run(cached.aclose())  # must not raise despite no aclose method


def test_send_until_accepted_uses_the_label_in_its_error(tmp_path: Path) -> None:
    def never_parses(_response):
        return None, "the reply was refused"

    cached = CachedClient(lambda: FakeClient(default_responder), tmp_path, token_budget=None)

    with pytest.raises(DialogueError, match=r"^pm pm_001: the reply was refused$"):
        asyncio.run(
            send_until_accepted(
                cached,
                _REQUEST,
                never_parses,
                scope="pm_001",
                max_retries=0,
                error_type=DialogueError,
                label="pm",
            )
        )
