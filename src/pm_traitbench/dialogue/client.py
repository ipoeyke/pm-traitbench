"""The single seam every Anthropic API request passes through.

`LlmClient` is the protocol every backend implements. `CachedClient` wraps a
backend with a disk cache keyed on the session id plus the request body, so
two sessions that happen to build byte-identical requests never share a
reply; because prompts are built deterministically within one session, that
cache also serves as the resume manifest for a crash-resumed or retried run.
`AnthropicClient` is the live backend.
"""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anthropic
import httpx2

from pm_traitbench.config import ModelPrice
from pm_traitbench.dialogue.usage import ZERO_USAGE, usage_of
from pm_traitbench.errors import DialogueBudgetError, DialogueError, PmTraitbenchError
from pm_traitbench.tables.schema import CallUsage

_NO_CREDENTIALS_MESSAGE = "no Anthropic credentials: run `ant auth login` or set ANTHROPIC_API_KEY"
# Looked up at call time so a test can replace it.
_sleep = asyncio.sleep


def _retry_delay(attempt: int) -> float:
    """The SDK's backoff shape: 0.5s doubling per retry, capped at 8s."""
    return min(0.5 * 2.0**attempt, 8.0)


def scope_prefix(label: str, scope: str) -> str:
    """The `{label} {scope}: ` lead of a per-unit error or warning."""
    return f"{label} {scope}: "


def session_prefix(session_id: str) -> str:
    """The `session {id}: ` lead of every per-session error and warning."""
    return scope_prefix("session", session_id)


class LlmClient(Protocol):
    """A backend that sends one Messages API request and returns its response."""

    async def send(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


def last_text_json(response: Mapping[str, Any]) -> Any | None:
    """The last text block's content parsed as JSON, or `None`.

    `None` when `stop_reason` is not `end_turn`, there is no text block, or
    the text is not valid JSON. Shared by every reply parser in the dialogue
    and validate stages so the content-block walk lives in one place.
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


def request_key(request: Mapping[str, Any], scope: str) -> str:
    """sha256 hex of `scope` plus the request serialised with sorted keys and no whitespace.

    `scope` is the session id: two sessions whose requests happen to render
    byte-identical must still land in distinct cache entries.
    """
    body = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{scope}\n{body}".encode()).hexdigest()


@dataclass(frozen=True)
class Reply:
    """One response to a request: its cache key, the response body and its cache origin."""

    key: str
    response: dict[str, Any]
    cached: bool


def _cost_usd(usage: CallUsage, price: ModelPrice) -> float:
    """One model's bill for `usage` at `price`, in USD."""
    per_token = 1e-6
    return per_token * (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    )


@dataclass
class UsageTotals:
    """Running call and token counts for a `CachedClient`; token fields exclude cache hits.

    `by_model` keeps the same fresh usage per request model, so a stage that mixes
    models (a narrator, an advisor, a refusal fallback) can be billed per model.
    """

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    by_model: dict[str, CallUsage] = field(default_factory=dict)

    def add(self, model: str, usage: CallUsage) -> None:
        """Count one fresh reply's usage, in the stage totals and under its model."""
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_input_tokens
        self.cache_creation_tokens += usage.cache_creation_input_tokens
        before = self.by_model.get(model, ZERO_USAGE)
        self.by_model[model] = CallUsage(
            input_tokens=before.input_tokens + usage.input_tokens,
            output_tokens=before.output_tokens + usage.output_tokens,
            cache_read_input_tokens=before.cache_read_input_tokens + usage.cache_read_input_tokens,
            cache_creation_input_tokens=before.cache_creation_input_tokens
            + usage.cache_creation_input_tokens,
        )

    def as_metadata(self, prices: Mapping[str, ModelPrice]) -> dict[str, Any]:
        """The counters, the usage and cost per model, and the total cost, for run metadata.

        `cost_usd` sums the models in `prices`; any other model is listed under
        `unpriced_models` rather than failing a finished stage.
        """
        by_model: dict[str, dict[str, Any]] = {}
        unpriced: list[str] = []
        total = 0.0
        for model, usage in sorted(self.by_model.items()):
            entry: dict[str, Any] = usage.model_dump()
            price = prices.get(model)
            if price is None:
                unpriced.append(model)
            else:
                cost = _cost_usd(usage, price)
                total += cost
                entry["cost_usd"] = round(cost, 4)
            by_model[model] = entry
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "usage_by_model": by_model,
            "cost_usd": round(total, 4),
            "unpriced_models": unpriced,
        }


class CachedClient:
    """Wraps an `LlmClient` with a disk cache keyed on the caller's scope plus the request body.

    A reply is written to disk only once the caller commits it, so a crash or
    a rejected response never leaves a cached entry a rerun would trust. The
    token budget is a soft stop, and counts only fresh tokens spent in this
    run (a resumed run starts back at zero): each in-flight call checks it
    before its own usage is added, so concurrent sends can overshoot it by up
    to the caller's own concurrency limit's worth of calls.
    """

    def __init__(
        self,
        inner_factory: Callable[[], LlmClient],
        cache_dir: Path,
        token_budget: int | None,
    ) -> None:
        self._inner_factory = inner_factory
        self._inner: LlmClient | None = None
        self._cache_dir = cache_dir
        self._token_budget = token_budget
        self._totals = UsageTotals()

    @property
    def totals(self) -> UsageTotals:
        return self._totals

    @property
    def token_budget(self) -> int | None:
        return self._token_budget

    def _path_for(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"

    async def send(self, request: Mapping[str, Any], *, scope: str, refresh: bool = False) -> Reply:
        """Send `request`; `refresh=True` skips the cache read, forcing a fresh inner call.

        A caller sets `refresh` after a cached reply it read failed its own
        validation, so a cache entry that no longer validates is never
        replayed forever; a valid fresh reply then overwrites it on commit.
        """
        key = request_key(request, scope)
        self._totals.calls += 1
        path = self._path_for(key)
        if not refresh and path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(stored, dict):
                    raise ValueError("cached entry is not a JSON object")
                cached_response = stored["response"]
            except (OSError, ValueError, TypeError, KeyError):
                # A partial or corrupt entry is a miss, rewritten on commit;
                # `ValueError` also covers `UnicodeDecodeError`/`JSONDecodeError`.
                pass
            else:
                self._totals.cache_hits += 1
                return Reply(key=key, response=cached_response, cached=True)

        if self._token_budget is not None:
            spent = self._totals.input_tokens + self._totals.output_tokens
            if spent >= self._token_budget:
                raise DialogueBudgetError("dialogue token budget spent")

        if self._inner is None:
            self._inner = self._inner_factory()
        response = await self._inner.send(request)

        self._totals.add(request["model"], usage_of(response))
        return Reply(key=key, response=response, cached=False)

    def commit(self, reply: Reply) -> None:
        """Write a fresh reply to the cache; a no-op for one already read from it."""
        if reply.cached:
            return
        path = self._path_for(reply.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": reply.key,
            "model": reply.response.get("model"),
            "response": reply.response,
        }
        body = json.dumps(payload, sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f"{reply.key}-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
            os.replace(tmp_path, path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    async def aclose(self) -> None:
        """Close the inner client's resources, if it was built and it supports closing."""
        if self._inner is not None:
            aclose = getattr(self._inner, "aclose", None)
            if aclose is not None:
                await aclose()


class AnthropicClient:
    """Live Anthropic backend, built lazily so a fully cached run never needs credentials.

    Streams each reply and returns the final message. Bounds concurrency; maps
    credential and 400 errors, and other API status and connection errors once
    the SDK's own retries are exhausted, to `DialogueError`. A connection lost
    mid-stream, which the SDK never retries, is retried here `max_retries` times
    with the SDK's backoff before it becomes one too.
    """

    def __init__(self, max_concurrency: int, max_retries: int = 2) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: anthropic.AsyncAnthropic | None = None
        self._max_retries = max_retries

    async def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is None:
            try:
                self._client = anthropic.AsyncAnthropic(max_retries=self._max_retries)
            except anthropic.CredentialsError as e:
                raise DialogueError(_NO_CREDENTIALS_MESSAGE) from e

        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    return await self._stream_once(request)
                except httpx2.TransportError as e:
                    # The SDK wraps transport errors only before the stream opens; a drop
                    # while reading it arrives raw, and the SDK does not retry it.
                    if attempt == self._max_retries:
                        raise DialogueError(
                            f"{type(e).__name__} while streaming the reply: {e} "
                            f"(gave up after {attempt} retries)"
                        ) from e
                    await _sleep(_retry_delay(attempt))
        raise AssertionError("unreachable: the retry loop returns or raises")

    async def _stream_once(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """One streamed attempt, mapping every SDK error but a mid-stream transport one."""
        assert self._client is not None
        try:
            async with self._client.messages.stream(**request) as stream:
                message = await stream.get_final_message()
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise DialogueError(_NO_CREDENTIALS_MESSAGE) from e
        except anthropic.BadRequestError as e:
            raise DialogueError(e.message) from e
        except anthropic.APIStatusError as e:
            # Anything past a mapped 4xx above: rate limits, overloads and other
            # 5xxs, once the SDK's own retries are spent.
            raise DialogueError(f"{type(e).__name__} (status {e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            # Covers `APITimeoutError`, its subclass.
            raise DialogueError(f"{type(e).__name__}: {e.message}") from e
        except TypeError as e:
            # With nothing configured, the SDK signals missing credentials with a bare
            # TypeError at request time (header resolution runs before any network call);
            # an unexpected keyword from a request-builder change raises the same way, so
            # this is worded to not assert credentials are the cause.
            raise DialogueError(
                f"request could not be built: {e}; if no credentials are configured, "
                "run `ant auth login` or set ANTHROPIC_API_KEY"
            ) from e
        return message.to_dict()

    async def aclose(self) -> None:
        """Close the SDK client's HTTP connections, if one was ever built."""
        if self._client is not None:
            await self._client.close()


async def send_until_accepted[T](
    client: CachedClient,
    request: Mapping[str, Any],
    classify: Callable[[Mapping[str, Any]], tuple[T | None, str]],
    *,
    scope: str,
    max_retries: int,
    error_type: type[PmTraitbenchError],
    label: str = "session",
) -> tuple[Reply, T, int]:
    """Send `request`, retrying a reply `classify` rejects up to `max_retries` times.

    A rejected reply that came from the cache makes every later attempt bypass
    the cache too, so an entry that no longer validates is never replayed forever
    and a run does not alternate between it and a fresh call. An accepted reply is
    committed before it is returned with the rejected count; at the cap, raises
    `error_type` with the last rejection's reason, led by `label` and `scope`
    (a session id by default, a unit id of another kind when `label` is set).
    """
    refresh = False
    rejected = 0
    reason = ""
    for _ in range(1 + max_retries):
        reply = await client.send(request, scope=scope, refresh=refresh)
        accepted, reason = classify(reply.response)
        if accepted is not None:
            client.commit(reply)
            return reply, accepted, rejected
        refresh = refresh or reply.cached
        rejected += 1
    raise error_type(f"{scope_prefix(label, scope)}{reason}")


async def send_parsed[T](
    client: CachedClient,
    request: Mapping[str, Any],
    parse: Callable[[Mapping[str, Any]], T | None],
    *,
    scope: str,
    max_retries: int,
    error_type: type[PmTraitbenchError],
    label: str = "session",
    reason: str,
) -> tuple[T, int]:
    """Send `request` until `parse` accepts a reply; that reply with the rejected count.

    A thin `send_until_accepted` whose every rejection carries the one `reason`.
    """
    _, parsed, rejected = await send_until_accepted(
        client,
        request,
        lambda response: (parse(response), reason),
        scope=scope,
        max_retries=max_retries,
        error_type=error_type,
        label=label,
    )
    return parsed, rejected
