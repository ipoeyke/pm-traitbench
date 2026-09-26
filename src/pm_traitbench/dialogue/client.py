"""The single seam every Anthropic API request passes through.

`LlmClient` is the protocol every backend implements. `CachedClient` wraps a
backend with a disk cache keyed on the request body; because prompts are
built deterministically, that cache also serves as the resume manifest for a
crash-resumed or retried run. `AnthropicClient` is the live backend.
"""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import anthropic

from pm_traitbench.errors import DialogueBudgetError, DialogueError

_NO_CREDENTIALS_MESSAGE = "no Anthropic credentials: run `ant auth login` or set ANTHROPIC_API_KEY"


class LlmClient(Protocol):
    """A backend that sends one Messages API request and returns its response."""

    async def send(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


def request_key(request: Mapping[str, Any]) -> str:
    """sha256 hex of the request serialised with sorted keys and no whitespace."""
    body = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Reply:
    """One response to a request: its cache key, the response body and its cache origin."""

    key: str
    response: dict[str, Any]
    cached: bool


@dataclass
class UsageTotals:
    """Running call and token counts for a `CachedClient`; token fields exclude cache hits."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


class CachedClient:
    """Wraps an `LlmClient` with a disk cache keyed on the request body.

    A reply is written to disk only once the caller commits it, so a crash or
    a rejected response never leaves a cached entry a rerun would trust. The
    token budget is a soft stop: each in-flight call checks it before its own
    usage is added, so concurrent sends can overshoot it by up to the
    concurrency limit's worth of calls.
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

    def _path_for(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"

    async def send(self, request: Mapping[str, Any]) -> Reply:
        key = request_key(request)
        self._totals.calls += 1
        path = self._path_for(key)
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                cached_response = stored["response"]
            except (OSError, json.JSONDecodeError, KeyError):
                # A partial or corrupt entry (e.g. after a power loss) is
                # treated as a miss: re-fetch and overwrite it on commit.
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

        usage = response.get("usage", {})
        # `to_dict()` keeps explicit nulls for optional usage fields.
        self._totals.input_tokens += usage.get("input_tokens") or 0
        self._totals.output_tokens += usage.get("output_tokens") or 0
        self._totals.cache_read_tokens += usage.get("cache_read_input_tokens") or 0

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


class AnthropicClient:
    """Live Anthropic backend, built lazily so a fully cached run never needs credentials.

    Bounds concurrency and maps credential and 400 errors to `DialogueError`.
    """

    def __init__(self, max_concurrency: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: anthropic.AsyncAnthropic | None = None

    async def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is None:
            try:
                self._client = anthropic.AsyncAnthropic()
            except anthropic.CredentialsError as e:
                raise DialogueError(_NO_CREDENTIALS_MESSAGE) from e

        async with self._semaphore:
            try:
                message = await self._client.messages.create(**request)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise DialogueError(_NO_CREDENTIALS_MESSAGE) from e
            except anthropic.BadRequestError as e:
                raise DialogueError(e.message) from e
            except TypeError as e:
                # With nothing configured, the SDK signals missing credentials with a bare
                # TypeError at request time (header resolution runs before any network call).
                raise DialogueError(f"{_NO_CREDENTIALS_MESSAGE} ({e})") from e
        return message.to_dict()
