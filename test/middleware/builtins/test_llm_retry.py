# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, MemoryStream
from ag2.config import LLMClient
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    TextInput,
    ToolCallEvent,
)
from ag2.middleware import RetryMiddleware
from ag2.testing import TestConfig
from ag2.tools import ToolResult, tool


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


@pytest.mark.asyncio()
async def test_llm_retry_calls_next_once_when_successful(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=3)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        return ModelResponse(ModelMessage("result"))

    middleware = retry_middleware(TextInput("Hi!"), mock)
    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    assert response == ModelResponse(ModelMessage("result"))
    mock.llm_call.assert_called_once_with([TextInput("Hi!")])


@pytest.mark.asyncio()
async def test_llm_retry_retries_matching_errors_until_success(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))
    attempts = 0

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        mock.llm_call(events)
        if attempts < 3:
            raise TransientError(f"transient failure {attempts}")
        return ModelResponse(ModelMessage("result"))

    middleware = retry_middleware(TextInput("Hi!"), mock)
    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    assert response == ModelResponse(ModelMessage("result"))
    assert mock.llm_call.call_count == attempts == 3


@pytest.mark.asyncio()
async def test_llm_retry_raises_after_exhausting_retries(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        raise TransientError("still failing")

    middleware = retry_middleware(TextInput("Hi!"), mock)
    with pytest.raises(TransientError, match="still failing"):
        await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    assert mock.llm_call.call_count == 3


@pytest.mark.asyncio()
async def test_llm_retry_does_not_retry_non_matching_errors(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=3, retry_on=(TransientError,))
    middleware = retry_middleware(TextInput("Hi!"), mock)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        raise PermanentError("do not retry")

    with pytest.raises(PermanentError, match="do not retry"):
        await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    mock.llm_call.assert_called_once_with([TextInput("Hi!")])


@pytest.mark.parametrize(
    "published",
    [
        ModelMessageChunk("stale "),
        ModelReasoning("half a thought"),
        ModelMessage("stale"),
        BuiltinToolCallEvent("web_search"),
        BuiltinToolResultEvent(parent_id="call-1", result=ToolResult("stale")),
    ],
    ids=["chunk", "reasoning", "message", "builtin_call", "builtin_result"],
)
@pytest.mark.asyncio()
async def test_llm_retry_stops_once_the_attempt_published_output(published: BaseEvent) -> None:
    """A retry cannot take back what consumers already have, so it must not happen.

    The cases are what a provider client publishes mid-call — streamed content,
    the message, server-side tool activity — but the middleware names no types:
    anything reaching the stream inside the call counts.
    """
    stream = MemoryStream()
    context = Context(stream=stream)
    middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))(TextInput("Hi!"), context)
    attempts = 0
    seen: list[BaseEvent] = []

    async def collect(event: BaseEvent) -> None:
        seen.append(event)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        await ctx.send(published)
        raise TransientError("stream disconnected")

    stream.subscribe(collect, sync_to_thread=False)

    with pytest.raises(TransientError, match="stream disconnected"):
        await middleware.on_llm_call(llm_call, [TextInput("Hi!")], context)

    assert attempts == 1
    # The middleware observes through an interrupter, which is allowed to drop or
    # replace an event — this one must do neither.
    assert seen == [published]
    assert seen[0] is published


@pytest.mark.asyncio()
async def test_llm_retry_retries_on_a_real_stream_when_nothing_was_published() -> None:
    """The control for the test above: the guard must not disarm retries wholesale.

    The other retry tests pass a ``MagicMock`` context, which makes the stream
    subscription a no-op — only a real stream exercises the guard's False branch.
    """
    stream = MemoryStream()
    context = Context(stream=stream)
    middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))(TextInput("Hi!"), context)
    attempts = 0

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientError(f"transient failure {attempts}")
        await ctx.send(ModelMessageChunk("complete"))
        return ModelResponse(ModelMessage("complete"))

    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], context)

    assert response == ModelResponse(ModelMessage("complete"))
    assert attempts == 3


@pytest.mark.asyncio()
async def test_llm_retry_scopes_published_output_to_a_single_call() -> None:
    """A chunk published by an earlier call must not disarm a later call's retries."""
    stream = MemoryStream()
    context = Context(stream=stream)
    middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))(TextInput("Hi!"), context)
    attempts = 0

    async def streaming_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        await ctx.send(ModelMessageChunk("first turn"))
        return ModelResponse(ModelMessage("first turn"))

    async def failing_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise TransientError("transient failure")
        return ModelResponse(ModelMessage("second turn"))

    await middleware.on_llm_call(streaming_call, [TextInput("Hi!")], context)
    response = await middleware.on_llm_call(failing_call, [TextInput("Hi!")], context)

    assert response == ModelResponse(ModelMessage("second turn"))
    assert attempts == 2


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class _FlakyClient(LLMClient):
    """Wraps a scripted client and fails the nth call with a transient error."""

    def __init__(self, client: LLMClient, fail_on: int) -> None:
        self.client = client
        self.fail_on = fail_on
        self.calls = 0

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.calls += 1
        if self.calls == self.fail_on:
            raise TransientError("transient failure")
        return await self.client(messages, context=context, **kwargs)


class _FlakyConfig(TestConfig):
    """``TestConfig`` whose single per-turn client fails one call."""

    def create(self) -> _FlakyClient:
        self.client = _FlakyClient(super().create(), fail_on=2)
        return self.client


@pytest.mark.asyncio()
async def test_llm_retry_survives_the_agent_loops_own_events() -> None:
    """The agent loop's events must not read as the attempt's own output.

    The guard counts *any* event published inside ``on_llm_call`` rather than a
    list of types, which holds only because the loop publishes its own events —
    ``ModelRequest``, ``ToolCallsEvent``, ``ToolResultEvent``, ``UsageEvent`` —
    outside the middleware chain. Were one to move inside it, every retry after
    the first tool call would silently stop happening; this pins that down.
    """
    config = _FlakyConfig(ToolCallEvent(name="add", arguments='{"a": 1, "b": 2}'), "done")
    agent = Agent(
        "retrying",
        config=config,
        tools=[add],
        middleware=[RetryMiddleware(max_retries=2, retry_on=(TransientError,))],
    )

    reply = await agent.ask("go")

    # Three client calls: the tool call, the failure, the retry that succeeds.
    assert config.client.calls == 3
    assert reply.body == "done"
