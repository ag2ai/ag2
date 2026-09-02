# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from ag2 import Context
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelResponse, TextInput
from ag2.middleware import RetryMiddleware
from ag2.stream import MemoryStream


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


@pytest.mark.asyncio()
async def test_llm_retry_preserves_context_contract() -> None:
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    retry_middleware = RetryMiddleware(max_retries=1, retry_on=(TransientError,))

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        assert ctx is context
        assert isinstance(ctx, ConversationContext)
        ctx.prompt = ["updated"]
        return ModelResponse(ModelMessage("result"))

    middleware = retry_middleware(TextInput("Hi!"), context)
    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], context)

    assert response == ModelResponse(ModelMessage("result"))
    assert context.prompt == ["updated"]
    assert context.stream is stream


@pytest.mark.asyncio()
async def test_llm_retry_propagates_when_chunks_emitted_before_error(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    emitted_chunks: list[str] = []
    stream.where(ModelMessageChunk).subscribe(lambda event: emitted_chunks.append(event.content))

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        await ctx.send(ModelMessageChunk("stale "))
        raise TransientError("network dropped mid-stream")

    middleware = retry_middleware(TextInput("Hi!"), context)
    with pytest.raises(TransientError, match="network dropped mid-stream"):
        await middleware.on_llm_call(llm_call, [TextInput("Hi!")], context)

    mock.llm_call.assert_called_once_with([TextInput("Hi!")])
    assert emitted_chunks == ["stale "]
    assert context.stream is stream


@pytest.mark.asyncio()
async def test_llm_retry_retries_when_error_before_any_emission(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    attempts = 0
    emitted_chunks: list[str] = []
    stream.where(ModelMessageChunk).subscribe(lambda event: emitted_chunks.append(event.content))

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        mock.llm_call(events)
        if attempts == 1:
            # Error occurs before any chunk is emitted (e.g. rate limit, DNS)
            raise TransientError("rate limited on handshake")
        await ctx.send(ModelMessageChunk("clean stream"))
        return ModelResponse(ModelMessage("clean stream"))

    middleware = retry_middleware(TextInput("Hi!"), context)
    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], context)

    assert response == ModelResponse(ModelMessage("clean stream"))
    assert mock.llm_call.call_count == attempts == 2
    assert emitted_chunks == ["clean stream"]
    assert context.stream is stream
