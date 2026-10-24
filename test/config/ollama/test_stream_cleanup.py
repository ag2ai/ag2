# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress

import httpx
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.ollama import OllamaConfig
from ag2.events import BaseEvent, ModelMessageChunk, ModelReasoning, ModelRequest, TextInput
from test.config.ollama._helpers import ask, make_chunk


class TrackedReplyStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield json.dumps(make_chunk(content="hello", thinking="thinking")).encode() + b"\n"
        yield json.dumps(make_chunk(done=True)).encode() + b"\n"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [ModelMessageChunk, ModelReasoning])
async def test_subscriber_error_closes_stream(event_type: type[BaseEvent]) -> None:
    body = TrackedReplyStream()
    next_body = TrackedReplyStream()
    replies = iter([body, next_body])
    error = RuntimeError("subscriber failed")

    async def fail(event: BaseEvent) -> None:
        raise error

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=next(replies)))
    async with httpx.AsyncClient(transport=transport) as http_client:
        stream = MemoryStream()
        stream.where(event_type).subscribe(fail)
        config = OllamaConfig("m1", streaming=True, http_client=http_client)
        client = config.create()

        with pytest.raises(RuntimeError) as caught:
            await client(
                [ModelRequest([TextInput("hello")])],
                Context(stream=stream),
                tools=[],
                response_schema=None,
                serializer=SerializerCls,
            )

        assert caught.value is error
        assert body.closed
        assert not http_client.is_closed

        result, _ = await ask(config)

        assert result.content == "hello"
        assert next_body.closed
        assert not http_client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [ModelMessageChunk, ModelReasoning])
async def test_cancellation_in_subscriber_closes_stream(event_type: type[BaseEvent]) -> None:
    body = TrackedReplyStream()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block(event: BaseEvent) -> None:
        entered.set()
        await release.wait()

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    async with httpx.AsyncClient(transport=transport) as http_client:
        stream = MemoryStream()
        stream.where(event_type).subscribe(block)
        client = OllamaConfig("m1", streaming=True, http_client=http_client).create()
        task = asyncio.create_task(
            client(
                [ModelRequest([TextInput("hello")])],
                Context(stream=stream),
                tools=[],
                response_schema=None,
                serializer=SerializerCls,
            )
        )

        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert body.closed
            assert not http_client.is_closed
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_completed_stream_closes_response_and_keeps_client_open() -> None:
    body = TrackedReplyStream()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    async with httpx.AsyncClient(transport=transport) as http_client:
        result, _ = await ask(OllamaConfig("m1", streaming=True, http_client=http_client))

        assert result.content == "hello"
        assert body.closed
        assert not http_client.is_closed
