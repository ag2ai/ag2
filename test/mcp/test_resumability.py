# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served agent's dropped stream, resumed over a real socket.

Resumability is the one transport setting the in-memory ASGI transport cannot
observe: it does not stream SSE, and the only way to see an event store working
is to drop a stream part-way and reconnect with a ``Last-Event-ID``. So this
module serves under ``uvicorn`` on a loopback port, as
``test/tools/test_mcp_live_transport.py`` does, behind the same import guard —
``uvicorn`` ships with ``ag2[acp]``, not ``ag2[mcp]``.

The guard is harmless here rather than silently skipping in CI: ``ag2[acp]``
pulls in ``uvicorn``, the ``optionals`` dependency group installs ``ag2[acp]``,
and the test workflow installs that group.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

pytest.importorskip("uvicorn")

import uvicorn
from mcp.server.streamable_http import EventCallback, EventId, EventMessage, EventStore, StreamId
from mcp.types import JSONRPCMessage
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelResponse
from ag2.mcp import MCPServer, TransportConfig

from ._helpers import JSON_HEADERS, initialize_request

_HEADERS = {**JSON_HEADERS, "MCP-Protocol-Version": "2025-06-18"}
"""The endpoint is a Starlette ``Mount``, so requests carry the canonical trailing slash."""
_ENDPOINT = "/mcp/"

_CALL = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "ask", "arguments": {"message": "go"}, "_meta": {"progressToken": "p1"}},
}
"""A call asking for progress: the notifications are what the dropped stream misses."""


class RecordingEventStore(EventStore):
    """The smallest store that can prove resumability: every event, in order.

    An operator's own store would bound its history and outlive the process; this
    one only has to answer "what did this client miss".
    """

    def __init__(self) -> None:
        self._events: list[tuple[StreamId, EventId, JSONRPCMessage | None]] = []

    async def store_event(self, stream_id: StreamId, message: "JSONRPCMessage | None") -> EventId:
        event_id = str(len(self._events))
        self._events.append((stream_id, event_id, message))
        return event_id

    async def replay_events_after(self, last_event_id: EventId, send_callback: EventCallback) -> "StreamId | None":
        anchor = next((i for i, (_, event_id, _) in enumerate(self._events) if event_id == last_event_id), None)
        if anchor is None:
            return None
        stream_id = self._events[anchor][0]
        for onward_stream, event_id, message in self._events[anchor + 1 :]:
            if onward_stream == stream_id and message is not None:
                await send_callback(EventMessage(message=message, event_id=event_id))
        return stream_id


class SlowChunkConfig(ModelConfig):
    """Streams chunks with a pause between them, so a stream can be dropped mid-flight."""

    def __init__(self, *chunks: str, pause: float) -> None:
        self._chunks = chunks
        self._pause = pause

    def copy(self) -> Self:
        return self

    def create(self) -> "SlowChunkClient":
        return SlowChunkClient(self._chunks, self._pause)

    def create_files_client(self) -> None:
        raise NotImplementedError


class SlowChunkClient(LLMClient):
    def __init__(self, chunks: Sequence[str], pause: float) -> None:
        self._chunks = chunks
        self._pause = pause

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        for chunk in self._chunks:
            await context.send(ModelMessageChunk(chunk))
            await asyncio.sleep(self._pause)
        message = ModelMessage("".join(self._chunks))
        await context.send(message)
        return ModelResponse(message=message)


@asynccontextmanager
async def _serving(app: Any) -> AsyncGenerator[str]:
    """Run ``app`` under ``uvicorn`` on a loopback port, yielding its base URL."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    # Bound here rather than inside `serve()`: the socket is already listening, so
    # the port is known and a connection can be made without waiting for start-up.
    sock = config.bind_socket()
    uv = uvicorn.Server(config)
    serving = asyncio.create_task(uv.serve(sockets=[sock]))
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        uv.should_exit = True
        await serving
        sock.close()


async def _handshake(client: httpx.AsyncClient) -> dict[str, str]:
    """Open an MCP session, returning the headers every later request carries."""
    opened = await client.post(_ENDPOINT, headers=_HEADERS, json=initialize_request(version="2025-06-18"))
    assert opened.status_code == 200, opened.text
    headers = {**_HEADERS, "mcp-session-id": opened.headers["mcp-session-id"]}
    await client.post(_ENDPOINT, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


async def _progress_until_dropped(client: httpx.AsyncClient, headers: dict[str, str]) -> tuple[str, str]:
    """Read one progress notification off the server's stream, then walk away.

    Returns what arrived and the id of the event that carried it — the
    ``Last-Event-ID`` a reconnecting client presents.
    """
    async with client.stream("GET", _ENDPOINT, headers=headers) as stream:
        assert stream.status_code == 200
        event_id = None
        async for line in stream.aiter_lines():
            if line.startswith("id: "):
                event_id = line.removeprefix("id: ")
            elif line.startswith("data: ") and event_id is not None:
                message = json.loads(line.removeprefix("data: "))
                if message.get("method") == "notifications/progress":
                    return message["params"]["message"], event_id
    raise AssertionError("the stream ended without a resumable progress notification")


async def _progress_after(client: httpx.AsyncClient, headers: dict[str, str], *, expected: int) -> list[str]:
    """Reconnect and collect the progress notifications the replay delivers.

    A resumed stream replays and then tails live, so it never ends on its own.
    """
    delivered: list[str] = []
    async with client.stream("GET", _ENDPOINT, headers=headers) as stream:
        assert stream.status_code == 200
        async for line in stream.aiter_lines():
            if not line.startswith("data: "):
                continue
            message = json.loads(line.removeprefix("data: "))
            if message.get("method") == "notifications/progress":
                delivered.append(message["params"]["message"])
                if len(delivered) == expected:
                    break
    return delivered


@pytest.mark.asyncio
async def test_a_dropped_stream_resumes_and_receives_what_it_missed() -> None:
    """The whole point of an event store, over a real socket.

    The client sees the first of three progress notifications, loses its
    connection, and the server issues the other two with nobody listening.
    Reconnecting with a ``Last-Event-ID`` is what gets them delivered.
    """
    agent = Agent("streamer", config=SlowChunkConfig("one ", "two ", "three ", pause=0.6))
    app = MCPServer(agent, transport=TransportConfig(event_store=RecordingEventStore()))

    async with _serving(app) as base_url, httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
        headers = await _handshake(client)
        call = asyncio.create_task(client.post(_ENDPOINT, headers=headers, json=_CALL))

        before_drop, last_event_id = await _progress_until_dropped(client, headers)
        answered = await call
        after_resume = await _progress_after(client, {**headers, "Last-Event-ID": last_event_id}, expected=2)

    assert answered.status_code == 200
    assert before_drop == "one "
    assert after_resume == ["two ", "three "], "the events issued while disconnected were not replayed"
