# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import string
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

from ..events.base import BaseEvent

DEBUG_SESSION_VAR = "_debug_session"

if TYPE_CHECKING:
    from ..context import Context
    from ..stream import MemoryStream

from .client import DebugClient, serialize_event


def _random_name() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=6))


class DebugSession:
    """
    A named debug session that can span multiple ``agent.ask()`` calls
    and multiple streams.

    Pass it via context variables so it flows naturally across calls::

        session = DebugSession(name="my-run")
        await agent.ask("hello", variables={"_debug_session": session})
        await agent.ask("follow up", variables={"_debug_session": session})
        await session.close()

    Or use as an async context manager::

        async with DebugSession(name="my-run") as session:
            await agent.ask("hello", variables={"_debug_session": session})
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        url: str | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.name = name or _random_name()
        server_url = url or os.environ.get("AG2_DEBUG_SERVER_URL", "")
        self._client: DebugClient | None = DebugClient(server_url) if server_url else None
        self._registered = False
        self._subscribed_streams: set[str] = set()
        self._stream_ids: list[str] = []
        # Live references (updated on each _attach call)
        self.stream: "MemoryStream | None" = None
        self.context: "Context | None" = None

    @property
    def stream_ids(self) -> list[str]:
        return list(self._stream_ids)

    async def _attach(self, stream: "MemoryStream", context: "Context") -> None:
        """Called internally by ``ask()`` to wire up event forwarding for this stream."""
        stream_id = str(stream.id)
        self.stream = stream
        self.context = context

        if not self._client:
            return

        # Register stream with server (idempotent — server resets if re-registered)
        await self._client.register_stream(stream_id, list(context.prompt))

        # Register session once (with the first stream)
        if not self._registered:
            await self._client.register_session(self.id, self.name, stream_id)
            self._registered = True
        elif stream_id not in self._subscribed_streams:
            # Add additional stream to existing session
            await self._client.add_stream_to_session(self.id, stream_id)

        # Subscribe to stream events (once per stream to avoid duplicates)
        if stream_id not in self._subscribed_streams:
            # Use partial to bind stream_id so each callback knows its stream
            stream.subscribe(partial(self._record_event_for_stream, stream_id))
            self._subscribed_streams.add(stream_id)
            self._stream_ids.append(stream_id)
            # Replay any events already in storage so the dashboard shows full history
            await self._replay_events_for_stream(stream_id, await stream.history.get_events())

    async def _record_event_for_stream(self, stream_id: str, event: BaseEvent) -> None:
        """Serialise and forward a stream event to the debug server."""
        if not self._client:
            return
        s = serialize_event(event)
        await self._client.send_event(stream_id, s["type"], s["data"])

    async def record_event(self, event: BaseEvent) -> None:
        """Serialise and forward an event to the most recently attached stream."""
        if self._stream_ids:
            await self._record_event_for_stream(self._stream_ids[-1], event)

    async def _replay_events_for_stream(self, stream_id: str, events: Iterable[BaseEvent]) -> None:
        """Replay historical events for a specific stream."""
        from ..events.types import ModelResponse

        for event in events:
            await self._record_event_for_stream(stream_id, event)
            if isinstance(event, ModelResponse) and event.tool_calls and event.tool_calls.calls:
                await self._record_event_for_stream(stream_id, event.tool_calls)

    async def replay_events(self, events: Iterable[BaseEvent]) -> None:
        """Replay historical events to the most recently attached stream."""
        if self._stream_ids:
            await self._replay_events_for_stream(self._stream_ids[-1], events)

    async def inject(self, event: BaseEvent) -> None:
        """Push an event directly into the live stream (in-process only)."""
        if self.stream and self.context:
            await self.stream.send(event, self.context)

    async def close(self) -> None:
        """Signal to the debug server that this session is done (freezes snapshot)."""
        if self._client:
            await self._client.end_session(self.id)

    async def __aenter__(self) -> "DebugSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
