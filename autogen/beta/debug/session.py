# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import string
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

    Pass :class:`~autogen.beta.debug.DebugMiddleware` via the
    ``middleware`` parameter::

        from autogen.beta.debug import DebugMiddleware

        session = DebugSession(name="my-run")
        await agent.ask("hello", middleware=[DebugMiddleware])
        await agent.ask("follow up", middleware=[DebugMiddleware])
        await session.close()

    Or use as an async context manager::

        async with DebugSession(name="my-run") as session:
            await agent.ask("hello", middleware=[DebugMiddleware])
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        url: str | None = None,
    ) -> None:
        print(f"[DebugSession] Initialising debug session {name!r} with url={url!r}")
        self.id = str(uuid4())
        self.name = name or _random_name()
        server_url = url or os.environ.get("AG2_DEBUG_SERVER_URL", "")
        self._client: DebugClient | None = DebugClient(server_url) if server_url else None
        self._registered = False
        self._registered_streams: set[str] = set()

    async def _ensure_stream(self, context: "Context") -> str:
        """Register the stream (and session on first call) with the debug server."""
        stream_id = str(context.stream.id)

        if stream_id in self._registered_streams:
            return stream_id

        self._registered_streams.add(stream_id)

        if not self._client:
            return stream_id

        await self._client.register_stream(stream_id, list(context.prompt))

        if not self._registered:
            await self._client.register_session(self.id, self.name, stream_id)
            self._registered = True
        else:
            await self._client.add_stream_to_session(self.id, stream_id)

        return stream_id

    async def record_event(self, event: BaseEvent, context: "Context") -> None:
        """Serialise and forward an event to the debug server."""
        if not self._client:
            return
        stream_id = await self._ensure_stream(context)
        s = serialize_event(event)
        await self._client.send_event(stream_id, s["type"], s["data"])

    async def close(self) -> None:
        """Signal to the debug server that this session is done (freezes snapshot)."""
        if self._client:
            await self._client.end_session(self.id)

    async def __aenter__(self) -> "DebugSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
