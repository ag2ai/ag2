# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..events.base import BaseEvent

if TYPE_CHECKING:
    from ..context import Context
    from ..stream import MemoryStream
    from .client import DebugClient


class DebugSession:
    """
    In-process agent-side session that holds **live references** to the agent's
    stream and context objects.

    Events are forwarded to the external debug server via :class:`DebugClient`.
    """

    def __init__(
        self,
        session_id: str,
        *,
        stream: "MemoryStream",
        context: "Context",
        client: "DebugClient",
    ) -> None:
        self.id = session_id
        self.stream = stream  # live reference to the agent's MemoryStream
        self.context = context  # live reference to the agent's Context
        self._client = client

    async def record_event(self, event: BaseEvent) -> None:
        """Called when a new event is persisted to storage — serialise and forward to the debug server."""
        from .client import serialize_event

        s = serialize_event(event)
        await self._client.send_event(self.id, s["type"], s["data"])

    async def replay_events(self, events: Iterable[BaseEvent]) -> None:
        """
        Replay historical events to the debug server.

        Storage backends may only persist high-level events (ModelRequest,
        ModelResponse, ToolResultsEvent) and not the intermediate stream
        events (ToolCallsEvent, ToolCallEvent) that are synthesised at
        runtime.  For each ModelResponse whose tool_calls are non-empty we
        emit a synthetic ToolCallsEvent so the debugger UI can show tool
        calls as discrete events.
        """
        from ..events.types import ModelResponse

        for event in events:
            await self.record_event(event)
            if isinstance(event, ModelResponse) and event.tool_calls and event.tool_calls.calls:
                await self.record_event(event.tool_calls)

    async def inject(self, event: BaseEvent) -> None:
        """Push an event directly into the live stream (in-process only)."""
        await self.stream.send(event, self.context)
