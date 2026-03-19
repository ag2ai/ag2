# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from ..events.base import BaseEvent

if TYPE_CHECKING:
    from .client import DebugClient
    from ..context import Context
    from ..stream import MemoryStream


class DebugSession:
    """
    In-process agent-side session that holds **live references** to the agent's
    stream and context objects.

    Events are forwarded to the external debug server via :class:`DebugClient`.
    Breakpoints block by issuing an HTTP long-poll to the server; the server
    returns any modifications the user requested at resume time, which are then
    applied to the live event / context before execution continues.
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
        """Stream subscriber — serialise and forward every emitted event."""
        from .client import serialize_event

        s = serialize_event(event)
        await self._client.send_event(self.id, s["type"], s["data"])

    async def pause(self, bp_type: str, event: BaseEvent) -> BaseEvent:
        """
        Block agent execution until the UI resumes the breakpoint.

        Returns the (potentially mutated) event so the caller can pass it on
        to the next middleware / handler.
        """
        from .client import serialize_event

        s = serialize_event(event)
        mods: dict[str, Any] = await self._client.hit_breakpoint(self.id, bp_type, s)

        # Apply context-level modifications returned by the server
        if mods.get("prompt") is not None:
            self.context.prompt[:] = mods["prompt"]
        if mods.get("variables"):
            self.context.variables.update(mods["variables"])

        # Apply field-level mutations to the live event object
        for key, value in (mods.get("event_modifications") or {}).items():
            try:
                setattr(event, key, value)
            except Exception:
                pass

        return event

    async def inject(self, event: BaseEvent) -> None:
        """Push an event directly into the live stream (in-process only)."""
        await self.stream.send(event, self.context)
