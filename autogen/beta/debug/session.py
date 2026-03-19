# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..events.base import BaseEvent

if TYPE_CHECKING:
    from ..context import Context
    from ..stream import MemoryStream


class _PendingBreakpoint:
    """Internal state for a breakpoint that is currently blocking agent execution."""

    __slots__ = ("id", "bp_type", "event", "waiter", "event_modifications", "prompt_override", "variable_updates")

    def __init__(self, bp_id: str, bp_type: str, event: BaseEvent) -> None:
        self.id = bp_id
        self.bp_type = bp_type
        self.event = event
        self.waiter: asyncio.Event = asyncio.Event()
        self.event_modifications: dict[str, Any] = {}
        self.prompt_override: list[str] | None = None
        self.variable_updates: dict[str, Any] = {}


class _BreakpointMeta:
    """Lightweight record kept after a breakpoint is created (serialised for the HTTP API)."""

    __slots__ = ("id", "type", "event_index", "timestamp", "resumed")

    def __init__(self, bp_id: str, bp_type: str, event_index: int) -> None:
        self.id = bp_id
        self.type = bp_type
        self.event_index = event_index
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.resumed: bool = False


class DebugSession:
    """
    In-process debug session that holds **live references** to the agent's
    stream and context objects.

    All blocking / modification logic runs in the same asyncio event loop as
    the agent, so ``asyncio.Event`` is sufficient — no cross-thread locking
    needed.
    """

    def __init__(
        self,
        session_id: str,
        *,
        stream: "MemoryStream",
        context: "Context",
        prompt: list[str] | None = None,
    ) -> None:
        self.id = session_id
        self.stream = stream  # live reference to the agent's MemoryStream
        self.context = context  # live reference to the agent's Context
        self.prompt: list[str] = list(prompt or [])

        # Live event objects — appended by the stream subscriber in agent.py
        self.events: list[BaseEvent] = []
        self.breakpoints: list[_BreakpointMeta] = []

        self._pending: _PendingBreakpoint | None = None
        self.status = "running"

    async def record_event(self, event: BaseEvent) -> None:
        """Stream subscriber — stores a live reference to every emitted event."""
        self.events.append(event)

    async def pause(self, bp_type: str, event: BaseEvent) -> BaseEvent:
        """
        Block agent execution until :meth:`resume` is called.

        Returns the (potentially mutated) event so the caller can pass it on
        to the next middleware / handler.
        """
        bp_id = str(uuid4())
        pending = _PendingBreakpoint(bp_id, bp_type, event)
        meta = _BreakpointMeta(bp_id, bp_type, max(len(self.events) - 1, 0))
        self.breakpoints.append(meta)
        self._pending = pending

        await pending.waiter.wait()

        # Apply context-level modifications requested via the HTTP API
        if pending.prompt_override is not None:
            self.context.prompt[:] = pending.prompt_override  # mutate in-place
        if pending.variable_updates:
            self.context.variables.update(pending.variable_updates)

        # Apply field-level mutations to the live event object
        for key, value in pending.event_modifications.items():
            try:
                setattr(event, key, value)
            except Exception:
                pass

        return event

    async def resume(
        self,
        bp_id: str,
        *,
        event_modifications: dict[str, Any] | None = None,
        prompt: list[str] | None = None,
        variables: dict[str, Any] | None = None,
    ) -> bool:
        """Unblock the pending breakpoint. Returns False if the id doesn't match."""
        pending = self._pending
        if not pending or pending.id != bp_id:
            return False

        if event_modifications:
            pending.event_modifications = event_modifications
        if prompt is not None:
            pending.prompt_override = prompt
        if variables:
            pending.variable_updates = variables

        for meta in self.breakpoints:
            if meta.id == bp_id:
                meta.resumed = True
                break

        pending.waiter.set()
        self._pending = None
        return True

    async def inject(self, event: BaseEvent) -> None:
        """Push an event directly into the live stream."""
        await self.stream.send(event, self.context)

    @property
    def pending_bp_id(self) -> str | None:
        return self._pending.id if self._pending else None
