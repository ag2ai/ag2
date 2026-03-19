# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Storage wrapper used by the debugger to intercept newly-persisted events.

Instead of subscribing a separate stream listener, DebugStorage wraps the
underlying MemoryStorage so that forwarding to the debug server happens at
the same point events are saved to history — not on every stream emission.
This means the debugger sees exactly what is in memory (deduplicated,
high-level events only).
"""

from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..annotations import Context
    from ..context import StreamId
    from ..events.base import BaseEvent
    from ..history import Storage


class DebugStorage:
    """
    Wraps any :class:`Storage` backend and calls *callback* each time a
    genuinely new event is persisted (duplicates are silently dropped by the
    inner storage and never reach the callback).
    """

    def __init__(self, inner: "Storage") -> None:
        self._inner = inner
        self._callback: Callable[["BaseEvent"], Awaitable[None]] | None = None

    def set_callback(self, callback: "Callable[[BaseEvent], Awaitable[None]]") -> None:
        """Wire up the debug forwarding callback (called once the DebugSession exists)."""
        self._callback = callback

    async def save_event(self, event: "BaseEvent", context: "Context") -> None:
        stream_id = context.stream.id
        # Snapshot history *before* saving so we can detect whether this event is new.
        existing = list(await self._inner.get_history(stream_id))
        is_new = event not in existing
        await self._inner.save_event(event, context)
        if is_new and self._callback is not None:
            await self._callback(event)

    async def get_history(self, stream_id: "StreamId") -> "Iterable[Any]":
        return await self._inner.get_history(stream_id)

    async def set_history(self, stream_id: "StreamId", events: "Iterable[Any]") -> None:
        await self._inner.set_history(stream_id, events)

    async def drop_history(self, stream_id: "StreamId") -> None:
        await self._inner.drop_history(stream_id)
