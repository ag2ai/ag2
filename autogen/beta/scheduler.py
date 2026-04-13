# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Scheduler — watch lifecycle manager.

The Scheduler manages Watch lifecycles: registering, arming, disarming,
and canceling watches. It is a watch lifecycle manager, not a rigid
scheduling engine.

Standalone, callback-driven. A user registers a ``Watch`` together with
an async callback; when the watch fires, the callback runs with the
accumulated events and the current context. The scheduler owns its own
:class:`MemoryStream` and does not take a reference to any network hub.

Callers that want a scheduler to drive hub-side behavior (e.g. the V3
TTL sweeper) instantiate their own :class:`Scheduler`, register a
callback, and have that callback perform the hub mutation directly.
``Scheduler`` itself is network-agnostic.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from autogen.beta.context import Context as ContextType
from autogen.beta.events import BaseEvent
from autogen.beta.stream import MemoryStream

from .watch import Watch

logger = logging.getLogger(__name__)


class WatchStatus(str, Enum):
    """Status of a registered watch."""

    PENDING = "pending"  # Registered but not yet armed
    ARMED = "armed"  # Active and watching
    PAUSED = "paused"  # Temporarily disarmed
    CANCELLED = "cancelled"  # Permanently removed


@dataclass
class _WatchEntry:
    """Internal registry entry for a scheduled watch."""

    id: str
    watch: Watch
    status: WatchStatus = WatchStatus.PENDING
    callback: Callable[[list[BaseEvent], ContextType], Awaitable[None]] | None = None


class Scheduler:
    """Manages watch lifecycles. Fires user-supplied callbacks when watches trigger.

    Example::

        scheduler = Scheduler()
        scheduler.add(IntervalWatch(300), callback=my_health_check)
        await scheduler.start()
    """

    def __init__(self) -> None:
        self._entries: dict[str, _WatchEntry] = {}
        self._running = False
        self._stream = MemoryStream()

    def add(
        self,
        watch: Watch,
        *,
        callback: Callable[[list[BaseEvent], ContextType], Awaitable[None]] | None = None,
    ) -> str:
        """Register a watch with an optional async callback.

        ``callback`` is invoked with the accumulated events and the
        current :class:`Context` whenever the watch fires. A watch with
        no callback is a silent no-op — useful for observability-only
        registrations where you want the watch to run but don't need to
        react to it.

        Returns the watch ID for lifecycle management.
        """
        entry_id = uuid4().hex[:12]
        entry = _WatchEntry(id=entry_id, watch=watch, callback=callback)
        self._entries[entry_id] = entry

        if self._running:
            self._arm_entry(entry)

        return entry_id

    async def start(self) -> None:
        """Arm all pending watches. Non-blocking."""
        if self._running:
            return
        self._running = True

        for entry in self._entries.values():
            if entry.status == WatchStatus.PENDING:
                self._arm_entry(entry)

    async def stop(self) -> None:
        """Disarm all watches and stop."""
        self._running = False
        for entry in self._entries.values():
            if entry.status == WatchStatus.ARMED:
                entry.watch.disarm()
                entry.status = WatchStatus.PAUSED

    def pause(self, watch_id: str) -> None:
        """Temporarily disarm one watch."""
        entry = self._entries.get(watch_id)
        if entry and entry.status == WatchStatus.ARMED:
            entry.watch.disarm()
            entry.status = WatchStatus.PAUSED

    def resume(self, watch_id: str) -> None:
        """Re-arm a paused watch. Only works when the scheduler is running."""
        if not self._running:
            return
        entry = self._entries.get(watch_id)
        if entry and entry.status == WatchStatus.PAUSED:
            self._arm_entry(entry)

    def cancel(self, watch_id: str) -> bool:
        """Permanently cancel a watch. Returns True if found."""
        entry = self._entries.get(watch_id)
        if entry:
            if entry.status == WatchStatus.ARMED:
                entry.watch.disarm()
            entry.status = WatchStatus.CANCELLED
            return True
        return False

    def status(self, watch_id: str) -> WatchStatus | None:
        """Get current status of a watch."""
        entry = self._entries.get(watch_id)
        return entry.status if entry else None

    @property
    def watches(self) -> list[tuple[str, Watch, WatchStatus]]:
        """List of (watch_id, watch, status) tuples."""
        return [(e.id, e.watch, e.status) for e in self._entries.values()]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _arm_entry(self, entry: _WatchEntry) -> None:
        """Arm a watch entry — create callback and arm the watch."""

        async def _on_fire(events: list[BaseEvent], ctx: ContextType) -> None:
            await self._handle_fire(entry, events, ctx)

        entry.watch.arm(self._stream, _on_fire)
        entry.status = WatchStatus.ARMED

    async def _handle_fire(
        self,
        entry: _WatchEntry,
        events: list[BaseEvent],
        ctx: ContextType,
    ) -> None:
        """Handle a watch firing."""
        if not self._running:
            return
        if entry.status == WatchStatus.CANCELLED:
            return

        if entry.callback is None:
            return

        try:
            await entry.callback(events, ctx)
        except Exception:
            logger.exception("Scheduler callback for watch %s failed", entry.id)
