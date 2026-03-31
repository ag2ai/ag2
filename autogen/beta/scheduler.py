# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Scheduler — watch lifecycle manager.

The Scheduler manages Watch lifecycles: registering, arming, disarming,
and canceling watches. It is a watch lifecycle manager, not a rigid
scheduling engine.

Works standalone (manages local watches with callbacks) or with a Hub
(delegates tasks to actors when watches fire).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from autogen.beta.context import Context as ContextType
from autogen.beta.events import BaseEvent
from autogen.beta.stream import MemoryStream

from .watch import Watch

if TYPE_CHECKING:
    from .network.events import SchedulerTriggerFired
    from .network.hub import Hub

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

    # Target mode (Hub-connected)
    target: str = ""
    task: str = ""
    task_factory: Callable[[list[BaseEvent]], str] | None = None
    priority: Any = None

    # Callback mode (standalone)
    callback: Callable[[list[BaseEvent], ContextType], Awaitable[None]] | None = None


class Scheduler:
    """Manages watch lifecycles. Fires callbacks or delegates tasks when watches trigger.

    Works standalone (manages local watches) or with a Hub (delegates to actors).

    Example (standalone)::

        scheduler = Scheduler()
        scheduler.add(IntervalWatch(300), callback=my_health_check)
        await scheduler.start()

    Example (with Hub)::

        scheduler = Scheduler(hub=hub)
        scheduler.add(IntervalWatch(300), target="monitor", task="Check health")
        await scheduler.start()
    """

    def __init__(self, hub: Hub | None = None) -> None:
        self._hub = hub
        self._entries: dict[str, _WatchEntry] = {}
        self._running = False
        self._stream = MemoryStream()

    def add(
        self,
        watch: Watch,
        *,
        target: str = "",
        task: str = "",
        task_factory: Callable[[list[BaseEvent]], str] | None = None,
        callback: Callable[[list[BaseEvent], ContextType], Awaitable[None]] | None = None,
        priority: Any = None,
    ) -> str:
        """Register a watch.

        Either provide ``target``+``task`` (Hub mode) or ``callback`` (standalone).

        Returns the watch ID for lifecycle management.
        """
        entry_id = uuid4().hex[:12]
        entry = _WatchEntry(
            id=entry_id,
            watch=watch,
            target=target,
            task=task,
            task_factory=task_factory,
            callback=callback,
            priority=priority,
        )
        self._entries[entry_id] = entry

        # If already running, arm immediately
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
        stream = self._hub.stream if self._hub else self._stream

        async def _on_fire(events: list[BaseEvent], ctx: ContextType) -> None:
            await self._handle_fire(entry, events, ctx)

        entry.watch.arm(stream, _on_fire)
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

        if entry.callback is not None:
            # Standalone mode: call the callback directly
            try:
                await entry.callback(events, ctx)
            except Exception:
                logger.exception("Scheduler callback for watch %s failed", entry.id)
            return

        if self._hub and entry.target:
            # Determine the task (only needed for hub mode)
            try:
                task = entry.task_factory(events) if entry.task_factory is not None else entry.task
            except Exception:
                logger.exception("Scheduler task_factory for watch %s failed", entry.id)
                return

            # Hub mode: emit event and delegate
            from .network.events import SchedulerTriggerFired

            hub_ctx = ContextType(stream=self._hub.stream)
            await self._hub.stream.send(
                SchedulerTriggerFired(
                    watch_id=entry.id,
                    target=entry.target,
                    task=task,
                ),
                hub_ctx,
            )
            try:
                await self._hub._delegate(entry.target, task, source="scheduler")
            except Exception:
                logger.exception(
                    "Scheduler trigger %s failed for agent '%s'",
                    entry.id,
                    entry.target,
                )
            return

        logger.warning(
            "Watch %s fired but has no callback or hub target — nothing to do",
            entry.id,
        )
