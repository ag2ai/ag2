# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Invocation lifecycle shared by model runs and delegations."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from .context import ConversationContext, Stream
from .events import BaseEvent, Usage
from .task import TERMINAL_TASK_STATES, Task, TaskSpec, TaskState
from .usage import UsageReport, usage_event_from

logger = logging.getLogger(__name__)


async def persist_accounting(context: ConversationContext, events: list[BaseEvent]) -> None:
    """Retain source records when cancellation interrupts ordinary delivery."""
    try:
        history = await context.stream.history.get_events()
    except Exception:
        return  # Preserve the original storage failure and the captured usage.
    seen = {record.event_id for event in history if (record := usage_event_from(event)) is not None}
    for event in events:
        record = usage_event_from(event)
        if record is not None and record.event_id not in seen:
            # Append rather than replacing the parent's history: siblings can
            # publish concurrently. Readers deduplicate by source identity.
            await context.stream.history.storage.save_event(event, context)
            seen.add(record.event_id)


async def snapshot_usage(stream: Stream, accounting: list[BaseEvent]) -> Usage:
    """Read a cumulative stream snapshot, retaining captured records on storage failure."""
    try:
        history = await stream.history.get_events()
    except Exception:
        history = []
    return UsageReport.from_events([*history, *accounting]).total


class RunLifecycle:
    """Task lifecycle of one model run or delegation.

    Wraps a ``Task`` without injecting it. ``start`` publishes ``TaskStarted``
    when the run is driven; ``scope`` publishes the terminal on exit, chosen
    from ``failure`` / ``result``. An undriven scope publishes nothing.
    """

    def __init__(
        self,
        owner_id: str,
        spec: TaskSpec,
        context: ConversationContext,
        *,
        ttl_seconds: int | None = None,
        task_id: str | None = None,
        parent_tool_call_id: str | None = None,
        emit_events: bool = True,
        accounting: list[BaseEvent] | None = None,
        publisher: Callable[[BaseEvent], Awaitable[Any]] | None = None,
        cancel_reason: str = "cancelled",
    ) -> None:
        self.task = Task(owner_id=owner_id, spec=spec, context=context, ttl_seconds=ttl_seconds)
        self.task_id = task_id
        self.parent_tool_call_id = parent_tool_call_id
        self.emit_events = emit_events
        self.accounting = accounting if accounting is not None else []
        self.publisher = publisher
        self.failure: BaseException | None = None
        self.result: str | None = None
        self.cancel_reason = cancel_reason

    async def start(self) -> None:
        if self.task.state == TaskState.CREATED:
            try:
                await self.task._start(
                    task_id=self.task_id,
                    parent_tool_call_id=self.parent_tool_call_id,
                    emit_events=self.emit_events,
                    inject=False,
                    publisher=self.publisher,
                )
            except BaseException as exc:
                self.failure = exc
                raise

    async def reject(self, failure: BaseException) -> None:
        """Record failure for a requested driver that never reached its first step."""
        if self.task.state == TaskState.CREATED:
            self.failure = failure
            await self.start()

    async def finish(self) -> None:
        """Publish the terminal, even when reading or retaining usage fails.

        A propagating cleanup error decides the terminal, replacing any
        earlier failure. A storage error behind an earlier failure is
        logged and suppressed so it does not replace that failure.
        """
        driven = self.task.state != TaskState.CREATED and self.task.state not in TERMINAL_TASK_STATES
        usage: Usage | None = None
        cleanup_error: BaseException | None = None
        try:
            if self.accounting:
                await persist_accounting(self.task.context, self.accounting)
            if driven and self.failure is None:
                usage = await snapshot_usage(self.task.context.stream, self.accounting)
        except BaseException as exc:
            if self.failure is None:
                self.failure = cleanup_error = exc
            elif isinstance(exc, Exception):
                logger.warning("Retaining usage failed after the run failed with %r", self.failure, exc_info=exc)
            else:
                self.failure = cleanup_error = exc
        if driven and self.task.state not in TERMINAL_TASK_STATES:
            await self.task._finish(
                self.failure,
                result=self.result,
                usage=usage,
                cancel_reason=self.cancel_reason,
            )
        if cleanup_error is not None:
            raise cleanup_error

    @asynccontextmanager
    async def scope(self) -> AsyncGenerator[None]:
        try:
            yield
        except BaseException as exc:
            self.failure = exc
            await self.start()
            raise
        finally:
            try:
                await self.finish()
            finally:
                self.task._unbind()
