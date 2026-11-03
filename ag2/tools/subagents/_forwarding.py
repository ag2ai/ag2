# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from copy import copy
from typing import Any
from uuid import uuid4

from ag2._run_lifecycle import RunLifecycle, persist_accounting, snapshot_usage
from ag2.annotations import Context
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ChildEvent,
    DeliveryFailed,
    DownEvent,
    HaltEvent,
    HumanInputRequest,
    InertEvent,
    TaskCancelled,
    TaskCompleted,
    TaskFailed,
    TransientChildEvent,
    TransientDownEvent,
    Usage,
    UsageEvent,
)
from ag2.events.delegation import Delivery, delivery_of, routed
from ag2.stream import Stream
from ag2.task import TaskSpec
from ag2.usage import usage_event_from

PARENT_CANCELLED = "parent cancelled"
"""Cancellation reason of a delegation stopped by its caller."""


class DeliveryError(RuntimeError):
    """The addressed delegation is not executing or no longer exists."""


class DelegationRegistry:
    """Internal bookkeeping for delegation cancellation and cleanup."""

    def __init__(self, context: Context) -> None:
        self.context = context
        self._delegations: dict[str, Forwarder] = {}

    @classmethod
    def for_context(cls, context: Context) -> "DelegationRegistry":
        registry: DelegationRegistry | None = context.dependencies.get(cls)
        if registry is None:
            registry = cls(context)
            context.dependencies[cls] = registry
        return registry

    def register(self, delegation: "Forwarder") -> None:
        delegation.caller_task = asyncio.current_task()
        self._delegations[delegation.task_id] = delegation

    def unregister(self, task_id: str) -> None:
        self._delegations.pop(task_id, None)

    def cancel_descendants(self, path: tuple[str, ...], reason: str) -> None:
        current = asyncio.current_task()
        for delegation in tuple(self._delegations.values()):
            if delegation.path != path and delegation.path[: len(path)] == path:
                origin = delegation.origin
                if origin is not None and origin.task is not None:
                    origin.halted = True
                    origin.reason = reason
                    origin.task.cancel()
                if delegation.caller_task is not None and delegation.caller_task is not current:
                    delegation.caller_task.cancel()

    async def settle_descendants(self, path: tuple[str, ...]) -> None:
        # Let already scheduled background delegation coroutines register before
        # taking the snapshot; this is a cooperative checkpoint, not a timeout.
        await asyncio.sleep(0)
        current = asyncio.current_task()
        tasks = [
            d.caller_task
            for d in tuple(self._delegations.values())
            if d.path != path
            and d.path[: len(path)] == path
            and d.caller_task is not None
            and d.caller_task is not current
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class _Endpoint:
    """A live delegation address, bound only while its turn owns the stream."""

    def __init__(self, context: Context, task_id: str, path: tuple[str, ...], agent_name: str, parent: Context) -> None:
        self.context = context
        self.task_id = task_id
        self.path = path
        self.agent_name = agent_name
        self.parent = parent
        self.alive = False
        self.task: asyncio.Task[Any] | None = None
        self.halted = False
        self.reason = ""

    async def halt(self, event: HaltEvent) -> None:
        self.halted = True
        self.reason = event.reason
        DelegationRegistry.for_context(self.parent).cancel_descendants(self.path, event.reason)
        if self.task is not None:
            self.task.cancel()

    def close(self) -> None:
        self.alive = False
        self.task = None

    def validate_delivery(self) -> None:
        if not self.alive:
            raise DeliveryError(f"Delegation {self.task_id} is not executing")

    async def send(self, event: BaseEvent, *, sender: Context) -> None:
        try:
            await self._deliver(event, sender)
        except DeliveryError:
            await sender.send(DeliveryFailed(task_id=self.task_id, reason="delegation is not executing"))
            raise

    async def _deliver(self, event: BaseEvent, sender: Context) -> None:
        self.validate_delivery()
        author = sender.dependencies.get(Forwarder)
        audit_type = TransientDownEvent if type(event).__transient__ else DownEvent
        audit = audit_type(
            inner=event,
            task_id=self.task_id,
            path=self.path,
            agent_name=self.agent_name,
            origin_task_id=author.task_id if author is not None else None,
            origin_path=author.path if author is not None else (),
            origin_stream_id=str(sender.stream.id),
        )
        await sender.send(audit)
        self.validate_delivery()
        delivery = Delivery(validate=self.validate_delivery)
        await self.context.send(routed(audit, delivery))
        self.validate_delivery()
        await self.context.send(routed(event, delivery))


class StreamHandle:
    """An address of the original author, with the sending ancestor's identity."""

    def __init__(self, endpoint: _Endpoint, sender: Context) -> None:
        self._endpoint = endpoint
        self._sender = sender

    @property
    def task_id(self) -> str:
        return self._endpoint.task_id

    @property
    def path(self) -> tuple[str, ...]:
        return self._endpoint.path

    @property
    def agent_name(self) -> str:
        return self._endpoint.agent_name

    async def send(self, event: BaseEvent) -> None:
        await self._endpoint.send(event, sender=self._sender)


class Forwarder:
    """Forward one active turn, preserving the original source across hops."""

    def __init__(
        self, parent: Context, task_id: str, agent_name: str, stream: Stream, objective: str, emit_events: bool
    ) -> None:
        self.parent = parent
        self.task_id = task_id
        self.agent_name = agent_name
        self.stream = stream
        self.objective = objective
        self.emit_events = emit_events
        self.bound = False
        self.accounting: list[BaseEvent] = []
        self.caller_task: asyncio.Task[Any] | None = None
        ancestor = parent.dependencies.get(Forwarder)
        self.path = (*ancestor.path, task_id) if ancestor is not None else (task_id,)
        self.origin: _Endpoint | None = None

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        self.origin = _Endpoint(context, self.task_id, self.path, self.agent_name, self.parent)
        stack.callback(self.origin.close)
        stack.callback(self._release_context, context)
        context.dependencies[Forwarder] = self
        stack.enter_context(context.stream.sub_scope(self.forward, interrupt=True, sync_to_thread=False))
        stack.enter_context(context.stream.where(HaltEvent).sub_scope(self.origin.halt, sync_to_thread=False))

    def _release_context(self, context: Context) -> None:
        if context.dependencies.get(Forwarder) is self:
            context.dependencies.pop(Forwarder)

    async def start(self, context: Context) -> None:
        if self.origin is None:
            self.origin = _Endpoint(context, self.task_id, self.path, self.agent_name, self.parent)
        self.origin.task = asyncio.current_task()
        self.origin.alive = True

    async def bridge(self, event: HumanInputRequest, ctx: Context) -> None:
        forwarded = routed(event, Delivery())
        forwarded.task_id = self.task_id
        # Dispatch deliberately under the requester's context: the parent's hook
        # must reply on the child's stream (ADR 0004).
        await self.parent.stream.send(forwarded, ctx)

    @asynccontextmanager
    async def execution(self, context: Context) -> AsyncGenerator[None]:
        """Keep forwarding and source accounting inside the stream's turn lock."""
        with ExitStack() as stack:
            self.register(stack, context)
            self.bound = True
            try:
                yield
            finally:
                if self.origin is not None:
                    self.origin.close()

    async def usage(self) -> Usage:
        return await snapshot_usage(self.stream, self.accounting)

    async def reject(self, failure: BaseException) -> None:
        """Use the common lifecycle for work rejected before acquiring its stream."""
        if self.bound:
            return
        context = ConversationContext(self.stream)
        self.origin = _Endpoint(context, self.task_id, self.path, self.agent_name, self.parent)
        lifecycle = self.lifecycle(context, publisher=self.forward)
        async with lifecycle.scope():
            await lifecycle.reject(failure)

    def lifecycle(
        self,
        context: ConversationContext,
        *,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        publisher: Callable[[BaseEvent], Awaitable[Any]] | None = None,
    ) -> RunLifecycle:
        """The task lifecycle this delegation reports to its parent."""
        return RunLifecycle(
            self.agent_name,
            TaskSpec(title=self.objective, capability=capability),
            context,
            ttl_seconds=ttl_seconds,
            task_id=self.task_id,
            parent_tool_call_id=self.parent.tool_call_id,
            emit_events=self.emit_events,
            accounting=self.accounting,
            publisher=publisher,
            cancel_reason=PARENT_CANCELLED,
        )

    def close(self) -> None:
        if self.origin is not None:
            self.origin.close()

    def stamp(self, event: InertEvent) -> InertEvent:
        """Name this delegation as the source of a local inert event; keep a forwarded one's source."""
        forwarded = copy(event)
        endpoint: _Endpoint | None
        if not event.path:
            forwarded.path = self.path
            if isinstance(forwarded, UsageEvent):
                forwarded.task_id = self.task_id
                forwarded.agent_name = self.agent_name
            endpoint = self.origin
        else:
            endpoint = event.origin._endpoint if event._origin is not None else None
        forwarded._origin = StreamHandle(endpoint, self.parent) if endpoint is not None else None
        return forwarded

    async def forward(self, event: BaseEvent) -> BaseEvent:
        if delivery_of(event) is not None:
            return event
        if usage_event_from(event) is not None:
            self.accounting.append(event)
        if (
            isinstance(event, (TaskCompleted, TaskFailed, TaskCancelled))
            and event.task_id == self.task_id
            and self.origin is not None
        ):
            self.origin.close()
        forwarded: BaseEvent
        if isinstance(event, InertEvent):
            forwarded = self.stamp(event)
        elif isinstance(event, ChildEvent):
            forwarded = copy(event)
            if event._origin is not None:
                forwarded._origin = StreamHandle(event.origin._endpoint, self.parent)
        else:
            envelope_type = TransientChildEvent if type(event).__transient__ else ChildEvent
            forwarded = envelope_type(
                inner=event,
                task_id=self.task_id,
                path=self.path,
                agent_name=self.agent_name,
                origin_event_id=getattr(event, "event_id", None) or uuid4().hex,
                _origin=StreamHandle(self.origin, self.parent) if self.origin is not None else None,
            )
        try:
            if isinstance(forwarded, InertEvent):
                # Forward the existing publication without Context.send starting
                # a new one, which drops its provenance and usage identity (ADR 0004).
                await self.parent.stream.send(forwarded, self.parent)
            else:
                await self.parent.send(forwarded)
        except BaseException:
            if usage_event_from(forwarded) is not None:
                await persist_accounting(self.parent, [forwarded])
            raise
        return event
