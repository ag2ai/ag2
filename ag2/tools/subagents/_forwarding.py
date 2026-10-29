# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from copy import copy
from typing import Any
from uuid import uuid4

from ag2.annotations import Context
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ChildEvent,
    DeliveryFailed,
    DownEvent,
    HaltEvent,
    HumanInputRequest,
    TaskCancelled,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    TransientChildEvent,
    TransientDownEvent,
    Usage,
)
from ag2.stream import Stream
from ag2.usage import UsageReport, usage_event_from


class DeliveryError(RuntimeError):
    """The addressed delegation is not executing or no longer exists."""


async def _persist_accounting(context: Context, events: list[BaseEvent]) -> None:
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
        child_audit = copy(audit)
        setattr(child_audit, "_downlink", True)
        setattr(child_audit, "_validate_delivery", self.validate_delivery)
        await self.context.send(child_audit)
        self.validate_delivery()
        down = copy(event)
        setattr(down, "_downlink", True)
        setattr(down, "_validate_delivery", self.validate_delivery)
        await self.context.send(down)


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
        self.started = False
        self.bound = False
        self.failure: BaseException | None = None
        self.result: str | None = None
        self.accounting: list[BaseEvent] = []
        self.caller_task: asyncio.Task[Any] | None = None
        ancestor = parent.dependencies.get(Forwarder)
        self.path = (*ancestor.path, task_id) if ancestor is not None else (task_id,)
        self.origin: _Endpoint | None = None

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        self.origin = _Endpoint(context, self.task_id, self.path, self.agent_name, self.parent)
        stack.callback(self.origin.close)
        context.dependencies[Forwarder] = self
        stack.enter_context(context.stream.sub_scope(self.forward, interrupt=True, sync_to_thread=False))
        stack.enter_context(context.stream.where(HaltEvent).sub_scope(self.origin.halt, sync_to_thread=False))

    async def start(self, context: Context) -> None:
        if self.origin is None:
            self.origin = _Endpoint(context, self.task_id, self.path, self.agent_name, self.parent)
        self.origin.task = asyncio.current_task()
        self.origin.alive = True
        self.started = True
        if self.emit_events:
            await context.send(self.started_event())

    async def bridge(self, event: HumanInputRequest, ctx: Context) -> None:
        forwarded = copy(event)
        forwarded.task_id = self.task_id
        setattr(forwarded, "_bridged", True)
        # Dispatch deliberately under the requester's context: the parent's hook
        # must reply on the child's stream (ADR 0004).
        await self.parent.stream.send(forwarded, ctx)

    @asynccontextmanager
    async def execution(self, context: Context) -> AsyncGenerator[None]:
        """Keep forwarding and the terminal inside the stream's turn lock."""
        with ExitStack() as stack:
            self.register(stack, context)
            self.bound = True
            try:
                yield
            except BaseException as exc:
                self.failure = exc
                raise
            finally:
                if self.origin is not None:
                    self.origin.close()
                await _persist_accounting(context, self.accounting)
                if self.emit_events:
                    if not self.started:
                        await context.send(self.started_event())
                    await context.send(await self.terminal_event())

    def started_event(self) -> TaskStarted:
        return TaskStarted(
            task_id=self.task_id,
            agent_name=self.agent_name,
            objective=self.objective,
            parent_tool_call_id=self.parent.tool_call_id,
        )

    async def usage(self) -> Usage:
        try:
            history = await self.stream.history.get_events()
        except Exception:
            return UsageReport.from_events(self.accounting).total
        return UsageReport.from_events([*history, *self.accounting]).total

    async def terminal_event(self) -> BaseEvent:
        if isinstance(self.failure, asyncio.CancelledError):
            reason = self.origin.reason if self.origin is not None and self.origin.halted else "parent cancelled"
            return TaskCancelled(
                task_id=self.task_id, agent_name=self.agent_name, objective=self.objective, reason=reason
            )
        if self.failure is not None:
            error = self.failure if isinstance(self.failure, Exception) else RuntimeError(str(self.failure))
            return TaskFailed(task_id=self.task_id, agent_name=self.agent_name, objective=self.objective, error=error)
        return TaskCompleted(
            task_id=self.task_id,
            agent_name=self.agent_name,
            objective=self.objective,
            result=self.result,
            task_stream=self.stream.id,
            usage=await self.usage(),
        )

    async def reject(self, failure: BaseException) -> None:
        """Report a delegation that never entered its turn, without touching a busy stream."""
        if self.bound:
            return
        self.failure = failure
        self.origin = _Endpoint(ConversationContext(self.stream), self.task_id, self.path, self.agent_name, self.parent)
        if self.emit_events:
            await self.forward(self.started_event())
            await self.forward(await self.terminal_event())

    def close(self) -> None:
        if self.origin is not None:
            self.origin.close()

    async def forward(self, event: BaseEvent) -> BaseEvent:
        if getattr(event, "_downlink", False) or getattr(event, "_bridged", False):
            return event
        if usage_event_from(event) is not None:
            self.accounting.append(event)
        if isinstance(event, ChildEvent):
            envelope = copy(event)
            if event._origin is not None:
                envelope._origin = StreamHandle(event.origin._endpoint, self.parent)
        else:
            envelope_type = TransientChildEvent if type(event).__transient__ else ChildEvent
            envelope = envelope_type(
                inner=event,
                task_id=self.task_id,
                path=self.path,
                agent_name=self.agent_name,
                origin_event_id=getattr(event, "event_id", None) or uuid4().hex,
                _origin=StreamHandle(self.origin, self.parent) if self.origin is not None else None,
            )
        try:
            await self.parent.send(envelope)
        except BaseException:
            if usage_event_from(envelope) is not None:
                await _persist_accounting(self.parent, [envelope])
            raise
        return event
