# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, overload, runtime_checkable
from uuid import UUID

from fast_depends import Provider

from autogen.beta.types import ClassInfo

from .events import BaseEvent, HumanInputRequest, HumanMessage, Input, ModelRequest
from .events.conditions import Condition

StreamId: TypeAlias = UUID
SubId: TypeAlias = UUID


@runtime_checkable
class Stream(Protocol):
    id: StreamId

    async def send(self, event: BaseEvent, context: "ConversationContext") -> None: ...

    def where(self, condition: ClassInfo | Condition) -> "Stream": ...

    def join(
        self,
        *,
        max_events: int | None = None,
    ) -> AbstractContextManager[AsyncIterator[BaseEvent]]: ...

    @overload
    def subscribe(
        self,
        func: Callable[..., Any],
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> SubId: ...

    @overload
    def subscribe(
        self,
        func: None = None,
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> Callable[[Callable[..., Any]], SubId]: ...

    def subscribe(
        self,
        func: Callable[..., Any] | None = None,
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> Callable[[Callable[..., Any]], SubId] | SubId: ...

    def unsubscribe(self, sub_id: SubId) -> None: ...

    def sub_scope(
        self,
        func: Callable[..., Any],
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
    ) -> AbstractContextManager[None]: ...

    def get(
        self,
        condition: ClassInfo | Condition,
    ) -> AbstractAsyncContextManager[asyncio.Future[BaseEvent]]: ...


@dataclass(slots=True)
class ConversationContext:
    stream: Stream = field(repr=False)
    dependency_provider: "Provider | None" = field(default=None, repr=False)

    # store Context Variables as separated serializable field
    variables: dict[str, Any] = field(default_factory=dict)

    dependencies: dict[Any, Any] = field(default_factory=dict)

    prompt: list[str] = field(default_factory=list)

    pending_messages: list[ModelRequest] | None = field(default=None, repr=False)

    _background_tasks: set[asyncio.Task[None]] | None = field(default=None, repr=False)

    def spawn_background(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Spawn a fire-and-forget task tied to the current ``Agent.ask`` run.

        The agent loop will not return until every task spawned this way has
        completed. Tasks are expected to deliver their outcome by calling
        ``self.enqueue(...)`` — the loop drains the pending queue after each
        completion and redirects the model with one more turn.

        Use this from tools that kick off work which must finish before the
        run can end (background subagents, long-running side effects whose
        results matter). Pure side-effects that should not delay the loop
        should use ``asyncio.create_task`` directly, outside this primitive.
        """
        if self._background_tasks is None:
            raise RuntimeError("spawn_background can only be called inside a live Agent.ask run.")

        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def enqueue(self, *content: "str | Input | ModelRequest") -> None:
        """Queue input for the currently running agent loop.

        Delivered at the next model-call opportunity — either prepended to the
        upcoming request, or, if the agent would otherwise terminate, used to
        redirect the run into one more turn.

        Enqueued messages are consumed only by a live ``Agent.ask`` run. If the
        run has already returned, callers should resume the conversation with a
        new ask call instead.
        """
        if not content:
            return

        if self.pending_messages is None:
            raise RuntimeError("Cannot enqueue a pending message outside a live Agent.ask run.")

        parts: list[Input] = []
        for item in content:
            if isinstance(item, ModelRequest):
                parts.extend(item.parts)
            else:
                parts.append(Input.ensure_input(item))

        self.pending_messages.append(ModelRequest(parts))

    async def input(self, message: str, timeout: float | None = None) -> str:
        request_msg = HumanInputRequest(message)
        async with self.stream.get(HumanMessage.parent_id == request_msg.id) as response:
            await self.send(request_msg)
            result: HumanMessage = await asyncio.wait_for(response, timeout)
            return result.content

    async def send(self, event: BaseEvent) -> None:
        await self.stream.send(event, self)
