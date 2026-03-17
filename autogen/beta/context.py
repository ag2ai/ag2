# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, overload, runtime_checkable
from uuid import UUID

from fast_depends import Provider

from .events import BaseEvent, HumanInputRequest, HumanMessage
from .events.conditions import ClassInfo, Condition

StreamId: TypeAlias = UUID
SubId: TypeAlias = UUID


@runtime_checkable
class Stream(Protocol):
    id: StreamId

    async def send(self, event: BaseEvent, context: "Context") -> None: ...

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
        condition: Condition | None = None,
    ) -> AbstractContextManager[None]: ...

    def get(
        self,
        condition: ClassInfo | Condition,
    ) -> AbstractAsyncContextManager[asyncio.Future[BaseEvent]]: ...


@dataclass(slots=True)
class Context:
    stream: Stream
    dependency_provider: "Provider | None" = None

    prompt: list[str] = field(default_factory=list)

    dependencies: dict[Any, Any] = field(default_factory=dict)
    # store Context Variables as separated serializable field
    variables: dict[str, Any] = field(default_factory=dict)

    async def input(self, message: str, timeout: float | None = None) -> str:
        async with self.stream.get(HumanMessage) as response:
            await self.send(HumanInputRequest(content=message))
            result: HumanMessage = await asyncio.wait_for(response, timeout)
            return result.content

    async def send(self, event: BaseEvent) -> None:
        await self.stream.send(event, self)
