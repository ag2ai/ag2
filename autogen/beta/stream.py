# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol, TypeAlias, overload, runtime_checkable
from uuid import UUID, uuid4

from .events import HITL, BaseEvent, UserMessage
from .events.conditions import ClassInfo, Condition, TypeCondition
from .history import History, MemoryStorage, Storage, StreamId

SubId: TypeAlias = UUID


@dataclass(slots=True)
class Context:
    stream: "Stream"

    prompt: list[str] = field(default_factory=list)

    async def input(self, message: str, timeout: float | None = None) -> str:
        async with self.stream.get(UserMessage) as response:
            await self.send(HITL(content=message))
            return (await asyncio.wait_for(response, timeout)).content

    async def send(self, event: BaseEvent) -> None:
        await self.stream.send(event, self)


class Subscriber(Protocol):
    async def __call__(self, event: BaseEvent, ctx: Context) -> Any: ...


class Interrupter(Protocol):
    async def __call__(self, event: BaseEvent, ctx: Context) -> BaseEvent | None: ...


@runtime_checkable
class Stream(Protocol):
    @overload
    def subscribe(
        self,
        func: Subscriber,
        *,
        interrupt: bool = False,
    ) -> SubId: ...

    @overload
    def subscribe(
        self,
        func: None = None,
        *,
        interrupt: bool = False,
    ) -> Callable[[Subscriber], SubId]: ...

    def subscribe(
        self,
        func: Subscriber | None = None,
        *,
        interrupt: bool = False,
    ) -> Callable[[Subscriber], SubId] | SubId: ...

    def unsubscribe(self, sub_id: SubId) -> None: ...

    @contextmanager
    def sub_scope(self, func: Subscriber) -> Iterator[None]:
        sub_id = self.subscribe(func)
        try:
            yield
        finally:
            self.unsubscribe(sub_id)

    def where(
        self,
        condition: ClassInfo | Condition,
    ) -> "Stream":
        if not isinstance(condition, Condition):
            condition = TypeCondition(condition)
        return SubStream(self, condition)

    @asynccontextmanager
    async def get(
        self,
        condition: ClassInfo | Condition,
    ) -> AsyncIterator[asyncio.Future[BaseEvent]]:
        result = asyncio.Future[BaseEvent]()

        async def wait_result(event: BaseEvent, ctx: Context) -> None:
            result.set_result(event)

        with self.where(condition).sub_scope(wait_result):
            yield result


class MemoryStream(Stream):
    def __init__(
        self,
        storage: Storage | None = None,
        *,
        id: StreamId | None = None,
    ) -> None:
        self.id: StreamId = id or uuid4()

        self._subscribers: dict[SubId, Subscriber] = {}
        # ordered dict
        self._interrupters: dict[SubId, Interrupter] = {}

        storage = storage or MemoryStorage()
        self.history = History(self.id, storage)
        self.subscribe(lambda ev, ctx: storage.save_event(ctx.stream.id, ev))

    @overload
    def subscribe(
        self,
        func: Subscriber,
        *,
        interrupt: bool = False,
    ) -> SubId: ...

    @overload
    def subscribe(
        self,
        func: None = None,
        *,
        interrupt: bool = False,
    ) -> Callable[[Subscriber], SubId]: ...

    def subscribe(
        self,
        func: Subscriber | None = None,
        *,
        interrupt: bool = False,
    ) -> Callable[[Subscriber], SubId] | SubId:
        def sub(s: Subscriber) -> SubId:
            sub_id = uuid4()
            if interrupt:
                self._interrupters[sub_id] = s
            else:
                self._subscribers[sub_id] = s
            return sub_id

        if func:
            return sub(func)
        return sub

    def unsubscribe(self, sub_id: SubId) -> None:
        self._subscribers.pop(sub_id, None)
        self._interrupters.pop(sub_id, None)

    async def send(
        self,
        event: BaseEvent,
        ctx: Context,
    ) -> None:
        # interrupters should follow registration order
        for interrupter in self._interrupters.values():
            if not (e := await interrupter(event, ctx)):
                return
            event = e

        for s in self._subscribers.values():
            await s(event, ctx)


class SubStream(Stream):
    def __init__(
        self,
        parent: Stream,
        condition: Condition,
    ) -> None:
        self._filter_condition = condition
        self._parent = parent

    @overload
    def subscribe(
        self,
        func: Subscriber,
        *,
        interrupt: bool = False,
    ) -> SubId: ...

    @overload
    def subscribe(
        self,
        func: None = None,
        *,
        interrupt: bool = False,
    ) -> Callable[[Subscriber], SubId]: ...

    def subscribe(
        self,
        func: Subscriber | None = None,
        *,
        interrupt: bool = False,
    ) -> Callable[[Subscriber], SubId] | SubId:
        def sub(s: Subscriber) -> SubId:
            @wraps(s)
            async def final_subscriber(event: BaseEvent, ctx: Context) -> Any:
                if self._filter_condition(event):
                    return await s(event, ctx)
                return event

            return self._parent.subscribe(final_subscriber, interrupt=interrupt)

        if func:
            return sub(func)
        return sub

    def unsubscribe(self, sub_id: SubId) -> None:
        return self._parent.unsubscribe(sub_id)
