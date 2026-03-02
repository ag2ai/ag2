# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from typing import Any, TypeAlias, overload
from uuid import UUID, uuid4

from fast_depends.core import CallModel

from .context import Context, StreamId, WritableStream
from .events import BaseEvent
from .events.conditions import ClassInfo, Condition, TypeCondition
from .history import History, MemoryStorage, Storage
from .utils import CONTEXT_OPTION_NAME, build_model

SubId: TypeAlias = UUID


class Stream(WritableStream):
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

    @contextmanager
    def sub_scope(self, func: Callable[..., Any]) -> Iterator[None]:
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

        async def wait_result(event: BaseEvent) -> None:
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

        self._subscribers: dict[SubId, tuple[Condition | None, CallModel]] = {}
        # ordered dict
        self._interrupters: dict[SubId, tuple[Condition | None, CallModel]] = {}

        storage = storage or MemoryStorage()
        self.history = History(self.id, storage)
        self.subscribe(storage.save_event)

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
    ) -> Callable[[Callable[..., Any]], SubId] | SubId:
        def sub(s: Callable[..., Any]) -> SubId:
            sub_id = uuid4()
            model = build_model(s, sync_to_thread=sync_to_thread, serialize_result=False)
            if interrupt:
                self._interrupters[sub_id] = (condition, model)
            else:
                self._subscribers[sub_id] = (condition, model)
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
        ctx: "Context",
    ) -> None:
        # interrupters should follow registration order
        for condition, interrupter in self._interrupters.values():
            if condition and not condition(event):
                continue

            async with AsyncExitStack() as stack:
                if not (
                    e := await interrupter.asolve(
                        event,
                        cache_dependencies={},
                        stack=stack,
                        dependency_provider=ctx.dependency_provider,
                        **{CONTEXT_OPTION_NAME: ctx},
                    )
                ):
                    return

            event = e

        for condition, s in self._subscribers.values():
            if condition and not condition(event):
                continue

            async with AsyncExitStack() as stack:
                await s.asolve(
                    event,
                    cache_dependencies={},
                    stack=stack,
                    dependency_provider=ctx.dependency_provider,
                    **{CONTEXT_OPTION_NAME: ctx},
                )


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
    ) -> Callable[[Callable[..., Any]], SubId] | SubId:
        def sub(s: Callable[..., Any]) -> SubId:
            c = self._filter_condition
            if condition:
                c = c & condition

            return self._parent.subscribe(
                s,
                condition=c,
                interrupt=interrupt,
                sync_to_thread=sync_to_thread,
            )

        if func:
            return sub(func)
        return sub

    def unsubscribe(self, sub_id: SubId) -> None:
        return self._parent.unsubscribe(sub_id)
