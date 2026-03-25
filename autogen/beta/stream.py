# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from typing import Any, overload
from uuid import uuid4

from fast_depends.core import CallModel

from autogen.beta.types import ClassInfo

from .context import Context, Stream, StreamId, SubId
from .events import BaseEvent
from .events.conditions import Condition, TypeCondition
from .history import History, MemoryStorage, Storage
from .utils import CONTEXT_OPTION_NAME, build_model

__all__ = ("MemoryStream", "Stream")


class ABCStream(Stream):
    @contextmanager
    def sub_scope(
        self,
        func: Callable[..., Any],
        *,
        interrupt: bool = False,
        sync_to_thread: bool = True,
        condition: Condition | None = None,
    ) -> Iterator[None]:
        sub_id = self.subscribe(
            func,
            interrupt=interrupt,
            sync_to_thread=sync_to_thread,
            condition=condition,
        )

        try:
            yield
        finally:
            self.unsubscribe(sub_id)

    @contextmanager
    def join(self, *, max_events: int | None = None) -> Iterator[AsyncIterator[BaseEvent]]:
        queue = asyncio.Queue[BaseEvent]()

        async def write_events(event: BaseEvent) -> None:
            await queue.put(event)

        if max_events:

            async def listen_events() -> AsyncIterator[BaseEvent]:
                for _ in range(max_events):
                    yield await queue.get()

        else:

            async def listen_events() -> AsyncIterator[BaseEvent]:
                while True:
                    yield await queue.get()

        with self.sub_scope(write_events):
            yield listen_events()

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


class MemoryStream(ABCStream):
    __slots__ = (
        "id",
        "_subscribers",
        "_interrupters",
        "history",
    )

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

    async def send(self, event: BaseEvent, context: "Context") -> None:
        # interrupters should follow registration order
        for condition, interrupter in tuple(self._interrupters.values()):
            if condition and not condition(event):
                continue

            async with AsyncExitStack() as stack:
                if not (
                    e := await interrupter.asolve(
                        event,
                        cache_dependencies={},
                        stack=stack,
                        dependency_provider=context.dependency_provider,
                        **{CONTEXT_OPTION_NAME: context},
                    )
                ):
                    return

            event = e

        # TODO: we need to publish under RWLock to prevent
        # subscribers dictionary mutation. Now it is protected by copy
        for condition, s in tuple(self._subscribers.values()):
            if condition and not condition(event):
                continue

            async with AsyncExitStack() as stack:
                await s.asolve(
                    event,
                    cache_dependencies={},
                    stack=stack,
                    dependency_provider=context.dependency_provider,
                    **{CONTEXT_OPTION_NAME: context},
                )


class SubStream(ABCStream):
    __slots__ = (
        "id",
        "_filter_condition",
        "_parent",
    )

    def __init__(
        self,
        parent: Stream,
        condition: Condition,
    ) -> None:
        self.id: StreamId = uuid4()

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

    async def send(self, event: BaseEvent, context: "Context") -> None:
        await self._parent.send(event, context)
