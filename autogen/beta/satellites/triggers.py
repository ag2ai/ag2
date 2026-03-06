# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.context import Stream, SubId
from autogen.beta.events import BaseEvent
from autogen.beta.events.conditions import ClassInfo, Condition, TypeCondition


@runtime_checkable
class Trigger(Protocol):
    """Protocol for satellite trigger strategies.

    Triggers determine when a satellite's ``process`` method is invoked.
    """

    def attach(
        self,
        stream: Stream,
        callback: Callable[[list[BaseEvent], Context], Awaitable[None]],
    ) -> SubId:
        """Subscribe to the stream and call ``callback`` when triggered."""
        ...

    def detach(self, stream: Stream, sub_id: SubId) -> None:
        """Remove the subscription from the stream."""
        ...


class OnEvent:
    """Fire immediately for each matching event."""

    def __init__(self, condition: ClassInfo | Condition) -> None:
        if not isinstance(condition, Condition):
            condition = TypeCondition(condition)
        self._condition = condition

    def attach(
        self,
        stream: Stream,
        callback: Callable[[list[BaseEvent], Context], Awaitable[None]],
    ) -> SubId:
        async def _handler(event: BaseEvent, ctx: Context) -> None:
            await callback([event], ctx)

        return stream.subscribe(_handler, condition=self._condition)

    def detach(self, stream: Stream, sub_id: SubId) -> None:
        stream.unsubscribe(sub_id)


class EveryNEvents:
    """Buffer *n* matching events, then fire with the batch."""

    def __init__(
        self,
        n: int,
        condition: ClassInfo | Condition | None = None,
    ) -> None:
        self._n = n
        self._condition: Condition | None = None
        if condition is not None:
            self._condition = (
                condition if isinstance(condition, Condition) else TypeCondition(condition)
            )
        self._buffer: list[BaseEvent] = []

    def attach(
        self,
        stream: Stream,
        callback: Callable[[list[BaseEvent], Context], Awaitable[None]],
    ) -> SubId:
        async def _handler(event: BaseEvent, ctx: Context) -> None:
            self._buffer.append(event)
            if len(self._buffer) >= self._n:
                batch = self._buffer[:]
                self._buffer.clear()
                await callback(batch, ctx)

        return stream.subscribe(_handler, condition=self._condition)

    def detach(self, stream: Stream, sub_id: SubId) -> None:
        stream.unsubscribe(sub_id)
        self._buffer.clear()
