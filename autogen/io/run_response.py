# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import queue
from asyncio import Queue as AsyncQueue
from typing import Any, AsyncIterable, Iterable, Optional, Protocol, Sequence
from uuid import UUID, uuid4

from ..agentchat.agent import Agent, LLMMessageType
from ..events.agent_events import ErrorEvent, InputRequestEvent, TerminationEvent
from ..events.base_event import BaseEvent
from .processors import (
    AsyncConsoleEventProcessor,
    AsyncEventProcessorProtocol,
    ConsoleEventProcessor,
    EventProcessorProtocol,
)
from .thread_io_stream import AsyncThreadIOStream, ThreadIOStream

Message = dict[str, Any]


class RunInfoProtocol(Protocol):
    @property
    def uuid(self) -> UUID: ...

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]: ...


class RunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> Iterable[BaseEvent]: ...

    @property
    def messages(self) -> Iterable[Message]: ...

    @property
    def summary(self) -> Optional[str]: ...

    @property
    def context_variables(self) -> Optional[dict[str, Any]]: ...

    @property
    def last_speaker(self) -> Optional[Agent]: ...

    def process(self, processor: Optional[EventProcessorProtocol] = None) -> None: ...


class AsyncRunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> AsyncIterable[BaseEvent]: ...

    @property
    async def messages(self) -> Iterable[Message]: ...

    @property
    async def summary(self) -> Optional[str]: ...

    @property
    async def context_variables(self) -> Optional[dict[str, Any]]: ...

    @property
    async def last_speaker(self) -> Optional[Agent]: ...

    async def process(self, processor: Optional[AsyncEventProcessorProtocol] = None) -> None: ...


class RunResponse:
    def __init__(self, iostream: ThreadIOStream):
        self.iostream = iostream
        self._summary: Optional[str] = None
        self._messages: Sequence[LLMMessageType] = []
        self._uuid = uuid4()
        self._context_variables: Optional[dict[str, Any]] = None
        self._last_speaker: Optional[Agent] = None

    def _queue_generator(self, q: queue.Queue) -> Iterable[BaseEvent]:  # type: ignore[type-arg]
        """A generator to yield items from the queue until the termination message is found."""
        while True:
            try:
                # Get an item from the queue
                event = q.get(timeout=0.1)  # Adjust timeout as needed

                if isinstance(event, InputRequestEvent):
                    event.content.respond = lambda response: self.iostream._output_stream.put(response)  # type: ignore[attr-defined]

                yield event

                if isinstance(event, TerminationEvent):
                    break

                if isinstance(event, ErrorEvent):
                    raise event.content.error  # type: ignore[attr-defined]
            except queue.Empty:
                continue  # Wait for more items in the queue

    @property
    def events(self) -> Iterable[BaseEvent]:
        return self._queue_generator(self.iostream.input_stream)

    @property
    def messages(self) -> Iterable[Message]:
        return self._messages

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]:
        return None

    @property
    def uuid(self) -> UUID:
        return self._uuid

    @property
    def context_variables(self) -> Optional[dict[str, Any]]:
        return self._context_variables

    @property
    def last_speaker(self) -> Optional[Agent]:
        return self._last_speaker

    def process(self, processor: Optional[EventProcessorProtocol] = None) -> None:
        processor = processor or ConsoleEventProcessor()
        processor.process(self)


class AsyncRunResponse:
    def __init__(self, iostream: AsyncThreadIOStream):
        self.iostream = iostream
        self._summary: Optional[str] = None
        self._messages: Sequence[LLMMessageType] = []
        self._uuid = uuid4()
        self._context_variables: Optional[dict[str, Any]] = None
        self._last_speaker: Optional[Agent] = None

    async def _queue_generator(self, q: AsyncQueue[Any]) -> AsyncIterable[BaseEvent]:  # type: ignore[type-arg]
        """A generator to yield items from the queue until the termination message is found."""
        while True:
            try:
                # Get an item from the queue
                event = await q.get()

                if isinstance(event, InputRequestEvent):

                    async def respond(response: str) -> None:
                        await self.iostream._output_stream.put(response)

                    event.content.respond = respond  # type: ignore[attr-defined]

                yield event

                if isinstance(event, TerminationEvent):
                    break

                if isinstance(event, ErrorEvent):
                    raise event.content.error  # type: ignore[attr-defined]
            except queue.Empty:
                continue

    @property
    def events(self) -> AsyncIterable[BaseEvent]:
        return self._queue_generator(self.iostream.input_stream)

    @property
    async def messages(self) -> Iterable[Message]:
        return self._messages

    @property
    async def summary(self) -> Optional[str]:
        return self._summary

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]:
        return None

    @property
    def uuid(self) -> UUID:
        return self._uuid

    @property
    async def context_variables(self) -> Optional[dict[str, Any]]:
        return self._context_variables

    @property
    async def last_speaker(self) -> Optional[Agent]:
        return self._last_speaker

    async def process(self, processor: Optional[AsyncEventProcessorProtocol] = None) -> None:
        processor = processor or AsyncConsoleEventProcessor()
        await processor.process(self)
