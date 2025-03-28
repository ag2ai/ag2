# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import queue
from typing import Any, AsyncIterable, Iterable, Optional, Protocol
from uuid import UUID, uuid4

from ..events.agent_events import ErrorEvent, InputRequestEvent, TerminationEvent
from ..events.base_event import BaseEvent
from .thread_io_stream import ThreadIOStream

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


class AsyncRunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> AsyncIterable[BaseEvent]: ...

    @property
    def messages(self) -> AsyncIterable[Message]: ...

    @property
    async def summary(self) -> str: ...


class RunResponse:
    def __init__(self, iostream: ThreadIOStream):
        self.iostream = iostream
        self._summary: Optional[str] = None
        self._uuid = uuid4()

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
        return []

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]:
        return None

    @property
    def uuid(self) -> UUID:
        return self._uuid
