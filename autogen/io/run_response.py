# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import queue
import threading
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Sequence
from typing import Any, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from autogen.tools.tool import Tool

from ..agentchat.agent import Agent, LLMMessageType
from ..agentchat.group.context_variables import ContextVariables
from ..events.agent_events import ErrorEvent, InputRequestEvent, RunCompletionEvent
from ..events.base_event import BaseEvent
from .processors import (
    AsyncConsoleEventProcessor,
    AsyncEventProcessorProtocol,
    ConsoleEventProcessor,
    EventProcessorProtocol,
)
from .step_controller import StepController
from .thread_io_stream import AsyncThreadIOStream, ThreadIOStream

Message = dict[str, Any]


@runtime_checkable
class RunInfoProtocol(Protocol):
    @property
    def uuid(self) -> UUID: ...

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]: ...


class Usage(BaseModel):
    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CostBreakdown(BaseModel):
    total_cost: float
    models: dict[str, Usage] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "CostBreakdown":
        # Extract total cost
        total_cost = data.get("total_cost", 0.0)

        # Remove total_cost key to extract models
        model_usages = {k: Usage(**v) for k, v in data.items() if k != "total_cost"}

        return cls(total_cost=total_cost, models=model_usages)


class Cost(BaseModel):
    usage_including_cached_inference: CostBreakdown
    usage_excluding_cached_inference: CostBreakdown

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "Cost":
        return cls(
            usage_including_cached_inference=CostBreakdown.from_raw(data.get("usage_including_cached_inference", {})),
            usage_excluding_cached_inference=CostBreakdown.from_raw(data.get("usage_excluding_cached_inference", {})),
        )


@runtime_checkable
class RunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> Iterable[BaseEvent]: ...

    @property
    def messages(self) -> Iterable[Message]: ...

    @property
    def summary(self) -> str | None: ...

    @property
    def context_variables(self) -> ContextVariables | None: ...

    @property
    def last_speaker(self) -> str | None: ...

    @property
    def cost(self) -> Cost | None: ...

    def process(self, processor: EventProcessorProtocol | None = None) -> None: ...

    def set_ui_tools(self, tools: list[Tool]) -> None: ...


@runtime_checkable
class AsyncRunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> AsyncIterable[BaseEvent]: ...

    @property
    async def messages(self) -> Iterable[Message]: ...

    @property
    async def summary(self) -> str | None: ...

    @property
    async def context_variables(self) -> ContextVariables | None: ...

    @property
    async def last_speaker(self) -> str | None: ...

    @property
    async def cost(self) -> Cost | None: ...

    async def process(self, processor: AsyncEventProcessorProtocol | None = None) -> None: ...

    def set_ui_tools(self, tools: list[Tool]) -> None: ...


class RunResponse:
    def __init__(
        self,
        iostream: ThreadIOStream,
        agents: Sequence[Agent],
    ):
        self.iostream = iostream
        self.agents = agents
        self._summary: str | None = None
        self._messages: Sequence[LLMMessageType] = []
        self._uuid = uuid4()
        self._context_variables: ContextVariables | None = None
        self._last_speaker: str | None = None
        self._cost: Cost | None = None

    def _queue_generator(self, q: queue.Queue) -> Iterable[BaseEvent]:  # type: ignore[type-arg]
        """A generator to yield items from the queue until the termination message is found."""
        while True:
            try:
                # Get an item from the queue
                event = q.get(timeout=0.1)  # Adjust timeout as needed

                if isinstance(event, InputRequestEvent):
                    event.content.respond = lambda response: self.iostream._output_stream.put(response)  # type: ignore[attr-defined]

                yield event

                if isinstance(event, RunCompletionEvent):
                    self._messages = event.content.history  # type: ignore[attr-defined]
                    self._last_speaker = event.content.last_speaker  # type: ignore[attr-defined]
                    self._summary = event.content.summary  # type: ignore[attr-defined]
                    self._context_variables = event.content.context_variables  # type: ignore[attr-defined]
                    self.cost = event.content.cost  # type: ignore[attr-defined]
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
    def summary(self) -> str | None:
        return self._summary

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]:
        return None

    @property
    def uuid(self) -> UUID:
        return self._uuid

    @property
    def context_variables(self) -> ContextVariables | None:
        return self._context_variables

    @property
    def last_speaker(self) -> str | None:
        return self._last_speaker

    @property
    def cost(self) -> Cost | None:
        return self._cost

    @cost.setter
    def cost(self, value: Cost | dict[str, Any]) -> None:
        if isinstance(value, dict):
            self._cost = Cost.from_raw(value)
        else:
            self._cost = value

    def process(self, processor: EventProcessorProtocol | None = None) -> None:
        processor = processor or ConsoleEventProcessor()
        processor.process(self)

    def set_ui_tools(self, tools: list[Tool]) -> None:
        """Set the UI tools for the agents."""
        for agent in self.agents:
            agent.set_ui_tools(tools)


class AsyncRunResponse:
    def __init__(
        self,
        iostream: AsyncThreadIOStream,
        agents: Sequence[Agent],
    ):
        self.iostream = iostream
        self.agents = agents
        self._summary: str | None = None
        self._messages: Sequence[LLMMessageType] = []
        self._uuid = uuid4()
        self._context_variables: ContextVariables | None = None
        self._last_speaker: str | None = None
        self._cost: Cost | None = None

    async def _queue_generator(self, q: asyncio.Queue[Any]) -> AsyncIterable[BaseEvent]:  # type: ignore[type-arg]
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

                if isinstance(event, RunCompletionEvent):
                    self._messages = event.content.history  # type: ignore[attr-defined]
                    self._last_speaker = event.content.last_speaker  # type: ignore[attr-defined]
                    self._summary = event.content.summary  # type: ignore[attr-defined]
                    self._context_variables = event.content.context_variables  # type: ignore[attr-defined]
                    self.cost = event.content.cost  # type: ignore[attr-defined]
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
    async def summary(self) -> str | None:
        return self._summary

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]:
        return None

    @property
    def uuid(self) -> UUID:
        return self._uuid

    @property
    async def context_variables(self) -> ContextVariables | None:
        return self._context_variables

    @property
    async def last_speaker(self) -> str | None:
        return self._last_speaker

    @property
    async def cost(self) -> Cost | None:
        return self._cost

    @cost.setter
    def cost(self, value: Cost | dict[str, Any]) -> None:
        if isinstance(value, dict):
            self._cost = Cost.from_raw(value)
        else:
            self._cost = value

    async def process(self, processor: AsyncEventProcessorProtocol | None = None) -> None:
        processor = processor or AsyncConsoleEventProcessor()
        await processor.process(self)

    def set_ui_tools(self, tools: list[Tool]) -> None:
        """Set the UI tools for the agents."""
        for agent in self.agents:
            agent.set_ui_tools(tools)


class RunIterResponse:
    """Iterator-based response for stepped execution.

    This class provides an iterator interface for stepping through agent execution.
    The background thread blocks after each event until you advance to the next iteration.

    Example:
        for event in agent.run_iter(message="Hello"):
            if isinstance(event, ToolCallEvent):
                print(f"Tool call: {event.content}")

    The generator's finally block ensures cleanup on break, exception, or normal completion.
    """

    def __init__(
        self,
        start_thread_func: Callable[[ThreadIOStream], "threading.Thread"],
        yield_on: Sequence[type[BaseEvent]] | None,
        agents: Sequence[Agent],
    ):
        """Initialize the iterator response.

        Args:
            start_thread_func: Function that creates and returns (but doesn't start) the background thread.
                              Takes the iostream as argument.
            yield_on: Event types to yield. If None, yields all events.
            agents: List of agents involved in the chat.
        """
        self._start_thread_func = start_thread_func
        self._yield_on = yield_on
        self._agents = agents
        self._started = False
        self._thread: threading.Thread | None = None

        # Set up step controller and iostream
        self._step_controller = StepController(yield_on=yield_on)
        self._iostream = ThreadIOStream(step_controller=self._step_controller)

        # State populated after completion
        self._summary: str | None = None
        self._messages: Sequence[LLMMessageType] = []
        self._context_variables: ContextVariables | None = None
        self._last_speaker: str | None = None
        self._cost: Cost | None = None
        self._uuid = uuid4()

    def __iter__(self) -> Iterator[BaseEvent]:
        """Return the generator iterator."""
        return self._generator()

    def _generator(self) -> Iterator[BaseEvent]:
        """Generate events from the background thread.

        Lazily starts the thread on first iteration.
        Cleanup happens in finally block on break, exception, or completion.
        """
        # Lazy start - only start thread on first iteration
        if not self._started:
            self._thread = self._start_thread_func(self._iostream)
            self._thread.start()
            self._started = True

        try:
            while True:
                # Signal producer to continue (unblock wait_for_step)
                self._step_controller.step()

                # Wait for next event
                event = self._iostream._input_stream.get()

                # Handle completion
                if isinstance(event, RunCompletionEvent):
                    self._extract_completion_data(event)
                    return  # StopIteration

                # Handle errors
                if isinstance(event, ErrorEvent):
                    raise event.content.error  # type: ignore[attr-defined]

                # Handle input requests - always yield these
                if isinstance(event, InputRequestEvent):
                    event.content.respond = lambda response: self._iostream._output_stream.put(response)  # type: ignore[attr-defined]
                    yield event
                    continue

                # Filter based on yield_on - yield if should_block returns True
                if self._step_controller.should_block(event):
                    yield event
        finally:
            # Cleanup - always terminate the step controller
            self._step_controller.terminate()

    def _extract_completion_data(self, event: RunCompletionEvent) -> None:
        """Extract data from completion event."""
        self._messages = event.content.history  # type: ignore[attr-defined]
        self._last_speaker = event.content.last_speaker  # type: ignore[attr-defined]
        self._summary = event.content.summary  # type: ignore[attr-defined]
        self._context_variables = event.content.context_variables  # type: ignore[attr-defined]
        if isinstance(event.content.cost, dict):  # type: ignore[attr-defined]
            self._cost = Cost.from_raw(event.content.cost)  # type: ignore[attr-defined]
        else:
            self._cost = event.content.cost  # type: ignore[attr-defined]

    @property
    def iostream(self) -> ThreadIOStream:
        """The IO stream for this response."""
        return self._iostream

    @property
    def agents(self) -> Sequence[Agent]:
        """The agents involved in this chat."""
        return self._agents

    @property
    def summary(self) -> str | None:
        """The summary of the chat (available after iteration completes)."""
        return self._summary

    @property
    def messages(self) -> Sequence[LLMMessageType]:
        """The message history (available after iteration completes)."""
        return self._messages

    @property
    def context_variables(self) -> ContextVariables | None:
        """The context variables (available after iteration completes)."""
        return self._context_variables

    @property
    def last_speaker(self) -> str | None:
        """The last speaker (available after iteration completes)."""
        return self._last_speaker

    @property
    def cost(self) -> Cost | None:
        """The cost information (available after iteration completes)."""
        return self._cost

    @property
    def uuid(self) -> UUID:
        """Unique identifier for this run."""
        return self._uuid

    def set_ui_tools(self, tools: list[Tool]) -> None:
        """Set the UI tools for the agents."""
        for agent in self._agents:
            agent.set_ui_tools(tools)


class AsyncRunIterResponse:
    """Async iterator-based response for stepped execution.

    This class provides an async iterator interface for stepping through agent execution.
    The background thread blocks after each event until you advance to the next iteration.

    Example:
        async for event in agent.a_run_iter(message="Hello"):
            if isinstance(event, ToolCallEvent):
                print(f"Tool call: {event.content}")

    The generator's finally block ensures cleanup on break, exception, or normal completion.

    Note: This uses threads internally (same as sync RunIterResponse) to avoid making
    iostream.send() async, which would require runtime type checks in calling code.
    """

    def __init__(
        self,
        start_thread_func: Callable[[ThreadIOStream], threading.Thread],
        yield_on: Sequence[type[BaseEvent]] | None,
        agents: Sequence[Agent],
    ):
        """Initialize the async iterator response.

        Args:
            start_thread_func: Function that creates and returns (but doesn't start) the background thread.
                              Takes the iostream as argument.
            yield_on: Event types to yield. If None, yields all events.
            agents: List of agents involved in the chat.
        """
        self._start_thread_func = start_thread_func
        self._yield_on = yield_on
        self._agents = agents
        self._started = False
        self._thread: threading.Thread | None = None

        # Set up step controller and iostream (sync versions, like RunIterResponse)
        self._step_controller = StepController(yield_on=yield_on)
        self._iostream = ThreadIOStream(step_controller=self._step_controller)

        # State populated after completion
        self._summary: str | None = None
        self._messages: Sequence[LLMMessageType] = []
        self._context_variables: ContextVariables | None = None
        self._last_speaker: str | None = None
        self._cost: Cost | None = None
        self._uuid = uuid4()

    def __aiter__(self) -> AsyncIterator[BaseEvent]:
        """Return the async generator iterator."""
        return self._generator()

    async def _generator(self) -> AsyncIterator[BaseEvent]:
        """Generate events from the background thread.

        Lazily starts the thread on first iteration.
        Cleanup happens in finally block on break, exception, or completion.
        """
        # Lazy start - only start thread on first iteration
        if not self._started:
            self._thread = self._start_thread_func(self._iostream)
            self._thread.start()
            self._started = True

        try:
            loop = asyncio.get_running_loop()
            while True:
                # Signal producer to continue (unblock wait_for_step)
                self._step_controller.step()

                # Wait for next event without blocking the event loop
                event = await loop.run_in_executor(None, self._iostream._input_stream.get)

                # Handle completion
                if isinstance(event, RunCompletionEvent):
                    self._extract_completion_data(event)
                    return  # StopAsyncIteration

                # Handle errors
                if isinstance(event, ErrorEvent):
                    raise event.content.error  # type: ignore[attr-defined]

                # Handle input requests - always yield these
                if isinstance(event, InputRequestEvent):
                    event.content.respond = lambda response: self._iostream._output_stream.put(response)  # type: ignore[attr-defined]
                    yield event
                    continue

                # Filter based on yield_on - yield if should_block returns True
                if self._step_controller.should_block(event):
                    yield event
        finally:
            # Cleanup - always terminate the step controller
            self._step_controller.terminate()

    def _extract_completion_data(self, event: RunCompletionEvent) -> None:
        """Extract data from completion event."""
        self._messages = event.content.history  # type: ignore[attr-defined]
        self._last_speaker = event.content.last_speaker  # type: ignore[attr-defined]
        self._summary = event.content.summary  # type: ignore[attr-defined]
        self._context_variables = event.content.context_variables  # type: ignore[attr-defined]
        if isinstance(event.content.cost, dict):  # type: ignore[attr-defined]
            self._cost = Cost.from_raw(event.content.cost)  # type: ignore[attr-defined]
        else:
            self._cost = event.content.cost  # type: ignore[attr-defined]

    @property
    def iostream(self) -> ThreadIOStream:
        """The IO stream for this response."""
        return self._iostream

    @property
    def agents(self) -> Sequence[Agent]:
        """The agents involved in this chat."""
        return self._agents

    @property
    def summary(self) -> str | None:
        """The summary of the chat (available after iteration completes)."""
        return self._summary

    @property
    def messages(self) -> Sequence[LLMMessageType]:
        """The message history (available after iteration completes)."""
        return self._messages

    @property
    def context_variables(self) -> ContextVariables | None:
        """The context variables (available after iteration completes)."""
        return self._context_variables

    @property
    def last_speaker(self) -> str | None:
        """The last speaker (available after iteration completes)."""
        return self._last_speaker

    @property
    def cost(self) -> Cost | None:
        """The cost information (available after iteration completes)."""
        return self._cost

    @property
    def uuid(self) -> UUID:
        """Unique identifier for this run."""
        return self._uuid

    def set_ui_tools(self, tools: list[Tool]) -> None:
        """Set the UI tools for the agents."""
        for agent in self._agents:
            agent.set_ui_tools(tools)
