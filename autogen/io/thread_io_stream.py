# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import queue
from asyncio import Queue as AsyncQueue
from typing import TYPE_CHECKING, Any, Optional

from autogen.io.base import AsyncIOStreamProtocol, IOStreamProtocol

from ..events.agent_events import InputRequestEvent
from ..events.print_event import PrintEvent

if TYPE_CHECKING:
    from .step_controller import AsyncStepController, StepController


class ThreadIOStream:
    def __init__(self, step_controller: Optional["StepController"] = None) -> None:
        self._input_stream: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        self._output_stream: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        self._step_controller = step_controller

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        self.send(InputRequestEvent(prompt=prompt, password=password))  # type: ignore[call-arg]
        return self._output_stream.get()  # type: ignore[no-any-return]

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        print_message = PrintEvent(*objects, sep=sep, end=end)
        self.send(print_message)

    def send(self, message: Any) -> None:
        self._input_stream.put(message)
        # Block if step controller says so (based on event type)
        if self._step_controller:
            self._step_controller.wait_for_step(message)

    @property
    def input_stream(self) -> queue.Queue:  # type: ignore[type-arg]
        return self._input_stream


class AsyncThreadIOStream:
    def __init__(self, step_controller: Optional["AsyncStepController"] = None) -> None:
        self._input_stream: AsyncQueue = AsyncQueue()  # type: ignore[type-arg]
        self._output_stream: AsyncQueue = AsyncQueue()  # type: ignore[type-arg]
        self._step_controller = step_controller

    async def input(self, prompt: str = "", *, password: bool = False) -> str:
        await self.send(InputRequestEvent(prompt=prompt, password=password))  # type: ignore[call-arg]
        return await self._output_stream.get()  # type: ignore[no-any-return]

    async def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        print_message = PrintEvent(*objects, sep=sep, end=end)
        await self.send(print_message)

    async def send(self, message: Any) -> None:
        self._input_stream.put_nowait(message)
        # Block if step controller says so (based on event type)
        if self._step_controller:
            await self._step_controller.wait_for_step(message)

    @property
    def input_stream(self) -> AsyncQueue[Any]:
        return self._input_stream


if TYPE_CHECKING:

    def check_type_1(x: ThreadIOStream) -> IOStreamProtocol:
        return x

    def check_type_2(x: AsyncThreadIOStream) -> AsyncIOStreamProtocol:
        return x
