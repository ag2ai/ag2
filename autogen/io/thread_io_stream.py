# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import queue
from typing import Any

from ..events.agent_events import InputRequestEvent
from ..events.print_event import PrintEvent


class ThreadIOStream:
    def __init__(self) -> None:
        self._input_stream: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        self._output_stream: queue.Queue = queue.Queue()  # type: ignore[type-arg]

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        self.send(InputRequestEvent(prompt=prompt, password=password))  # type: ignore[call-arg]
        return self._output_stream.get()  # type: ignore[no-any-return]

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        print_message = PrintEvent(*objects, sep=sep, end=end)
        self.send(print_message)

    def send(self, message: Any) -> None:
        self._input_stream.put(message)

    @property
    def input_stream(self) -> queue.Queue:  # type: ignore[type-arg]
        return self._input_stream
