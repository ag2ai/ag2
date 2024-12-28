# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional

from asyncer import TaskGroup, create_task_group
from openai.types.beta.realtime.realtime_server_event import RealtimeServerEvent

if TYPE_CHECKING:
    from .oai_realtime_client import OpenAIRealtimeClient

RunningState = Literal["not started", "running", "shutdown requested", "shutdown completed"]


class RealtimeObserver(ABC):
    """Observer for the OpenAI Realtime API."""

    def __init__(self) -> None:
        self._client: Optional["OpenAIRealtimeClient"] = None
        self._shutdown_state: RunningState = "not started"
        self._tg: Optional[TaskGroup] = None

    @property
    def running_state(self) -> RunningState:
        """Get the shutdown state of the observer."""
        return self._shutdown_state

    @property
    def client(self) -> "OpenAIRealtimeClient":
        """Get the client associated with the observer."""
        if self._client is None:
            raise ValueError("Observer client is not registered.")

        return self._client

    def register_client(self, client: "OpenAIRealtimeClient") -> None:
        """Register a client with the observer."""
        self._client = client

    def request_shutdown(self) -> None:
        """Shutdown the observer."""
        if self._shutdown_state != "running":
            raise RuntimeError("Observer is not running.")
        if self._tg is None:
            raise RuntimeError("Task group is not initialized.")

        self._shutdown_state = "shutdown requested"
        self._tg.cancel_scope.cancel()

    async def run(self) -> None:
        try:
            async with create_task_group() as tg:
                self._tg = tg
                self._shutdown_state = "running"
                tg.start_soon(self._run)
        finally:
            self._shutdown_state = "shutdown completed"

    @abstractmethod
    async def _run(self) -> None:
        """Run the observer."""
        ...

    @abstractmethod
    async def update(self, message: RealtimeServerEvent) -> None:
        """Update the observer with a message from the OpenAI Realtime API."""
        ...
