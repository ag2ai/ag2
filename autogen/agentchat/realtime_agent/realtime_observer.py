# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from anyio import Event

from .realtime_client import RealtimeClientProtocol

if TYPE_CHECKING:
    from .realtime_agent import RealtimeAgent

__all__ = ["RealtimeObserver"]


class RealtimeObserver(ABC):
    """Observer for the OpenAI Realtime API."""

    def __init__(self) -> None:
        self._ready_event = Event()
        self._agent: Optional["RealtimeAgent"] = None

    @property
    def agent(self) -> "RealtimeAgent":
        if self._agent is None:
            raise RuntimeError("Agent has not been set.")
        return self._agent

    @property
    def realtime_client(self) -> RealtimeClientProtocol:
        if self._agent is None:
            raise RuntimeError("Agent has not been set.")
        if self._agent.realtime_client is None:
            raise RuntimeError("Realtime client has not been set.")

        return self._agent.realtime_client

    async def run(self, agent: "RealtimeAgent") -> None:
        """Run the observer with the agent.

        When implementing, be sure to call `self._ready_event.set()` when the observer is ready to process events.

        Args:
            agent (RealtimeAgent): The realtime agent attached to the observer.
        """
        ...

    async def wait_for_ready(self) -> None:
        """Get the event that is set when the observer is ready."""
        await self._ready_event.wait()

    @abstractmethod
    async def on_event(self, event: dict[str, Any]) -> None:
        """Handle an event from the OpenAI Realtime API.

        Args:
            event (RealtimeServerEvent): The event from the OpenAI Realtime API.
        """
        ...
