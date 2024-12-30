# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, Optional

from anyio import Event

from .realtime_client import RealtimeClientProtocol

if TYPE_CHECKING:
    from .realtime_agent import RealtimeAgent

__all__ = ["RealtimeObserver"]

global_logger = getLogger(__name__)


class RealtimeObserver(ABC):
    """Observer for the OpenAI Realtime API."""

    def __init__(self, *, logger: Optional[Logger] = None) -> None:
        """Observer for the OpenAI Realtime API.

        Args:
            logger (Logger): The logger for the observer.
        """
        self._ready_event = Event()
        self._agent: Optional["RealtimeAgent"] = None
        self._logger = logger

    @property
    def logger(self) -> Logger:
        return self._logger or global_logger

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
        self._agent = agent
        await self.initialize_session()
        self._ready_event.set()

        await self.run_loop()

    @abstractmethod
    async def run_loop(self) -> None:
        """Run the loop if needed.

        This method is called after the observer is ready to process events.
        Events will be processed by the on_event method, this is just a hook for additional processing.
        Use initialize_session to set up the session.
        """
        ...

    @abstractmethod
    async def initialize_session(self) -> None:
        """Initialize the session for the observer."""
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
