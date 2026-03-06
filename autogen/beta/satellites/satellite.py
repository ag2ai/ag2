# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.context import Stream, SubId
from autogen.beta.events import BaseEvent

from .events import SatelliteFlag
from .triggers import Trigger


@runtime_checkable
class Satellite(Protocol):
    """Protocol for satellite plug-ins.

    Any object implementing ``attach`` / ``detach`` can serve as a satellite.
    """

    name: str

    def attach(self, stream: Stream, ctx: Context) -> None: ...
    def detach(self) -> None: ...


class BaseSatellite(ABC):
    """Convenience base class for trigger-driven satellites.

    Subclasses only need to implement :meth:`process`.  The trigger handles
    stream subscription and event buffering.
    """

    def __init__(self, name: str, trigger: Trigger) -> None:
        self.name = name
        self._trigger = trigger
        self._sub_id: SubId | None = None
        self._stream: Stream | None = None

    def attach(self, stream: Stream, ctx: Context) -> None:
        self._stream = stream
        self._sub_id = self._trigger.attach(stream, self._on_trigger)

    def detach(self) -> None:
        if self._stream and self._sub_id:
            self._trigger.detach(self._stream, self._sub_id)
        self._stream = None
        self._sub_id = None

    async def _on_trigger(self, events: list[BaseEvent], ctx: Context) -> None:
        flag = await self.process(events, ctx)
        if flag is not None:
            await ctx.send(flag)

    @abstractmethod
    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        """Analyse *events* and optionally return a flag for the planet."""
        ...
