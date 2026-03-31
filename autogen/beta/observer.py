# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer — monitors an event stream and produces alerts.

An Observer attaches to a stream, uses a Watch to monitor for conditions,
and produces ObserverAlert events when those conditions are met.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from autogen.beta.annotations import Context
from autogen.beta.context import Stream
from autogen.beta.events import BaseEvent

from .watch import Watch

if TYPE_CHECKING:
    from .network.primitives.signal import Signal


@runtime_checkable
class Observer(Protocol):
    """Monitors an event stream and produces alerts."""

    name: str

    def attach(self, stream: Stream, ctx: Context) -> None:
        """Begin observing. Arm watches, start monitoring."""
        ...

    def detach(self) -> None:
        """Stop observing. Disarm watches, clean up."""
        ...


class BaseObserver(ABC):
    """Trigger-driven observer. Subclasses implement process().

    The Watch handles stream subscription and event buffering.
    When the Watch fires, process() is called with the collected events.
    If process() returns a Signal/ObserverAlert, it is emitted on the stream.

    Parameters
    ----------
    name:
        Observer display name (used in alert ``source`` field).
    watch:
        Watch strategy that determines when ``process`` is called.
    """

    def __init__(self, name: str, watch: Watch) -> None:
        self.name = name
        self._watch = watch
        self._stream: Stream | None = None

    def attach(self, stream: Stream, ctx: Context) -> None:
        if self._watch.is_armed:
            self._watch.disarm()
        self._stream = stream
        self._watch.arm(stream, self._on_watch)

    def detach(self) -> None:
        self._watch.disarm()
        self._stream = None

    async def _on_watch(self, events: list[BaseEvent], ctx: Context) -> None:
        try:
            signal = await self.process(events, ctx)
            if signal is not None:
                await ctx.send(signal)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Observer '%s' process() failed", self.name)

    @abstractmethod
    async def process(self, events: list[BaseEvent], ctx: Context) -> Signal | None:
        """Analyze events and optionally return a signal/alert."""
        ...
