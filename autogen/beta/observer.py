# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer — event-stream subscribers with lifecycle registration.

An ``Observer`` is anything that registers itself against an actor's stream
inside a caller-owned ``ExitStack``.  Two canonical shapes ship out of the box:

1. ``StreamObserver`` — lightweight ``condition → callback`` subscription,
   produced by the :func:`observer` decorator/factory.  Ideal for one-off
   event hooks (e.g. ``observer(ModelResponse, on_response)``).
2. ``BaseObserver`` — trigger-driven monitoring primitive backed by a
   :class:`~autogen.beta.watch.Watch`.  Subclasses implement ``process`` and
   optionally return an :class:`~autogen.beta.events.alert.ObserverAlert`
   which is emitted on the stream.  Ideal for long-running health checks
   (``TokenMonitor``, ``LoopDetector``, …).

Both shapes satisfy the same :class:`Observer` protocol so the Actor
can register either kind via a single ``register(stack, ctx)`` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

from autogen.beta.types import ClassInfo

from .annotations import Context
from .events import BaseEvent
from .events.alert import ObserverAlert
from .events.conditions import Condition, TypeCondition
from .watch import Watch

__all__ = (
    "BaseObserver",
    "Observer",
    "StreamObserver",
    "observer",
)


@runtime_checkable
class Observer(Protocol):
    """Registers stream subscriptions under the caller's ExitStack."""

    def register(self, stack: ExitStack, context: Context) -> None: ...


@dataclass(slots=True)
class StreamObserver:
    """Lightweight ``condition → callback`` stream subscription.

    Produced by :func:`observer`. Enters a ``sub_scope`` on the filtered
    stream; the ``ExitStack`` cleans up the subscription when the actor
    finishes.
    """

    condition: Condition
    callback: Callable[..., Any]
    interrupt: bool = False
    sync_to_thread: bool = True

    def register(self, stack: ExitStack, context: Context) -> None:
        stack.enter_context(
            context.stream.where(self.condition).sub_scope(
                self.callback,
                interrupt=self.interrupt,
                sync_to_thread=self.sync_to_thread,
            )
        )


class BaseObserver(ABC):
    """Trigger-driven observer. Subclasses implement :meth:`process`.

    The :class:`Watch` handles stream subscription and event buffering.
    When the watch fires, :meth:`process` is called with the collected
    events. If ``process`` returns an :class:`ObserverAlert`, it is emitted
    on the stream.

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
        self._ctx: Context | None = None

    def register(self, stack: ExitStack, context: Context) -> None:
        if self._watch.is_armed:
            self._watch.disarm()
        self._ctx = context
        self._watch.arm(context.stream, self._on_watch)
        stack.callback(self._disarm)

    def _disarm(self) -> None:
        self._watch.disarm()
        self._ctx = None

    async def _on_watch(self, events: list[BaseEvent], ctx: Context) -> None:
        try:
            alert = await self.process(events, ctx)
            if alert is not None:
                await ctx.send(alert)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Observer '%s' process() failed", self.name)

    @abstractmethod
    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        """Analyze events and optionally return an alert."""
        ...


def _ensure_condition(condition: ClassInfo | Condition) -> Condition:
    if isinstance(condition, Condition):
        return condition
    return TypeCondition(condition)


@overload
def observer(
    condition: ClassInfo | Condition,
    callback: Callable[..., Any],
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> StreamObserver: ...


@overload
def observer(
    condition: ClassInfo | Condition,
    callback: None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], StreamObserver]: ...


def observer(
    condition: ClassInfo | Condition,
    callback: Callable[..., Any] | None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
) -> StreamObserver | Callable[[Callable[..., Any]], StreamObserver]:
    cond = _ensure_condition(condition)

    if callback is not None:
        return StreamObserver(
            condition=cond,
            callback=callback,
            interrupt=interrupt,
            sync_to_thread=sync_to_thread,
        )

    def decorator(func: Callable[..., Any]) -> StreamObserver:
        return StreamObserver(
            condition=cond,
            callback=func,
            interrupt=interrupt,
            sync_to_thread=sync_to_thread,
        )

    return decorator
