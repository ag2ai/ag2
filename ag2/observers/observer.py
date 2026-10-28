# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer — event-stream subscribers with lifecycle registration.

An ``Observer`` is anything that registers itself against an agent's stream
inside a caller-owned ``ExitStack``.  Two canonical shapes ship out of the box:

1. ``StreamObserver`` — lightweight ``condition → callback`` subscription,
   produced by the :func:`observer` decorator/factory.  Ideal for one-off
   event hooks (e.g. ``observer(ModelResponse, on_response)``).  ``condition``
   is optional; when omitted the observer fires for every event.
2. ``BaseObserver`` — trigger-driven monitoring primitive backed by a
   :class:`~ag2.watch.Watch`.  Subclasses implement ``process`` and
   optionally return an :class:`~ag2.events.ObserverAlert`
   which is emitted on the stream.  Ideal for long-running health checks
   (``TokenMonitor``, ``LoopDetector``, …).

Both shapes satisfy the same :class:`Observer` protocol so the Agent
can register either kind via a single ``register(stack, ctx)`` call.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

from ag2.annotations import Context
from ag2.events import BaseEvent, ObserverAlert
from ag2.events.conditions import Condition, TypeCondition
from ag2.types import ClassInfo
from ag2.utils import _validate_timeout
from ag2.watch import Watch

__all__ = (
    "BaseObserver",
    "CompositeObserver",
    "Observer",
    "StreamObserver",
    "observer",
)


@runtime_checkable
class Observer(Protocol):
    """Registers stream subscriptions under the caller's ExitStack."""

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None: ...


class CompositeObserver(Observer):
    def __init__(self, *observers: Observer) -> None:
        self._observers = observers

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        for observer in self._observers:
            observer.register(stack, context)

    def __repr__(self) -> str:
        return f"CompositeObserver({', '.join(repr(o) for o in self._observers)})"


@dataclass(slots=True, kw_only=True)
class SimpleObserver:
    """Lightweight ``callback`` stream subscription.

    Produced by :func:`observer`. Enters a ``sub_scope`` on the filtered
    stream; the ``ExitStack`` cleans up the subscription when the agent
    finishes. When ``condition`` is ``None`` the observer fires on every
    event.
    """

    callback: Callable[..., Any]
    interrupt: bool = False
    sync_to_thread: bool = True
    timeout: float | None = None

    def __post_init__(self) -> None:
        _validate_timeout(self.timeout)

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        stack.enter_context(
            context.stream.sub_scope(
                self.callback,
                interrupt=self.interrupt,
                sync_to_thread=self.sync_to_thread,
                timeout=self.timeout,
            )
        )


@dataclass(slots=True, kw_only=True)
class StreamObserver(SimpleObserver):
    """Lightweight ``condition → callback`` stream subscription.

    Produced by :func:`observer`. Enters a ``sub_scope`` on the filtered
    stream; the ``ExitStack`` cleans up the subscription when the agent
    finishes. When ``condition`` is ``None`` the observer fires on every
    event.
    """

    condition: Condition

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        stack.enter_context(
            context.stream.where(self.condition).sub_scope(
                self.callback,
                interrupt=self.interrupt,
                sync_to_thread=self.sync_to_thread,
                timeout=self.timeout,
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
    timeout:
        Per-call limit for ``process`` in seconds, or ``None`` to wait indefinitely.
        Must be finite and positive. Expiry is logged like other process errors.
    """

    def __init__(self, name: str, watch: Watch, *, timeout: float | None = None) -> None:
        _validate_timeout(timeout)
        self.name = name
        self.timeout = timeout
        self._watch = watch
        self._ctx: Context | None = None

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
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
            if self.timeout is None:
                alert = await self.process(events, ctx)
            else:
                alert = await asyncio.wait_for(self.process(events, ctx), timeout=self.timeout)
            if alert is not None:
                await ctx.send(alert)
        except Exception:
            logging.getLogger(__name__).exception("Observer '%s' process() failed", self.name)

    @abstractmethod
    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        """Analyze events and optionally return an alert."""
        ...


@overload
def observer(
    condition: ClassInfo | Condition | None = None,
    callback: None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
    timeout: float | None = None,
) -> Callable[[Callable[..., Any]], StreamObserver]: ...


@overload
def observer(
    condition: ClassInfo | Condition | None,
    callback: Callable[..., Any],
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
    timeout: float | None = None,
) -> StreamObserver: ...


def observer(
    condition: ClassInfo | Condition | None = None,
    callback: Callable[..., Any] | None = None,
    *,
    interrupt: bool = False,
    sync_to_thread: bool = True,
    timeout: float | None = None,
) -> StreamObserver | Callable[[Callable[..., Any]], StreamObserver]:
    """Register an event callback with an optional per-invocation timeout.

    ``timeout`` is a finite positive number of seconds or ``None`` (no limit).
    It covers dependency resolution, the callback, and dependency cleanup.
    Expiry raises ``asyncio.TimeoutError`` to the event publisher. Cancellation is
    cooperative and cannot stop a synchronous callback's worker thread.
    """
    if condition is None:
        cond: Condition | None = None
    elif isinstance(condition, Condition):
        cond = condition
    else:
        cond = TypeCondition(condition)

    def decorator(func: Callable[..., Any]) -> StreamObserver:
        if cond is None:
            return SimpleObserver(
                callback=func,
                interrupt=interrupt,
                sync_to_thread=sync_to_thread,
                timeout=timeout,
            )
        return StreamObserver(
            condition=cond,
            callback=func,
            interrupt=interrupt,
            sync_to_thread=sync_to_thread,
            timeout=timeout,
        )

    if callback is not None:
        return decorator(callback)
    return decorator
