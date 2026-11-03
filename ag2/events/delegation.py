# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

if TYPE_CHECKING:
    from ag2.tools.subagents._forwarding import StreamHandle

from .base import BaseEvent, Field

_E = TypeVar("_E", bound=BaseEvent)

# Kept out of the event's fields: a delivery is never serialized or compared.
_DELIVERY_ATTRIBUTE = "_delivery"


@dataclass(frozen=True, slots=True)
class Delivery:
    """How a copy entered a stream it was not published on.

    A routed copy is never forwarded back to the stream it came from. A stream
    calls ``validate`` before each handler; it raises once the addressed
    invocation has ended, so a delivery cannot reach a later turn.
    """

    validate: Callable[[], None] | None = None


def routed(event: _E, delivery: Delivery) -> _E:
    """A copy of ``event`` marked as delivered from another stream."""
    marked = copy(event)
    marked.__dict__[_DELIVERY_ATTRIBUTE] = delivery
    return marked


def delivery_of(event: BaseEvent) -> Delivery | None:
    """The delivery of a routed copy; ``None`` for an event published on its stream."""
    delivery = event.__dict__.get(_DELIVERY_ATTRIBUTE)
    return delivery if isinstance(delivery, Delivery) else None


class InertEvent(BaseEvent):
    """An event that triggers nothing, so ancestors receive it as itself.

    Accounting and task lifecycle events are forwarded without a ``ChildEvent``
    envelope (ADR 0030), so typed subscriptions keep receiving them. The first
    forwarding hop stamps where the event came from; later hops keep it.
    """

    __conversational__ = False

    path: tuple[str, ...] = Field(default=(), compare=False)
    """Task ids of the delegations the event came through, outermost first; empty for a local event."""
    _origin: Any = Field(default=None, compare=False, repr=False)

    @property
    def origin(self) -> "StreamHandle":
        """The live address of a forwarded event's author; absent locally or on replay."""
        if self._origin is None:
            raise RuntimeError("A local or replayed event has no live origin")
        return cast("StreamHandle", self._origin)

    def as_local(self) -> Self:
        """This event as a new publication on the current stream, without inherited provenance."""
        if not self.path and self._origin is None:
            return self
        event = copy(self)
        event.path = ()
        event._origin = None
        return event


class ChildEvent(BaseEvent):
    """An observation of an event authored by a descendant, never a local trigger."""

    __conversational__ = False

    inner: BaseEvent
    task_id: str
    path: tuple[str, ...]
    agent_name: str
    origin_event_id: str
    direction: str = "up"
    _origin: Any = Field(default=None, compare=False, repr=False)

    @property
    def source_event(self) -> BaseEvent:
        """The original authored event, including replayed observations."""
        return self.inner.source_event

    @property
    def origin(self) -> "StreamHandle":
        """The live address of the original author; absent on replayed events."""
        if self._origin is None:
            raise RuntimeError("A replayed event has no live origin")
        return cast("StreamHandle", self._origin)


class TransientChildEvent(ChildEvent):
    """A descendant's transient event retains its storage policy."""

    __transient__ = True


class DeliveryFailed(BaseEvent):
    """A downlink could not reach an executing delegation."""

    __conversational__ = False
    task_id: str | None = None
    reason: str


class DownEvent(BaseEvent):
    """Audit of an ancestor's intervention; the inner event is delivered normally."""

    __conversational__ = False
    inner: BaseEvent
    task_id: str
    path: tuple[str, ...]
    agent_name: str
    direction: str = "down"
    origin_task_id: str | None = None
    origin_path: tuple[str, ...] = ()
    origin_stream_id: str = ""


class TransientDownEvent(DownEvent):
    __transient__ = True
