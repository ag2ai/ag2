# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from .base import BaseEvent


class ChildEvent(BaseEvent):
    """An observation of an event authored by a descendant, never a local trigger."""

    __conversational__ = False

    inner: BaseEvent
    task_id: str
    path: tuple[str, ...]
    agent_name: str
    origin_event_id: str
    direction: str = "up"

    @property
    def source_event(self) -> BaseEvent:
        """The original authored event, including replayed observations."""
        return self.inner.source_event


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
