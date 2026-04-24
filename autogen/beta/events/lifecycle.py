# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Actor lifecycle events: observer, compaction, aggregation, and deserialization.

These events are emitted by Actor (framework core) during execution. They are
not network-specific — any Actor emits them regardless of Hub registration.

Task-subagent lifecycle events live in ``autogen.beta.events.task_events``
(``TaskStarted`` / ``TaskProgress`` / ``TaskCompleted`` / ``TaskFailed``).
"""

from .base import BaseEvent, Field


class ObserverStarted(BaseEvent):
    """Emitted when an observer attaches to the actor's stream."""

    __transient__ = True

    name: str


class ObserverCompleted(BaseEvent):
    """Emitted when an observer detaches from the actor's stream."""

    __transient__ = True

    name: str


class CompactionCompleted(BaseEvent):
    """Emitted on the actor's stream when compaction finishes."""

    __transient__ = True

    actor: str
    strategy: str
    events_before: int
    events_after: int
    llm_calls: int = 0
    usage: dict = Field(default_factory=dict)


class AggregationCompleted(BaseEvent):
    """Emitted on the actor's stream when aggregation finishes."""

    __transient__ = True

    actor: str
    strategy: str
    event_count: int
    llm_calls: int = 0
    usage: dict = Field(default_factory=dict)


class UnknownEvent(BaseEvent):
    """Placeholder for events whose type cannot be resolved during deserialization.

    Preserves the raw data so nothing is lost.
    """

    type_name: str
    data: dict = Field(default_factory=dict)
