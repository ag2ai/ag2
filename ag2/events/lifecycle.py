# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent lifecycle events: observer, compaction, aggregation, and deserialization.

These events are emitted by Agent (framework core) during execution. They are
not network-specific — any Agent emits them regardless of Hub registration.

Task-subagent lifecycle events live in ``ag2.events``
(``TaskStarted`` / ``TaskProgress`` / ``TaskCompleted`` / ``TaskFailed``).
"""

from .base import BaseEvent, Field


class ObserverStarted(BaseEvent):
    """Emitted when an observer attaches to the agent's stream."""

    __transient__ = True

    name: str


class ObserverCompleted(BaseEvent):
    """Emitted when an observer detaches from the agent's stream."""

    __transient__ = True

    name: str


class CompactionStarted(BaseEvent):
    """Emitted on the agent's stream when compaction begins.

    Paired with either ``CompactionCompleted`` (success) or
    ``CompactionFailed`` (the strategy raised). Subscribe to both if you
    need to know that a compaction attempt was made — relying on
    ``CompactionCompleted`` alone hides errors.
    """

    __transient__ = True

    agent: str
    strategy: str
    event_count: int


class CompactionCompleted(BaseEvent):
    """Emitted on the agent's stream when compaction finishes."""

    __transient__ = True

    agent: str
    strategy: str
    events_before: int
    events_after: int
    llm_calls: int = 0
    usage: dict = Field(default_factory=dict)


class CompactionFailed(BaseEvent):
    """Emitted on the agent's stream when a compaction strategy raises.

    The exception is also logged via the module logger, but the stream
    event is the durable signal — observers and tests should subscribe
    here rather than configuring Python logging.
    """

    __transient__ = True

    agent: str
    strategy: str
    error_type: str
    error: str


class AggregationStarted(BaseEvent):
    """Emitted on the agent's stream when aggregation begins.

    Paired with either ``AggregationCompleted`` (success) or
    ``AggregationFailed`` (the strategy raised). Subscribe to both if you
    need to know that an aggregation attempt was made.
    """

    __transient__ = True

    agent: str
    strategy: str
    event_count: int


class AggregationCompleted(BaseEvent):
    """Emitted on the agent's stream when aggregation finishes."""

    __transient__ = True

    agent: str
    strategy: str
    event_count: int
    llm_calls: int = 0
    usage: dict = Field(default_factory=dict)


class AggregationFailed(BaseEvent):
    """Emitted on the agent's stream when an aggregation strategy raises.

    The exception is also logged via the module logger, but the stream
    event is the durable signal — observers and tests should subscribe
    here rather than configuring Python logging.
    """

    __transient__ = True

    agent: str
    strategy: str
    error_type: str
    error: str


class EventLogFailed(BaseEvent):
    """Emitted when the post-``ask`` event-log writer raises.

    Fires after the agent's turn completes but before ``ask`` returns. The
    failure does not interrupt the turn — the reply is still delivered —
    but log persistence into ``/log/{stream_id}.jsonl`` did not happen.
    """

    __transient__ = True

    agent: str
    error_type: str
    error: str


class UnknownEvent(BaseEvent):
    """Placeholder for events whose type cannot be resolved during deserialization.

    Preserves the raw data so nothing is lost.
    """

    type_name: str
    data: dict = Field(default_factory=dict)
