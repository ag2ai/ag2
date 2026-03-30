# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network event types: actor lifecycle, task lifecycle, delegation, and scheduling."""

from autogen.beta.events.base import BaseEvent, Field

# ------------------------------------------------------------------
# Actor lifecycle events
# ------------------------------------------------------------------


class ObserverStarted(BaseEvent):
    """Emitted when an observer attaches to the actor's stream."""

    name: str


class ObserverCompleted(BaseEvent):
    """Emitted when an observer detaches from the actor's stream."""

    name: str


# ------------------------------------------------------------------
# Task lifecycle events
# ------------------------------------------------------------------


class TaskRequest(BaseEvent):
    """Emitted when the actor spawns a task sub-agent."""

    task: str
    task_name: str


class TaskProgress(BaseEvent):
    """Streamed progress from a running task sub-agent."""

    task_name: str
    content: str


class TaskResult(BaseEvent):
    """Emitted when a task sub-agent completes its work."""

    task: str
    task_name: str
    result: str
    usage: dict = Field(default_factory=dict)


# ------------------------------------------------------------------
# Delegation events
# ------------------------------------------------------------------


class DelegationRequest(BaseEvent):
    """Emitted when an actor delegates work to another actor via the Hub."""

    source: str
    target: str
    task: str


class DelegationResult(BaseEvent):
    """Emitted when a delegation completes successfully."""

    source: str
    target: str
    result: str


class DelegationRejected(BaseEvent):
    """Emitted when a delegation is rejected by the Hub topology."""

    source: str
    target: str
    task: str
    reason: str


class DelegationError(BaseEvent):
    """Emitted when a delegation fails with an exception."""

    source: str
    target: str
    task: str
    error: str


# ------------------------------------------------------------------
# Scheduler events
# ------------------------------------------------------------------


class SchedulerTriggerFired(BaseEvent):
    """Emitted when the scheduler fires a watch."""

    watch_id: str
    target: str
    task: str


# ------------------------------------------------------------------
# Topic events (pub/sub)
# ------------------------------------------------------------------


class TopicMessage(BaseEvent):
    """A message published to a network topic."""

    topic: str
    sender: str
    message: str
    data: dict = Field(default_factory=dict)


class TopicSubscription(BaseEvent):
    """Emitted when an actor subscribes to a topic."""

    actor: str
    topic: str


class TopicUnsubscription(BaseEvent):
    """Emitted when an actor unsubscribes from a topic."""

    actor: str
    topic: str


# ------------------------------------------------------------------
# Harness lifecycle events
# ------------------------------------------------------------------


class CompactionCompleted(BaseEvent):
    """Emitted on the actor's stream when compaction finishes."""

    actor: str
    strategy: str
    events_before: int
    events_after: int
    llm_calls: int = 0
    usage: dict = Field(default_factory=dict)


class AggregationCompleted(BaseEvent):
    """Emitted on the actor's stream when aggregation finishes."""

    actor: str
    strategy: str
    event_count: int
    llm_calls: int = 0
    usage: dict = Field(default_factory=dict)


# ------------------------------------------------------------------
# Deserialization fallback
# ------------------------------------------------------------------


class UnknownEvent(BaseEvent):
    """Placeholder for events whose type cannot be resolved during deserialization.

    Preserves the raw data so nothing is lost.
    """

    type_name: str
    data: dict = Field(default_factory=dict)
