# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network event types: delegation, scheduling, and topics.

Actor lifecycle events (observer, task, compaction, aggregation) have been
promoted to ``autogen.beta.events.lifecycle`` since they are framework-core
concerns. They are re-exported here for backward compatibility.
"""

from autogen.beta.events.base import BaseEvent, Field

# Re-export promoted lifecycle events for backward compatibility
from autogen.beta.events.lifecycle import (  # noqa: F401
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
    TaskProgress,
    TaskRequest,
    TaskResult,
    UnknownEvent,
)

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
