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
    delegation_id: str = ""


class DelegationResult(BaseEvent):
    """Emitted when a delegation completes successfully."""

    source: str
    target: str
    result: str
    delegation_id: str = ""


class DelegationRejected(BaseEvent):
    """Emitted when a delegation is rejected by the Hub topology."""

    source: str
    target: str
    task: str
    reason: str
    delegation_id: str = ""


class DelegationError(BaseEvent):
    """Emitted when a delegation fails with an exception."""

    source: str
    target: str
    task: str
    error: str
    delegation_id: str = ""


class DelegationStarted(BaseEvent):
    """Emitted when a background (non-blocking) delegation is spawned.

    Unlike ``DelegationRequest`` (which fires for every delegation, blocking
    or background), this event fires only when ``Hub.request_background`` is
    called. Subscribers can use it to correlate progress/completion events
    with a specific background job.
    """

    source: str
    target: str
    task: str
    delegation_id: str


class DelegationProgress(BaseEvent):
    """Incremental progress from a long-running delegation.

    Reported by the delegatee (directly or via an adapter) while it is still
    executing. ``progress`` is optional — set it when the total work is
    quantifiable (0.0–1.0); otherwise use ``message`` for a human-readable
    status update.
    """

    source: str
    target: str
    delegation_id: str
    progress: float | None = None
    message: str = ""


class DelegationCancelled(BaseEvent):
    """Emitted when a background delegation is cancelled."""

    source: str
    target: str
    delegation_id: str
    reason: str = ""


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
