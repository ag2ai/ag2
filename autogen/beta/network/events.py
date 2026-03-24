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
