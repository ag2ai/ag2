# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import traceback
from typing import TYPE_CHECKING, Any

from .base import BaseEvent, Field
from .types import Usage

if TYPE_CHECKING:
    from autogen.beta.context import StreamId
    from autogen.beta.task import TaskSpec


class TaskEvent(BaseEvent):
    task_id: str
    agent_name: str
    objective: str


class TaskStarted(TaskEvent):
    # Optional ``TaskSpec`` describing what the task is doing. Set by the
    # framework-core ``Task`` primitive (``autogen.beta.task``); legacy
    # ``run_task`` callers leave it ``None``.
    spec: "TaskSpec | None" = Field(None)
    # Absolute ISO deadline (start + ttl_seconds) if the task was created
    # with a TTL, else ``None``. Carried on the event so the network's
    # ``TaskMirror`` can hand it to the hub, whose TTL sweeper expires
    # tasks past ``expires_at``. ``TaskSpec`` does not hold the TTL.
    expires_at: str | None = Field(None)


class TaskProgress(TaskEvent):
    """Progress update for a running task.

    Two distinct uses on one event:

    * ``content`` carries streamed sub-agent output (``run_subtask`` /
      ``subagent_tool``). Marked transient — superseded by the final
      ``TaskCompleted``.
    * ``payload`` carries structured checkpoint data emitted by the task
      owner via ``Task.progress({...})``. Persistent within the parent
      stream until the task terminates.
    """

    __transient__ = True

    content: str = Field("")
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskCompleted(TaskEvent):
    # Widened from ``str | None`` to ``Any`` so framework-core ``Task``
    # owners can return structured results. ``run_task`` still passes a
    # string, so existing callers are unaffected.
    result: Any = Field(None)
    task_stream: "StreamId"  # Stream reference for inspection
    usage: Usage = Field(default_factory=Usage)


class TaskFailed(TaskEvent):
    error: Exception

    _content: str = Field(
        default_factory=str,
        init=False,
        compare=False,
    )

    @property
    def content(self) -> str:
        if not self._content:
            self._content = "".join(
                traceback.format_exception(
                    type(self.error),
                    self.error,
                    self.error.__traceback__,
                )
            )
        return self._content


class TaskExpired(TaskEvent):
    """Terminal event — task TTL elapsed without a terminal event.

    Emitted by whichever observer holds the TTL clock. Standalone agents
    receive no expiry unless app code arranges it; networked agents
    receive it via the hub's TTL sweeper, mirrored back through
    ``AgentClient``.
    """


class TaskCancelled(TaskEvent):
    """Terminal event — owner explicitly cancelled the task.

    Distinct from ``TaskFailed`` (work could not complete) and
    ``TaskExpired`` (TTL elapsed). Carries an optional ``reason`` for
    operational visibility — surfaced through the network mirror as
    ``ag2.task.cancelled`` so peers subscribed to the task observe
    the cancellation alongside the other terminal events.
    """

    reason: str = Field("")
