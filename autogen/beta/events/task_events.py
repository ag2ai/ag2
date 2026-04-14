# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import traceback
from typing import TYPE_CHECKING

from .base import BaseEvent, Field

if TYPE_CHECKING:
    from autogen.beta.context import StreamId


class TaskEvent(BaseEvent):
    task_id: str
    agent_name: str
    objective: str


class TaskStarted(TaskEvent):
    pass


class TaskProgress(TaskEvent):
    """Streamed progress update from a running task sub-agent.

    Transient: superseded by the final ``TaskCompleted``.
    """

    __transient__ = True

    content: str


class TaskCompleted(TaskEvent):
    result: str | None
    task_stream: "StreamId"  # Stream reference for inspection
    usage: dict = Field(default_factory=dict)


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
