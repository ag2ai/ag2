# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import httpx
from a2a.client import A2AClientHTTPError
from a2a.types import Task, TaskState

HttpxClientFactory: TypeAlias = Callable[[], httpx.AsyncClient]


TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset({
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
})


TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    A2AClientHTTPError,
)


@dataclass(slots=True)
class StreamOutcome:
    """Result of consuming one A2A streaming or polling session."""

    text: str = ""
    task: Task | None = None
    input_required: bool = False
    input_prompt: str | None = None
