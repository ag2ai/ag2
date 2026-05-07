# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import Task, TaskState


class A2AError(Exception):
    """Base class for A2A integration errors."""


class A2AClientToolsNotSupportedError(A2AError):
    """Raised when the user passes ``tools=`` to an Agent backed by an A2A
    server that does not advertise support for the
    ``urn:ag2:client-tools:v1`` extension in its AgentCard.
    """


class A2AReconnectError(A2AError):
    """Raised when reconnect attempts on a streaming Task are exhausted."""

    def __init__(self, attempts: int) -> None:
        super().__init__(f"Exhausted {attempts} reconnect attempts on streaming task")
        self.attempts = attempts


class A2ATaskTerminalError(A2AError):
    """Base class for terminal Task error states (failed / rejected)."""

    def __init__(self, task: Task) -> None:
        self.task = task
        state_name = TaskState.Name(task.status.state) if task.status else "<no status>"
        super().__init__(f"Task {task.id} ended in state {state_name}")


class A2ATaskFailedError(A2ATaskTerminalError):
    """Task ended in TASK_STATE_FAILED."""


class A2ATaskRejectedError(A2ATaskTerminalError):
    """Task ended in TASK_STATE_REJECTED."""
