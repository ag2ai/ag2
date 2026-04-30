# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import Task

from autogen.beta.exceptions import AG2Error

from .mappers import text_from_message


class A2AError(AG2Error):
    """Base exception for A2A integration errors."""


class A2AClientToolsNotSupportedError(A2AError):
    """Raised when tools or response_schema are passed to A2AClient.__call__.

    A2A operates in Model A: tools and response_schema must be defined on the
    server-side Agent. Passing them on the client side is a configuration error
    (forwarding client-side tools is Model B, not implemented).
    """

    def __init__(self) -> None:
        super().__init__(
            "A2AConfig does not support tools or response_schema on the client side. "
            "Define them on the server-side Agent (Model A). "
            "If you intended to give tools to the orchestrator, attach them to the "
            "main agent (the one calling `remote.as_tool()`), not to the remote handle."
        )


class A2AReconnectError(A2AError):
    """Raised when reconnect attempts (streaming `resubscribe` or polling
    `get_task`) exceed `max_reconnects`."""

    def __init__(self, attempts: int, last_error: BaseException | None = None) -> None:
        self.attempts = attempts
        self.last_error = last_error
        msg = f"A2A reconnect failed after {attempts} attempts"
        if last_error is not None:
            msg += f": {last_error!r}"
        super().__init__(msg)


class A2ANoTaskError(A2AError):
    """Raised when the server completed `send_message` without producing a
    Task — there is nothing to drive to a terminal state."""

    def __init__(self) -> None:
        super().__init__(
            "A2A server returned a Message without an associated Task. A2AClient requires a Task-producing flow."
        )


class A2AAuthRequiredError(A2AError):
    """Raised when the server signals `TaskState.auth_required`.

    The remote agent demands client-side authentication before it can proceed
    (analogous to `input_required`, but for credentials rather than text).
    AG2 beta does not implement an auth-challenge flow — wire credentials
    through `A2AConfig.client_factory` (a custom `httpx.AsyncClient`) instead.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        super().__init__(
            f"Task {task.id} requires authentication to proceed. Configure credentials via A2AConfig.client_factory."
        )


class A2ATaskTerminalError(A2AError):
    """Base for terminal task error states (`failed` / `rejected`).

    The remote agent ended the task in an error state. The original `Task`
    object is preserved on `self.task` so callers can inspect status,
    artifacts, and history.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        msg = task.status.message and text_from_message(task.status.message)
        super().__init__(f"Task {task.id} ended in state {task.status.state.value}: {msg or '<no message>'}")


class A2ATaskFailedError(A2ATaskTerminalError):
    """Task ended in `failed` state — the remote agent encountered an error."""


class A2ATaskRejectedError(A2ATaskTerminalError):
    """Task ended in `rejected` state — the remote agent refused the request."""
