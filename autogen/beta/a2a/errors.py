# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import Task, TaskState

from autogen.beta.exceptions import AG2Error

from .mappers import text_from_message


class A2AError(AG2Error):
    """Base exception for A2A integration errors."""


class A2AResponseSchemaNotSupportedError(A2AError):
    """Raised when ``response_schema`` is passed to ``A2AClient.__call__``.

    Structured-output enforcement is the remote agent's job — the A2A protocol
    has no native ``response_format`` field. If you need structured output,
    declare it on the remote ``Agent`` at server-build time.
    """

    def __init__(self) -> None:
        super().__init__(
            "A2AConfig does not forward `response_schema`. Define structured-output "
            "expectations on the remote Agent instead."
        )


class A2AReconnectError(A2AError):
    """Raised when reconnect attempts (streaming ``resubscribe`` or polling
    ``get_task``) exceed ``max_reconnects``."""

    def __init__(self, attempts: int, last_error: BaseException | None = None) -> None:
        self.attempts = attempts
        self.last_error = last_error
        msg = f"A2A reconnect failed after {attempts} attempts"
        if last_error is not None:
            msg += f": {last_error!r}"
        super().__init__(msg)


class A2ANoTaskError(A2AError):
    """Raised when the server completed ``send_message`` without producing a
    Task — there is nothing to drive to a terminal state."""

    def __init__(self) -> None:
        super().__init__(
            "A2A server returned a Message without an associated Task. A2AClient requires a Task-producing flow."
        )


class A2AAuthRequiredError(A2AError):
    """Raised when the server signals ``TaskState.auth_required``.

    The remote agent demands client-side authentication. AG2 beta has no
    auth-challenge flow — wire credentials through ``A2AConfig.client_factory``
    (a custom ``httpx.AsyncClient`` configured with httpx-auth or Authlib).
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        super().__init__(
            f"Task {task.id} requires authentication to proceed. Configure credentials via A2AConfig.client_factory."
        )


class A2ATaskTerminalError(A2AError):
    """Base for terminal task error states (``failed`` / ``rejected``)."""

    def __init__(self, task: Task) -> None:
        self.task = task
        text = text_from_message(task.status.message) if task.status.HasField("message") else None
        super().__init__(f"Task {task.id} ended in state {TaskState.Name(task.status.state)}: {text or '<no message>'}")


class A2ATaskFailedError(A2ATaskTerminalError):
    """Task ended in ``failed`` state — the remote agent encountered an error."""


class A2ATaskRejectedError(A2ATaskTerminalError):
    """Task ended in ``rejected`` state — the remote agent refused the request."""
