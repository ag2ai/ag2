# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import AG2Error


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
    """Raised when streaming/polling reconnect attempts exceed max_reconnects."""

    def __init__(self, attempts: int, last_error: BaseException | None = None) -> None:
        self.attempts = attempts
        self.last_error = last_error
        msg = f"A2A streaming reconnect failed after {attempts} attempts"
        if last_error is not None:
            msg += f": {last_error!r}"
        super().__init__(msg)


class InputRequiredError(A2AError):
    """Internal signal raised by the A2A replay HITL hook to suspend a task.

    The executor catches this in `execute()` and translates it into an A2A
    `requires_input(...)` event so the client can supply the answer. On the
    follow-up call, the executor replays the agent with the user's answer
    pre-loaded in the replay queue.
    """

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__(f"Human input required: {prompt!r}")
