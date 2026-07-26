# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


class NlipError(Exception):
    """Base class for NLIP integration errors."""


class NlipConnectionError(NlipError):
    """Raised when the HTTP connection to the remote NLIP server fails."""


class NlipTimeoutError(NlipError):
    """Raised when a request to a remote NLIP server times out."""


class NlipServerError(NlipError):
    """Raised when a remote NLIP server returns a non-2xx HTTP response."""

    def __init__(self, *, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"NLIP server returned HTTP {status_code}: {body}")


class RehydratedNlipToolError(Exception):
    """Placeholder error type for a ``ToolErrorEvent`` rebuilt from the wire.

    The original exception type is lost in transit — only the rendered
    string is carried. Subclassing ``Exception`` keeps ``ToolErrorEvent``'s
    invariants intact (``str(ev.error)`` round-trips) without pretending we
    have the real type.
    """


class NlipInputRequiredError(NlipError):
    """Raised when a remote NLIP server asks for human input.

    NLIP is stateless and has no HITL hook on the wire; the caller is
    expected to catch this, obtain the answer, and resend a fresh
    request with the user's reply appended to the conversation history.
    """

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__(f"NLIP server requires input: {prompt}")
