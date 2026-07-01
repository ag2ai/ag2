# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import Task, TaskState
from a2a.utils.constants import PROTOCOL_VERSION_1_0


class A2AError(Exception):
    """Base class for A2A integration errors."""


class A2AClientToolsNotSupportedError(A2AError):
    """Raised when the user passes ``tools=`` to an Agent backed by an A2A
    server that does not advertise support for the
    ``urn:ag2:client-tools:v1`` extension in its AgentCard.
    """


class A2AInvalidCardError(A2AError):
    """Raised when an ``AgentCard`` is missing data required to connect."""


class A2AIncompatibleProtocolVersionError(A2AError):
    """Raised when the interface AG2 selected on an ``AgentCard`` advertises an
    A2A protocol version older than 1.0.

    The 1.0 protocol is not backwards-compatible with the earlier 0.x drafts
    (see https://a2a-protocol.org/latest/announcing-1.0/), so AG2 refuses to
    connect rather than silently negotiating a session that fails later with
    opaque RPC-level errors. A missing or unparsable ``protocol_version`` is
    treated as compatible — it is an optional field and the A2A SDK itself
    defaults it to the current version.
    """

    def __init__(self, *, url: str, transport: str, protocol_version: str) -> None:
        self.url = url
        self.transport = transport
        self.protocol_version = protocol_version
        super().__init__(
            f"AgentCard at {url!r} declares an incompatible A2A protocol version "
            f"{protocol_version!r} for transport {transport!r}; AG2 requires "
            f">= {PROTOCOL_VERSION_1_0} (see https://a2a-protocol.org/latest/announcing-1.0/)."
        )


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


class A2ATaskAuthRequiredError(A2ATaskTerminalError):
    """Task ended in TASK_STATE_AUTH_REQUIRED.

    Per A2A spec §7.6 the agent expects credentials to arrive out-of-band.
    AG2 does not currently wire an auth hook, so the client surfaces this
    state as an error — the application is expected to obtain credentials,
    apply them (e.g. via interceptor headers or a refreshed config), and
    retry the request.
    """


class RehydratedToolError(Exception):
    """Placeholder error type for ``ToolErrorEvent`` rebuilt from the wire.

    The original exception type is lost in transit — we only carry the
    rendered string. Subclassing ``Exception`` keeps ``ToolErrorEvent``'s
    invariants intact (e.g. ``str(ev.error)`` round-trips) without pretending
    we have the real type. Used by both ``tool-result+json`` Part decoding
    and ``ag2.history+json`` event decoding.
    """
