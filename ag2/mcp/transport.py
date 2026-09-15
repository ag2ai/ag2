# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from mcp.server.streamable_http import EventStore
from mcp.server.transport_security import TransportSecuritySettings

# "Nobody supplied one", distinct from every value the field accepts — ``None``
# already means "never reap". Typed ``Any`` so it can stand in as the declared
# default; ``__post_init__`` replaces it before anyone reads the field.
_MISSING: Any = object()

_DEFAULT_MCP_SESSION_IDLE_TIMEOUT = 1800.0
_DEFAULT_MAX_MCP_SESSIONS = 10_000
_DEFAULT_MAX_REQUEST_BODY_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Streamable-HTTP transport settings for :class:`~ag2.mcp.MCPServer`.

    Carries AG2's own defaults rather than the resolved ``mcp`` release's: the
    server passes every one through explicitly. Only settings with no other
    route onto the server live here — ``stateless``, ``json_response``, ``path``
    and ``security`` stay flat parameters on :class:`~ag2.mcp.MCPServer`.

    The guide is ``website/docs/user-guide/tools/serving_mcp.mdx``.

    Attributes:
        mcp_session_idle_timeout: Seconds an *MCP* session may idle before the
            transport closes it; its next request is answered ``404``. ``None``
            never reaps. Handshake-era only, so supplying one alongside
            ``stateless=True`` is refused at construction — see
            :attr:`asks_to_reap`. Omitted, it reads back as the default.
        max_mcp_sessions: Concurrent *MCP* sessions; one beyond the cap is
            answered ``503``. ``None`` removes the cap. Not
            :attr:`~ag2.mcp.SessionConfig.max_sessions`, which bounds
            *conversation* sessions.
        max_request_body_size: Bytes a request body may carry, refused ``413``
            before it is parsed and before a session is created.
        security_settings: DNS rebinding protection. **``None`` disables it** —
            the SDK reads an absent value that way. A default-constructed
            ``TransportSecuritySettings`` enables it with empty allow-lists and
            so refuses everything; name your hosts and origins, ports included.
        sse_retry_interval: Milliseconds suggested to a client before it
            reconnects a dropped SSE stream. A passthrough AG2 has no opinion on
            and exercises no behaviour of.
        event_store: Makes connections resumable — a client reconnecting with a
            ``Last-Event-ID`` receives what it missed. ``None`` leaves them
            non-resumable. A passthrough; bring your own. What is proven is one
            end-to-end test over a real socket (``test/mcp/test_resumability.py``).
        asks_to_reap: Whether reaping was *asked for* rather than inherited.
            Provenance, not value: supplying the default number is still
            supplying one, and ``None`` asks not to reap at all. Derived at
            construction; do not pass it.
    """

    mcp_session_idle_timeout: float | None = _MISSING
    max_mcp_sessions: int | None = _DEFAULT_MAX_MCP_SESSIONS
    max_request_body_size: int = _DEFAULT_MAX_REQUEST_BODY_SIZE
    security_settings: TransportSecuritySettings | None = None
    sse_retry_interval: int | None = None
    event_store: EventStore | None = None
    asks_to_reap: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        supplied = self.mcp_session_idle_timeout is not _MISSING
        if not supplied:
            object.__setattr__(self, "mcp_session_idle_timeout", _DEFAULT_MCP_SESSION_IDLE_TIMEOUT)
        object.__setattr__(self, "asks_to_reap", supplied and self.mcp_session_idle_timeout is not None)

        if self.mcp_session_idle_timeout is not None and self.mcp_session_idle_timeout <= 0:
            raise ValueError(
                f"mcp_session_idle_timeout must be > 0 when set, got {self.mcp_session_idle_timeout} "
                "(pass None to never reap an idle MCP session)."
            )
        if self.max_mcp_sessions is not None and self.max_mcp_sessions < 1:
            raise ValueError(
                f"max_mcp_sessions must be >= 1 when set, got {self.max_mcp_sessions} "
                "(pass None to remove the cap on concurrent MCP sessions)."
            )
        if self.max_request_body_size < 1:
            raise ValueError(f"max_request_body_size must be >= 1 byte, got {self.max_request_body_size}.")
        if self.sse_retry_interval is not None and self.sse_retry_interval < 1:
            raise ValueError(f"sse_retry_interval must be >= 1 millisecond when set, got {self.sse_retry_interval}.")


__all__ = ("TransportConfig",)
