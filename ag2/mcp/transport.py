# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from urllib.parse import urlparse

from mcp.server.streamable_http import EventStore
from mcp.server.transport_security import TransportSecuritySettings

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
            never reaps. Handshake-era only, so choosing a value other than the
            default alongside ``stateless=True`` is refused at construction —
            see :attr:`asks_to_reap`.
        max_mcp_sessions: Concurrent *MCP* sessions; one beyond the cap is
            answered ``503``. ``None`` removes the cap. Not
            :attr:`~ag2.mcp.SessionConfig.max_sessions`, which bounds
            *conversation* sessions.
        max_request_body_size: Bytes a request body may carry, refused ``413``
            before it is parsed and before a session is created.
        security_settings: DNS rebinding protection. Left ``None``, AG2 derives
            it from ``security.resource_url`` when an authorization requirement
            is configured — see :meth:`security_settings_for`. To serve without
            it, say so: ``TransportSecuritySettings(enable_dns_rebinding_protection=False)``.
            A *default-constructed* ``TransportSecuritySettings()`` is not that —
            it enables the protection with empty allow-lists and so refuses
            everything.
        sse_retry_interval: Milliseconds suggested to a client before it
            reconnects a dropped SSE stream. It rides the priming event, which
            the SDK mints only for a resumable stream, so it does nothing unless
            :attr:`event_store` is set *and* the client is on 2025-11-25 or
            later. A passthrough otherwise: AG2 holds no opinion about the number.
        event_store: Makes connections resumable — a client reconnecting with a
            ``Last-Event-ID`` receives what it missed. ``None`` leaves them
            non-resumable. A passthrough; bring your own. What is proven is one
            end-to-end test over a real socket (``test/mcp/test_resumability.py``).
    """

    mcp_session_idle_timeout: float | None = _DEFAULT_MCP_SESSION_IDLE_TIMEOUT
    max_mcp_sessions: int | None = _DEFAULT_MAX_MCP_SESSIONS
    max_request_body_size: int = _DEFAULT_MAX_REQUEST_BODY_SIZE
    security_settings: TransportSecuritySettings | None = None
    sse_retry_interval: int | None = None
    event_store: EventStore | None = None

    def security_settings_for(self, resource_url: "str | None") -> TransportSecuritySettings | None:
        """The DNS rebinding protection to serve with, given the endpoint's public URL.

        An explicit :attr:`security_settings` always wins, including an explicit
        refusal. Otherwise AG2 derives one from the URL the operator already
        named in ``security.resource_url``: that host is the one thing about the
        deployment's topology AG2 can know without guessing, and a served agent
        answering on it is exactly what the operator declared. Any port is
        allowed when the URL names none, so a proxy that forwards ``Host`` with
        one still reaches the server.

        With no authorization requirement there is no such URL and protection
        stays off, which is what an absent value has always meant to the SDK.
        """
        if self.security_settings is not None:
            return self.security_settings
        if resource_url is None:
            return None
        parsed = urlparse(resource_url)
        if not parsed.netloc:
            return None
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.port is not None:
            return TransportSecuritySettings(allowed_hosts=[parsed.netloc], allowed_origins=[origin])
        return TransportSecuritySettings(
            allowed_hosts=[parsed.netloc, f"{parsed.netloc}:*"],
            allowed_origins=[origin, f"{origin}:*"],
        )

    @property
    def asks_to_reap(self) -> bool:
        """Whether this config asks the transport to reap idle MCP sessions.

        Neither off value asks: ``None`` asks it *not* to reap, and the default
        number is what a caller who expressed nothing would get either way.
        """
        return self.mcp_session_idle_timeout not in (None, _DEFAULT_MCP_SESSION_IDLE_TIMEOUT)

    def __post_init__(self) -> None:
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
