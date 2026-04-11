# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""DesktopProxyAgent — proxy for an agent hosted on a user's desktop.

The companion to ``DesktopChannel``. A DesktopProxyAgent looks like a
regular Agent to the cloud Hub but forwards ``ask()`` calls over the
desktop channel and waits for the matching reply.

Usage::

    channel = DesktopChannel(send_callback=ws_sender)
    proxy = DesktopProxyAgent(
        name="claude-code",
        channel=channel,
        capabilities=["coding", "filesystem", "terminal"],
        description="Coding agent on user's desktop",
    )
    await hub.register(proxy, capabilities=proxy.capabilities)

    # Pilot delegates via the standard network tool
    await hub.ask(pilot, 'network(action="request", target="claude-code", message="...")')

The proxy doesn't know or care which adapter (CLI, API, MCP) actually runs
on the desktop side — that's the AgentManager's job. From the Hub's
perspective, this looks identical to any other registered agent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .events import DelegationRequest
from .primitives.envelope import Envelope

if TYPE_CHECKING:
    from .channels.desktop import DesktopChannel

logger = logging.getLogger(__name__)


class DesktopAgentReply:
    """Reply from a desktop-hosted agent.

    Provides the same ``.content`` / ``.body`` interface as ``AgentReply``
    so callers (the network tool, downstream code) can treat it uniformly.
    Multi-turn chaining via ``.ask()`` makes a fresh call through the same
    proxy.
    """

    def __init__(
        self,
        content: str | None,
        *,
        proxy: DesktopProxyAgent,
        error: str | None = None,
    ) -> None:
        self._content = content
        self._proxy = proxy
        self._error = error

    @property
    def content(self) -> str | None:
        return self._content

    @property
    def body(self) -> str | None:
        return self._content

    @property
    def error(self) -> str | None:
        return self._error

    async def ask(self, msg: str, **kwargs: Any) -> DesktopAgentReply:
        """Continue the conversation with another delegation."""
        return await self._proxy.ask(msg, **kwargs)


class DesktopProxyAgent:
    """Cloud-side proxy for a desktop-hosted agent.

    Implements the minimal Agent interface (``name`` + ``ask()``) so it can
    be registered on the Hub. When ``ask()`` is called, it builds a
    DelegationRequest envelope and sends it through the DesktopChannel,
    awaiting the matching DelegationResult.

    Multi-turn conversations are supported by passing a ``session_id`` in
    the metadata — the desktop adapter uses it to resume an existing
    session (e.g., Claude Code session continuation). The default empty
    session_id starts a fresh session per ask.
    """

    def __init__(
        self,
        name: str,
        channel: DesktopChannel,
        *,
        capabilities: list[str] | None = None,
        description: str = "",
        timeout: float = 600.0,
        cloud_sender: str = "cloud",
    ) -> None:
        self.name = name
        self._channel = channel
        self.capabilities = list(capabilities or [])
        self.description = description
        self._timeout = timeout
        self._cloud_sender = cloud_sender

    async def ask(
        self,
        msg: str,
        **kwargs: Any,  # noqa: ARG002
    ) -> DesktopAgentReply:
        """Send a delegation to the desktop and wait for the result.

        Accepts the same kwargs as ``Agent.ask()`` for interface compatibility,
        but only the message text and metadata are forwarded — the desktop
        adapter manages its own model, tools, and middleware.
        """
        # Read source and metadata from delegation context vars (set by Hub._delegate)
        source = self._cloud_sender
        metadata: dict[str, Any] = {}
        try:
            from .hub import _delegation_metadata, _delegation_source  # lazy import

            source = _delegation_source.get(self._cloud_sender) or self._cloud_sender
            ctx_meta = _delegation_metadata.get({})
            if ctx_meta:
                metadata = dict(ctx_meta)
        except (ImportError, LookupError):
            pass

        request_event = DelegationRequest(
            source=source,
            target=self.name,
            task=msg,
        )
        envelope = Envelope(
            event=request_event,
            sender=source,
            recipient=self.name,
            metadata=metadata,
            ttl=self._timeout,
        )

        try:
            reply_envelope = await self._channel.request(
                envelope, timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            err = f"Timed out waiting for {self.name} after {self._timeout}s"
            logger.warning("DesktopProxy timeout: %s", err)
            return DesktopAgentReply(content=None, proxy=self, error=err)
        except Exception as e:
            err = f"Channel error talking to {self.name}: {e}"
            logger.warning("DesktopProxy error: %s", err)
            return DesktopAgentReply(content=None, proxy=self, error=err)

        # Extract result from the reply envelope. Desktop replies use
        # DelegationResult (success) or DelegationError (failure).
        return self._reply_from_envelope(reply_envelope)

    def _reply_from_envelope(self, env: Envelope) -> DesktopAgentReply:
        from .events import DelegationError, DelegationResult

        event = env.event
        if isinstance(event, DelegationResult):
            return DesktopAgentReply(content=event.result, proxy=self)
        if isinstance(event, DelegationError):
            return DesktopAgentReply(
                content=None, proxy=self, error=event.error,
            )

        # Unknown event type — best-effort: try to coerce
        content = getattr(event, "result", None) or getattr(event, "content", None)
        if content:
            return DesktopAgentReply(content=str(content), proxy=self)

        return DesktopAgentReply(
            content=None,
            proxy=self,
            error=f"Unexpected reply event type: {type(event).__name__}",
        )

    def __repr__(self) -> str:
        return f"DesktopProxyAgent(name={self.name!r}, capabilities={self.capabilities})"
