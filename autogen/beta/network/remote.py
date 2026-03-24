# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""RemoteAgent — proxy for agents hosted on remote Hubs.

A RemoteAgent looks like a regular Agent to the local Hub but forwards
ask() calls over HTTP to a remote Hub's delegation endpoint.

Usage::

    # Register a remote agent manually
    remote = RemoteAgent("researcher", "http://server-a:8900")
    await hub.register(remote, capabilities=["research"])

    # Or auto-discover via Hub.connect()
    await hub.connect("http://server-a:8900")
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)


class RemoteAgentReply:
    """Reply from a remote agent delegation.

    Provides the same ``.content`` interface as ``AgentReply``.
    Multi-turn chaining via ``.ask()`` makes a fresh remote call.
    """

    def __init__(
        self,
        content: str | None,
        *,
        remote_agent: RemoteAgent,
    ) -> None:
        self._content = content
        self._remote_agent = remote_agent

    @property
    def content(self) -> str | None:
        """Text content of the remote agent's response."""
        return self._content

    async def ask(self, msg: str, **kwargs: Any) -> RemoteAgentReply:
        """Continue conversation with a fresh remote call."""
        return await self._remote_agent.ask(msg, **kwargs)


class RemoteAgent:
    """Proxy for an agent hosted on a remote Hub.

    Implements the minimal Agent interface (``name`` + ``ask()``) so it can be
    registered with a local Hub. When ``ask()`` is called, it POSTs the task
    to the remote Hub's ``/delegate`` endpoint via HTTP.

    The remote Hub runs the real agent with its full capabilities (observers,
    signals, network tools) and returns the result.

    Example::

        remote = RemoteAgent("researcher", "http://server-a:8900")
        await local_hub.register(remote, capabilities=["research"])

        # When local_hub delegates to "researcher", it goes over HTTP
        reply = await local_hub.ask(writer, "Research and write about AI")
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        *,
        capabilities: list[str] | None = None,
        description: str = "",
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.name = name
        self._endpoint = endpoint.rstrip("/")
        self.capabilities = list(capabilities or [])
        self.description = description
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._session: ClientSession | None = None

    async def ask(self, msg: str, **kwargs: Any) -> RemoteAgentReply:  # noqa: ARG002
        """Send task to remote Hub and return the result.

        Accepts the same keyword arguments as ``Agent.ask()`` for interface
        compatibility, but only the message text is forwarded — the remote
        Hub configures its own tools and middleware.
        """
        # Read source from context variable if available (set by Hub._delegate)
        source = ""
        try:
            from .hub import _delegation_source  # lazy import to avoid circular dep

            source = _delegation_source.get("")
        except (ImportError, LookupError):
            pass

        payload: dict[str, Any] = {
            "agent": self.name,
            "task": msg,
            "source": source,
        }

        result = await self._post("/delegate", payload)

        if result is None:
            return RemoteAgentReply(
                content=f"Error: failed to reach remote agent '{self.name}' at {self._endpoint}",
                remote_agent=self,
            )

        if result.get("status") == "error":
            return RemoteAgentReply(
                content=f"Error: {result.get('reason', 'unknown error')}",
                remote_agent=self,
            )

        return RemoteAgentReply(
            content=result.get("result"),
            remote_agent=self,
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """POST to remote endpoint with retry."""
        if self._session is None:
            self._session = ClientSession(
                timeout=ClientTimeout(total=self._timeout)
            )

        url = f"{self._endpoint}{path}"
        data = json.dumps(payload)
        last_error: Exception | None = None
        assert self._session is not None  # guaranteed by check above
        session = self._session

        for attempt in range(self._max_retries + 1):
            try:
                async with session.post(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body = await resp.json()
                    if resp.status == 200:
                        return body
                    last_error = RuntimeError(
                        f"HTTP {resp.status}: {body.get('reason', 'unknown')}"
                    )
            except Exception as e:
                last_error = e

            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay * (attempt + 1))

        logger.error(
            "Failed to reach remote agent '%s' at %s after %d attempts: %s",
            self.name,
            self._endpoint,
            self._max_retries + 1,
            last_error,
        )
        return None

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def endpoint(self) -> str:
        """The remote Hub's base URL."""
        return self._endpoint

    def __repr__(self) -> str:
        return f"RemoteAgent(name={self.name!r}, endpoint={self._endpoint!r})"
