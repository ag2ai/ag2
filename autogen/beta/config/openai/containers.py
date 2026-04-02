# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, not_given


@dataclass(slots=True)
class ExpiresAfter:
    """Container expiry policy.

    Args:
        minutes: Number of inactivity minutes before the container expires.
        anchor: Reference point for the timer. Currently only ``"last_active_at"`` is supported.
    """

    minutes: int
    anchor: Literal["last_active_at"] = "last_active_at"


@dataclass(slots=True)
class ContainerInfo:
    """Metadata returned after creating a container."""

    id: str
    name: str | None = None
    status: str | None = None


class ContainerManager:
    """Manages OpenAI hosted shell containers.

    Use this to create reusable containers and reference them in :class:`ShellTool`
    via :class:`~autogen.beta.tools.builtin.shell.ContainerReferenceEnvironment`.

    Example::

        manager = ContainerManager(api_key="sk-...")
        container = await manager.create(
            name="my-env",
            expires_after=ExpiresAfter(minutes=20),
        )

        agent = Agent(
            ...,
            tools=[ShellTool(environment=ContainerReferenceEnvironment(container_id=container.id))],
        )

    See: https://developers.openai.com/api/docs/guides/tools-shell#reuse-a-container-across-requests
    """

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        timeout: Any = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )

    async def create(
        self,
        *,
        name: str | None = None,
        memory_limit: str | None = None,
        expires_after: ExpiresAfter | None = None,
    ) -> ContainerInfo:
        """Create a new hosted container and return its metadata.

        Args:
            name: Optional human-readable name for the container.
            memory_limit: Memory cap, e.g. ``"1g"`` or ``"512m"``.
            expires_after: Expiry policy. Defaults to OpenAI's platform default
                (20 minutes of inactivity) when omitted.

        Returns:
            :class:`ContainerInfo` with the ``id`` to pass to
            :class:`~autogen.beta.tools.builtin.shell.ContainerReferenceEnvironment`.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if memory_limit is not None:
            body["memory_limit"] = memory_limit
        if expires_after is not None:
            body["expires_after"] = {
                "anchor": expires_after.anchor,
                "minutes": expires_after.minutes,
            }

        container = await self._client.containers.create(**body)
        return ContainerInfo(
            id=container.id,
            name=getattr(container, "name", None),
            status=getattr(container, "status", None),
        )

    async def delete(self, container_id: str) -> None:
        """Delete a container by ID.

        Args:
            container_id: The ``id`` returned by :meth:`create`.
        """
        await self._client.containers.delete(container_id)
