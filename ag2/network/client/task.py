# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ClientTask`` — client-side handle for an observed remote task.

Wraps a ``TaskMetadata`` snapshot from the hub and provides refresh
helpers without exposing the full hub surface.

The framework-core ``ag2.task.Task`` is the *owner-side*
lifecycle (the one constructed inside ``async with agent.task(...)``).
This handle is *observer-side* — held by an agent that delegated work
to a peer.
"""

from typing import TYPE_CHECKING

from ag2.task import TaskMetadata, TaskState

if TYPE_CHECKING:
    from .agent_client import AgentClient

__all__ = ("ClientTask",)


class ClientTask:
    """Per-observer handle for a remote task this agent is tracking."""

    def __init__(
        self,
        *,
        metadata: TaskMetadata,
        client: "AgentClient",
    ) -> None:
        # __init__ stores params; no side effects.
        self._metadata = metadata
        self._client = client

    @property
    def task_id(self) -> str:
        return self._metadata.task_id

    @property
    def metadata(self) -> TaskMetadata:
        return self._metadata

    @property
    def state(self) -> TaskState:
        return self._metadata.state

    @property
    def owner_id(self) -> str:
        return self._metadata.owner_id

    async def info(self) -> TaskMetadata:
        """Re-fetch metadata from the hub."""
        refreshed = await self._client._hub_client.get_task(self.task_id)
        self._metadata = refreshed
        return refreshed
