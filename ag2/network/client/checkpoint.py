# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub-backed :class:`CheckpointStore` impl.

``HubBackedCheckpointStore`` is the canonical default for networked
agents: it delegates :meth:`write` / :meth:`read` to
``checkpoint_task`` / ``read_task_checkpoint``, which persist a single
JSON blob per ``task_id`` under the hub's ``KnowledgeStore``. It
accepts either an in-process :class:`Hub` or a :class:`HubClient` —
both expose those two methods, so the checkpoint path is direct
in-process and an RPC round-trip cross-process without the owner code
changing. Standalone agents that don't need cross-process durability
can use any other :class:`CheckpointStore` impl — or omit
checkpointing entirely.
"""

from typing import TYPE_CHECKING, Any

from ag2.task import CheckpointStore

if TYPE_CHECKING:
    from ..hub import Hub
    from .hub_client import HubClient

__all__ = ("HubBackedCheckpointStore",)


class HubBackedCheckpointStore:
    """Persists task checkpoints through a :class:`Hub` or :class:`HubClient`.

    Satisfies the framework-core :class:`CheckpointStore` Protocol by
    delegating to ``checkpoint_task`` / ``read_task_checkpoint`` on the
    supplied backend. Pass a :class:`Hub` for in-process durability or a
    :class:`HubClient` to route checkpoints to a remote hub over the
    wire. Deployments that need different durability supply another
    :class:`CheckpointStore` — the Protocol is the seam.
    """

    def __init__(self, hub: "Hub | HubClient") -> None:
        # __init__ stores params; no side effects.
        self._hub = hub

    async def write(self, task_id: str, state: dict[str, Any]) -> None:
        await self._hub.checkpoint_task(task_id, dict(state))

    async def read(self, task_id: str) -> dict[str, Any] | None:
        return await self._hub.read_task_checkpoint(task_id)


if TYPE_CHECKING:
    _check: CheckpointStore = HubBackedCheckpointStore.__new__(HubBackedCheckpointStore)
