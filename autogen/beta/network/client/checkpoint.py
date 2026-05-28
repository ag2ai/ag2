# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub-backed :class:`CheckpointStore` impl.

``HubBackedCheckpointStore`` is the canonical default for networked
agents: it delegates :meth:`write` / :meth:`read` to
:meth:`Hub.checkpoint_task` / :meth:`Hub.read_task_checkpoint`, which
persist a single JSON blob per ``task_id`` under the hub's
``KnowledgeStore``. Standalone agents that don't need cross-process
durability can use any other :class:`CheckpointStore` impl — or omit
checkpointing entirely.
"""

from typing import TYPE_CHECKING, Any

from autogen.beta.task import CheckpointStore

if TYPE_CHECKING:
    from ..hub import Hub

__all__ = ("HubBackedCheckpointStore",)


class HubBackedCheckpointStore:
    """Persists task checkpoints through a :class:`Hub` instance.

    Satisfies the framework-core :class:`CheckpointStore` Protocol by
    delegating to :meth:`Hub.checkpoint_task` and
    :meth:`Hub.read_task_checkpoint`. Constructing with a hub gives an
    in-process flow direct access; a wire-mode transport would supply
    an equivalent RPC-backed store (out of scope here — tenants build
    that on top of the existing Protocol).
    """

    def __init__(self, hub: "Hub") -> None:
        # __init__ stores params; no side effects.
        self._hub = hub

    async def write(self, task_id: str, state: dict[str, Any]) -> None:
        await self._hub.checkpoint_task(task_id, dict(state))

    async def read(self, task_id: str) -> dict[str, Any] | None:
        return await self._hub.read_task_checkpoint(task_id)


# Structural conformance check: confirm at import time that
# ``HubBackedCheckpointStore`` satisfies the Protocol. The check is
# free (Protocol membership) and surfaces any drift between the
# Protocol surface and this canonical impl immediately.
_protocol_check: CheckpointStore = HubBackedCheckpointStore.__new__(HubBackedCheckpointStore)
del _protocol_check
