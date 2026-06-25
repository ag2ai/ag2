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

import json
from typing import TYPE_CHECKING, Any

from autogen.beta._telemetry_consts import (
    ATTR_CHECKPOINT_BYTES,
    ATTR_CHECKPOINT_HIT,
    ATTR_CHECKPOINT_TASK_ID,
)
from autogen.beta.task import CheckpointStore

try:
    # Soft dependency: checkpointing works without the tracing extras.
    # When OpenTelemetry is installed, save/restore are pinned as
    # span-events on the active span; with no provider configured,
    # ``add_event`` is a no-op on a non-recording span.
    from opentelemetry import trace as _otel_trace
except ImportError:
    _otel_trace = None

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
        payload = dict(state)
        await self._hub.checkpoint_task(task_id, payload)
        if _otel_trace is not None:
            # Pin a marker on the active task/turn span — checkpoints bypass
            # the envelope path, so this is the only place they surface in a
            # trace. ``payload`` already serialised cleanly inside the hub
            # call above, so re-dumping for the byte count cannot raise.
            _otel_trace.get_current_span().add_event(
                "checkpoint.write",
                attributes={
                    ATTR_CHECKPOINT_TASK_ID: task_id,
                    ATTR_CHECKPOINT_BYTES: len(json.dumps(payload)),
                },
            )

    async def read(self, task_id: str) -> dict[str, Any] | None:
        result = await self._hub.read_task_checkpoint(task_id)
        if _otel_trace is not None:
            # On a resume read, ``task_id`` is the prior task being resumed
            # from — so this event on the new run's span is the link back.
            _otel_trace.get_current_span().add_event(
                "checkpoint.read",
                attributes={
                    ATTR_CHECKPOINT_TASK_ID: task_id,
                    ATTR_CHECKPOINT_HIT: result is not None,
                },
            )
        return result


if TYPE_CHECKING:
    _check: CheckpointStore = HubBackedCheckpointStore.__new__(HubBackedCheckpointStore)
