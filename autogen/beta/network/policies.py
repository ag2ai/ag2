# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Qualified-key constants for ``ConversationContext.dependencies`` injection.

Network plugin tools resolve their bindings via these keys — mirroring
how framework-core's ``TaskInject`` resolves to ``ag2.task``.
Centralising the keys here prevents drift between the side that stamps
into ``context.dependencies`` and the side that injects them.

The ``ag2.network.*`` namespace is reserved for network-only injects;
``ag2.task`` is re-exported here so consumers see one canonical key list.
"""

__all__ = (
    "AGENT_CLIENT_DEP",
    "CHANNEL_DEP",
    "CHANNEL_STATE_DEP",
    "HUB_DEP",
    "TASK_DEP",
)


CHANNEL_DEP = "ag2.network.channel"
CHANNEL_STATE_DEP = "ag2.network.channel_state"
AGENT_CLIENT_DEP = "ag2.network.agent_client"
HUB_DEP = "ag2.network.hub"
TASK_DEP = "ag2.task"  # framework-core key; re-exported for symmetry
