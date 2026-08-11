# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tenant-side clients — ``NetworkClient`` Protocol, ``HubClient``, ``AgentClient``.

The trust boundary runs through this package: tenant code (notify
handlers, future transforms, LLM tool execution) only runs inside the
tenant process. The hub never imports anything from here.
"""

from .agent_client import AgentClient
from .channel import Channel
from .checkpoint import HubBackedCheckpointStore
from .handlers import (
    default_handler,
    read_wal_until,
    resolve_view_policy,
    stamp_dependencies,
)
from .hub_client import HubClient
from .human_client import HumanClient
from .inject import AgentClientInject, ChannelInject, ChannelStateInject, HubInject, TaskInject
from .network_client import NetworkClient
from .plugin import NetworkContextPolicy, NetworkPlugin
from .skill_render import ParsedSkill, parse_skill_frontmatter, render_fallback_skill
from .task import ClientTask

__all__ = (
    "AgentClient",
    "AgentClientInject",
    "Channel",
    "ChannelInject",
    "ChannelStateInject",
    "ClientTask",
    "HubBackedCheckpointStore",
    "HubClient",
    "HubInject",
    "HumanClient",
    "NetworkClient",
    "NetworkContextPolicy",
    "NetworkPlugin",
    "ParsedSkill",
    "TaskInject",
    "default_handler",
    "parse_skill_frontmatter",
    "read_wal_until",
    "render_fallback_skill",
    "resolve_view_policy",
    "stamp_dependencies",
)
