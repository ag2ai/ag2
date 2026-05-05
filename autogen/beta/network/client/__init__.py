# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tenant-side clients — ``NetworkClient`` Protocol, ``HubClient``, ``AgentClient``.

The trust boundary runs through this package: tenant code (notify
handlers, future transforms, LLM tool execution) only runs inside the
tenant process. The hub never imports anything from here.
"""

from .agent_client import AgentClient
from .hub_client import HubClient
from .network_client import NetworkClient

__all__ = ("AgentClient", "HubClient", "NetworkClient")
