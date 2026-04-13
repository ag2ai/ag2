# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Client-side network package — HubClient, ActorClient, Session handles.

The two-client narrative: every actor process holds a :class:`HubClient`
(outbound: registration, discovery, session creation, send) and one
:class:`ActorClient` per registered identity (inbound: inbox loop, handler
registry, local transform execution, reply posting). The split exists so
rule transforms run in the tenant's address space rather than the hub's.
"""

from __future__ import annotations

from .actor_client import ActorClient, NotifyHandler, TaskHandler
from .hub_client import HubClient
from .human import (
    HumanClient,
    HumanCliSurface,
    HumanScriptedSurface,
    HumanSurface,
    human_cli_client,
)
from .session import Session
from .task import Task

__all__ = (
    "ActorClient",
    "HubClient",
    "HumanCliSurface",
    "HumanClient",
    "HumanScriptedSurface",
    "HumanSurface",
    "NotifyHandler",
    "Session",
    "Task",
    "TaskHandler",
    "human_cli_client",
)
