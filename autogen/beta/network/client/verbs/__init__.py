# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — auto-injected LLM network verb surface.

Eight verbs (per §10.5 of the design doc) factored as small,
single-purpose ``@tool``-decorated coroutines that close over the
owning :class:`~autogen.beta.network.client.actor_client.ActorClient`:

================  ===========================================================
Verb              Action
================  ===========================================================
find_actors       Discover actors by capability + free-text query
describe_actor    Fetch an :class:`ActorIdentity` + SKILL.md by name or id
open_session      Open a new session (type, target, intent) → ``session_id``
say               Send content into a session (defaults to current)
listen            Read recent envelopes from inbox or session WAL
run_task          Create a network task; awaits terminal state by default
track_task        Look up a task's current state by id
leave             Close (leave) a session
================  ===========================================================

Each module exports a ``build_*_verbs(client)`` factory; this package's
:func:`build_network_verbs` returns the union, which is what
:meth:`ActorClient._build_network_verbs` calls per-turn from
``client/handlers.py::_ask``.

Verbs that need a "current session" (every session/task verb except
``find_actors`` / ``describe_actor`` / ``track_task``) accept an
optional ``session_id`` argument and fall back to
``context.dependencies[SESSION_DEP]`` when the LLM omits it. The DI
keys are populated by the same notify-handler dispatch that builds
the verb list, so the fallback is always populated when a verb is
running through a real handler turn.

These verbs route through :meth:`Session.send` / :meth:`ActorClient.open`
/ :meth:`Hub.create_task` etc. — they do **not** bypass any access,
limit, or transform enforcement. A verb that violates the actor's
outbound rule raises an :class:`AccessDeniedError` to the LLM, which
the model can read in the tool result and recover from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .discovery import build_discovery_verbs
from .session import build_session_verbs
from .task import build_task_verbs

if TYPE_CHECKING:
    from ..actor_client import ActorClient
    from ..session import Session
    from ..task import Task


def build_network_verbs(
    client: ActorClient,
    *,
    session: Session | None = None,
    task: Task | None = None,
) -> list[Any]:
    """Return the eight Phase 6 LLM verb tools wired against ``client``.

    ``session`` and ``task`` are unused at construction time but are
    passed by the caller for symmetry with future verb factories that
    might want to specialize on them — for example, a verb registry
    that strips ``run_task`` when the actor is already inside a task
    handler. Today every turn gets the same eight verbs.
    """

    del session, task  # currently informational

    verbs: list[Any] = []
    verbs.extend(build_discovery_verbs(client))
    verbs.extend(build_session_verbs(client))
    verbs.extend(build_task_verbs(client))
    return verbs


__all__ = (
    "build_discovery_verbs",
    "build_network_verbs",
    "build_session_verbs",
    "build_task_verbs",
)
