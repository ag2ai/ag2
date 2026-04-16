# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-canned ``Annotated`` aliases for user-tool dependency injection.

The Phase 6 LLM verb surface lives in
``autogen.beta.network.client.verbs`` and is auto-injected by the
:class:`ActorClient` whenever a notify handler dispatches a real
:meth:`Actor.ask` turn. Alongside the verbs the ``ActorClient`` also
populates :class:`~autogen.beta.context.ConversationContext`'s
``dependencies`` map with three live handles — the current
:class:`~autogen.beta.network.client.session.Session`, the owning
:class:`~autogen.beta.network.client.actor_client.ActorClient`, and
(for task handlers) the in-flight
:class:`~autogen.beta.network.client.task.Task` — under qualified key
strings declared in
:mod:`autogen.beta.network.policies.session_inbox`.

User-defined ``@tool``s opt into these handles via
``Annotated[T, Inject("ag2.network.<key>")]``. To save users from
having to remember the exact key strings (and to keep type checkers
happy without a manual ``# type: ignore``) Phase 6 ships pre-canned
aliases here:

.. code-block:: python

    from autogen.beta.network.client.inject import (
        SessionInject,
        TaskInject,
        ActorClientInject,
        HubInject,
    )

    @tool
    async def my_tool(
        question: str,
        session: SessionInject,
        client: ActorClientInject,
    ) -> str:
        # ``session`` and ``client`` come from the ActorClient's
        # context.dependencies — populated by the notify handler before
        # actor.ask was called.
        await session.send(f"forwarded: {question}")
        return f"actor={client.identity.name}"

The aliases resolve to the same qualified keys the framework's own
auto-injected verbs use, so a tool that injects ``SessionInject`` sees
the same :class:`Session` instance the ``say`` verb would target by
default.

Outside a network turn (for example a standalone ``Actor.ask`` with no
hub attachment) the keys are absent. ``Inject(...)`` with no default
will raise at tool-resolution time; if you want a tool that works both
on and off the network, pass an explicit ``default`` to the wrapped
``Inject``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from autogen.beta.annotations import Inject

from ..policies.session_inbox import (
    ACTOR_CLIENT_DEP,
    HUB_DEP,
    SESSION_DEP,
    TASK_DEP,
)

if TYPE_CHECKING:
    from ..hub import Hub
    from .actor_client import ActorClient
    from .session import Session
    from .task import Task


# ---------------------------------------------------------------------------
# Pre-canned ``Annotated`` aliases
# ---------------------------------------------------------------------------
#
# Each alias resolves to ``Annotated[T, Inject("ag2.network.<key>")]``
# so a parameter typed with the alias automatically picks up the
# corresponding entry from ``context.dependencies``. The key strings
# are the same constants the framework-injected verbs use — see
# ``autogen/beta/network/policies/session_inbox.py`` for the source.

SessionInject = Annotated["Session", Inject(SESSION_DEP)]
"""The current :class:`Session` handle for the notify-handler turn."""

TaskInject = Annotated["Task", Inject(TASK_DEP)]
"""The in-flight :class:`Task` handle when a task handler is dispatching."""

ActorClientInject = Annotated["ActorClient", Inject(ACTOR_CLIENT_DEP)]
"""The owning :class:`ActorClient` (identity + rule + transport view)."""

HubInject = Annotated["Hub", Inject(HUB_DEP)]
"""The :class:`Hub` the actor is attached to."""


__all__ = (
    "ActorClientInject",
    "HubInject",
    "SessionInject",
    "TaskInject",
)
