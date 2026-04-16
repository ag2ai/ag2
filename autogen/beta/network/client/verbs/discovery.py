# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Discovery verbs — ``find_actors``, ``describe_actor``.

Both verbs are read-only: they go through :meth:`Hub.find` and
:meth:`Hub.describe` respectively, do not touch the WAL, and never
mutate hub state. They are safe to call from any actor turn,
inside or outside a session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autogen.beta.tools.final.function_tool import tool

from ...errors import UnknownActorError

if TYPE_CHECKING:
    from ..actor_client import ActorClient


def _summarise_identity(identity: Any) -> dict[str, Any]:
    """Return the LLM-friendly summary view of an :class:`ActorIdentity`.

    ``find_actors`` returns a *list* of these, so each entry intentionally
    omits the SKILL.md sidecar and the ``capabilities``/``domains`` are
    truncated to keep the LLM's working set small. ``describe_actor`` is
    the verb that returns the full identity (including SKILL.md) when
    the model decides it wants more detail on a candidate.
    """

    return {
        "name": identity.name,
        "actor_id": identity.actor_id,
        "display": identity.display,
        "summary": identity.summary,
        "capabilities": list(identity.capabilities or ()),
        "domains": list(identity.domains or ()),
        "owner": identity.owner,
        "version": identity.version,
        "runtime_kind": identity.runtime_kind,
    }


def build_discovery_verbs(client: ActorClient) -> list[Any]:
    """Return the ``find_actors`` and ``describe_actor`` verb tools."""

    @tool
    async def find_actors(
        capability: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Discover registered actors on the hub.

        Args:
            capability: Optional exact capability string to require
                (matches an entry in the actor's ``capabilities`` list,
                e.g. ``"research"``, ``"summarization"``).
            query: Optional free-text query — case-insensitive substring
                match against the actor's name, display, summary,
                domains, and strengths.

        When both filters are supplied the result is the AND. Returns
        a list of summary dicts (no SKILL.md, no rule, no auth). Use
        :func:`describe_actor` to fetch the full identity for a single
        candidate.
        """

        identities = await client._hub.find(capability=capability, query=query)
        return [_summarise_identity(i) for i in identities]

    @tool
    async def describe_actor(name: str) -> dict[str, Any]:
        """Fetch the full :class:`ActorIdentity` for a single actor.

        Args:
            name: The actor's registered name (e.g. ``"ag2:writer:1"``)
                or its hub-stamped ``actor_id``.

        Returns the identity as a dict — same shape as the
        ``GET /v1/actors/{id}`` HTTP response, including the SKILL.md
        sidecar under the ``skill_md`` key when the actor uploaded one.
        Use this after ``find_actors`` narrows candidates down to one.
        """

        try:
            identity = await client._hub.describe(name)
        except UnknownActorError:
            return {"error": f"unknown actor {name!r}"}
        return identity.to_dict()

    return [find_actors, describe_actor]


__all__ = ("build_discovery_verbs",)
