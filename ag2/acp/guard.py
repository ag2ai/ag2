# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Reject ACP requests whose ``_meta`` would overwrite a validated parameter.

The SDK's router expands a request into keyword arguments and then merges the
request's ``_meta`` on top (``acp/router.py``: ``params.update(meta)``). Metadata
therefore *wins* over the protocol's own fields: a request carrying
``_meta: {"session_id": "..."}`` reaches the handler with its ``session_id``
silently replaced, and with no trace of the substitution left in ``**kwargs``.

That matters because ``_meta`` is the one part of a request an application is
encouraged to fill with data from elsewhere — AG2 Space provenance, say. Data
that came from a chat room must not be able to name the session a prompt runs
against.

By the time a handler runs the original value is gone, so the check has to
happen earlier: this module wraps the router and refuses any request whose
metadata collides with a field of that request, before the merge takes place.
"""

import asyncio
import logging
from typing import Any

from acp import schema
from acp.exceptions import RequestError

logger = logging.getLogger(__name__)

# ``_meta`` as it appears on the wire; the request models call it ``field_meta``.
META_KEY = "_meta"


def _reserved_names(model: type) -> frozenset[str]:
    """Every name a request's fields answer to — field name and JSON alias."""
    names: set[str] = set()
    for name, field in getattr(model, "model_fields", {}).items():
        names.add(name)
        if field.alias:
            names.add(field.alias)
    names.discard("field_meta")
    return frozenset(names)


# Built once from the SDK's own models, so a protocol addition is covered
# automatically rather than needing this list to be maintained by hand.
_RESERVED: dict[str, frozenset[str]] = {
    "initialize": _reserved_names(schema.InitializeRequest),
    "authenticate": _reserved_names(schema.AuthenticateRequest),
    "session/new": _reserved_names(schema.NewSessionRequest),
    "session/load": _reserved_names(schema.LoadSessionRequest),
    "session/prompt": _reserved_names(schema.PromptRequest),
    "session/cancel": _reserved_names(schema.CancelNotification),
}


def colliding_meta_keys(method: str, params: Any) -> frozenset[str]:
    """Metadata keys in ``params`` that would displace one of ``method``'s fields."""
    reserved = _RESERVED.get(method)
    if not reserved or not isinstance(params, dict):
        return frozenset()
    meta = params.get(META_KEY)
    if not isinstance(meta, dict):
        return frozenset()
    return frozenset(reserved.intersection(meta))


def guard_router(handler: Any) -> Any:
    """Wrap an ACP message handler so colliding ``_meta`` is refused, not merged.

    Rejecting is the only honest option: the substitution is indistinguishable
    from a legitimate request once the merge has happened, so there is nothing
    left to sanitize downstream.
    """

    async def guarded(method: str, params: Any = None, is_notification: bool = False) -> Any:
        collisions = colliding_meta_keys(method, params)
        if collisions:
            listed = ", ".join(sorted(collisions))
            logger.warning("rejected %s: _meta would overwrite the request field(s) %s", method, listed)
            if is_notification:
                # No response channel on a notification — dropping it is the
                # whole remedy, and acting on it is what we must not do.
                return None
            raise RequestError.invalid_params({
                "reason": f"_meta may not contain the reserved request field(s): {listed}",
            })
        return await handler(method, params, is_notification)

    return guarded


async def serve(agent: Any, reader: Any, writer: Any) -> None:
    """Run ``agent`` over the given streams with the ``_meta`` guard installed.

    Equivalent to :func:`acp.run_agent`, plus :func:`guard_router`. The guard has
    to sit between the transport and the router, which is one layer below what
    ``run_agent`` exposes — hence assembling the connection here instead.

    ``agent`` may be a callable taking the Client, in which case it is invoked
    once for this connection. That is how :class:`~ag2.acp.agent.ACPAgent` gives
    each connection its own authorization and sessions — and since that scope is
    created here, releasing it is this function's job too: nothing else ever sees
    it, so nobody else could.
    """
    from acp.agent.connection import AgentSideConnection

    # ``AgentSideConnection`` invokes the factory and keeps the result to itself,
    # so intercept it on the way past. Only what the factory produced is closed
    # below: a caller who passes a ready-made agent object still owns it.
    scope: Any = None

    def bind(client: Any) -> Any:
        nonlocal scope
        scope = agent(client)
        return scope

    # ``AgentSideConnection`` takes (input=writer, output=reader).
    conn = AgentSideConnection(bind if callable(agent) else agent, writer, reader, listening=False)
    # The router is built inside ``AgentSideConnection`` and is not exposed, so
    # the guard is installed by wrapping it here — before anything is read off
    # the wire, and before any handler can be reached.
    conn._conn._handler = guard_router(conn._conn._handler)
    try:
        await conn.listen()
    finally:
        try:
            # Shielded like ``acp.run_agent`` does: teardown must finish even when
            # the surrounding task is being cancelled.
            await asyncio.shield(conn.close())
        finally:
            # Strictly after the connection, which is what stops the in-flight
            # handlers: closing the scope cancels live turns and deletes their
            # histories, so a handler still running would be writing into a store
            # that had just been purged. In its own ``finally`` because a failed
            # ``conn.close()`` is no reason to strand a Client's sessions.
            await asyncio.shield(_release(scope))


async def _release(scope: Any) -> None:
    """Close the per-connection scope, if the factory produced a closable one.

    ``serve`` is usable with any object implementing the SDK's Agent protocol, and
    that protocol says nothing about teardown — so this is opt-in rather than
    required, and an agent with no per-connection state to drop simply has no
    ``aclose``.
    """
    aclose = getattr(scope, "aclose", None)
    if aclose is None:
        return
    await aclose()
