# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Reject ACP requests whose ``_meta`` would overwrite a validated parameter.

The SDK's router expands a request into keyword arguments and then merges the
request's ``_meta`` on top (``acp/utils.py``: ``model_to_kwargs``). Metadata
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
from collections.abc import Callable
from typing import Protocol, cast

from acp import schema
from acp.agent.connection import AgentSideConnection
from acp.connection import Connection, JsonValue, MethodHandler
from acp.exceptions import RequestError
from acp.interfaces import Agent, Client
from acp.meta import AGENT_METHODS
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ``_meta`` as it appears on the wire; the request models call it ``field_meta``.
META_KEY = "_meta"

# The requests ACPAgent serves, by method: the SDK's own models, so a field a
# newer protocol adds is covered without a list maintained by hand.
_REQUESTS: dict[str, type[BaseModel]] = {
    AGENT_METHODS["initialize"]: schema.InitializeRequest,
    AGENT_METHODS["authenticate"]: schema.AuthenticateRequest,
    AGENT_METHODS["session_new"]: schema.NewSessionRequest,
    AGENT_METHODS["session_load"]: schema.LoadSessionRequest,
    AGENT_METHODS["session_prompt"]: schema.PromptRequest,
    AGENT_METHODS["session_cancel"]: schema.CancelNotification,
}


def request_fields(method: str) -> frozenset[str]:
    """The names ``method``'s fields reach its handler under: their Python names.

    The SDK never passes a field by its JSON alias, so a camelCase key in a
    handler's ``**kwargs`` can only have come from ``_meta``.
    """
    return frozenset(_REQUESTS[method].model_fields) - {"field_meta"}


def colliding_meta_keys(method: str, params: JsonValue) -> frozenset[str]:
    """Metadata keys in ``params`` that would displace one of ``method``'s fields.

    Both spellings count: the Python name, which the merge would overwrite, and
    the JSON alias, which names the same field on the wire. That makes every
    field a protocol release adds reserved too — ``systemPrompt`` and
    ``configOptions`` since ACP 1.1 — so ``_meta`` cannot use either spelling of
    them, whether or not the request sets the field itself.
    """
    model = _REQUESTS.get(method)
    if model is None or not isinstance(params, dict):
        return frozenset()
    meta = params.get(META_KEY)
    if not isinstance(meta, dict):
        return frozenset()
    aliases = {field.alias for field in model.model_fields.values() if field.alias}
    return (request_fields(method) | aliases).intersection(meta)


def guard_router(handler: MethodHandler) -> MethodHandler:
    """Wrap an ACP message handler so colliding ``_meta`` is refused, not merged.

    Rejecting is the only honest option: the substitution is indistinguishable
    from a legitimate request once the merge has happened, so there is nothing
    left to sanitize downstream.
    """

    async def guarded(method: str, params: JsonValue = None, is_notification: bool = False) -> JsonValue:
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


class _Scope(Protocol):
    """A per-connection scope ``serve`` created and must close when the loop ends.

    The SDK's ``Agent`` protocol has no teardown of its own, so this is AG2's.
    """

    async def aclose(self) -> None: ...


async def serve(
    agent: "Callable[[Client], _Scope] | object",
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Run ``agent`` over the given streams with the ``_meta`` guard installed.

    Equivalent to :func:`acp.run_agent`, plus :func:`guard_router`. The guard has
    to sit between the transport and the router, which is one layer below what
    ``run_agent`` exposes — hence assembling the connection here instead.

    ``agent`` may be a factory taking the Client, invoked once for this
    connection — that is how :class:`~ag2.acp.agent.ACPAgent` gives each
    connection its own authorization and sessions. Because that scope is created
    here, closing it is this function's job too; a ready-made agent object passed
    instead is the caller's to own, and is never closed here.
    """

    # ``AgentSideConnection`` invokes the factory and keeps the result to itself,
    # so intercept it on the way past to hold a handle for teardown below. Only
    # what the factory produced is closed: ``scope`` stays None for a pre-built
    # agent, so a passed-in object is never torn down here even when it has an
    # ``aclose`` of its own.
    scope: _Scope | None = None

    def bind(client: Client) -> _Scope:
        nonlocal scope
        assert callable(agent)  # bind is only installed when ``agent`` is a factory
        scope = agent(client)
        return scope

    # Own the ``Connection`` so the guard sits between the transport and the
    # router without reaching into SDK internals. ``Connection`` needs a handler
    # at construction but ``attach`` only yields one afterwards, so ``forward``
    # stands in and reads ``guarded`` when a frame arrives — which is never before
    # ``main_loop`` below, by when ``guarded`` is set. ``(input=writer,
    # output=reader)`` as the constructor takes them.
    guarded: MethodHandler | None = None

    async def forward(method: str, params: JsonValue = None, is_notification: bool = False) -> JsonValue:
        assert guarded is not None  # set below, before ``main_loop`` reads a frame
        return await guarded(method, params, is_notification)

    conn = Connection(forward, writer, reader, listening=False)
    # ``attach`` runs the factory (``bind``) with the wrapper — so each connection
    # still gets its own scope — and returns the SDK's own message handler, built
    # before anything is read off the wire. That handler is what the guard wraps.
    # ``bind`` returns ACPAgent's scope, which implements only the subset of the
    # SDK's ``Agent`` it serves; ``attach`` wants a full ``Agent`` and dispatches
    # by getattr, so this cast is unavoidable — the SDK's own ``_prepare`` casts
    # identically. A factory goes in wrapped as ``bind``; a pre-built agent passes
    # straight through.
    to_agent = cast("Callable[[AgentSideConnection], Agent] | Agent", bind if callable(agent) else agent)
    _wrapper, handler = AgentSideConnection.attach(to_agent, conn)
    guarded = guard_router(handler)
    try:
        await conn.main_loop()
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
            # ``conn.close()`` is no reason to strand a Client's sessions. ``scope``
            # is None only when ``attach`` failed before it ever ran the factory.
            if scope is not None:
                await asyncio.shield(scope.aclose())
