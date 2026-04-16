# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Session verbs — ``open_session``, ``say``, ``listen``, ``leave``.

The three verbs that operate on a session (``say`` / ``listen`` /
``leave``) accept an optional ``session_id`` argument and fall back to
the "current" session — the one whose handler is running this turn —
which is stamped into ``context.dependencies[SESSION_DEP]`` by
``client/handlers.py::_ask`` before ``actor.ask`` is invoked.

This is what makes the ergonomic ``say("hi")`` pattern work without
the LLM having to thread the session id through every call. If the
LLM does want to talk to a *different* session it opened earlier in
the same turn, it passes the explicit ``session_id`` and the verb
routes to that one instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autogen.beta.annotations import Context
from autogen.beta.tools.final.function_tool import tool

from ...envelope import EV_TEXT, Envelope
from ...errors import (
    AccessDeniedError,
    NetworkError,
    SessionClosedError,
    UnknownSessionError,
)
from ..session import Session

if TYPE_CHECKING:
    from ..actor_client import ActorClient


def _resolve_session(
    ctx: Context,
    client: ActorClient,
    session_id: str | None,
) -> Session:
    """Resolve an explicit-or-implicit session reference into a live handle.

    * Explicit ``session_id`` → look up via ``client._hub.peek_session``
      and rebuild a :class:`Session` handle. Raises
      :class:`UnknownSessionError` if the hub has no record.
    * ``session_id is None`` → fall back to
      ``ctx.dependencies[SESSION_DEP]`` (populated by the notify
      handler dispatch). Raises ``ValueError`` if the dispatch did not
      stamp one — that usually means the verb was called outside a
      notify handler, which the framework should never do, but the
      explicit error beats a confusing ``KeyError`` from the
      dependency map.
    """

    from ...policies.session_inbox import SESSION_DEP

    if session_id is not None:
        metadata = client._hub.peek_session(session_id)
        if metadata is None:
            raise UnknownSessionError(session_id)
        return Session(client=client, metadata=metadata)

    current = ctx.dependencies.get(SESSION_DEP)
    if current is None:
        raise ValueError(
            "no current session: pass an explicit session_id or call "
            "this verb from inside a notify handler"
        )
    return current


def _envelope_summary(envelope: Envelope) -> dict[str, Any]:
    """LLM-friendly view of one envelope.

    Strips structural fields the model does not need (signatures, ttl,
    causation chain) and surfaces the readable bits — sender, type,
    payload, timestamp.
    """

    summary: dict[str, Any] = {
        "envelope_id": envelope.envelope_id,
        "session_id": envelope.session_id,
        "sender_id": envelope.sender_id,
        "recipient_id": envelope.recipient_id,
        "event_type": envelope.event_type,
        "created_at": envelope.created_at,
    }
    if envelope.task_id:
        summary["task_id"] = envelope.task_id
    if envelope.event_type == EV_TEXT:
        try:
            summary["content"] = envelope.content()
        except KeyError:
            summary["content"] = None
    else:
        summary["event_data"] = dict(envelope.event_data or {})
    return summary


def build_session_verbs(client: ActorClient) -> list[Any]:
    """Return the four session verbs: ``open_session`` / ``say`` / ``listen`` / ``leave``."""

    @tool
    async def open_session(
        ctx: Context,
        session_type: str,
        target: str,
        intent: str = "",
    ) -> dict[str, Any]:
        """Open a new session with another actor and return its id.

        Args:
            session_type: One of ``"consulting"``, ``"conversation"``,
                ``"notification"``, ``"broadcast"``, ``"discussion"``,
                ``"auction"``, or any operator-registered session type.
            target: The actor name or id of the session's sole peer
                participant. Multi-recipient session types take the
                first invitee here; use the lower-level ``Session``
                API if you need to invite multiple at once.
            intent: Optional free-text reason for opening the session,
                stored on the session's ``labels`` so the audit log
                and downstream tooling can see why the LLM chose this
                action.

        Returns a dict with ``session_id`` and ``state`` (one of
        ``"active"`` / ``"pending"`` / ``"closed"``). On failure
        returns a dict with ``error`` set instead of raising — the
        LLM can read the error and recover.
        """

        from ...policies.session_inbox import SESSION_DEP, SESSION_ID_VAR

        try:
            session = await client.open(
                session_type,
                target=target,
                intent=intent or None,
            )
        except NetworkError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        # Stash the freshly-opened session as the new "current" so a
        # following ``say``/``listen``/``leave`` without explicit
        # ``session_id`` targets the new one. The original incoming
        # session is still reachable via its explicit id.
        ctx.dependencies[SESSION_DEP] = session
        ctx.variables[SESSION_ID_VAR] = session.session_id

        return {
            "session_id": session.session_id,
            "state": session.metadata.state.value
            if hasattr(session.metadata.state, "value")
            else str(session.metadata.state),
            "type": session.metadata.type,
        }

    @tool
    async def say(
        ctx: Context,
        content: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a text message into a session.

        Args:
            content: The message body. Will land in the recipient's
                inbox as an ``ag2.msg.text`` envelope.
            session_id: Optional explicit session id. If omitted the
                verb targets the current notify-handler session
                (stamped into the context by the framework).

        Returns ``{envelope_id, session_id}`` on success, or a dict
        with ``error`` set on failure (closed session, access denied,
        rule violation, …). Errors do **not** raise — the LLM reads
        the result and decides whether to retry.
        """

        try:
            session = _resolve_session(ctx, client, session_id)
        except (ValueError, UnknownSessionError) as exc:
            return {"error": str(exc)}

        try:
            envelope_id = await session.send(content)
        except (
            AccessDeniedError,
            SessionClosedError,
            NetworkError,
        ) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        return {"envelope_id": envelope_id, "session_id": session.session_id}

    @tool
    async def listen(
        ctx: Context,
        scope: str = "session",
        session_id: str | None = None,
        since: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Read recent envelopes from the actor's inbox or a session WAL.

        Args:
            scope: ``"session"`` reads from the bounded WAL of a
                session (defaults to the current one); ``"inbox"``
                reads pending envelopes from the actor's structured
                inbox (envelopes the actor has not yet acknowledged).
            session_id: Optional explicit session id, only meaningful
                for ``scope="session"``. Defaults to the current
                notify-handler session.
            since: WAL byte offset to start reading from. Use ``0``
                for the entire WAL and the offset returned by a
                previous ``listen`` for incremental reads.
            limit: Maximum number of envelopes to return. Default 50.

        Returns ``{scope, envelopes: [...]}`` where each envelope is
        an LLM-friendly summary dict. On error returns a dict with
        ``error`` set.
        """

        scope_normalised = (scope or "session").lower()

        if scope_normalised == "inbox":
            envelopes = await client._hub.peek_inbox(
                client.actor_id, limit=limit
            )
            return {
                "scope": "inbox",
                "envelopes": [_envelope_summary(e) for e in envelopes],
            }

        if scope_normalised != "session":
            return {"error": f"unknown scope {scope!r}; use 'inbox' or 'session'"}

        try:
            session = _resolve_session(ctx, client, session_id)
        except (ValueError, UnknownSessionError) as exc:
            return {"error": str(exc)}

        try:
            envelopes = await client._hub.read_wal(
                session.session_id, since=since
            )
        except NetworkError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        return {
            "scope": "session",
            "session_id": session.session_id,
            "envelopes": [_envelope_summary(e) for e in envelopes[:limit]],
        }

    @tool
    async def leave(
        ctx: Context,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Close (leave) a session.

        Args:
            session_id: Optional explicit session id. Defaults to the
                current notify-handler session.

        Closes the session for everyone — there is no "leave but
        others stay" semantic at the network layer. For multi-party
        sessions where you only want to mute yourself, use a session
        adapter that supports per-participant departure (e.g. a
        custom ``discussion`` adapter), not the generic ``leave`` verb.
        """

        try:
            session = _resolve_session(ctx, client, session_id)
        except (ValueError, UnknownSessionError) as exc:
            return {"error": str(exc)}

        try:
            await session.close()
        except NetworkError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        return {"session_id": session.session_id, "state": "closed"}

    return [open_session, say, listen, leave]


__all__ = (
    "build_session_verbs",
    "_envelope_summary",
    "_resolve_session",
)
