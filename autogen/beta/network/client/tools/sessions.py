# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``sessions(action)`` — session lifecycle for the LLM.

Four actions:

* ``list``  — sessions this agent participates in.
* ``open``  — create a new session (mirrors :meth:`AgentClient.open`).
* ``info``  — full ``SessionMetadata`` for a session this agent can see.
* ``close`` — close the current (or specified) session.

The grouped surface keeps the LLM's tool list short — discovery,
state, and lifecycle live behind one tool.
"""

from typing import TYPE_CHECKING, Any, Literal

from autogen.beta.tools import tool

from ..inject import AgentClientInject, SessionInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient

__all__ = ("make_sessions_tool",)


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    return {
        "session_id": metadata.session_id,
        "type": metadata.manifest.type,
        "version": metadata.manifest.version,
        "state": metadata.state.value,
        "creator_id": metadata.creator_id,
        "participants": [
            {"agent_id": p.agent_id, "role": p.role.value, "order": p.order} for p in metadata.participants
        ],
        "knobs": dict(metadata.knobs),
        "labels": dict(metadata.labels),
        "expectations": [
            {
                "name": e.name,
                "on_violation": e.on_violation,
                "params": dict(e.params),
            }
            for e in metadata.manifest.expectations
        ],
        "created_at": metadata.created_at,
        "expires_at": metadata.expires_at,
        "closed_at": metadata.closed_at,
        "close_reason": metadata.close_reason,
    }


def make_sessions_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``sessions`` tool."""

    @tool
    async def sessions(
        action: Literal["list", "open", "info", "close"],
        *,
        type: str | None = None,
        target: str | list[str] | None = None,
        knobs: dict | None = None,
        intent: str | None = None,
        ttl: str | int | None = None,
        session_id: str | None = None,
        state: Literal["active", "all"] = "active",
        client: AgentClientInject = None,
        current: SessionInject = None,
    ) -> list[dict] | dict | str:
        """Session lifecycle.

        ``list``:  args state="active"|"all"
        ``open``:  args type, target, knobs?, intent?, ttl?
        ``info``:  args session_id
        ``close``: args session_id? (defaults to current)
        """
        actual = client if client is not None else agent_client
        hub = actual._hub_client

        if action == "list":
            include_terminal = state == "all"
            metas = await hub.list_sessions(agent_id=actual.agent_id, include_terminal=include_terminal)
            return [
                {
                    "session_id": m.session_id,
                    "type": m.manifest.type,
                    "state": m.state.value,
                    "participants": [p.agent_id for p in m.participants],
                }
                for m in metas
            ]

        if action == "open":
            if not type or not target:
                return "Error: open requires `type` and `target`"
            try:
                session = await actual.open(
                    type=type,
                    target=target,
                    knobs=knobs,
                    intent=intent,
                    ttl=ttl,
                )
            except Exception as exc:
                return f"Error: open failed: {exc}"
            return {
                "session_id": session.session_id,
                "type": type,
                "participants": [p.agent_id for p in session.metadata.participants],
            }

        if action == "info":
            if not session_id:
                return "Error: info requires `session_id`"
            try:
                meta = await hub.get_session(session_id)
            except Exception:
                return f"Error: session {session_id!r} not found"
            if not any(p.agent_id == actual.agent_id for p in meta.participants):
                return f"Error: not a participant of session {session_id!r}"
            return _metadata_dict(meta)

        if action == "close":
            sid = session_id or (current.session_id if current is not None else None)
            if not sid:
                return "Error: close requires `session_id` or an active session"
            try:
                closed = await hub.close_session(sid, reason="closed_by_agent")
            except Exception as exc:
                return f"Error: close failed: {exc}"
            return {
                "session_id": sid,
                "state": closed.state.value,
                "close_reason": closed.close_reason,
            }

        return f"Error: unknown action {action!r}; choose from list, open, info, close"

    return sessions
