# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``channels(action)`` — channel lifecycle for the LLM.

Four actions:

* ``list``  — channels this agent participates in.
* ``open``  — create a new channel (mirrors :meth:`AgentClient.open`).
* ``info``  — full ``ChannelMetadata`` for a channel this agent can see.
* ``close`` — close the current (or specified) channel.

The grouped surface keeps the LLM's tool list short — discovery,
state, and lifecycle live behind one tool.
"""

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from ag2.tools import tool

from ..inject import AgentClientInject, ChannelInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient

__all__ = ("make_channels_tool",)


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    return {
        "channel_id": metadata.channel_id,
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


def make_channels_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``channels`` tool."""

    @tool
    async def channels(
        action: Literal["list", "open", "info", "close"],
        *,
        type: str | None = None,
        target: str | list[str] | None = None,
        knobs: dict | None = None,
        intent: str | None = None,
        ttl: str | int | None = None,
        message: str | None = None,
        channel_id: str | None = None,
        state: Literal["active", "all"] = "active",
        client: AgentClientInject = None,
        current: ChannelInject = None,
    ) -> list[dict] | dict | str:
        """Channel lifecycle.

        ``list``:  args state="active"|"all"
        ``open``:  args type, target, knobs?, intent?, ttl?, message?
                   ``message`` seeds the first envelope on the
                   initiator's behalf after the channel transitions
                   to ``OPENED`` — useful for short-lived channels
                   where the initiator wants to atomically open + send.
        ``info``:  args channel_id
        ``close``: args channel_id? (defaults to current)
        """
        actual = client if client is not None else agent_client
        hub = actual._hub_client

        if action == "list":
            include_terminal = state == "all"
            metas = await hub.list_channels(agent_id=actual.agent_id, include_terminal=include_terminal)
            return [
                {
                    "channel_id": m.channel_id,
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
                channel = await actual.open(
                    type=type,
                    target=target,
                    knobs=knobs,
                    intent=intent,
                    ttl=ttl,
                )
            except Exception as exc:
                return f"Error: open failed: {exc}"
            seeded_envelope_id: str | None = None
            if message:
                try:
                    seeded_envelope_id = await channel.send(message)
                except Exception as exc:
                    # Rollback so we deliver atomic-ish semantics: a
                    # failed seed should not leave a dangling-open
                    # channel that no one ever sends into.
                    with contextlib.suppress(Exception):
                        await hub.close_channel(channel.channel_id, reason="seed_failed")
                    return f"Error: seed send failed: {exc}"
            result: dict = {
                "channel_id": channel.channel_id,
                "type": type,
                "participants": [p.agent_id for p in channel.metadata.participants],
            }
            if seeded_envelope_id:
                result["seed_envelope_id"] = seeded_envelope_id
            return result

        if action == "info":
            if not channel_id:
                return "Error: info requires `channel_id`"
            try:
                meta = await hub.get_channel(channel_id)
            except Exception:
                return f"Error: channel {channel_id!r} not found"
            if not any(p.agent_id == actual.agent_id for p in meta.participants):
                return f"Error: not a participant of channel {channel_id!r}"
            return _metadata_dict(meta)

        if action == "close":
            sid = channel_id or (current.channel_id if current is not None else None)
            if not sid:
                return "Error: close requires `channel_id` or an active channel"
            try:
                closed = await hub.close_channel(sid, reason="closed_by_agent")
            except Exception as exc:
                return f"Error: close failed: {exc}"
            return {
                "channel_id": sid,
                "state": closed.state.value,
                "close_reason": closed.close_reason,
            }

        return f"Error: unknown action {action!r}; choose from list, open, info, close"

    return channels
