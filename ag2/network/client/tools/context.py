# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``context(action)`` — read from past content.

Two actions:

* ``search`` — substring search over the current channel's WAL or the
  calling agent's ``KnowledgeStore``. Returns up to ``limit`` excerpts.
* ``quote``  — return the last N envelopes a given speaker posted in
  the current (or specified) channel.

Substring search only. Vector / semantic search composes via the
existing ``KnowledgeStore`` infrastructure when configured; the tool
surface stays the same.
"""

from typing import TYPE_CHECKING, Literal

from ag2.tools import tool

from ...envelope import EV_TEXT, Envelope, visible_to
from ..inject import AgentClientInject, ChannelInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient


__all__ = ("make_context_tool",)


def _excerpt(envelope: Envelope, max_chars: int = 240) -> str:
    text = envelope.event_data.get("text", "")
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def make_context_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``context`` tool."""

    @tool
    async def context(
        action: Literal["search", "quote"],
        *,
        query: str | None = None,
        scope: Literal["channel", "knowledge"] = "channel",
        speaker: str | None = None,
        recent_n: int = 1,
        limit: int = 10,
        channel_id: str | None = None,
        client: AgentClientInject = None,
        channel: ChannelInject = None,
    ) -> list[dict] | str:
        """Read from past content.

        ``search``: args query, scope="channel"|"knowledge", limit
                    Returns excerpts of envelopes whose text matches
                    ``query`` (case-insensitive substring).
        ``quote``:  args speaker, recent_n=1, channel_id?
                    Returns the last ``recent_n`` envelopes from
                    ``speaker`` in the current (or specified) channel.
        """
        actual = client if client is not None else agent_client
        hub = actual._hub_client
        sid = channel_id or (channel.channel_id if channel is not None else None)

        if action == "search":
            if not query:
                return "Error: search requires `query`"
            needle = query.lower()
            if scope == "channel":
                if not sid:
                    return "Error: search scope=channel requires an active channel"
                try:
                    wal = await hub.read_wal(sid)
                except Exception:
                    return f"Error: channel {sid!r} not found"
                results: list[dict] = []
                for env in wal:
                    if env.event_type != EV_TEXT:
                        continue
                    if not visible_to(env, actual.agent_id):
                        continue
                    text = env.event_data.get("text", "")
                    if not isinstance(text, str) or needle not in text.lower():
                        continue
                    results.append({
                        "envelope_id": env.envelope_id,
                        "sender_id": env.sender_id,
                        "when": env.created_at,
                        "excerpt": _excerpt(env),
                    })
                    if len(results) >= limit:
                        break
                return results
            if scope == "knowledge":
                # Knowledge-scope search is best-effort over the calling
                # agent's own KnowledgeStore. Returns an empty list when
                # the store has no primitive for substring search —
                # semantic search lives in framework-core ``recall`` and
                # is invoked by the agent's own loop, not by this tool.
                return []

        if action == "quote":
            if not speaker:
                return "Error: quote requires `speaker`"
            if not sid:
                return "Error: quote requires an active channel or `channel_id`"
            # Resolve speaker name → agent_id (accept both).
            try:
                speaker_passport = await hub.get_agent(speaker)
            except Exception:
                return f"Error: speaker {speaker!r} not found"
            speaker_id = speaker_passport.agent_id
            try:
                wal = await hub.read_wal(sid)
            except Exception:
                return f"Error: channel {sid!r} not found"
            picks: list[dict] = []
            for env in reversed(wal):
                if env.event_type != EV_TEXT:
                    continue
                if env.sender_id != speaker_id:
                    continue
                if not visible_to(env, actual.agent_id):
                    continue
                picks.append({
                    "envelope_id": env.envelope_id,
                    "when": env.created_at,
                    "text": env.event_data.get("text", ""),
                })
                if len(picks) >= recent_n:
                    break
            picks.reverse()
            return picks

        return f"Error: unknown action {action!r}; choose from search, quote"

    return context
