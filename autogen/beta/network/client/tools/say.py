# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``say`` — flat LLM tool to post into a channel.

Defaults to the ``Channel`` resolved from ``ChannelInject`` (the active
notify-handler context). ``channel_id`` overrides for the rare case
the LLM wants to post into a different channel it's also a participant
of. ``audience`` is a list of agent **names**; the tool resolves them
to ids via the hub. ``audience=None`` broadcasts within the channel.
"""

from typing import TYPE_CHECKING

from autogen.beta.tools import tool

from ..channel import Channel
from ..inject import AgentClientInject, ChannelInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient

__all__ = ("make_say_tool",)


def make_say_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``say`` tool.

    The closure captures ``agent_client`` once at registration; the
    resulting ``FunctionTool`` is stable across turns (no nested
    allocation in the hot path). The LLM-facing parameter is
    ``client: AgentClientInject`` — the framework resolves it from
    ``context.dependencies`` when the tool runs inside a notify
    handler; the closure binding is the fallback for direct invocation.
    """

    @tool
    async def say(
        content: str,
        *,
        audience: list[str] | None = None,
        channel_id: str | None = None,
        channel: ChannelInject = None,
        client: AgentClientInject = None,
    ) -> str:
        """Post a text envelope into the current (or specified) channel.

        ``audience``: list of peer **names**. ``None`` broadcasts within
        the channel. The tool resolves names to agent ids via the hub.

        Returns the hub-stamped envelope_id, or an error string if the
        send fails.
        """
        # Resolve the channel handle.
        target_channel = channel
        actual_client = client if client is not None else agent_client

        if target_channel is None:
            if channel_id is None:
                return "Error: no current channel and no channel_id provided"
            try:
                metadata = await actual_client._hub_client.get_channel(channel_id)
            except Exception as exc:
                return f"Error: channel {channel_id!r} not found: {exc}"
            target_channel = Channel(metadata=metadata, client=actual_client)

        # Resolve audience names → agent ids.
        audience_ids: list[str] | None = None
        if audience is not None:
            audience_ids = []
            for name in audience:
                try:
                    passport = await actual_client._hub_client.get_agent(name)
                except Exception:
                    return f"Error: peer {name!r} not found"
                if passport.agent_id is None:
                    return f"Error: peer {name!r} has no agent_id"
                audience_ids.append(passport.agent_id)

        try:
            envelope_id = await target_channel.send(content, audience=audience_ids)
        except Exception as exc:
            return f"Error: send failed: {exc}"
        return f"posted envelope {envelope_id}"

    return say
