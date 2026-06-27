# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``say`` — flat LLM tool to post into a channel.

Defaults to the ``Channel`` resolved from ``ChannelInject`` (the active
notify-handler context). ``channel_id`` overrides for the rare case
the LLM wants to post into a different channel it's also a participant
of. ``audience`` is a list of agent **names**; the tool resolves them
to ids via the hub. ``audience=None`` broadcasts within the channel.

Envelope construction goes through ``adapter.build_text_envelope`` so
the produced envelope is shaped for the channel's adapter — the same
helper a non-AG2 bridge would call to drive a turn manually.
"""

from typing import TYPE_CHECKING

from ag2.tools import tool

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

        Envelope shape comes from ``adapter.build_text_envelope`` so
        per-adapter customization (e.g. an adapter that wraps text in
        a richer event shape) is honored automatically.

        Returns the hub-stamped envelope_id, or an error string if the
        send fails.
        """
        # Resolve the channel handle + metadata.
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

        # Build the envelope through the adapter's Layer-2 helper so the
        # tool produces the same shape a hand-written bridge would. The
        # adapter is fetched from the public HubClient surface — the
        # tool never reaches into hub internals.
        try:
            adapter = actual_client._hub_client.adapter_for_metadata(target_channel.metadata)
        except Exception as exc:
            return f"Error: adapter unavailable for channel {target_channel.channel_id!r}: {exc}"
        envelope = adapter.build_text_envelope(
            channel_id=target_channel.channel_id,
            sender_id=actual_client.agent_id,
            text=content,
            audience=audience_ids,
        )

        try:
            envelope_id = await actual_client.send_envelope(envelope)
        except Exception as exc:
            return f"Error: send failed: {exc}"
        return f"posted envelope {envelope_id}"

    return say
