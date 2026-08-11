# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Channel`` — client-side handle for a channel this agent participates in.

Wraps ``ChannelMetadata`` plus a back-pointer to the ``AgentClient`` so
tools / handlers can ``send`` envelopes, ``close`` early, or ``info``
the current state without reaching back through the hub directly.
"""

from typing import TYPE_CHECKING, Any

from ..channel import ChannelMetadata, ChannelState
from ..envelope import EV_TEXT, Envelope

if TYPE_CHECKING:
    from .agent_client import AgentClient

__all__ = ("Channel",)


class Channel:
    """Per-participant handle for a channel.

    Constructed by ``AgentClient.open(...)`` / hydrated when an
    ``EV_CHANNEL_OPENED`` lands. The ``metadata`` attribute is a
    snapshot — call :meth:`info` to refresh from the hub.
    """

    def __init__(
        self,
        *,
        metadata: ChannelMetadata,
        client: "AgentClient",
    ) -> None:
        # __init__ stores params; no side effects.
        self._metadata = metadata
        self._client = client

    @property
    def channel_id(self) -> str:
        return self._metadata.channel_id

    @property
    def metadata(self) -> ChannelMetadata:
        return self._metadata

    @property
    def state(self) -> ChannelState:
        return self._metadata.state

    async def send(
        self,
        content: str,
        *,
        audience: list[str] | None = None,
        causation_id: str | None = None,
        event_type: str = EV_TEXT,
        event_data: dict[str, Any] | None = None,
        depth: int | None = None,
    ) -> str:
        """Post an envelope into this channel.

        ``audience=None`` broadcasts within the channel (all
        participants except sender). ``content`` is the substantive
        body for ``EV_TEXT`` envelopes; for non-text events pass
        ``event_data`` and ``event_type`` instead. ``depth`` overrides
        the default 0 so callers (e.g. ``delegate``) can stamp the
        delegation hop count for ``Rule.limits.delegation_depth``
        enforcement.
        """
        if event_data is None:
            event_data = {"text": content}
        envelope = Envelope(
            channel_id=self.channel_id,
            sender_id=self._client.agent_id,
            audience=audience,
            event_type=event_type,
            event_data=event_data,
            causation_id=causation_id,
            depth=depth if depth is not None else 0,
        )
        return await self._client.send_envelope(envelope)

    async def info(self) -> ChannelMetadata:
        """Re-fetch metadata from the hub (refreshes cached state)."""
        refreshed = await self._client._hub_client.get_channel(self.channel_id)
        self._metadata = refreshed
        return refreshed

    async def close(self, reason: str = "") -> ChannelMetadata:
        """Close the channel. Auto-cascades expiry to non-terminal tasks."""
        return await self._client._hub_client.close_channel(self.channel_id, reason=reason)

    def is_terminal(self) -> bool:
        return self._metadata.is_terminal()
