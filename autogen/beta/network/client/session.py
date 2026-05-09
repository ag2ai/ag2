# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Session`` — client-side handle for a session this agent participates in.

Wraps ``SessionMetadata`` plus a back-pointer to the ``AgentClient`` so
tools / handlers can ``send`` envelopes, ``close`` early, or ``info``
the current state without reaching back through the hub directly.
"""

from typing import TYPE_CHECKING, Any

from ..envelope import EV_TEXT, Envelope
from ..session import SessionMetadata, SessionState

if TYPE_CHECKING:
    from .agent_client import AgentClient

__all__ = ("Session",)


class Session:
    """Per-participant handle for a session.

    Constructed by ``AgentClient.open(...)`` / hydrated when an
    ``EV_SESSION_OPENED`` lands. The ``metadata`` attribute is a
    snapshot — call :meth:`info` to refresh from the hub.
    """

    def __init__(
        self,
        *,
        metadata: SessionMetadata,
        client: "AgentClient",
    ) -> None:
        # __init__ stores params; no side effects.
        self._metadata = metadata
        self._client = client

    @property
    def session_id(self) -> str:
        return self._metadata.session_id

    @property
    def metadata(self) -> SessionMetadata:
        return self._metadata

    @property
    def state(self) -> SessionState:
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
        """Post an envelope into this session.

        ``audience=None`` broadcasts within the session (all
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
            session_id=self.session_id,
            sender_id=self._client.agent_id,
            audience=audience,
            event_type=event_type,
            event_data=event_data,
            causation_id=causation_id,
            depth=depth if depth is not None else 0,
        )
        return await self._client.send_envelope(envelope)

    async def info(self) -> SessionMetadata:
        """Re-fetch metadata from the hub (refreshes cached state)."""
        refreshed = await self._client._hub_client.get_session(self.session_id)
        self._metadata = refreshed
        return refreshed

    async def close(self, reason: str = "") -> SessionMetadata:
        """Close the session. Auto-cascades expiry to non-terminal tasks."""
        return await self._client._hub_client.close_session(self.session_id, reason=reason)

    def is_terminal(self) -> bool:
        return self._metadata.is_terminal()
