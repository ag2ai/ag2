# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Session — client-side handle for a live session.

Holds a reference back to its owning :class:`ActorClient` so ``send`` /
``ask`` / ``subscribe`` / ``close`` all route through the link rather than
talking to the hub directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..envelope import EV_TEXT, Envelope
from ..errors import TimeoutError as NetTimeoutError
from ..session_types import SessionMetadata

if TYPE_CHECKING:
    from .actor_client import ActorClient


class _Unset:
    """Sentinel for kwargs that distinguish "not provided" from ``None``."""


_UNSET: _Unset = _Unset()


class Session:
    """Client-side session handle.

    Construction is internal — obtain a :class:`Session` via
    :meth:`ActorClient.open` or :meth:`ActorClient.session_for`.
    """

    def __init__(self, *, client: ActorClient, metadata: SessionMetadata) -> None:
        self._client = client
        self._metadata = metadata

    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._metadata.session_id

    @property
    def type(self) -> str:
        return self._metadata.type

    @property
    def metadata(self) -> SessionMetadata:
        return self._metadata

    def _refresh_metadata(self) -> SessionMetadata:
        refreshed = self._client.lookup_session(self.session_id)
        if refreshed is not None:
            self._metadata = refreshed
        return self._metadata

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def send(
        self,
        content: str,
        *,
        recipient_id: str | None = None,
        causation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Post a text envelope into the session. Returns the envelope id.

        ``idempotency_key`` (optional, client-provided UUID) makes the
        send safe to retry after a transient failure: within the hub's
        dedup TTL window a second call with the same key returns the
        same envelope without re-running adapters / rate limits.
        """

        target = recipient_id or self._default_recipient()
        envelope = Envelope.text(
            session_id=self.session_id,
            sender_id=self._client.actor_id,
            content=content,
            recipient_id=target,
            causation_id=causation_id,
        )
        if idempotency_key is not None:
            envelope.idempotency_key = idempotency_key
        envelope_id, _ = await self._client._send_envelope(envelope)
        return envelope_id

    async def ask(self, content: str, *, timeout: float | None = 10.0) -> str:
        """Send ``content`` and await the reply correlated by causation_id.

        Subscribes with ``since=wal_offset_after_send`` so the hub replays
        only envelopes that land after this send — no O(N) full-WAL replay
        on every ask.
        """

        envelope = Envelope.text(
            session_id=self.session_id,
            sender_id=self._client.actor_id,
            content=content,
            recipient_id=self._default_recipient(),
        )
        envelope_id, wal_offset = await self._client._send_envelope(envelope)
        queue = await self._client._open_subscription(
            session_id=self.session_id,
            causation_id=envelope_id,
            since=wal_offset,
        )
        try:
            deadline = None if timeout is None else timeout
            reply = await asyncio.wait_for(queue.get(), timeout=deadline)
            if reply.event_type != EV_TEXT:
                raise NetTimeoutError("reply was not a text envelope")
            return reply.content()
        except asyncio.TimeoutError as exc:
            raise NetTimeoutError(f"session.ask timed out after {timeout}s") from exc
        finally:
            await self._client._close_subscription(queue)

    async def subscribe(self, *, since: int = 0) -> AsyncIterator[Envelope]:
        queue = await self._client._open_subscription(
            session_id=self.session_id, since=since
        )
        try:
            while True:
                envelope = await queue.get()
                yield envelope
        finally:
            await self._client._close_subscription(queue)

    # ------------------------------------------------------------------
    # Streaming (Phase 2)
    # ------------------------------------------------------------------

    async def send_chunk(
        self,
        *,
        envelope_id: str,
        chunk_index: int,
        content: str,
        recipient_id: str | None | _Unset = _UNSET,
        final: bool = False,
    ) -> None:
        """Emit a chunk for an in-flight response envelope.

        ``envelope_id`` is the id the recipient will correlate chunks
        with — typically the ``causation_id`` of the incoming request
        this reply answers. ``recipient_id`` behavior:

        * Not passed at all → use the default recipient (2-party
          sessions) or fan out to every non-sender participant
          (multi-party).
        * Passed as ``None`` → explicit broadcast, fan out to every
          non-sender participant even in a 2-party session.
        * Passed as a string → direct unicast to that participant.
        """

        target = (
            self._default_recipient()
            if isinstance(recipient_id, _Unset)
            else recipient_id
        )
        await self._client._send_chunk(
            envelope_id=envelope_id,
            session_id=self.session_id,
            chunk_index=chunk_index,
            content=content,
            recipient_id=target,
            final=final,
        )

    async def iter_chunks(self, envelope_id: str) -> AsyncIterator[str]:
        """Yield chunk content strings as they arrive for ``envelope_id``.

        Terminates once a chunk with ``final=True`` is received.
        Callers must call this *before* sending the envelope whose
        reply they expect to stream — otherwise a very fast sender
        can race the queue registration and lose the first chunks.
        """

        queue = self._client._chunk_queue_for(envelope_id)
        try:
            while True:
                chunk = await queue.get()
                yield chunk.content
                if chunk.final:
                    return
        finally:
            self._client._discard_chunk_queue(envelope_id)

    async def close(self) -> None:
        await self._client.close_session(self.session_id)

    def _default_recipient(self) -> str | None:
        participants = [
            p
            for p in self._metadata.participants
            if p.actor_id != self._client.actor_id
        ]
        if len(participants) == 1:
            return participants[0].actor_id
        return None
