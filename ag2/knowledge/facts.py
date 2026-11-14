# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Facts — the knowledge store's records with an explicit lifecycle.

A :class:`Fact` is one durable statement about the world (a preference, an
instruction, an event) with provenance and state. Unlike the working-memory
blob, facts are individually addressable: a corrected preference *supersedes*
its stale version instead of silently coexisting with it, an expired fact is
*forgotten*, and access statistics can inform later curation. Lifecycle states:
``active``, ``superseded`` (kept for provenance, never injected), ``forgotten``
(expired or explicitly dropped, never injected).

:class:`FactStore` persists one JSON file per fact under
:data:`~ag2.knowledge.FACTS_PREFIX` through any :class:`~ag2.knowledge.KnowledgeStore`.
Writes are whole-file replacements, so a fact's file is always internally
consistent; the one multi-write sequence — supersede — has a defined partial
state that :meth:`FactStore.repair` completes (see its docstring, and ADR 0036).
"""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from ag2.knowledge.base import KnowledgeStore
from ag2.knowledge.constants import FACTS_PREFIX

__all__ = ("Fact", "FactKind", "FactSource", "FactStatus", "FactStore")

logger = logging.getLogger(__name__)

FactKind = Literal["preference", "fact", "instruction", "event"]
FactStatus = Literal["active", "superseded", "forgotten"]


def new_fact_id() -> str:
    """A fresh fact identifier — uuid4 hex, the naming scheme of the fact files."""
    return uuid4().hex


class FactSource(BaseModel):
    """Where a fact came from.

    Attributes:
        stream_id: Stream the fact was extracted from, when known.
        episode_path: Conversation summary the fact is grounded in, when known.
        event_ids: Identifiers of the source events, where the events carry
            one — partial provenance is recorded, not invented.
    """

    stream_id: str | None = None
    episode_path: str | None = None
    event_ids: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    """One durable statement with identity, provenance, and lifecycle state.

    Attributes:
        id: Stable identifier (uuid4 hex); names the fact's store file.
        text: The statement itself, standalone — readable without its context.
        subject: The entity the statement is about, when there is one; updates
            and supersession match on it.
        kind: What kind of statement this is.
        status: Lifecycle state — only ``active`` facts are ever injected.
        supersedes: Ids of facts this one replaces.
        superseded_by: Id of the fact that replaced this one, when superseded.
        source: Where the fact came from.
        confidence: Extractor's confidence in ``[0, 1]``.
        created_at: When the fact was first written.
        updated_at: When the fact was last written (any field, including access).
        expires_at: When the fact stops holding, if known; :meth:`FactStore.expire`
            forgets it past this point.
        access_count: How many times the fact was touched since creation.
        last_accessed_at: When it was last touched, if ever.
    """

    id: str = Field(default_factory=new_fact_id)
    text: str
    subject: str | None = None
    kind: FactKind = "fact"
    status: FactStatus = "active"
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    source: FactSource = Field(default_factory=FactSource)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: datetime | None = None


class FactStore:
    """Persists facts as one JSON file per fact through a KnowledgeStore.

    Args:
        store: The knowledge store to persist through.
        prefix: Directory the fact files live under.
        track_access: When True, :meth:`touch` records access statistics;
            when False (the default) it is a no-op, so retrieval paths never
            write merely by having retrieved.
        access_flush_threshold: How many pending touches accumulate before
            they flush to the store without an explicit :meth:`flush_access`.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        prefix: str = FACTS_PREFIX,
        *,
        track_access: bool = False,
        access_flush_threshold: int = 32,
    ) -> None:
        self._store = store
        self._prefix = prefix if prefix.endswith("/") else f"{prefix}/"
        self._track_access = track_access
        self._access_flush_threshold = max(1, access_flush_threshold)
        self._pending_access: set[str] = set()

    def path_for(self, fact_id: str) -> str:
        """Store path of one fact's file."""
        return f"{self._prefix}{fact_id}.json"

    def fact_id(self, path: str) -> str | None:
        """The fact id a store path under this store's prefix holds; ``None`` otherwise."""
        if not path.startswith(self._prefix) or not path.endswith(".json"):
            return None
        return path[len(self._prefix) : -len(".json")]

    async def put(self, fact: Fact) -> None:
        """Write one fact's file in full."""
        await self._store.write(self.path_for(fact.id), fact.model_dump_json())

    async def get(self, fact_id: str) -> Fact | None:
        """Read one fact; ``None`` when the file is missing or unreadable."""
        raw = await self._store.read(self.path_for(fact_id))
        if raw is None:
            return None
        try:
            return Fact.model_validate_json(raw)
        except ValueError as exc:
            logger.warning("Unreadable fact file %s: %s", self.path_for(fact_id), exc)
            return None

    async def active(self) -> list[Fact]:
        """Every ``active`` fact, oldest first — the injectable set."""
        return [fact for fact in await self._load_all() if fact.status == "active"]

    async def supersede(self, old_id: str, new_fact: Fact) -> Fact:
        """Replace ``old_id`` with ``new_fact``, keeping the old record for provenance.

        The new fact is written *first*, the old one marked superseded second —
        the order that leaves a readable store at every intermediate point. If
        the process dies between the two writes, the new fact exists while the
        old one is still ``active``: a one-sided link that :meth:`repair`
        completes. A ``superseded_by`` link pointing at a file that never got
        written is the mirror case, also fixed by :meth:`repair`.
        """
        if old_id not in new_fact.supersedes:
            new_fact.supersedes = [*new_fact.supersedes, old_id]
        await self.put(new_fact)
        old = await self.get(old_id)
        if old is None:
            logger.warning("Superseding %s, which has no fact file; repair() will clear the link", old_id)
            return new_fact
        old.status = "superseded"
        old.superseded_by = new_fact.id
        old.updated_at = datetime.now(UTC)
        await self.put(old)
        return new_fact

    async def forget(self, fact_id: str, reason: str) -> bool:
        """Mark a fact ``forgotten`` — it stays on disk, never injected again.

        Returns ``False`` when there is no such fact. ``reason`` is recorded in
        the log line, not on the fact: the lifecycle state is the durable fact,
        the reason is operational context.
        """
        fact = await self.get(fact_id)
        if fact is None:
            return False
        fact.status = "forgotten"
        fact.updated_at = datetime.now(UTC)
        await self.put(fact)
        logger.info("Fact %s forgotten: %s", fact_id, reason)
        return True

    async def expire(self, now: datetime) -> list[str]:
        """Forget every active fact whose ``expires_at`` has passed; returns their ids."""
        expired: list[str] = []
        for fact in await self._load_all():
            if fact.status == "active" and fact.expires_at is not None and fact.expires_at <= now:
                fact.status = "forgotten"
                fact.updated_at = now
                await self.put(fact)
                expired.append(fact.id)
        return expired

    async def repair(self) -> list[str]:
        """Fix one-sided supersede links; returns the ids it rewrote.

        Two partial states are repaired:

        * a fact ``F`` lists ``X`` in ``supersedes`` while ``X`` is still
          ``active`` — the supersede's second write never ran: ``X`` is marked
          ``superseded`` by ``F``;
        * a fact references, in ``supersedes`` or ``superseded_by``, a file
          that does not exist — the referenced fact never materialized: the
          dangling reference is dropped, and a fact left ``superseded`` by it
          returns to ``active``.
        """
        all_facts = await self._load_all()
        by_id = {fact.id: fact for fact in all_facts}
        now = datetime.now(UTC)
        repaired: set[str] = set()

        for fact in all_facts:
            kept: list[str] = []
            for referenced in fact.supersedes:
                target = by_id.get(referenced)
                if target is None:
                    repaired.add(fact.id)
                    continue
                if target.status == "active":
                    target.status = "superseded"
                    target.superseded_by = fact.id
                    target.updated_at = now
                    repaired.add(target.id)
                kept.append(referenced)
            if len(kept) != len(fact.supersedes):
                fact.supersedes = kept
                fact.updated_at = now
                repaired.add(fact.id)

            if fact.superseded_by is not None and fact.superseded_by not in by_id:
                fact.superseded_by = None
                if fact.status == "superseded":
                    fact.status = "active"
                fact.updated_at = now
                repaired.add(fact.id)

        for fact_id in sorted(repaired):
            fact = by_id[fact_id]
            await self.put(fact)
        return sorted(repaired)

    async def touch(self, fact_ids: Iterable[str]) -> None:
        """Record one access of each fact — batched, not written immediately.

        Touches accumulate in a pending buffer and flush when it reaches
        ``access_flush_threshold``; call :meth:`flush_access` to force the
        write (e.g. at the end of a retrieval-heavy turn). A no-op unless
        ``track_access`` was set at construction.
        """
        if not self._track_access:
            return
        self._pending_access.update(fact_ids)
        if len(self._pending_access) >= self._access_flush_threshold:
            await self.flush_access()

    async def flush_access(self) -> int:
        """Write the pending access statistics; returns how many facts were written."""
        pending = self._pending_access
        self._pending_access = set()
        now = datetime.now(UTC)
        written = 0
        for fact_id in sorted(pending):
            fact = await self.get(fact_id)
            if fact is None:
                continue
            fact.access_count += 1
            fact.last_accessed_at = now
            await self.put(fact)
            written += 1
        return written

    async def _load_all(self) -> list[Fact]:
        """Every fact file under the prefix, oldest first; unreadable files are skipped."""
        facts: list[Fact] = []
        for entry in await self._store.list(self._prefix):
            if not entry.endswith(".json"):
                continue
            fact = await self.get(entry[: -len(".json")])
            if fact is not None:
                facts.append(fact)
        facts.sort(key=lambda fact: (fact.created_at, fact.id))
        return facts
