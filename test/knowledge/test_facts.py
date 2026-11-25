# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Fact and FactStore: persistence, lifecycle transitions, repair,
expiry, and access tracking."""

from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.knowledge.facts import Fact, FactStore


def _fact(
    text: str,
    *,
    subject: str = "user",
    kind: Literal["preference", "fact", "instruction", "event"] = "preference",
    **kwargs,
) -> Fact:
    return Fact(text=text, subject=subject, kind=kind, **kwargs)


@pytest.mark.asyncio
class TestFactPersistence:
    async def test_put_and_get_round_trip(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        fact = _fact("user likes tea", expires_at=datetime(2027, 1, 1, tzinfo=UTC), confidence=0.8)

        await facts.put(fact)
        loaded = await facts.get(fact.id)

        assert loaded == fact
        assert loaded.expires_at == datetime(2027, 1, 1, tzinfo=UTC)

    async def test_get_missing_returns_none(self) -> None:
        facts = FactStore(MemoryKnowledgeStore())

        assert await facts.get("nope") is None

    async def test_fact_file_lives_under_the_prefix_and_holds_json(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        fact = _fact("user likes tea")

        await facts.put(fact)

        path = facts.path_for(fact.id)
        assert path.startswith("/memory/facts/")
        assert path.endswith(".json")
        assert '"user likes tea"' in await store.read(path)

    async def test_fact_id_extracts_from_paths(self) -> None:
        facts = FactStore(MemoryKnowledgeStore())

        assert facts.fact_id("/memory/facts/abc.json") == "abc"
        assert facts.fact_id("/memory/conversations/abc.md") is None

    async def test_unreadable_fact_file_returns_none(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        await store.write(facts.path_for("broken"), "not json")

        assert await facts.get("broken") is None

    async def test_active_lists_only_active_facts(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        active = _fact("active one")
        superseded = _fact("old one", status="superseded")
        forgotten = _fact("dropped one", status="forgotten")
        for fact in (active, superseded, forgotten):
            await facts.put(fact)

        assert [fact.id for fact in await facts.active()] == [active.id]


@pytest.mark.asyncio
class TestSupersede:
    async def test_supersede_marks_the_old_fact_and_links_both_ways(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        old = await _stored(facts, _fact("user likes tea"))
        replacement = _fact("user prefers coffee")

        await facts.supersede(old.id, replacement)

        stored_old = await facts.get(old.id)
        assert stored_old.status == "superseded"
        assert stored_old.superseded_by == replacement.id
        assert replacement.supersedes == [old.id]
        assert [fact.id for fact in await facts.active()] == [replacement.id]

    async def test_supersede_of_a_missing_old_fact_leaves_a_link_for_repair(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        replacement = _fact("user prefers coffee")

        await facts.supersede("ghost", replacement)

        assert replacement.supersedes == ["ghost"]
        assert [fact.id for fact in await facts.active()] == [replacement.id]

        await facts.repair()
        assert (await facts.get(replacement.id)).supersedes == []


@pytest.mark.asyncio
class TestRepair:
    async def test_repair_fixes_a_crash_between_the_two_supersede_writes(self) -> None:
        store = _CrashOnSecondWriteStore()
        facts = FactStore(store)
        old = _fact("user likes tea")
        await facts.put(old)
        replacement = _fact("user prefers coffee")
        store.arm()  # the next two writes are supersede's: new fact, then the old one

        with pytest.raises(OSError, match="crashed"):
            await facts.supersede(old.id, replacement)  # dies before marking the old fact

        # the partial state: the new fact is on disk and linked, the old one is still active
        assert (await facts.get(old.id)).status == "active"
        assert (await facts.get(replacement.id)).supersedes == [old.id]

        repaired = await facts.repair()

        assert old.id in repaired
        stored_old = await facts.get(old.id)
        assert stored_old.status == "superseded"
        assert stored_old.superseded_by == replacement.id
        stored_replacement = await facts.get(replacement.id)
        assert stored_replacement.status == "active"
        assert stored_replacement.supersedes == [old.id]
        assert [fact.id for fact in await facts.active()] == [replacement.id]

    async def test_repair_completes_a_one_sided_supersede(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        old = await _stored(facts, _fact("user likes tea"))
        # The crash between the two writes: the new fact is on disk, the old one
        # was never marked.
        replacement = _fact("user prefers coffee", supersedes=[old.id])
        await facts.put(replacement)

        repaired = await facts.repair()

        assert old.id in repaired
        stored_old = await facts.get(old.id)
        assert stored_old.status == "superseded"
        assert stored_old.superseded_by == replacement.id

    async def test_repair_clears_a_link_to_a_missing_file(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        orphan = await _stored(facts, _fact("user likes tea", status="superseded", superseded_by="ghost"))

        repaired = await facts.repair()

        assert orphan.id in repaired
        stored = await facts.get(orphan.id)
        assert stored.status == "active"
        assert stored.superseded_by is None

    async def test_repair_drops_a_supersedes_reference_to_a_missing_file(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        fact = await _stored(facts, _fact("user prefers coffee", supersedes=["ghost"]))

        repaired = await facts.repair()

        assert fact.id in repaired
        assert (await facts.get(fact.id)).supersedes == []

    async def test_repair_is_idempotent(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        old = await _stored(facts, _fact("user likes tea"))
        await facts.put(_fact("user prefers coffee", supersedes=[old.id]))

        first = await facts.repair()
        second = await facts.repair()

        assert first and not second


@pytest.mark.asyncio
class TestExpiryAndForget:
    async def test_expire_forgets_only_past_expiries(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        past = await _stored(facts, _fact("expired already", expires_at=datetime.now(UTC) - timedelta(hours=1)))
        future = await _stored(facts, _fact("still valid", expires_at=datetime.now(UTC) + timedelta(days=1)))
        never = await _stored(facts, _fact("no expiry"))

        expired = await facts.expire(datetime.now(UTC))

        assert expired == [past.id]
        assert (await facts.get(past.id)).status == "forgotten"
        assert (await facts.get(future.id)).status == "active"
        assert (await facts.get(never.id)).status == "active"

    async def test_forget_marks_and_reports_missing(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        fact = await _stored(facts, _fact("temporary"))

        assert await facts.forget(fact.id, "superseded manually") is True
        assert (await facts.get(fact.id)).status == "forgotten"
        assert await facts.forget("ghost", "nowhere") is False


@pytest.mark.asyncio
class TestTouch:
    async def test_touch_without_track_access_never_writes(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store)
        fact = await _stored(facts, _fact("user likes tea"))

        await facts.touch([fact.id])

        stored = await facts.get(fact.id)
        assert stored.access_count == 0
        assert stored.last_accessed_at is None

    async def test_touch_batches_until_flush(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store, track_access=True)
        first = await _stored(facts, _fact("one"))
        second = await _stored(facts, _fact("two"))

        await facts.touch([first.id])
        stored = await facts.get(first.id)
        assert stored.access_count == 0  # batched, not yet written

        written = await facts.flush_access()

        assert written == 1
        stored = await facts.get(first.id)
        assert stored.access_count == 1
        assert stored.last_accessed_at is not None
        assert (await facts.get(second.id)).access_count == 0

    async def test_touch_flushes_itself_at_the_threshold(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store, track_access=True, access_flush_threshold=2)
        one = await _stored(facts, _fact("one"))
        two = await _stored(facts, _fact("two"))

        await facts.touch([one.id])
        assert (await facts.get(one.id)).access_count == 0
        await facts.touch([one.id, two.id])  # crosses the threshold of 2 pending

        assert (await facts.get(one.id)).access_count == 1
        assert (await facts.get(two.id)).access_count == 1

    async def test_repeated_touches_count_once_per_flush(self) -> None:
        store = MemoryKnowledgeStore()
        facts = FactStore(store, track_access=True)
        fact = await _stored(facts, _fact("one"))

        await facts.touch([fact.id, fact.id, fact.id])
        await facts.flush_access()

        assert (await facts.get(fact.id)).access_count == 1


async def _stored(facts: FactStore, fact: Fact) -> Fact:
    await facts.put(fact)
    return fact


class _CrashOnSecondWriteStore(MemoryKnowledgeStore):
    """A store that dies on the second write after :meth:`arm` — supersede's second put.

    ``arm()`` keeps the setup writes (writing the facts themselves) from
    consuming the crash, so the kill lands exactly between the two writes
    :meth:`FactStore.supersede` performs.
    """

    def __init__(self) -> None:
        super().__init__()
        self._armed = False
        self._writes_since_arm = 0

    def arm(self) -> None:
        self._armed = True
        self._writes_since_arm = 0

    async def write(self, path: str, content: str) -> None:
        if self._armed:
            self._writes_since_arm += 1
            if self._writes_since_arm == 2:
                raise OSError("crashed between the two writes")
        await super().write(path, content)
