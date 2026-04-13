# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the V3 KnowledgeStore extensions: append / read_range / watch.

These tests run against both ``MemoryKnowledgeStore`` and
``DiskKnowledgeStore`` so the two backends are held to the same contract.
``LockedKnowledgeStore`` is included to verify the locking wrapper proxies
the new methods correctly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autogen.beta.knowledge import (
    DiskKnowledgeStore,
    KnowledgeStore,
    LockedKnowledgeStore,
    MemoryKnowledgeStore,
)


class _StubLock:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, key: str, ttl: float = 30.0) -> bool:
        lock = self._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        return True

    async def release(self, key: str) -> None:
        lock = self._locks.get(key)
        if lock is not None and lock.locked():
            lock.release()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mem() -> MemoryKnowledgeStore:
    return MemoryKnowledgeStore()


def _disk(tmp_path: Path) -> DiskKnowledgeStore:
    return DiskKnowledgeStore(str(tmp_path / "store"))


def _locked_mem() -> LockedKnowledgeStore:
    return LockedKnowledgeStore(MemoryKnowledgeStore(), _StubLock())


@pytest.fixture(params=["mem", "disk", "locked_mem"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> KnowledgeStore:
    kind = request.param
    if kind == "mem":
        return _mem()
    if kind == "disk":
        return _disk(tmp_path)
    if kind == "locked_mem":
        return _locked_mem()
    raise RuntimeError(f"unknown backend {kind}")


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


def test_append_creates_file_and_returns_zero(store: KnowledgeStore) -> None:
    offset = asyncio.run(store.append("/wal/session.jsonl", "first line\n"))
    assert offset == 0


def test_append_returns_byte_offset_of_each_write(store: KnowledgeStore) -> None:
    async def run() -> list[int]:
        return [
            await store.append("/wal/session.jsonl", "aaa\n"),
            await store.append("/wal/session.jsonl", "bb\n"),
            await store.append("/wal/session.jsonl", "cccc\n"),
        ]

    offsets = asyncio.run(run())
    assert offsets == [0, 4, 7]  # 4 + 3 + 5 bytes cumulative


def test_append_content_accumulates(store: KnowledgeStore) -> None:
    async def run() -> str | None:
        await store.append("/log.txt", "a")
        await store.append("/log.txt", "b")
        await store.append("/log.txt", "c")
        return await store.read("/log.txt")

    assert asyncio.run(run()) == "abc"


def test_append_handles_unicode_byte_offset(store: KnowledgeStore) -> None:
    async def run() -> list[int]:
        first = await store.append("/log.txt", "héllo")  # 'é' is 2 bytes in UTF-8
        second = await store.append("/log.txt", "!")
        return [first, second]

    first, second = asyncio.run(run())
    assert first == 0
    assert second == 6  # len("héllo".encode()) == 6


def test_append_is_atomic_under_concurrency(store: KnowledgeStore) -> None:
    async def run() -> None:
        async def writer(tag: str) -> None:
            for i in range(20):
                await store.append("/race.log", f"{tag}:{i}\n")

        await asyncio.gather(writer("A"), writer("B"), writer("C"))

    asyncio.run(run())
    final = asyncio.run(store.read("/race.log"))
    assert final is not None
    lines = [line for line in final.split("\n") if line]
    assert len(lines) == 60
    # Each line must be intact (no interleaving mid-line).
    for line in lines:
        tag, index = line.split(":")
        assert tag in ("A", "B", "C")
        assert 0 <= int(index) < 20


# ---------------------------------------------------------------------------
# read_range
# ---------------------------------------------------------------------------


def test_read_range_returns_slice(store: KnowledgeStore) -> None:
    async def run() -> str:
        await store.append("/log.txt", "hello world")
        return await store.read_range("/log.txt", 6, 11)

    assert asyncio.run(run()) == "world"


def test_read_range_without_end_reads_to_eof(store: KnowledgeStore) -> None:
    async def run() -> str:
        await store.append("/log.txt", "abcdef")
        return await store.read_range("/log.txt", 2)

    assert asyncio.run(run()) == "cdef"


def test_read_range_missing_file_returns_empty(store: KnowledgeStore) -> None:
    assert asyncio.run(store.read_range("/nope.txt", 0)) == ""


def test_read_range_start_beyond_file_returns_empty(store: KnowledgeStore) -> None:
    async def run() -> str:
        await store.append("/a.txt", "short")
        return await store.read_range("/a.txt", 100)

    assert asyncio.run(run()) == ""


def test_append_and_read_range_emulate_wal_tailer(store: KnowledgeStore) -> None:
    """Simulates how a session subscriber tails a WAL: append, read_range
    from the last cursor, advance the cursor by the bytes consumed."""

    async def tailer() -> list[str]:
        lines: list[str] = []
        cursor = 0
        await store.append("/wal.jsonl", "one\n")
        chunk = await store.read_range("/wal.jsonl", cursor)
        cursor += len(chunk.encode("utf-8"))
        lines.extend(x for x in chunk.split("\n") if x)
        await store.append("/wal.jsonl", "two\n")
        chunk = await store.read_range("/wal.jsonl", cursor)
        cursor += len(chunk.encode("utf-8"))
        lines.extend(x for x in chunk.split("\n") if x)
        return lines

    assert asyncio.run(tailer()) == ["one", "two"]


# ---------------------------------------------------------------------------
# on_change
# ---------------------------------------------------------------------------


def test_memory_on_change_fires_on_append() -> None:
    store = MemoryKnowledgeStore()
    events: list[str] = []

    async def run() -> None:
        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/wal", cb)
        await store.append("/wal/a.jsonl", "hi\n")
        await store.append("/wal/b.jsonl", "hi\n")
        await sub.close()

    asyncio.run(run())
    assert events == ["/wal/a.jsonl", "/wal/b.jsonl"]


def test_memory_on_change_close_stops_delivery() -> None:
    store = MemoryKnowledgeStore()
    events: list[str] = []

    async def run() -> None:
        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/wal", cb)
        await store.append("/wal/a.jsonl", "hi\n")
        await sub.close()
        await store.append("/wal/b.jsonl", "hi\n")

    asyncio.run(run())
    assert events == ["/wal/a.jsonl"]


def test_memory_on_change_write_also_notifies() -> None:
    store = MemoryKnowledgeStore()
    events: list[str] = []

    async def run() -> None:
        async def cb(path: str) -> None:
            events.append(path)

        await store.on_change("/rules", cb)
        await store.write("/rules/actor-1.json", "{}")

    asyncio.run(run())
    assert events == ["/rules/actor-1.json"]


# ---------------------------------------------------------------------------
# on_change — DiskKnowledgeStore (Phase 3a watchdog bridge)
# ---------------------------------------------------------------------------


async def _wait_until(
    predicate: "callable",  # type: ignore[valid-type]
    *,
    timeout: float = 2.0,
    interval: float = 0.02,
) -> None:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses.

    Watchdog events cross a thread boundary — the native backend
    delivers events on a background thread and they are bridged to the
    main asyncio loop via ``run_coroutine_threadsafe``. Tests need a
    bounded wait loop instead of a fixed sleep to stay stable on CI.
    """

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"predicate {predicate!r} did not become true in {timeout}s")


class TestDiskOnChange:
    """Phase 3a real ``on_change`` delivery on ``DiskKnowledgeStore``."""

    @pytest.mark.asyncio
    async def test_fires_on_append(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/wal", cb)
        try:
            await store.append("/wal/a.jsonl", "hi\n")
            await _wait_until(lambda: any("a.jsonl" in e for e in events))
            assert any(e.endswith("/wal/a.jsonl") for e in events)
        finally:
            await sub.close()

    @pytest.mark.asyncio
    async def test_fires_on_write(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/rules", cb)
        try:
            await store.write("/rules/actor-1.json", "{}")
            await _wait_until(lambda: any("actor-1.json" in e for e in events))
            assert any(e.endswith("/rules/actor-1.json") for e in events)
        finally:
            await sub.close()

    @pytest.mark.asyncio
    async def test_multiple_files_under_same_prefix(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/wal", cb)
        try:
            await store.append("/wal/a.jsonl", "a\n")
            await store.append("/wal/b.jsonl", "b\n")
            await store.append("/wal/c.jsonl", "c\n")
            await _wait_until(
                lambda: all(
                    any(e.endswith(name) for e in events)
                    for name in ("a.jsonl", "b.jsonl", "c.jsonl")
                )
            )
        finally:
            await sub.close()

    @pytest.mark.asyncio
    async def test_fires_on_nested_subdirectory(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/hub", cb)
        try:
            await store.append("/hub/sessions/abc/wal.jsonl", "x\n")
            await _wait_until(lambda: any("wal.jsonl" in e for e in events))
            assert any(e.endswith("/hub/sessions/abc/wal.jsonl") for e in events)
        finally:
            await sub.close()

    @pytest.mark.asyncio
    async def test_close_stops_delivery(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/wal", cb)
        await store.append("/wal/a.jsonl", "a\n")
        await _wait_until(lambda: any("a.jsonl" in e for e in events))
        before = len(events)
        await sub.close()
        # Subsequent writes must not be observed once the sub is closed.
        await store.append("/wal/b.jsonl", "b\n")
        await asyncio.sleep(0.2)
        assert not any("b.jsonl" in e for e in events)
        assert len(events) == before

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))

        async def cb(_: str) -> None:
            return None

        sub = await store.on_change("/wal", cb)
        await sub.close()
        await sub.close()  # second close must not raise

    @pytest.mark.asyncio
    async def test_events_outside_prefix_are_filtered(
        self, tmp_path: Path
    ) -> None:
        """A subscription on ``/wal`` must not fire for writes under ``/rules``."""

        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        # Pre-create /rules so an observer on /wal doesn't accidentally
        # see it created at watch time.
        await store.append("/rules/seed.json", "{}")
        sub = await store.on_change("/wal", cb)
        try:
            await store.append("/wal/in.jsonl", "in\n")
            await store.write("/rules/other.json", "{}")
            await _wait_until(lambda: any("in.jsonl" in e for e in events))
            # Let the cross-prefix event have a chance to land.
            await asyncio.sleep(0.15)
            assert not any("other.json" in e for e in events)
        finally:
            await sub.close()

    @pytest.mark.asyncio
    async def test_two_subscriptions_fire_independently(
        self, tmp_path: Path
    ) -> None:
        store = DiskKnowledgeStore(str(tmp_path / "disk"))
        a_events: list[str] = []
        b_events: list[str] = []

        async def cb_a(path: str) -> None:
            a_events.append(path)

        async def cb_b(path: str) -> None:
            b_events.append(path)

        sub_a = await store.on_change("/a", cb_a)
        sub_b = await store.on_change("/b", cb_b)
        try:
            await store.append("/a/one.jsonl", "1\n")
            await store.append("/b/two.jsonl", "2\n")
            await _wait_until(lambda: a_events and b_events)
            assert any("one.jsonl" in e for e in a_events)
            assert not any("one.jsonl" in e for e in b_events)
            assert any("two.jsonl" in e for e in b_events)
            assert not any("two.jsonl" in e for e in a_events)
        finally:
            await sub_a.close()
            await sub_b.close()

    @pytest.mark.asyncio
    async def test_virtual_path_is_store_relative_not_physical(
        self, tmp_path: Path
    ) -> None:
        """Callback must receive the store-relative path, not the OS path."""

        root = tmp_path / "deeply/nested/store"
        store = DiskKnowledgeStore(str(root))
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await store.on_change("/wal", cb)
        try:
            await store.append("/wal/x.jsonl", "x\n")
            await _wait_until(lambda: any("x.jsonl" in e for e in events))
            # No event should leak the physical tmp_path prefix.
            for e in events:
                assert str(root) not in e
                assert e.startswith("/wal/")
        finally:
            await sub.close()

    @pytest.mark.asyncio
    async def test_locked_disk_store_proxies_on_change(
        self, tmp_path: Path
    ) -> None:
        """LockedKnowledgeStore wrapping a Disk store proxies subscriptions."""

        inner = DiskKnowledgeStore(str(tmp_path / "disk"))
        locked = LockedKnowledgeStore(inner, _StubLock())
        events: list[str] = []

        async def cb(path: str) -> None:
            events.append(path)

        sub = await locked.on_change("/wal", cb)
        try:
            await locked.append("/wal/a.jsonl", "a\n")
            await _wait_until(lambda: any("a.jsonl" in e for e in events))
            assert any(e.endswith("/wal/a.jsonl") for e in events)
        finally:
            await sub.close()
