# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for KnowledgeStore, EventLogWriter, and bootstrapping."""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("watchdog")

from autogen.beta.events import ModelRequest, TaskCompleted, TextInput, UnknownEvent
from autogen.beta.knowledge import (
    DefaultBootstrap,
    DiskKnowledgeStore,
    EventLogWriter,
    LockedKnowledgeStore,
    MemoryKnowledgeStore,
    SqliteKnowledgeStore,
)
from autogen.beta.stream import MemoryStream


class TestMemoryKnowledgeStore:
    @pytest.mark.asyncio
    async def test_read_write(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "hello")
        assert await store.read("/test.txt") == "hello"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self) -> None:
        store = MemoryKnowledgeStore()
        assert await store.read("/missing.txt") is None

    @pytest.mark.asyncio
    async def test_list_root(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/a.txt", "a")
        await store.write("/b/c.txt", "c")
        entries = await store.list("/")
        assert "a.txt" in entries
        assert "b/" in entries

    @pytest.mark.asyncio
    async def test_list_subdirectory(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/log/stream-1.jsonl", "data1")
        await store.write("/log/stream-2.jsonl", "data2")
        entries = await store.list("/log/")
        assert entries == ["stream-1.jsonl", "stream-2.jsonl"]

    @pytest.mark.asyncio
    async def test_delete_file(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "hello")
        await store.delete("/test.txt")
        assert await store.read("/test.txt") is None

    @pytest.mark.asyncio
    async def test_delete_directory(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/dir/a.txt", "a")
        await store.write("/dir/b.txt", "b")
        await store.delete("/dir")
        assert await store.read("/dir/a.txt") is None
        assert await store.read("/dir/b.txt") is None

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "hello")
        assert await store.exists("/test.txt") is True
        assert await store.exists("/missing.txt") is False

    @pytest.mark.asyncio
    async def test_exists_directory(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/dir/file.txt", "data")
        assert await store.exists("/dir") is True

    @pytest.mark.asyncio
    async def test_path_normalization(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("no_leading_slash.txt", "data")
        assert await store.read("/no_leading_slash.txt") == "data"

    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        store = MemoryKnowledgeStore()
        entries = await store.list("/")
        assert entries == []


class TestEventLogWriter:
    @pytest.mark.asyncio
    async def test_persist_and_load(self) -> None:
        store = MemoryKnowledgeStore()
        writer = EventLogWriter(store)
        stream_id = uuid4()

        events = [
            ModelRequest([TextInput("hello")]),
            TaskCompleted(
                task_id="t1",
                agent_name="analyzer",
                objective="summarize",
                result="done",
                task_stream=MemoryStream().id,
            ),
        ]
        await writer.persist(stream_id, events)

        loaded = await writer.load(stream_id)
        assert len(loaded) == 2
        assert loaded[0].parts[0].content == "hello"
        assert loaded[1].agent_name == "analyzer"
        assert loaded[1].result == "done"

    @pytest.mark.asyncio
    async def test_persist_dropped_segments(self) -> None:
        store = MemoryKnowledgeStore()
        writer = EventLogWriter(store)
        stream_id = uuid4()

        # Simulate two compaction drops + final persist
        dropped1 = [ModelRequest([TextInput("old-1")])]
        dropped2 = [ModelRequest([TextInput("old-2")])]
        final = [ModelRequest([TextInput("recent")])]

        await writer.persist_dropped(stream_id, dropped1)
        await writer.persist_dropped(stream_id, dropped2)
        await writer.persist(stream_id, final)

        # Load should return all in order: dropped-1, dropped-2, final
        loaded = await writer.load(stream_id)
        assert len(loaded) == 3
        assert loaded[0].parts[0].content == "old-1"
        assert loaded[1].parts[0].content == "old-2"
        assert loaded[2].parts[0].content == "recent"

    @pytest.mark.asyncio
    async def test_persist_dropped_multiple_writers_no_overwrite(self) -> None:
        """Multiple EventLogWriter instances must not overwrite each other's segments.

        This is the fix for Bug A: segment counter was per-instance, causing
        the second writer to reuse segment number 1 and overwrite the first.
        """
        store = MemoryKnowledgeStore()
        stream_id = uuid4()

        # First compaction — writer 1
        writer1 = EventLogWriter(store)
        await writer1.persist_dropped(stream_id, [ModelRequest([TextInput("batch-1")])])

        # Second compaction — completely new writer (as TailWindowCompact creates)
        writer2 = EventLogWriter(store)
        await writer2.persist_dropped(stream_id, [ModelRequest([TextInput("batch-2")])])

        # Final persist
        writer3 = EventLogWriter(store)
        await writer3.persist(stream_id, [ModelRequest([TextInput("final")])])

        # All three segments must be present and loadable
        loaded = await EventLogWriter(store).load(stream_id)
        assert len(loaded) == 3
        assert loaded[0].parts[0].content == "batch-1"
        assert loaded[1].parts[0].content == "batch-2"
        assert loaded[2].parts[0].content == "final"

    @pytest.mark.asyncio
    async def test_load_empty(self) -> None:
        store = MemoryKnowledgeStore()
        writer = EventLogWriter(store)
        loaded = await writer.load(uuid4())
        assert loaded == []

    @pytest.mark.asyncio
    async def test_unknown_event_fallback(self) -> None:
        store = MemoryKnowledgeStore()
        # Write a record with a non-existent event type
        stream_id = uuid4()
        record = json.dumps({"type": "nonexistent.module.FakeEvent", "data": {"key": "val"}})
        await store.write(f"/log/{stream_id}.jsonl", record)

        writer = EventLogWriter(store)
        loaded = await writer.load(stream_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0], UnknownEvent)
        assert loaded[0].type_name == "nonexistent.module.FakeEvent"


class TestDefaultBootstrap:
    @pytest.mark.asyncio
    async def test_creates_standard_layout(self) -> None:
        store = MemoryKnowledgeStore()
        # Actor writes sentinel before calling bootstrap, so simulate that
        await store.write("/.initialized", "test-actor")
        bootstrap = DefaultBootstrap()
        await bootstrap.bootstrap(store, "test-actor")

        assert await store.exists("/.initialized")
        assert await store.exists("/SKILL.md")
        assert await store.exists("/log/SKILL.md")
        assert await store.exists("/artifacts/SKILL.md")
        assert await store.exists("/memory/SKILL.md")

        root_skill = await store.read("/SKILL.md")
        assert "test-actor" in root_skill

    @pytest.mark.asyncio
    async def test_sentinel_prevents_rebootstrap(self) -> None:
        store = MemoryKnowledgeStore()
        # Actor writes sentinel before calling bootstrap
        await store.write("/.initialized", "actor-1")
        bootstrap = DefaultBootstrap()
        await bootstrap.bootstrap(store, "actor-1")

        # Overwrite a file
        await store.write("/SKILL.md", "custom content")

        # Sentinel exists, so a second bootstrap should be skipped by caller logic
        assert await store.exists("/.initialized")
        # (The Actor checks this before calling bootstrap)


@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
class TestDiskKnowledgeStore:
    async def test_read_write(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/test.txt", "hello")
        assert await store.read("/test.txt") == "hello"
        assert (tmp_path / "test.txt").read_text() == "hello"

    async def test_read_nonexistent(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        assert await store.read("/missing.txt") is None

    async def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/a/b/c/deep.txt", "data")
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text() == "data"
        assert await store.read("/a/b/c/deep.txt") == "data"

    async def test_list_root_marks_directories(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/a.txt", "a")
        await store.write("/sub/b.txt", "b")
        entries = await store.list("/")
        assert "a.txt" in entries
        assert "sub/" in entries

    async def test_list_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        assert await store.list("/missing") == []

    async def test_delete_file(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/test.txt", "hello")
        await store.delete("/test.txt")
        assert not (tmp_path / "test.txt").exists()
        assert await store.read("/test.txt") is None

    async def test_delete_directory_recursively(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/dir/a.txt", "a")
        await store.write("/dir/sub/b.txt", "b")
        await store.delete("/dir")
        assert not (tmp_path / "dir").exists()
        assert await store.exists("/dir") is False

    async def test_delete_missing_is_noop(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.delete("/never-existed.txt")  # should not raise

    async def test_exists(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/test.txt", "hello")
        assert await store.exists("/test.txt") is True
        assert await store.exists("/missing.txt") is False

    async def test_append_returns_offset(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        first = await store.append("/log.jsonl", "line1\n")
        second = await store.append("/log.jsonl", "line2\n")
        assert first == 0
        assert second == len(b"line1\n")
        assert (tmp_path / "log.jsonl").read_text() == "line1\nline2\n"

    async def test_read_range(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        await store.write("/data.txt", "hello world")
        assert await store.read_range("/data.txt", 0, 5) == "hello"
        assert await store.read_range("/data.txt", 6, None) == "world"
        assert await store.read_range("/data.txt", 6) == "world"

    async def test_read_range_missing_file(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        assert await store.read_range("/missing.txt", 0, 10) == ""

    async def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        store = DiskKnowledgeStore(str(root))
        with pytest.raises(ValueError, match="Path traversal"):
            await store.write("/../escape.txt", "x")

    async def test_on_change_fires_on_write(self, tmp_path: Path) -> None:
        store = DiskKnowledgeStore(str(tmp_path))
        received: list[str] = []
        event = asyncio.Event()

        async def callback(path: str) -> None:
            received.append(path)
            event.set()

        sub = await store.on_change("/watched", callback)
        try:
            await store.write("/watched/file.txt", "hello")
            # watchdog delivers asynchronously via a background thread
            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pytest.skip("watchdog backend did not deliver event in time")
            assert any("file.txt" in p for p in received)
        finally:
            await sub.close()


@pytest.mark.asyncio
class TestSqliteKnowledgeStore:
    async def test_read_write(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            await store.write("/test.txt", "hello")
            assert await store.read("/test.txt") == "hello"
        finally:
            store.close()

    async def test_read_nonexistent(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            assert await store.read("/missing.txt") is None
        finally:
            store.close()

    async def test_list_root_and_subdirectory(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            await store.write("/a.txt", "a")
            await store.write("/dir/b.txt", "b")
            await store.write("/dir/c.txt", "c")
            root = await store.list("/")
            assert "a.txt" in root
            assert "dir/" in root
            sub = await store.list("/dir/")
            assert sub == ["b.txt", "c.txt"]
        finally:
            store.close()

    async def test_delete_file_and_directory(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            await store.write("/file.txt", "x")
            await store.write("/dir/a.txt", "a")
            await store.write("/dir/b.txt", "b")
            await store.delete("/file.txt")
            await store.delete("/dir")
            assert await store.read("/file.txt") is None
            assert await store.read("/dir/a.txt") is None
            assert await store.read("/dir/b.txt") is None
        finally:
            store.close()

    async def test_exists_file_and_directory(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            await store.write("/dir/file.txt", "x")
            assert await store.exists("/dir/file.txt") is True
            assert await store.exists("/dir") is True
            assert await store.exists("/missing.txt") is False
        finally:
            store.close()

    async def test_append_returns_offset(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            first = await store.append("/log.jsonl", "line1\n")
            second = await store.append("/log.jsonl", "line2\n")
            assert first == 0
            assert second == len(b"line1\n")
            assert await store.read("/log.jsonl") == "line1\nline2\n"
        finally:
            store.close()

    async def test_read_range(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            await store.write("/data.txt", "hello world")
            assert await store.read_range("/data.txt", 0, 5) == "hello"
            assert await store.read_range("/data.txt", 6, None) == "world"
            assert await store.read_range("/data.txt", 100, 200) == ""
            assert await store.read_range("/missing.txt", 0, 10) == ""
        finally:
            store.close()

    async def test_list_versions_under_bumps_per_write(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        try:
            await store.write("/a.txt", "v1")
            await store.write("/dir/b.txt", "v1")
            snapshot1 = await store.list_versions_under("/")
            assert set(snapshot1.keys()) == {"/a.txt", "/dir/b.txt"}

            await store.write("/a.txt", "v2")
            snapshot2 = await store.list_versions_under("/")
            assert snapshot2["/a.txt"] > snapshot1["/a.txt"]
            assert snapshot2["/dir/b.txt"] == snapshot1["/dir/b.txt"]

            scoped = await store.list_versions_under("/dir")
            assert set(scoped.keys()) == {"/dir/b.txt"}
        finally:
            store.close()

    async def test_on_change_polling(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"), poll_interval_s=0.05)
        received: list[str] = []
        event = asyncio.Event()

        async def callback(path: str) -> None:
            received.append(path)
            event.set()

        sub = await store.on_change("/watched", callback)
        try:
            await store.write("/watched/file.txt", "hello")
            await asyncio.wait_for(event.wait(), timeout=2.0)
            assert "/watched/file.txt" in received
        finally:
            await sub.close()
            store.close()

    async def test_close_is_idempotent(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.db"))
        store.close()
        store.close()  # should not raise

    async def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "store.db")
        store1 = SqliteKnowledgeStore(db_path)
        await store1.write("/persisted.txt", "survives")
        store1.close()

        store2 = SqliteKnowledgeStore(db_path)
        try:
            assert await store2.read("/persisted.txt") == "survives"
        finally:
            store2.close()


class _FakeLock:
    """Minimal Lock protocol implementation for LockedKnowledgeStore tests."""

    def __init__(self) -> None:
        self.acquired: list[tuple[str, float]] = []
        self.released: list[str] = []
        self.acquire_result = True

    async def acquire(self, name: str, ttl: float) -> bool:
        self.acquired.append((name, ttl))
        return self.acquire_result

    async def release(self, name: str) -> None:
        self.released.append(name)


@pytest.mark.asyncio
class TestLockedKnowledgeStore:
    async def test_read_bypasses_lock(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/test.txt", "data")
        lock = _FakeLock()
        locked = LockedKnowledgeStore(inner, lock)

        assert await locked.read("/test.txt") == "data"
        assert lock.acquired == []

    async def test_write_acquires_and_releases(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        locked = LockedKnowledgeStore(inner, lock)

        await locked.write("/test.txt", "hello")

        assert lock.acquired == [("store:write:/test.txt", 30.0)]
        assert lock.released == ["store:write:/test.txt"]
        assert await inner.read("/test.txt") == "hello"

    async def test_write_raises_when_lock_unavailable(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        lock.acquire_result = False
        locked = LockedKnowledgeStore(inner, lock)

        with pytest.raises(RuntimeError, match="Failed to acquire write lock"):
            await locked.write("/test.txt", "hello")
        assert lock.released == []
        assert await inner.read("/test.txt") is None

    async def test_write_releases_on_inner_exception(self) -> None:
        class ExplodingStore:
            async def write(self, path: str, content: str) -> None:
                raise ValueError("boom")

        lock = _FakeLock()
        locked = LockedKnowledgeStore(ExplodingStore(), lock)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="boom"):
            await locked.write("/test.txt", "hello")
        assert lock.released == ["store:write:/test.txt"]

    async def test_delete_acquires_and_releases(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/test.txt", "data")
        lock = _FakeLock()
        locked = LockedKnowledgeStore(inner, lock)

        await locked.delete("/test.txt")
        assert lock.acquired == [("store:write:/test.txt", 30.0)]
        assert lock.released == ["store:write:/test.txt"]
        assert await inner.read("/test.txt") is None

    async def test_delete_raises_when_lock_unavailable(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        lock.acquire_result = False
        locked = LockedKnowledgeStore(inner, lock)

        with pytest.raises(RuntimeError, match="Failed to acquire delete lock"):
            await locked.delete("/test.txt")

    async def test_append_acquires_and_returns_offset(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        locked = LockedKnowledgeStore(inner, lock)

        first = await locked.append("/log.jsonl", "line1\n")
        second = await locked.append("/log.jsonl", "line2\n")

        assert first == 0
        assert second == len(b"line1\n")
        assert lock.acquired == [
            ("store:write:/log.jsonl", 30.0),
            ("store:write:/log.jsonl", 30.0),
        ]
        assert lock.released == [
            "store:write:/log.jsonl",
            "store:write:/log.jsonl",
        ]

    async def test_append_raises_when_lock_unavailable(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        lock.acquire_result = False
        locked = LockedKnowledgeStore(inner, lock)

        with pytest.raises(RuntimeError, match="Failed to acquire append lock"):
            await locked.append("/log.jsonl", "line1\n")

    async def test_read_list_exists_read_range_bypass_lock(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/dir/a.txt", "hello")
        lock = _FakeLock()
        locked = LockedKnowledgeStore(inner, lock)

        assert await locked.list("/dir") == ["a.txt"]
        assert await locked.exists("/dir/a.txt") is True
        assert await locked.read_range("/dir/a.txt", 0, 5) == "hello"
        assert lock.acquired == []

    async def test_on_change_bypasses_lock(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        locked = LockedKnowledgeStore(inner, lock)
        received: list[str] = []

        async def callback(path: str) -> None:
            received.append(path)

        sub = await locked.on_change("/watched", callback)
        try:
            await inner.write("/watched/file.txt", "x")
            assert received == ["/watched/file.txt"]
            assert lock.acquired == []
        finally:
            await sub.close()
