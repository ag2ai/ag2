# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SqliteKnowledgeStore`` — the Phase 3b Sqlite backend.

The cross-backend contract tests live in
``test_knowledge_store_extensions.py`` (parameterized across Memory /
Disk / Sqlite / LockedMemory). This module covers Sqlite-specific
behavior:

* Polling ``on_change`` fires for writes / appends / deletes.
* Close behavior.
* Persistence across ``close`` → reopen.
* Concurrent writes serialize safely.
"""

from __future__ import annotations

import asyncio

import pytest

from autogen.beta.knowledge import SqliteKnowledgeStore


async def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.05) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Basic protocol round-trip (Sqlite-specific sanity)
# ---------------------------------------------------------------------------


class TestSqliteBasics:
    @pytest.mark.asyncio
    async def test_write_read_round_trip(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.sqlite"))
        await store.write("/foo/bar.txt", "hello")
        assert await store.read("/foo/bar.txt") == "hello"
        store.close()

    @pytest.mark.asyncio
    async def test_list_direct_children(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.sqlite"))
        await store.write("/foo/a.txt", "a")
        await store.write("/foo/b.txt", "b")
        await store.write("/foo/sub/c.txt", "c")
        children = await store.list("/foo/")
        assert sorted(children) == ["a.txt", "b.txt", "sub/"]
        store.close()

    @pytest.mark.asyncio
    async def test_delete_file_and_subtree(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.sqlite"))
        await store.write("/foo/a.txt", "a")
        await store.write("/foo/b.txt", "b")
        await store.delete("/foo/a.txt")
        assert await store.read("/foo/a.txt") is None
        assert await store.read("/foo/b.txt") == "b"

        await store.delete("/foo")
        assert await store.read("/foo/b.txt") is None
        store.close()

    @pytest.mark.asyncio
    async def test_append_and_read_range(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.sqlite"))
        off1 = await store.append("/wal/log.jsonl", "one\n")
        off2 = await store.append("/wal/log.jsonl", "two\n")
        assert off1 == 0
        assert off2 == len("one\n".encode())
        whole = await store.read("/wal/log.jsonl")
        assert whole == "one\ntwo\n"
        slice_ = await store.read_range("/wal/log.jsonl", 0, off2)
        assert slice_ == "one\n"
        store.close()

    @pytest.mark.asyncio
    async def test_exists(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(str(tmp_path / "store.sqlite"))
        assert not await store.exists("/foo/bar")
        await store.write("/foo/bar.txt", "x")
        assert await store.exists("/foo/bar.txt")
        assert await store.exists("/foo")  # directory exists
        store.close()


# ---------------------------------------------------------------------------
# Polling on_change
# ---------------------------------------------------------------------------


class TestSqliteOnChange:
    @pytest.mark.asyncio
    async def test_fires_on_write(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)

            await store.write("/watched/new.txt", "hello")
            ok = await _wait_for(lambda: bool(events))
            assert ok
            assert "/watched/new.txt" in events

            await sub.close()
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_fires_on_append(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            await store.write("/watched/log.jsonl", "first\n")
            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)  # let the baseline snapshot settle

            await store.append("/watched/log.jsonl", "second\n")
            ok = await _wait_for(
                lambda: "/watched/log.jsonl" in events, timeout=2.0
            )
            assert ok
            await sub.close()
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_fires_on_delete(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            await store.write("/watched/doomed.txt", "bye")
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)
            await store.delete("/watched/doomed.txt")
            ok = await _wait_for(lambda: bool(events), timeout=2.0)
            assert ok
            assert "/watched/doomed.txt" in events
            await sub.close()
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_scope_isolates_siblings(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)

            # A sibling write outside the watched prefix must not fire.
            await store.write("/other/file.txt", "silent")
            await asyncio.sleep(0.15)
            assert events == []

            await store.write("/watched/visible.txt", "visible")
            ok = await _wait_for(lambda: bool(events), timeout=2.0)
            assert ok
            assert "/watched/visible.txt" in events
            await sub.close()
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_close_stops_delivery(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)
            await sub.close()

            # No further events after close.
            await store.write("/watched/after-close.txt", "late")
            await asyncio.sleep(0.15)
            assert not any("after-close" in e for e in events)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            sub = await store.on_change("/", lambda _p: asyncio.sleep(0))
            await sub.close()
            await sub.close()  # no raise
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_two_independent_subscriptions(self, tmp_path) -> None:
        store = SqliteKnowledgeStore(
            str(tmp_path / "store.sqlite"), poll_interval_s=0.05
        )
        try:
            events_a: list[str] = []
            events_b: list[str] = []

            async def cb_a(path: str) -> None:
                events_a.append(path)

            async def cb_b(path: str) -> None:
                events_b.append(path)

            sub_a = await store.on_change("/a/", cb_a)
            sub_b = await store.on_change("/b/", cb_b)
            await asyncio.sleep(0.1)

            await store.write("/a/file.txt", "a")
            await store.write("/b/file.txt", "b")
            ok = await _wait_for(
                lambda: events_a and events_b, timeout=2.0
            )
            assert ok
            assert any("/a/file.txt" in e for e in events_a)
            assert not any("/a/file.txt" in e for e in events_b)
            assert any("/b/file.txt" in e for e in events_b)
            assert not any("/b/file.txt" in e for e in events_a)

            await sub_a.close()
            await sub_b.close()
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Persistence across close / reopen
# ---------------------------------------------------------------------------


class TestSqlitePersistence:
    @pytest.mark.asyncio
    async def test_reopen_reads_previous_writes(self, tmp_path) -> None:
        db = tmp_path / "store.sqlite"
        store1 = SqliteKnowledgeStore(str(db))
        await store1.write("/persisted.txt", "kept")
        store1.close()

        store2 = SqliteKnowledgeStore(str(db))
        assert await store2.read("/persisted.txt") == "kept"
        store2.close()

    @pytest.mark.asyncio
    async def test_reopen_preserves_append_offsets(self, tmp_path) -> None:
        db = tmp_path / "store.sqlite"
        store1 = SqliteKnowledgeStore(str(db))
        off1 = await store1.append("/wal/log.jsonl", "a\n")
        assert off1 == 0
        store1.close()

        store2 = SqliteKnowledgeStore(str(db))
        off2 = await store2.append("/wal/log.jsonl", "b\n")
        assert off2 == len("a\n".encode())
        assert await store2.read("/wal/log.jsonl") == "a\nb\n"
        store2.close()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestSqliteConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_appends_preserve_content(self, tmp_path) -> None:
        """Appends under a lock must not interleave bytes."""

        store = SqliteKnowledgeStore(str(tmp_path / "store.sqlite"))
        try:
            async def append_many(prefix: str, n: int) -> None:
                for i in range(n):
                    await store.append("/wal/log.jsonl", f"{prefix}{i}\n")

            await asyncio.gather(
                append_many("a", 20),
                append_many("b", 20),
                append_many("c", 20),
            )

            content = await store.read("/wal/log.jsonl")
            assert content is not None
            lines = [line for line in content.split("\n") if line]
            assert len(lines) == 60
            # Each line is one contiguous token (no interleaving).
            for line in lines:
                assert line[0] in "abc"
                assert line[1:].isdigit()
        finally:
            store.close()
