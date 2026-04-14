# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``RedisKnowledgeStore`` — the Phase 3b Redis backend.

Uses ``fakeredis.aioredis`` as an in-process Redis replacement so the
test suite does not require a running server. If ``fakeredis`` is not
installed, every test in this module is skipped cleanly.

Covers the full KnowledgeStore protocol plus the polling ``on_change``
contract (no keyspace notifications — see §14 Phase 3b "Deferred").
"""

from __future__ import annotations

import asyncio

import pytest

fakeredis = pytest.importorskip("fakeredis")
fakeredis_aioredis = pytest.importorskip("fakeredis.aioredis")

from autogen.beta.knowledge import RedisKnowledgeStore


def _new_store(**kwargs) -> RedisKnowledgeStore:
    client = fakeredis_aioredis.FakeRedis()
    return RedisKnowledgeStore(client, **kwargs)


async def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.05) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


class TestRedisBasics:
    @pytest.mark.asyncio
    async def test_write_read(self) -> None:
        store = _new_store()
        await store.write("/foo/bar.txt", "hello")
        assert await store.read("/foo/bar.txt") == "hello"
        await store.close()

    @pytest.mark.asyncio
    async def test_read_missing_returns_none(self) -> None:
        store = _new_store()
        assert await store.read("/missing.txt") is None
        await store.close()

    @pytest.mark.asyncio
    async def test_list_immediate_children(self) -> None:
        store = _new_store()
        await store.write("/foo/a.txt", "a")
        await store.write("/foo/b.txt", "b")
        await store.write("/foo/sub/c.txt", "c")
        children = await store.list("/foo/")
        assert sorted(children) == ["a.txt", "b.txt", "sub/"]
        await store.close()

    @pytest.mark.asyncio
    async def test_delete_file(self) -> None:
        store = _new_store()
        await store.write("/foo.txt", "content")
        await store.delete("/foo.txt")
        assert await store.read("/foo.txt") is None
        await store.close()

    @pytest.mark.asyncio
    async def test_delete_subtree(self) -> None:
        store = _new_store()
        await store.write("/dir/a.txt", "a")
        await store.write("/dir/b.txt", "b")
        await store.write("/other.txt", "other")
        await store.delete("/dir")
        assert await store.read("/dir/a.txt") is None
        assert await store.read("/dir/b.txt") is None
        assert await store.read("/other.txt") == "other"
        await store.close()

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        store = _new_store()
        assert not await store.exists("/missing")
        await store.write("/foo/bar.txt", "x")
        assert await store.exists("/foo/bar.txt")
        assert await store.exists("/foo")  # directory via prefix
        await store.close()


# ---------------------------------------------------------------------------
# append + read_range
# ---------------------------------------------------------------------------


class TestRedisAppend:
    @pytest.mark.asyncio
    async def test_append_returns_offsets(self) -> None:
        store = _new_store()
        off1 = await store.append("/wal.jsonl", "one\n")
        off2 = await store.append("/wal.jsonl", "two\n")
        assert off1 == 0
        assert off2 == len("one\n".encode())
        assert await store.read("/wal.jsonl") == "one\ntwo\n"
        await store.close()

    @pytest.mark.asyncio
    async def test_read_range_slices(self) -> None:
        store = _new_store()
        await store.append("/wal.jsonl", "abcdefgh")
        assert await store.read_range("/wal.jsonl", 0, 3) == "abc"
        assert await store.read_range("/wal.jsonl", 3, 6) == "def"
        assert await store.read_range("/wal.jsonl", 3, None) == "defgh"
        await store.close()

    @pytest.mark.asyncio
    async def test_read_range_of_missing_returns_empty(self) -> None:
        store = _new_store()
        assert await store.read_range("/missing.txt", 0, 10) == ""
        await store.close()


# ---------------------------------------------------------------------------
# Polling on_change
# ---------------------------------------------------------------------------


class TestRedisOnChange:
    @pytest.mark.asyncio
    async def test_fires_on_write(self) -> None:
        store = _new_store(poll_interval_s=0.05)
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)  # let baseline settle

            await store.write("/watched/file.txt", "content")
            ok = await _wait_for(lambda: bool(events), timeout=2.0)
            assert ok
            assert "/watched/file.txt" in events
            await sub.close()
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_fires_on_append(self) -> None:
        store = _new_store(poll_interval_s=0.05)
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            await store.write("/watched/log.jsonl", "first\n")
            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)

            await store.append("/watched/log.jsonl", "second\n")
            ok = await _wait_for(
                lambda: "/watched/log.jsonl" in events, timeout=2.0
            )
            assert ok
            await sub.close()
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_fires_on_delete(self) -> None:
        store = _new_store(poll_interval_s=0.05)
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
            await store.close()

    @pytest.mark.asyncio
    async def test_scope_filters_sibling_writes(self) -> None:
        store = _new_store(poll_interval_s=0.05)
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)

            await store.write("/other/file.txt", "silent")
            await asyncio.sleep(0.15)
            assert events == []

            await store.write("/watched/visible.txt", "visible")
            ok = await _wait_for(lambda: bool(events), timeout=2.0)
            assert ok
            assert "/watched/visible.txt" in events
            await sub.close()
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_close_stops_delivery(self) -> None:
        store = _new_store(poll_interval_s=0.05)
        try:
            events: list[str] = []

            async def cb(path: str) -> None:
                events.append(path)

            sub = await store.on_change("/watched/", cb)
            await asyncio.sleep(0.1)
            await sub.close()

            await store.write("/watched/after-close.txt", "late")
            await asyncio.sleep(0.15)
            assert not any("after-close" in e for e in events)
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        store = _new_store(poll_interval_s=0.05)
        try:
            sub = await store.on_change("/", lambda _p: asyncio.sleep(0))
            await sub.close()
            await sub.close()  # no raise
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_two_subscriptions_are_independent(self) -> None:
        store = _new_store(poll_interval_s=0.05)
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

            await store.write("/a/alpha.txt", "a")
            await store.write("/b/beta.txt", "b")
            ok = await _wait_for(
                lambda: events_a and events_b, timeout=2.0
            )
            assert ok
            assert "/a/alpha.txt" in events_a
            assert "/b/beta.txt" in events_b
            assert not any("/a/" in e for e in events_b)
            assert not any("/b/" in e for e in events_a)

            await sub_a.close()
            await sub_b.close()
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Isolation — key prefix customization
# ---------------------------------------------------------------------------


class TestRedisKeyPrefix:
    @pytest.mark.asyncio
    async def test_custom_prefix(self) -> None:
        client = fakeredis_aioredis.FakeRedis()
        store_a = RedisKnowledgeStore(client, key_prefix="tenant_a")
        store_b = RedisKnowledgeStore(client, key_prefix="tenant_b")
        try:
            await store_a.write("/shared.txt", "from-a")
            await store_b.write("/shared.txt", "from-b")
            assert await store_a.read("/shared.txt") == "from-a"
            assert await store_b.read("/shared.txt") == "from-b"
        finally:
            await store_a.close()
            await store_b.close()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestRedisConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_appends_do_not_corrupt(self) -> None:
        store = _new_store()
        try:
            async def append_many(prefix: str, n: int) -> None:
                for i in range(n):
                    await store.append("/wal.jsonl", f"{prefix}{i}\n")

            await asyncio.gather(
                append_many("a", 10),
                append_many("b", 10),
            )

            content = await store.read("/wal.jsonl")
            assert content is not None
            lines = [line for line in content.split("\n") if line]
            assert len(lines) == 20
            for line in lines:
                assert line[0] in "ab"
                assert line[1:].isdigit()
        finally:
            await store.close()
