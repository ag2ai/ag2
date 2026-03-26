# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.network.primitives.infra import (
    ActorInfo,
    LocalLock,
    LocalRegistry,
    MemoryCache,
    MemoryStateStore,
)


class TestMemoryStateStore:
    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        store = MemoryStateStore()
        await store.set("key1", "value1")
        assert await store.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        store = MemoryStateStore()
        assert await store.get("missing") is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = MemoryStateStore()
        await store.set("key1", "value1")
        await store.delete("key1")
        assert await store.get("key1") is None

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        store = MemoryStateStore()
        await store.set("key1", "value1")
        assert await store.exists("key1") is True
        assert await store.exists("missing") is False

    @pytest.mark.asyncio
    async def test_overwrite(self) -> None:
        store = MemoryStateStore()
        await store.set("key1", "v1")
        await store.set("key1", "v2")
        assert await store.get("key1") == "v2"


class TestMemoryCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        cache = MemoryCache()
        assert await cache.get("missing") is None

    @pytest.mark.asyncio
    async def test_invalidate_exact(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "v1")
        await cache.invalidate("key1")
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self) -> None:
        cache = MemoryCache()
        await cache.set("user:1", "a")
        await cache.set("user:2", "b")
        await cache.set("other:1", "c")
        await cache.invalidate("user:*")
        assert await cache.get("user:1") is None
        assert await cache.get("user:2") is None
        assert await cache.get("other:1") == "c"


class TestLocalLock:
    @pytest.mark.asyncio
    async def test_acquire_and_release(self) -> None:
        lock = LocalLock()
        assert await lock.acquire("resource1") is True
        await lock.release("resource1")

    @pytest.mark.asyncio
    async def test_double_acquire_fails(self) -> None:
        lock = LocalLock()
        assert await lock.acquire("resource1") is True
        assert await lock.acquire("resource1") is False
        await lock.release("resource1")

    @pytest.mark.asyncio
    async def test_held_context_manager(self) -> None:
        lock = LocalLock()
        async with lock.held("resource1"):
            # Lock should be held
            assert await lock.acquire("resource1") is False
        # Lock should be released
        assert await lock.acquire("resource1") is True
        await lock.release("resource1")


class TestLocalRegistry:
    @pytest.mark.asyncio
    async def test_register_and_discover(self) -> None:
        registry = LocalRegistry()
        info = ActorInfo(name="researcher", capabilities=["research", "analysis"])
        await registry.register("researcher", info)

        results = await registry.discover("research")
        assert len(results) == 1
        assert results[0].name == "researcher"

    @pytest.mark.asyncio
    async def test_discover_all(self) -> None:
        registry = LocalRegistry()
        await registry.register("a", ActorInfo(name="a", capabilities=["x"]))
        await registry.register("b", ActorInfo(name="b", capabilities=["y"]))

        results = await registry.discover()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_discover_no_match(self) -> None:
        registry = LocalRegistry()
        await registry.register("a", ActorInfo(name="a", capabilities=["x"]))

        results = await registry.discover("nonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_unregister(self) -> None:
        registry = LocalRegistry()
        await registry.register("a", ActorInfo(name="a", capabilities=["x"]))
        await registry.unregister("a")

        results = await registry.discover()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_heartbeat(self) -> None:
        registry = LocalRegistry()
        info = ActorInfo(name="a", capabilities=["x"])
        await registry.register("a", info)
        old_hb = info.last_heartbeat

        await asyncio.sleep(0.01)
        await registry.heartbeat("a")

        results = await registry.discover()
        assert results[0].last_heartbeat > old_hb
