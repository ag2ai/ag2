# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta.network.primitives.infra import (
    LocalLock,
    MemoryCache,
    MemoryStateStore,
)


class TestMemoryStateStoreTTL:
    """MemoryStateStore TTL behavior — values expire after the TTL elapses."""

    @pytest.mark.asyncio
    async def test_get_before_expiry_succeeds(self) -> None:
        store = MemoryStateStore()
        await store.set("key", "value", ttl=1.0)

        # Immediately after set, get should succeed
        result = await store.get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_get_after_expiry_returns_none(self) -> None:
        store = MemoryStateStore()
        await store.set("key", "value", ttl=0.1)

        # Wait for TTL to expire (with margin)
        await asyncio.sleep(0.2)

        result = await store.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists_respects_ttl_before_expiry(self) -> None:
        store = MemoryStateStore()
        await store.set("key", "value", ttl=1.0)

        assert await store.exists("key") is True

    @pytest.mark.asyncio
    async def test_exists_respects_ttl_after_expiry(self) -> None:
        store = MemoryStateStore()
        await store.set("key", "value", ttl=0.1)

        await asyncio.sleep(0.2)

        assert await store.exists("key") is False


class TestMemoryCacheTTL:
    """MemoryCache TTL behavior — cached values expire after the TTL elapses."""

    @pytest.mark.asyncio
    async def test_get_before_expiry_succeeds(self) -> None:
        cache = MemoryCache()
        await cache.set("key", "value", ttl=1.0)

        result = await cache.get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_get_after_expiry_returns_none(self) -> None:
        cache = MemoryCache()
        await cache.set("key", "value", ttl=0.1)

        await asyncio.sleep(0.2)

        result = await cache.get("key")
        assert result is None


class TestLocalLockTTLAutoRelease:
    """LocalLock TTL auto-release — lock is automatically released after TTL."""

    @pytest.mark.asyncio
    async def test_auto_release_after_ttl(self) -> None:
        lock = LocalLock()
        acquired = await lock.acquire("resource", ttl=0.1)
        assert acquired is True

        # Lock is held, second acquire should fail
        assert await lock.acquire("resource") is False

        # Wait for TTL auto-release (with margin)
        await asyncio.sleep(0.2)

        # Lock should now be released automatically
        acquired_again = await lock.acquire("resource", ttl=30.0)
        assert acquired_again is True
        await lock.release("resource")


class TestLocalLockConcurrentAcquire:
    """LocalLock concurrent acquire — first coroutine wins, second fails."""

    @pytest.mark.asyncio
    async def test_first_wins_second_fails(self) -> None:
        lock = LocalLock()
        results: dict[str, bool] = {}

        async def attempt(name: str, delay: float) -> None:
            await asyncio.sleep(delay)
            results[name] = await lock.acquire("resource", ttl=5.0)

        # Launch two coroutines; first has no delay, second has a small delay
        await asyncio.gather(
            attempt("first", 0.0),
            attempt("second", 0.01),
        )

        assert results["first"] is True
        assert results["second"] is False

        await lock.release("resource")


class TestLocalLockHeldContextManager:
    """LocalLock held() context manager — acquire/release and error on contention."""

    @pytest.mark.asyncio
    async def test_held_acquires_and_releases(self) -> None:
        lock = LocalLock()

        async with lock.held("resource", ttl=5.0):
            # Inside the context, lock should be held
            assert await lock.acquire("resource") is False

        # After the context exits, lock should be released
        assert await lock.acquire("resource") is True
        await lock.release("resource")

    @pytest.mark.asyncio
    async def test_held_raises_when_already_locked(self) -> None:
        lock = LocalLock()

        # Acquire the lock first
        assert await lock.acquire("resource", ttl=5.0) is True

        # Attempting held() on the same resource should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to acquire lock"):
            async with lock.held("resource", ttl=5.0):
                pass  # pragma: no cover

        await lock.release("resource")
