# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from ag2.knowledge import LockedKnowledgeStore, MemoryKnowledgeStore


class ExclusiveLock:
    def __init__(self) -> None:
        self.held: set[str] = set()

    async def acquire(self, name: str, ttl: float) -> bool:
        if name in self.held:
            return False
        self.held.add(name)
        return True

    async def release(self, name: str) -> None:
        self.held.remove(name)


class BlockingMemoryStore(MemoryKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    async def write(self, path: str, content: str) -> None:
        self.started.set()
        await self.resume.wait()
        await super().write(path, content)


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["log/events.jsonl", "//log//events.jsonl/", "/log/events.jsonl/"])
@pytest.mark.parametrize("operation", ["write", "append", "delete"])
async def test_aliases_cannot_bypass_an_in_progress_write(alias: str, operation: str) -> None:
    inner = BlockingMemoryStore()
    lock = ExclusiveLock()
    first = LockedKnowledgeStore(inner, lock)
    second = LockedKnowledgeStore(inner, lock)
    task = asyncio.create_task(first.write("/log/events.jsonl", "initial"))
    try:
        await asyncio.wait_for(inner.started.wait(), timeout=2)
        with pytest.raises(RuntimeError, match=f"Failed to acquire {operation} lock"):
            if operation == "delete":
                await second.delete(alias)
            elif operation == "append":
                await second.append(alias, "extra")
            else:
                # Let the backing write finish if the alias incorrectly bypasses the lock.
                inner.resume.set()
                await second.write(alias, "replacement")
    finally:
        inner.resume.set()
        await asyncio.wait_for(task, timeout=2)

    assert lock.held == set()
    assert await inner.read("/log/events.jsonl") == "initial"
    assert await second.append(alias, "extra") == len(b"initial")
    assert await inner.read("/log/events.jsonl") == "initialextra"
