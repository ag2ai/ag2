# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from ag2 import Agent, Context
from ag2.aggregate import AggregateTrigger, WorkingMemoryAggregate
from ag2.events import AggregationCompleted, ModelMessage, ModelRequest, ModelResponse, Usage, render_for_prompt
from ag2.knowledge import KnowledgePlugin, LockedKnowledgeStore, MemoryKnowledgeStore
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig


class OverlappingReads(MemoryKnowledgeStore):
    def __init__(self):
        super().__init__()
        self.reads = 0
        self.ready = asyncio.Event()

    async def read(self, path):
        content = await super().read(path)
        self.reads += 1
        if self.reads == 2:
            self.ready.set()
        if self.reads <= 2:
            await asyncio.wait_for(self.ready.wait(), 2)
        return content


class ConflictingStore(MemoryKnowledgeStore):
    async def compare_and_swap(self, path, expected, content):
        await self.write(path, "concurrent update")
        return False


class CancelledStore(MemoryKnowledgeStore):
    async def compare_and_swap(self, path, expected, content):
        raise asyncio.CancelledError


class OneConflictStore(MemoryKnowledgeStore):
    def __init__(self):
        super().__init__()
        self.conflicted = False

    async def compare_and_swap(self, path, expected, content):
        if not self.conflicted:
            self.conflicted = True
            await self.write(path, "concurrent fact")
            return False
        return await super().compare_and_swap(path, expected, content)


class LegacyStore:
    def __init__(self):
        self.inner = MemoryKnowledgeStore()

    async def read(self, path):
        return await self.inner.read(path)

    async def write(self, path, content):
        await self.inner.write(path, content)


class SharedLock:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def acquire(self, name, ttl):
        await self.lock.acquire()
        return True

    async def release(self, name):
        self.lock.release()


@pytest.mark.asyncio
async def test_independent_strategies_preserve_both_concurrent_updates():
    store = OverlappingReads()
    await store.write("/memory/working.md", "old fact")
    configs = [
        TrackingConfig(TestConfig("old fact; fact A", "old fact; fact A; fact B")),
        TrackingConfig(TestConfig("old fact; fact B", "old fact; fact A; fact B")),
    ]
    await asyncio.gather(
        *(
            WorkingMemoryAggregate(config).aggregate(
                [ModelRequest.ensure_request([f"remember fact {fact}"])], Context(MemoryStream()), store
            )
            for config, fact in zip(configs, ("A", "B"))
        )
    )
    assert await store.read("/memory/working.md") == "old fact; fact A; fact B"
    assert sorted(len(config.calls) for config in configs) == [1, 2]
    retried = next(config for config in configs if len(config.calls) == 2)
    retry_prompt = "\n".join(render_for_prompt(event) for event in retried.calls[1].messages)
    assert "old fact; fact A" in retry_prompt or "old fact; fact B" in retry_prompt


@pytest.mark.asyncio
async def test_exhaustion_preserves_concurrent_update_and_bounds_model_calls():
    store = ConflictingStore()
    config = TrackingConfig(TestConfig("stale", "stale"))
    with pytest.raises(RuntimeError, match="all 2 merge attempts"):
        await WorkingMemoryAggregate(config, max_attempts=2).aggregate(
            [ModelRequest.ensure_request(["remember"])], Context(MemoryStream()), store
        )
    assert len(config.calls) == 2
    assert await store.read("/memory/working.md") == "concurrent update"


@pytest.mark.asyncio
async def test_cancellation_does_not_retry_or_write():
    store = CancelledStore()
    config = TrackingConfig(TestConfig("new"))
    with pytest.raises(asyncio.CancelledError):
        await WorkingMemoryAggregate(config).aggregate(
            [ModelRequest.ensure_request(["remember"])], Context(MemoryStream()), store
        )
    assert len(config.calls) == 1
    assert await store.read("/memory/working.md") is None


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_attempt_limit_must_be_a_positive_integer(attempts):
    with pytest.raises(ValueError, match="positive integer"):
        WorkingMemoryAggregate(TestConfig("unused"), max_attempts=attempts)


@pytest.mark.asyncio
async def test_plugin_reports_usage_from_all_merge_attempts():
    store = OneConflictStore()
    stream = MemoryStream()
    completed = []
    stream.where(AggregationCompleted).subscribe(lambda event: completed.append(event))
    merge_config = TrackingConfig(
        TestConfig(
            ModelResponse(ModelMessage("stale"), usage=Usage(prompt_tokens=2, completion_tokens=1)),
            ModelResponse(ModelMessage("concurrent fact; new fact"), usage=Usage(prompt_tokens=3, completion_tokens=1)),
        )
    )
    agent = Agent(
        "rememberer",
        config=TestConfig("done"),
        plugins=[
            KnowledgePlugin(
                store=store,
                aggregate=WorkingMemoryAggregate(merge_config),
                aggregate_trigger=AggregateTrigger(on_end=True),
            )
        ],
    )
    await agent.ask("remember new fact", stream=stream)
    assert await store.read("/memory/working.md") == "concurrent fact; new fact"
    assert len(merge_config.calls) == 2
    assert len(completed) == 1
    assert completed[0].llm_calls == 2
    assert completed[0].usage["prompt_tokens"] == 5
    assert completed[0].usage["completion_tokens"] == 2


@pytest.mark.asyncio
async def test_custom_store_fails_before_spending_tokens_without_atomic_capability():
    config = TrackingConfig(TestConfig("unused"))
    with pytest.raises(TypeError, match="compare_and_swap"):
        await WorkingMemoryAggregate(config).aggregate(
            [ModelRequest.ensure_request(["remember"])], Context(MemoryStream()), LegacyStore()
        )
    assert config.calls == []


@pytest.mark.asyncio
async def test_locked_adapter_supplies_atomic_updates_for_custom_store():
    inner = LegacyStore()
    lock = SharedLock()
    stores = [LockedKnowledgeStore(inner, lock), LockedKnowledgeStore(inner, lock)]
    outcomes = await asyncio.gather(
        *(store.compare_and_swap("/note", None, value) for store, value in zip(stores, ("A", "B")))
    )
    assert sorted(outcomes) == [False, True]
    assert await inner.read("/note") == ("A" if outcomes[0] else "B")
