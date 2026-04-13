# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for agent harness: compaction/aggregation middleware,
knowledge/memory tools, LockedKnowledgeStore, and unregister cleanup.

Covers gaps identified in the design review.
"""

from __future__ import annotations

import pytest

from autogen.beta import Actor, KnowledgeConfig
from autogen.beta.actor import (
    _AggregationMiddleware,
    _CompactionMiddleware,
)
from autogen.beta.aggregate import AggregateTrigger
from autogen.beta.compact import CompactTrigger, TailWindowCompact
from autogen.beta.context import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse
from autogen.beta.knowledge import (
    DefaultBootstrap,
    LockedKnowledgeStore,
    MemoryKnowledgeStore,
)
from autogen.beta.events.lifecycle import AggregationCompleted, CompactionCompleted
from autogen.beta.policies import TokenBudgetPolicy
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAggregate:
    """In-memory aggregate that writes a fixed summary without LLM."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_usage: dict = {}

    async def aggregate(self, events, context, store):
        self.call_count += 1
        await store.write("/memory/working.md", f"aggregated-{self.call_count}")


class _FakeLock:
    """Minimal lock satisfying LockedKnowledgeStore's interface."""

    def __init__(self) -> None:
        self._held: set[str] = set()

    async def acquire(self, key: str, ttl: float = 0) -> bool:
        if key in self._held:
            return False
        self._held.add(key)
        return True

    async def release(self, key: str) -> None:
        self._held.discard(key)


async def _populate_history(stream: MemoryStream, events: list[BaseEvent]) -> None:
    """Set events directly in stream history."""
    existing = list(await stream.history.get_events())
    await stream.history.replace(existing + events)


# ---------------------------------------------------------------------------
# CompactionMiddleware tests
# ---------------------------------------------------------------------------


class TestCompactionMiddleware:
    @pytest.mark.asyncio
    async def test_triggers_on_max_events(self) -> None:
        """Compaction fires when event count exceeds max_events threshold."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        trigger = CompactTrigger(max_events=5)
        strategy = TailWindowCompact(target=3)

        # Populate stream with 8 events
        events = [ModelRequest(content=f"msg-{i}") for i in range(8)]
        await _populate_history(stream, events)

        initial_event = ModelRequest(content="start")

        mw = _CompactionMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=None,
            trigger=trigger,
        )

        # Simulate a turn — call_next returns a result, then compaction runs
        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_turn(call_next, initial_event, ctx)

        # History should now be compacted to 3 events + 1 CompactionCompleted
        remaining = list(await stream.history.get_events())
        non_lifecycle = [e for e in remaining if not isinstance(e, CompactionCompleted)]
        assert len(non_lifecycle) == 3

    @pytest.mark.asyncio
    async def test_triggers_on_max_tokens(self) -> None:
        """Compaction fires when token estimate exceeds max_tokens threshold."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        # chars_per_token=1 means each char counts as a token
        trigger = CompactTrigger(max_tokens=50, chars_per_token=1)
        strategy = TailWindowCompact(target=2)

        # Each event's str() representation is well over 10 chars
        events = [ModelRequest(content="x" * 30) for _ in range(5)]
        await _populate_history(stream, events)

        initial_event = ModelRequest(content="start")
        mw = _CompactionMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=None,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_turn(call_next, initial_event, ctx)
        remaining = list(await stream.history.get_events())
        non_lifecycle = [e for e in remaining if not isinstance(e, CompactionCompleted)]
        assert len(non_lifecycle) == 2

    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self) -> None:
        """No compaction when event count is below threshold."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        trigger = CompactTrigger(max_events=100)
        strategy = TailWindowCompact(target=3)

        events = [ModelRequest(content=f"msg-{i}") for i in range(5)]
        await _populate_history(stream, events)

        initial_event = ModelRequest(content="start")
        mw = _CompactionMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=None,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_turn(call_next, initial_event, ctx)
        remaining = list(await stream.history.get_events())
        assert len(remaining) == 5  # Unchanged

    @pytest.mark.asyncio
    async def test_emits_compaction_completed_event(self) -> None:
        """CompactionCompleted event is emitted on the stream after compaction."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        trigger = CompactTrigger(max_events=3)
        strategy = TailWindowCompact(target=2)

        events = [ModelRequest(content=f"msg-{i}") for i in range(5)]
        await _populate_history(stream, events)

        collected: list[CompactionCompleted] = []

        async def _collect(e: CompactionCompleted) -> None:
            collected.append(e)

        sub = stream.where(CompactionCompleted).subscribe(_collect)

        initial_event = ModelRequest(content="start")
        mw = _CompactionMiddleware(
            initial_event,
            ctx,
            actor_name="test-actor",
            strategy=strategy,
            store=None,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_turn(call_next, initial_event, ctx)
        stream.unsubscribe(sub)

        assert len(collected) == 1
        assert collected[0].actor == "test-actor"
        assert collected[0].events_before == 5
        assert collected[0].events_after == 2
        assert collected[0].strategy == "TailWindowCompact"

    @pytest.mark.asyncio
    async def test_no_double_compaction(self) -> None:
        """Compaction shouldn't fire again if count hasn't grown since last compaction."""
        stream = MemoryStream()
        ctx = Context(stream=stream)
        trigger = CompactTrigger(max_events=3)
        strategy = TailWindowCompact(target=2)

        events = [ModelRequest(content=f"msg-{i}") for i in range(5)]
        await _populate_history(stream, events)

        initial_event = ModelRequest(content="start")
        mw = _CompactionMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=None,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        # First turn compacts
        await mw.on_turn(call_next, initial_event, ctx)
        remaining = list(await stream.history.get_events())
        non_lifecycle = [e for e in remaining if not isinstance(e, CompactionCompleted)]
        assert len(non_lifecycle) == 2

        # Second turn — no new events, should not compact again
        await mw.on_turn(call_next, initial_event, ctx)
        remaining2 = list(await stream.history.get_events())
        non_lifecycle2 = [e for e in remaining2 if not isinstance(e, CompactionCompleted)]
        assert len(non_lifecycle2) == len(non_lifecycle)


# ---------------------------------------------------------------------------
# AggregationMiddleware tests
# ---------------------------------------------------------------------------


class TestAggregationMiddleware:
    @pytest.mark.asyncio
    async def test_triggers_every_n_turns(self) -> None:
        """Aggregation fires every N turns."""
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        ctx = Context(stream=stream)
        strategy = _FakeAggregate()
        trigger = AggregateTrigger(every_n_turns=2, on_end=False)

        events = [ModelRequest(content=f"msg-{i}") for i in range(3)]
        await _populate_history(stream, events)

        initial_event = ModelRequest(content="start")
        mw = _AggregationMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=store,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        # Turn 1 — no aggregation
        await mw.on_turn(call_next, initial_event, ctx)
        assert strategy.call_count == 0

        # Turn 2 — aggregation fires
        await mw.on_turn(call_next, initial_event, ctx)
        assert strategy.call_count == 1

        # Turn 3 — no aggregation
        await mw.on_turn(call_next, initial_event, ctx)
        assert strategy.call_count == 1

        # Turn 4 — aggregation fires again
        await mw.on_turn(call_next, initial_event, ctx)
        assert strategy.call_count == 2

    @pytest.mark.asyncio
    async def test_triggers_every_n_events(self) -> None:
        """Aggregation fires when enough new events accumulate."""
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        ctx = Context(stream=stream)
        strategy = _FakeAggregate()
        trigger = AggregateTrigger(every_n_events=3, on_end=False)

        initial_event = ModelRequest(content="start")
        mw = _AggregationMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=store,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        # Push 2 events — not enough
        await _populate_history(stream, [ModelRequest(content=f"e-{i}") for i in range(2)])
        await mw.on_turn(call_next, initial_event, ctx)
        assert strategy.call_count == 0

        # Push 1 more — now 3 total, should trigger
        await _populate_history(stream, [ModelRequest(content="e-2")])
        await mw.on_turn(call_next, initial_event, ctx)
        assert strategy.call_count == 1

    @pytest.mark.asyncio
    async def test_emits_aggregation_completed_event(self) -> None:
        """AggregationCompleted event is emitted after aggregation."""
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        ctx = Context(stream=stream)
        strategy = _FakeAggregate()
        trigger = AggregateTrigger(every_n_turns=1, on_end=False)

        events = [ModelRequest(content="msg")]
        await _populate_history(stream, events)

        collected: list[AggregationCompleted] = []

        async def _collect(e: AggregationCompleted) -> None:
            collected.append(e)

        sub = stream.where(AggregationCompleted).subscribe(_collect)

        initial_event = ModelRequest(content="start")
        mw = _AggregationMiddleware(
            initial_event,
            ctx,
            actor_name="test-actor",
            strategy=strategy,
            store=store,
            trigger=trigger,
        )

        async def call_next(event, context):
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_turn(call_next, initial_event, ctx)
        stream.unsubscribe(sub)

        assert len(collected) == 1
        assert collected[0].actor == "test-actor"


# ---------------------------------------------------------------------------
# LockedKnowledgeStore tests
# ---------------------------------------------------------------------------


class TestLockedKnowledgeStore:
    @pytest.mark.asyncio
    async def test_read_not_locked(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/test.txt", "data")
        lock = _FakeLock()
        store = LockedKnowledgeStore(inner, lock)
        # Read should work without acquiring lock
        assert await store.read("/test.txt") == "data"
        assert not lock._held  # No lock held

    @pytest.mark.asyncio
    async def test_write_acquires_lock(self) -> None:
        inner = MemoryKnowledgeStore()
        lock = _FakeLock()
        store = LockedKnowledgeStore(inner, lock)
        await store.write("/test.txt", "hello")
        # Lock should be released after write
        assert not lock._held
        assert await inner.read("/test.txt") == "hello"

    @pytest.mark.asyncio
    async def test_delete_acquires_lock(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/test.txt", "hello")
        lock = _FakeLock()
        store = LockedKnowledgeStore(inner, lock)
        await store.delete("/test.txt")
        assert not lock._held
        assert await inner.read("/test.txt") is None

    @pytest.mark.asyncio
    async def test_list_not_locked(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/dir/a.txt", "a")
        lock = _FakeLock()
        store = LockedKnowledgeStore(inner, lock)
        entries = await store.list("/dir/")
        assert entries == ["a.txt"]
        assert not lock._held

    @pytest.mark.asyncio
    async def test_exists_not_locked(self) -> None:
        inner = MemoryKnowledgeStore()
        await inner.write("/test.txt", "data")
        lock = _FakeLock()
        store = LockedKnowledgeStore(inner, lock)
        assert await store.exists("/test.txt") is True
        assert not lock._held

    @pytest.mark.asyncio
    async def test_write_fails_if_lock_unavailable(self) -> None:
        inner = MemoryKnowledgeStore()

        class _BusyLock:
            async def acquire(self, key, ttl=0):
                return False  # Always busy

            async def release(self, key):
                pass

        store = LockedKnowledgeStore(inner, _BusyLock())
        with pytest.raises(RuntimeError, match="Failed to acquire write lock"):
            await store.write("/test.txt", "data")


# ---------------------------------------------------------------------------
# Hub.unregister cleanup tests (V2) were dropped during the V3 rewrite.
# Equivalent coverage for the V3 Hub lives in test/beta/network/test_hub_registry.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Knowledge tool tests
# ---------------------------------------------------------------------------


def _get_raw_fn(tool_list, index=0):
    """Extract the underlying async function from a FunctionTool."""
    return tool_list[index].model.call


class TestKnowledgeTool:
    @pytest.mark.asyncio
    async def test_read_action(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "hello world")
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="read", path="/test.txt")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self) -> None:

        store = MemoryKnowledgeStore()
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="read", path="/missing.txt")
        assert "Not found" in result

    @pytest.mark.asyncio
    async def test_write_action(self) -> None:

        store = MemoryKnowledgeStore()
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="write", path="/note.txt", content="my note")
        assert "Written" in result
        assert await store.read("/note.txt") == "my note"

    @pytest.mark.asyncio
    async def test_write_requires_content(self) -> None:

        store = MemoryKnowledgeStore()
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="write", path="/note.txt")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_list_action_with_skill_md(self) -> None:

        store = MemoryKnowledgeStore()
        await store.write("/dir/SKILL.md", "This directory stores artifacts.")
        await store.write("/dir/file1.txt", "data")
        await store.write("/dir/file2.txt", "data")
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="list", path="/dir/")
        assert "This directory stores artifacts." in result
        assert "file1.txt" in result
        assert "file2.txt" in result

    @pytest.mark.asyncio
    async def test_list_empty_directory(self) -> None:

        store = MemoryKnowledgeStore()
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="list", path="/empty/")
        assert "Empty" in result

    @pytest.mark.asyncio
    async def test_delete_action(self) -> None:

        store = MemoryKnowledgeStore()
        await store.write("/test.txt", "data")
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="delete", path="/test.txt")
        assert "Deleted" in result
        assert await store.read("/test.txt") is None

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:

        store = MemoryKnowledgeStore()
        actor = Actor("test", knowledge=KnowledgeConfig(store=store))
        fn = _get_raw_fn(actor._build_knowledge_tool())
        result = await fn(action="bogus")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# Memory tool tests
# ---------------------------------------------------------------------------


class TestMemoryTool:
    @pytest.mark.asyncio
    async def test_compact_action(self) -> None:

        store = MemoryKnowledgeStore()
        strategy = TailWindowCompact(target=2)
        actor = Actor("test", knowledge=KnowledgeConfig(store=store, compact=strategy))
        fn = _get_raw_fn(actor._build_memory_tool())

        stream = MemoryStream()
        ctx = Context(stream=stream)
        events = [ModelRequest(content=f"msg-{i}") for i in range(10)]
        await _populate_history(stream, events)

        result = await fn(action="compact", ctx=ctx)
        assert "10 events -> 2 events" in result

        remaining = list(await stream.history.get_events())
        assert len(remaining) == 2

    @pytest.mark.asyncio
    async def test_compact_not_configured(self) -> None:

        actor = Actor("test")
        fn = _get_raw_fn(actor._build_memory_tool())

        stream = MemoryStream()
        ctx = Context(stream=stream)
        result = await fn(action="compact", ctx=ctx)
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_summarize_action(self) -> None:

        store = MemoryKnowledgeStore()
        strategy = _FakeAggregate()
        actor = Actor("test", knowledge=KnowledgeConfig(store=store, aggregate=strategy))
        fn = _get_raw_fn(actor._build_memory_tool())

        stream = MemoryStream()
        ctx = Context(stream=stream)
        events = [ModelRequest(content=f"msg-{i}") for i in range(3)]
        await _populate_history(stream, events)

        result = await fn(action="summarize", ctx=ctx)
        assert "updated" in result.lower()
        assert strategy.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_not_configured(self) -> None:

        actor = Actor("test")
        fn = _get_raw_fn(actor._build_memory_tool())

        stream = MemoryStream()
        ctx = Context(stream=stream)
        result = await fn(action="summarize", ctx=ctx)
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:

        store = MemoryKnowledgeStore()
        strategy = TailWindowCompact(target=2)
        actor = Actor("test", knowledge=KnowledgeConfig(store=store, compact=strategy))
        fn = _get_raw_fn(actor._build_memory_tool())

        stream = MemoryStream()
        ctx = Context(stream=stream)
        result = await fn(action="bogus", ctx=ctx)
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# TokenBudgetPolicy transparent mode
# ---------------------------------------------------------------------------


class TestTokenBudgetPolicyTransparent:
    @pytest.mark.asyncio
    async def test_transparent_adds_note(self) -> None:
        policy = TokenBudgetPolicy(max_tokens=10, chars_per_token=1, transparent=True)
        events = [ModelRequest(content="a" * 20), ModelRequest(content="b" * 5)]
        ctx = Context(stream=MemoryStream())
        prompts, _ = await policy.apply([], events, ctx)
        assert len(prompts) == 1
        assert "token budget" in prompts[0].lower()


# ---------------------------------------------------------------------------
# CompactTrigger max_tokens path
# ---------------------------------------------------------------------------


class TestCompactTriggerMaxTokens:
    def test_defaults(self) -> None:
        trigger = CompactTrigger()
        assert trigger.max_events == 0
        assert trigger.max_tokens == 0
        assert trigger.chars_per_token == 4

    def test_custom_chars_per_token(self) -> None:
        trigger = CompactTrigger(max_tokens=100, chars_per_token=2)
        assert trigger.chars_per_token == 2


# ---------------------------------------------------------------------------
# ConversationSummaryAggregate uses full stream_id
# ---------------------------------------------------------------------------


class TestConversationSummaryAggregateStreamId:
    @pytest.mark.asyncio
    async def test_uses_full_stream_id(self) -> None:
        """The summary filename must use the full stream_id, not truncated."""
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        ctx = Context(stream=stream)
        full_id = str(stream.id)

        # Use a fake aggregate that writes to the same path pattern
        # to verify the filename contains the full ID
        from unittest.mock import AsyncMock, MagicMock

        from autogen.beta.aggregate import ConversationSummaryAggregate

        config = MagicMock()
        mock_client = AsyncMock()
        mock_client.return_value = MagicMock(content="summary text", usage={})
        config.create.return_value = mock_client

        agg = ConversationSummaryAggregate(config)
        events = [ModelRequest(content="hello")]
        await agg.aggregate(events, ctx, store)

        entries = await store.list("/memory/conversations/")
        assert len(entries) == 1
        filename = entries[0]
        # Full UUID (36 chars with hyphens) must be in the filename
        assert full_id in filename


# ---------------------------------------------------------------------------
# DefaultBootstrap no longer writes sentinel
# ---------------------------------------------------------------------------


class TestDefaultBootstrapNoSentinel:
    @pytest.mark.asyncio
    async def test_bootstrap_does_not_write_sentinel(self) -> None:
        """DefaultBootstrap should NOT write /.initialized — Actor owns that."""
        store = MemoryKnowledgeStore()
        bootstrap = DefaultBootstrap()
        await bootstrap.bootstrap(store, "test-actor")

        # Bootstrap creates SKILL.md files but NOT /.initialized
        assert await store.exists("/SKILL.md")
        assert not await store.exists("/.initialized")
