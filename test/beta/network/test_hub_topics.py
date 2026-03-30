# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hub topic pub/sub and cross-actor knowledge queries."""

import pytest

from autogen.beta.network.events import TopicMessage, TopicSubscription
from autogen.beta.network.hub import Hub
from autogen.beta.network.primitives.knowledge import MemoryKnowledgeStore


class _FakeAgent:
    """Minimal agent stub for registration."""

    def __init__(self, name: str, knowledge_store=None) -> None:
        self.name = name
        self._knowledge_store = knowledge_store

    async def ask(self, message, **kwargs):
        return type("Reply", (), {"body": "ok", "content": "ok", "response": None})()


class TestHubTopics:
    @pytest.mark.asyncio
    async def test_publish_and_read(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("reader", "news")
        await hub.publish("writer", "news", "Breaking news!")

        messages = await hub.read_topic("reader", "news")
        assert len(messages) == 1
        assert messages[0].message == "Breaking news!"
        assert messages[0].sender == "writer"
        assert messages[0].topic == "news"

    @pytest.mark.asyncio
    async def test_cursor_advances(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("reader", "news")

        await hub.publish("w", "news", "msg-1")
        await hub.publish("w", "news", "msg-2")

        first_read = await hub.read_topic("reader", "news")
        assert len(first_read) == 2

        # Second read should return empty (cursor advanced)
        second_read = await hub.read_topic("reader", "news")
        assert len(second_read) == 0

        # New message after cursor
        await hub.publish("w", "news", "msg-3")
        third_read = await hub.read_topic("reader", "news")
        assert len(third_read) == 1
        assert third_read[0].message == "msg-3"

    @pytest.mark.asyncio
    async def test_subscribe_starts_at_end(self) -> None:
        hub = Hub()
        # Publish before subscribing
        await hub.publish("w", "news", "old-msg")
        await hub.subscribe_topic("late-reader", "news")

        # Late subscriber should not see old messages
        messages = await hub.read_topic("late-reader", "news")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("reader", "news")
        await hub.unsubscribe_topic("reader", "news")

        subs = hub.subscriptions_for("reader")
        assert "news" not in subs

    @pytest.mark.asyncio
    async def test_list_topics(self) -> None:
        hub = Hub()
        await hub.publish("w", "topic-a", "msg")
        await hub.publish("w", "topic-b", "msg")
        topics = await hub.list_topics()
        assert set(topics) == {"topic-a", "topic-b"}

    @pytest.mark.asyncio
    async def test_subscriptions_for(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("actor", "news")
        await hub.subscribe_topic("actor", "alerts")
        subs = hub.subscriptions_for("actor")
        assert set(subs) == {"news", "alerts"}

    @pytest.mark.asyncio
    async def test_emits_topic_events(self) -> None:
        hub = Hub()
        events = []
        hub.stream.where(TopicSubscription).subscribe(lambda e: events.append(e))
        hub.stream.where(TopicMessage).subscribe(lambda e: events.append(e))

        await hub.subscribe_topic("reader", "news")
        await hub.publish("writer", "news", "hello")

        # Allow events to propagate
        sub_events = [e for e in events if isinstance(e, TopicSubscription)]
        msg_events = [e for e in events if isinstance(e, TopicMessage)]
        assert len(sub_events) == 1
        assert len(msg_events) == 1


class TestHubPeekAdvance:
    @pytest.mark.asyncio
    async def test_peek_does_not_advance_cursor(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("reader", "news")
        await hub.publish("w", "news", "msg-1")
        await hub.publish("w", "news", "msg-2")

        peeked = await hub.peek_topic("reader", "news")
        assert len(peeked) == 2

        # Peek again — same messages (cursor not advanced)
        peeked2 = await hub.peek_topic("reader", "news")
        assert len(peeked2) == 2

        # read_topic still returns same messages
        read = await hub.read_topic("reader", "news")
        assert len(read) == 2

    @pytest.mark.asyncio
    async def test_advance_moves_cursor(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("reader", "news")
        await hub.publish("w", "news", "msg-1")
        await hub.publish("w", "news", "msg-2")
        await hub.publish("w", "news", "msg-3")

        await hub.advance_topic("reader", "news", 2)

        # Only the third message should be unread
        remaining = await hub.read_topic("reader", "news")
        assert len(remaining) == 1
        assert remaining[0].message == "msg-3"

    @pytest.mark.asyncio
    async def test_advance_clamped_to_max(self) -> None:
        hub = Hub()
        await hub.subscribe_topic("reader", "news")
        await hub.publish("w", "news", "msg-1")

        # Advance by more than available — should clamp
        await hub.advance_topic("reader", "news", 100)

        remaining = await hub.peek_topic("reader", "news")
        assert len(remaining) == 0


class TestHubKnowledgeQueries:
    @pytest.mark.asyncio
    async def test_query_exposed_path(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/working.md", "current state")

        agent = _FakeAgent("researcher", knowledge_store=store)
        hub = Hub()
        await hub.register(agent, exposed_paths=["/memory/"])

        result = await hub.query_knowledge("analyst", "researcher", "/memory/working.md")
        assert result == "current state"

    @pytest.mark.asyncio
    async def test_query_non_exposed_path_returns_none(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/secrets/key.txt", "secret")

        agent = _FakeAgent("researcher", knowledge_store=store)
        hub = Hub()
        await hub.register(agent, exposed_paths=["/memory/"])

        result = await hub.query_knowledge("analyst", "researcher", "/secrets/key.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_no_store_returns_none(self) -> None:
        agent = _FakeAgent("researcher")
        hub = Hub()
        await hub.register(agent)

        result = await hub.query_knowledge("analyst", "researcher", "/anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_unknown_agent_returns_none(self) -> None:
        hub = Hub()
        result = await hub.query_knowledge("analyst", "unknown", "/path")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_knowledge(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/conversations/s1.md", "summary 1")
        await store.write("/memory/conversations/s2.md", "summary 2")

        agent = _FakeAgent("researcher", knowledge_store=store)
        hub = Hub()
        await hub.register(agent, exposed_paths=["/memory/"])

        entries = await hub.list_knowledge("analyst", "researcher", "/memory/conversations/")
        assert entries is not None
        assert "s1.md" in entries
        assert "s2.md" in entries

    @pytest.mark.asyncio
    async def test_default_no_exposed_paths(self) -> None:
        store = MemoryKnowledgeStore()
        await store.write("/memory/working.md", "data")

        agent = _FakeAgent("researcher", knowledge_store=store)
        hub = Hub()
        await hub.register(agent)  # No exposed_paths

        result = await hub.query_knowledge("analyst", "researcher", "/memory/working.md")
        assert result is None  # Private by default
