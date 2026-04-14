# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for KnowledgeStore, EventLogWriter, and bootstrapping."""

from uuid import uuid4

import pytest

from autogen.beta.events import ModelRequest, TaskCompleted, TextInput
from autogen.beta.events.lifecycle import UnknownEvent
from autogen.beta.knowledge import (
    DefaultBootstrap,
    EventLogWriter,
    MemoryKnowledgeStore,
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
        assert loaded[0].inputs[0].content == "hello"
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
        assert loaded[0].inputs[0].content == "old-1"
        assert loaded[1].inputs[0].content == "old-2"
        assert loaded[2].inputs[0].content == "recent"

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
        assert loaded[0].inputs[0].content == "batch-1"
        assert loaded[1].inputs[0].content == "batch-2"
        assert loaded[2].inputs[0].content == "final"

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
        import json

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
