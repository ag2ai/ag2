# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for transient event filtering in MemoryStream persistence.

Verifies that events marked ``__transient__ = True`` are:
- Delivered to subscribers (streaming still works)
- NOT persisted to storage by default
- Persisted when ``persist_all=True``
"""

import pytest

from autogen.beta import Context, MemoryStream
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelRequest, ModelResponse
from autogen.beta.events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
    TaskProgress,
    TaskRequest,
    TaskResult,
)
from autogen.beta.events.tool_events import ToolCallEvent
from autogen.beta.events.types import ModelReasoning


# ---------------------------------------------------------------------------
# Transient flag on event types
# ---------------------------------------------------------------------------


class TestTransientFlag:
    """Verify __transient__ is set correctly on event types."""

    def test_base_event_not_transient(self):
        assert BaseEvent.__transient__ is False

    def test_model_request_not_transient(self):
        assert ModelRequest.__transient__ is False

    def test_model_response_not_transient(self):
        assert ModelResponse.__transient__ is False

    def test_tool_call_event_not_transient(self):
        assert ToolCallEvent.__transient__ is False

    def test_task_request_not_transient(self):
        assert TaskRequest.__transient__ is False

    def test_task_result_not_transient(self):
        assert TaskResult.__transient__ is False

    def test_model_message_chunk_transient(self):
        assert ModelMessageChunk.__transient__ is True

    def test_model_message_transient(self):
        assert ModelMessage.__transient__ is True

    def test_model_reasoning_transient(self):
        assert ModelReasoning.__transient__ is True

    def test_task_progress_transient(self):
        assert TaskProgress.__transient__ is True

    def test_observer_started_transient(self):
        assert ObserverStarted.__transient__ is True

    def test_observer_completed_transient(self):
        assert ObserverCompleted.__transient__ is True

    def test_compaction_completed_transient(self):
        assert CompactionCompleted.__transient__ is True

    def test_aggregation_completed_transient(self):
        assert AggregationCompleted.__transient__ is True


# ---------------------------------------------------------------------------
# Transient events still delivered to subscribers
# ---------------------------------------------------------------------------


class TestTransientDelivery:
    """Transient events must still reach subscribers (for real-time streaming)."""

    @pytest.mark.asyncio
    async def test_chunk_delivered_to_subscriber(self):
        stream = MemoryStream()
        received = []
        stream.where(ModelMessageChunk).subscribe(lambda ev: received.append(ev))

        chunk = ModelMessageChunk(content="hello")
        await stream.send(chunk, context=Context(stream))

        assert len(received) == 1
        assert received[0].content == "hello"

    @pytest.mark.asyncio
    async def test_task_progress_delivered_to_subscriber(self):
        stream = MemoryStream()
        received = []
        stream.where(TaskProgress).subscribe(lambda ev: received.append(ev))

        progress = TaskProgress(task_name="t1", content="working...")
        await stream.send(progress, context=Context(stream))

        assert len(received) == 1
        assert received[0].content == "working..."


# ---------------------------------------------------------------------------
# Transient events NOT persisted by default
# ---------------------------------------------------------------------------


class TestTransientNotPersisted:
    """Default MemoryStream should NOT store transient events in history."""

    @pytest.mark.asyncio
    async def test_chunk_not_in_history(self):
        stream = MemoryStream()
        ctx = Context(stream)

        await stream.send(ModelRequest(content="hi"), ctx)
        await stream.send(ModelMessageChunk(content="hel"), ctx)
        await stream.send(ModelMessageChunk(content="lo"), ctx)
        await stream.send(ModelResponse(message=ModelMessage(content="hello")), ctx)

        events = list(await stream.history.get_events())
        types = [type(e).__name__ for e in events]
        assert "ModelRequest" in types
        assert "ModelResponse" in types
        assert "ModelMessageChunk" not in types
        assert "ModelMessage" not in types

    @pytest.mark.asyncio
    async def test_lifecycle_events_not_in_history(self):
        stream = MemoryStream()
        ctx = Context(stream)

        await stream.send(ObserverStarted(name="loop_detector"), ctx)
        await stream.send(ModelRequest(content="hi"), ctx)
        await stream.send(ModelResponse(message=ModelMessage(content="hello")), ctx)
        await stream.send(ObserverCompleted(name="loop_detector"), ctx)

        events = list(await stream.history.get_events())
        types = [type(e).__name__ for e in events]
        assert "ObserverStarted" not in types
        assert "ObserverCompleted" not in types
        assert "ModelRequest" in types
        assert "ModelResponse" in types

    @pytest.mark.asyncio
    async def test_task_progress_not_in_history(self):
        stream = MemoryStream()
        ctx = Context(stream)

        await stream.send(TaskRequest(task="research", task_name="t1"), ctx)
        await stream.send(TaskProgress(task_name="t1", content="step 1"), ctx)
        await stream.send(TaskProgress(task_name="t1", content="step 2"), ctx)
        await stream.send(TaskResult(task="research", task_name="t1", result="done"), ctx)

        events = list(await stream.history.get_events())
        types = [type(e).__name__ for e in events]
        assert "TaskRequest" in types
        assert "TaskResult" in types
        assert "TaskProgress" not in types

    @pytest.mark.asyncio
    async def test_compaction_aggregation_not_in_history(self):
        stream = MemoryStream()
        ctx = Context(stream)

        await stream.send(ModelRequest(content="hi"), ctx)
        await stream.send(CompactionCompleted(actor="pilot", strategy="s", events_before=10, events_after=5), ctx)
        await stream.send(AggregationCompleted(actor="pilot", strategy="s", event_count=5), ctx)

        events = list(await stream.history.get_events())
        types = [type(e).__name__ for e in events]
        assert "CompactionCompleted" not in types
        assert "AggregationCompleted" not in types

    @pytest.mark.asyncio
    async def test_non_transient_events_all_persisted(self):
        """Ensure conversation events are all stored."""
        stream = MemoryStream()
        ctx = Context(stream)

        await stream.send(ModelRequest(content="hi"), ctx)
        await stream.send(ToolCallEvent(name="search", arguments="q"), ctx)
        await stream.send(ModelResponse(message=ModelMessage(content="hello")), ctx)

        events = list(await stream.history.get_events())
        assert len(events) == 3


# ---------------------------------------------------------------------------
# persist_all=True stores everything
# ---------------------------------------------------------------------------


class TestPersistAll:
    """When persist_all=True, ALL events including transient ones are stored."""

    @pytest.mark.asyncio
    async def test_chunks_persisted_when_persist_all(self):
        stream = MemoryStream(persist_all=True)
        ctx = Context(stream)

        await stream.send(ModelRequest(content="hi"), ctx)
        await stream.send(ModelMessageChunk(content="hel"), ctx)
        await stream.send(ModelMessageChunk(content="lo"), ctx)
        await stream.send(ModelResponse(message=ModelMessage(content="hello")), ctx)

        events = list(await stream.history.get_events())
        types = [type(e).__name__ for e in events]
        assert types.count("ModelMessageChunk") == 2
        assert "ModelMessage" not in types  # ModelMessage is standalone, not sent here as top-level

    @pytest.mark.asyncio
    async def test_lifecycle_persisted_when_persist_all(self):
        stream = MemoryStream(persist_all=True)
        ctx = Context(stream)

        await stream.send(ObserverStarted(name="monitor"), ctx)
        await stream.send(TaskProgress(task_name="t1", content="working"), ctx)

        events = list(await stream.history.get_events())
        types = [type(e).__name__ for e in events]
        assert "ObserverStarted" in types
        assert "TaskProgress" in types
