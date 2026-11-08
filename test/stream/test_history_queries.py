# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterable

import pytest

from ag2 import Context, MemoryStream
from ag2.context import StreamId
from ag2.events import BaseEvent, ModelRequest, ModelResponse, TextInput, ToolCallEvent, ToolCallsEvent
from ag2.history import MemoryStorage


class SpecialToolCall(ToolCallEvent):
    pass


class RecordedStorage(MemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        self.reads += 1
        return iter(tuple(await super().get_history(stream_id)))


class UnavailableStorage(MemoryStorage):
    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        raise OSError("storage unavailable")


@pytest.mark.asyncio
async def test_history_selects_retained_events_in_publication_order() -> None:
    stream = MemoryStream()
    context = Context(stream)
    first = ToolCallEvent(name="lookup", arguments="{}")
    last = ToolCallEvent(name="lookup", arguments='{"page": 2}')
    await context.send(first)
    await context.send(ModelRequest([TextInput("question")]))
    await context.send(last)

    query = stream.history.where(ToolCallEvent)

    assert await query.all() == [first, last]
    assert await query.first() is first
    assert await query.last() is last
    assert await query.count() == 2


@pytest.mark.asyncio
async def test_history_query_combines_conditions_without_changing_original() -> None:
    stream = MemoryStream()
    context = Context(stream)
    first = ToolCallEvent(name="lookup", arguments="{}")
    second = SpecialToolCall(name="other", arguments="{}")
    await context.send(first)
    await context.send(second)

    calls = stream.history.where(ToolCallEvent)
    lookups = calls.where(ToolCallEvent.name == "lookup")

    assert await lookups.all() == [first]
    assert await calls.all() == [first, second]
    assert await stream.history.where(ToolCallEvent.name == "other").all() == [second]
    assert await lookups.where(ToolCallEvent.name == "other").all() == []


@pytest.mark.asyncio
async def test_empty_history_queries_complete_without_waiting_for_publication() -> None:
    query = MemoryStream().history.where(ToolCallEvent)

    assert await asyncio.wait_for(query.all(), timeout=1) == []
    assert await asyncio.wait_for(query.first(), timeout=1) is None
    assert await asyncio.wait_for(query.last(), timeout=1) is None
    assert await asyncio.wait_for(query.count(), timeout=1) == 0


@pytest.mark.asyncio
async def test_history_queries_keep_top_level_occurrences_and_return_fresh_lists() -> None:
    stream = MemoryStream()
    context = Context(stream)
    call = ToolCallEvent(name="lookup", arguments="{}")
    batch = ToolCallsEvent([call])
    await context.send(ModelResponse(tool_calls=batch))
    await context.send(batch)
    await context.send(call)
    await context.send(call)

    query = stream.history.where(ToolCallEvent)
    selected = await query.all()
    assert selected == [call, call]
    selected.clear()

    assert await query.count() == 2
    assert await query.all() == [call, call]
    assert len(list(await stream.history.get_events())) == 4


@pytest.mark.asyncio
async def test_history_query_uses_one_fresh_custom_storage_snapshot_per_terminal() -> None:
    storage = RecordedStorage()
    stream = MemoryStream(storage=storage)
    context = Context(stream)
    first = ToolCallEvent(name="first", arguments="{}")
    second = ToolCallEvent(name="second", arguments="{}")
    await context.send(first)
    query = stream.history.where(ToolCallEvent).where(ToolCallEvent.name != "ignored")
    assert storage.reads == 0

    assert await query.all() == [first]
    assert storage.reads == 1
    await context.send(second)
    assert await query.last() is second
    assert storage.reads == 2
    assert await query.first() is first
    assert storage.reads == 3
    assert await query.count() == 2
    assert storage.reads == 4

    await stream.history.replace([second])
    restored = MemoryStream(storage=storage, id=stream.id)
    assert await restored.history.where(ToolCallEvent).all() == [second]


@pytest.mark.asyncio
async def test_filtered_stream_query_uses_parent_history_without_live_filter() -> None:
    stream = MemoryStream()
    context = Context(stream)
    request = ModelRequest([TextInput("question")])
    call = ToolCallEvent(name="lookup", arguments="{}")
    await context.send(request)
    await context.send(call)

    filtered = stream.where(ToolCallEvent)

    assert await filtered.history.where(ModelRequest).all() == [request]
    assert await filtered.history.where(BaseEvent).all() == [request, call]


@pytest.mark.asyncio
async def test_history_query_propagates_storage_errors() -> None:
    query = MemoryStream(storage=UnavailableStorage()).history.where(BaseEvent)

    with pytest.raises(OSError, match="storage unavailable"):
        await query.count()
