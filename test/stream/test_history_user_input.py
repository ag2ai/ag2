# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable

import pytest

from ag2 import Agent, Context, MemoryStream, tool
from ag2.compact import CompactionSummary
from ag2.context import StreamId
from ag2.events import (
    BaseEvent,
    DrainedModelRequest,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.history import MemoryStorage
from ag2.testing import TestConfig


@tool
def lookup() -> str:
    """Look up the next item."""
    return "item"


@tool
def queue_update(context: Context) -> str:
    """Queue an update for the model after this tool finishes."""
    context.enqueue("background update")
    return "queued"


async def queue_after_answer(event: ModelResponse, context: Context) -> None:
    if event.message is not None and event.message.content == "answer before inbox":
        context.enqueue("final inbox update")


class JsonStorage:
    def __init__(self) -> None:
        self.records: dict[StreamId, list[tuple[type[BaseEvent], str]]] = {}
        self.reads = 0

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        self.records.setdefault(context.stream.id, []).append((type(event), json.dumps(event.to_dict())))

    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        self.reads += 1
        return [event_type.from_dict(json.loads(data)) for event_type, data in self.records.get(stream_id, [])]

    async def set_history(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        self.records[stream_id] = [(type(event), json.dumps(event.to_dict())) for event in events]

    async def drop_history(self, stream_id: StreamId) -> None:
        self.records.pop(stream_id, None)


class UnavailableStorage(MemoryStorage):
    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        raise OSError("storage unavailable")


@pytest.mark.asyncio
class TestUserInputBoundaries:
    async def test_reply_continuation_starts_a_new_interval(self) -> None:
        call = ToolCallEvent(name="lookup", arguments="{}")
        agent = Agent("test", config=TestConfig(call, "first answer", "second answer"), tools=[lookup])
        reply = await agent.ask("first question")
        history = reply.context.stream.history

        assert await history.since_last_user_input().where(ToolCallEvent).all() == [call]
        assert await history.since_last_user_input().where(ModelRequest).count() == 0

        await reply.ask("second question")

        assert await history.since_last_user_input().where(ToolCallEvent).count() == 0
        assert await history.where(ToolCallEvent).count() == 1

    async def test_run_and_reply_run_start_new_intervals(self) -> None:
        call = ToolCallEvent(name="lookup", arguments="{}")
        agent = Agent("test", config=TestConfig(call, "first answer", "second answer"), tools=[lookup])

        async with agent.run("first question") as run:
            reply = await run.result()
        history = reply.context.stream.history
        assert await history.since_last_user_input().where(ToolCallEvent).all() == [call]

        async with reply.run("second question") as run:
            await run.result()

        assert await history.since_last_user_input().where(ToolCallEvent).count() == 0

    async def test_run_enqueue_mid_turn_starts_a_new_interval(self) -> None:
        first_call = ToolCallEvent(name="queue_update", arguments="{}")
        second_call = ToolCallEvent(name="lookup", arguments="{}")
        agent = Agent("test", config=TestConfig(first_call, second_call, "answer"), tools=[queue_update, lookup])

        steered = False
        async with agent.run("question") as run:

            @run.stream.where(ToolResultEvent).subscribe
            async def steer(_event: BaseEvent) -> None:
                nonlocal steered
                if not steered:
                    steered = True
                    run.enqueue("use another source")

            reply = await run.result()

        history = reply.context.stream.history
        assert await history.since_last_user_input().where(ToolCallEvent).all() == [second_call]
        assert await history.where(ToolCallEvent).count() == 2
        assert await history.where(ModelRequest).last() == DrainedModelRequest([
            TextInput("background update"),
            TextInput("use another source"),
        ])

    async def test_pending_input_before_ask_keeps_the_new_boundary(self) -> None:
        stream = MemoryStream()
        config = TestConfig(ToolCallEvent(name="lookup", arguments="{}"), "first answer", "second answer")
        agent = Agent("test", config=config, tools=[lookup])
        reply = await agent.ask("first question", stream=stream)
        stream.enqueue("old background result")

        await reply.ask("second question")

        assert await stream.history.since_last_user_input().where(ToolCallEvent).count() == 0
        assert await stream.history.where(ModelRequest).last() == ModelRequest([
            TextInput("old background result"),
            TextInput("second question"),
        ])


@pytest.mark.asyncio
class TestServiceInput:
    async def test_mid_turn_inbox_drain_keeps_the_interval(self) -> None:
        call = ToolCallEvent(name="queue_update", arguments="{}")
        agent = Agent("test", config=TestConfig(call, "answer"), tools=[queue_update])

        reply = await agent.ask("question")

        assert await reply.context.stream.history.since_last_user_input().where(ToolCallEvent).all() == [call]

    async def test_final_inbox_drain_keeps_the_interval(self) -> None:
        stream = MemoryStream()
        stream.where(ModelResponse).subscribe(queue_after_answer)
        call = ToolCallEvent(name="lookup", arguments="{}")
        agent = Agent("test", config=TestConfig(call, "answer before inbox", "final answer"), tools=[lookup])

        reply = await agent.ask("question", stream=stream)

        assert reply.body == "final answer"
        assert await stream.history.since_last_user_input().where(ToolCallEvent).all() == [call]


@pytest.mark.asyncio
class TestStoredHistory:
    async def test_scope_survives_custom_storage_round_trip(self) -> None:
        storage = JsonStorage()
        stream = MemoryStream(storage=storage)
        stream.where(ModelResponse).subscribe(queue_after_answer)
        call = ToolCallEvent(name="queue_update", arguments="{}")
        agent = Agent("test", config=TestConfig(call, "answer before inbox", "final answer"), tools=[queue_update])
        await agent.ask("question", stream=stream)
        restored = MemoryStream(storage=storage, id=stream.id)
        calls = restored.history.where(ToolCallEvent)
        scoped_calls = calls.since_last_user_input().where(ToolCallEvent.name == "queue_update")
        reads_before = storage.reads

        assert await scoped_calls.all() == [call]
        assert storage.reads == reads_before + 1
        assert await scoped_calls.first() == call
        assert await scoped_calls.last() == call
        assert await scoped_calls.count() == 1
        assert await calls.count() == 1

    async def test_legacy_requests_establish_a_boundary(self) -> None:
        old_request = ModelRequest([TextInput("old question")]).to_dict()
        old_request.pop("origin")
        restored = ModelRequest.from_dict(old_request)
        call = ToolCallEvent(name="lookup", arguments="{}")
        stream = MemoryStream()
        await stream.history.replace([restored, call])

        assert await stream.history.since_last_user_input().where(ToolCallEvent).all() == [call]

    async def test_legacy_drained_requests_do_not_establish_a_boundary(self) -> None:
        old_drain = DrainedModelRequest([TextInput("background update")]).to_dict()
        old_drain.pop("origin")
        restored = DrainedModelRequest.from_dict(old_drain)
        first_call = ToolCallEvent(name="lookup", arguments="{}")
        second_call = ToolCallEvent(name="queue_update", arguments="{}")
        stream = MemoryStream()
        await stream.history.replace([ModelRequest([TextInput("question")]), first_call, restored, second_call])

        assert await stream.history.since_last_user_input().where(ToolCallEvent).all() == [first_call, second_call]

    async def test_no_retained_user_input_produces_empty_scope(self) -> None:
        call = ToolCallEvent(name="lookup", arguments="{}")
        agent = Agent("test", config=TestConfig(call, "answer"), tools=[lookup])
        reply = await agent.ask("question")
        history = reply.context.stream.history
        all_calls = history.where(ToolCallEvent)
        scoped = all_calls.since_last_user_input()
        assert await scoped.count() == 1

        await history.replace([
            CompactionSummary(summary="earlier conversation", event_count=2),
            ModelRequest([TextInput("background message")], origin="service"),
            call,
        ])

        assert await scoped.all() == []
        assert await scoped.first() is None
        assert await scoped.last() is None
        assert await scoped.count() == 0
        assert await all_calls.all() == [call]

    async def test_storage_errors_propagate(self) -> None:
        history = MemoryStream(storage=UnavailableStorage()).history

        with pytest.raises(OSError, match="storage unavailable"):
            await history.since_last_user_input().where(ToolCallEvent).count()
