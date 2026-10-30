# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest

from ag2 import Agent, Context, DataInput, ImageInput, Inject, KnowledgeConfig, MemoryStream, TextInput
from ag2.compact import CompactTrigger, SummarizeCompact, TailWindowCompact
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StatusObservation,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
    UsageEvent,
    is_conversational,
)
from ag2.history import MemoryStorage
from ag2.knowledge import MemoryKnowledgeStore
from ag2.middleware import BaseMiddleware, LLMCall, RetryMiddleware, statusbar_middleware
from ag2.plugin import Plugin
from ag2.policies import ConversationPolicy, SlidingWindowPolicy
from ag2.testing import TestConfig, TrackingConfig
from test._helpers import DurableReasoning, ProviderTurnState


@statusbar_middleware(persist=True)
def progress(context: Context) -> TextInput:
    return TextInput(f"progress:{context.variables.get('step', 0)}")


def advance(context: Context) -> str:
    context.variables["step"] = context.variables.get("step", 0) + 1
    return "advanced"


def texts(events: Sequence[BaseEvent]) -> list[str]:
    return [
        part.content
        for event in events
        if isinstance(event, ModelRequest)
        for part in event.parts
        if isinstance(part, TextInput)
    ]


@pytest.mark.asyncio
async def test_persistent_status_accumulates_without_creating_user_turns_or_extra_calls() -> None:
    config = TrackingConfig(TestConfig(ToolCallEvent("advance"), "done", "continued"))
    agent = Agent("status", config=config, middleware=[progress], tools=[advance])

    reply = await agent.ask("start")
    await reply.ask("continue")

    assert [texts(call.messages) for call in config.calls] == [
        ["start", "progress:0"],
        ["start", "progress:0", "progress:1"],
        ["start", "progress:0", "progress:1", "continue", "progress:1"],
    ]
    history = list(await reply.context.stream.history.get_events())
    retained = [event for event in history if isinstance(event, StatusObservation)]
    assert [event.parts for event in retained] == [
        [TextInput("progress:0")],
        [TextInput("progress:1")],
        [TextInput("progress:1")],
    ]
    assert all(event.source == f"{__name__}.progress" for event in retained)
    assert all(not is_conversational(event) and not isinstance(event, ModelRequest) for event in retained)
    assert texts(history) == ["start", "continue"]
    assert [texts([event]) for event in config.calls[-1].messages if not is_conversational(event)] == [
        ["progress:0"],
        ["progress:1"],
        ["progress:1"],
    ]


@statusbar_middleware
def dynamic_progress(context: Context) -> TextInput:
    return TextInput(f"dynamic:{context.variables.get('step', 0)}")


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent_first", [False, True])
async def test_persistent_and_dynamic_callbacks_compose_in_wrapper_order(persistent_first: bool) -> None:
    config = TrackingConfig(TestConfig(ToolCallEvent("advance"), "done", "continued"))
    middleware = [progress, dynamic_progress] if persistent_first else [dynamic_progress, progress]
    agent = Agent("status", config=config, tools=[advance], middleware=middleware)

    reply = await agent.ask("start")
    await reply.ask("continue")

    tails = [["progress:0", "dynamic:0"], ["progress:1", "dynamic:1"], ["progress:1", "dynamic:1"]]
    if not persistent_first:
        tails = [list(reversed(tail)) for tail in tails]
    assert [texts(call.messages) for call in config.calls] == [
        ["start", *tails[0]],
        ["start", "progress:0", *tails[1]],
        ["start", "progress:0", "progress:1", "continue", *tails[2]],
    ]
    assert all("dynamic:" not in str(event) for event in await reply.history.get_events())


@statusbar_middleware(persist=True)
def chosen_status(value: Annotated[Any, Inject()]) -> Any:
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", [None, [], ()])
async def test_skipping_update_keeps_earlier_persistent_observations(skip: Any) -> None:
    config = TrackingConfig(TestConfig("one", "two"))
    agent = Agent("status", config=config, middleware=[chosen_status])

    reply = await agent.ask("start", dependencies={"value": TextInput("saved")})
    await reply.ask("continue", dependencies={"value": skip})

    assert [texts(call.messages) for call in config.calls] == [
        ["start", "saved"],
        ["start", "saved", "continue"],
    ]
    assert len([event for event in await reply.history.get_events() if isinstance(event, StatusObservation)]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("via_plugin", [False, True])
async def test_observations_remain_visible_after_invocation_scope_ends(via_plugin: bool) -> None:
    config = TrackingConfig(TestConfig("one", "two"))
    agent = Agent("status", config=config, assembly=[ConversationPolicy()])
    options = {"plugins": [Plugin(middleware=[progress])]} if via_plugin else {"middleware": [progress]}

    reply = await agent.ask("start", **options)
    await reply.ask("continue")

    assert [texts(call.messages) for call in config.calls] == [
        ["start", "progress:0"],
        ["start", "progress:0", "continue"],
    ]
    assert len([event for event in await reply.history.get_events() if isinstance(event, StatusObservation)]) == 1


@pytest.mark.asyncio
async def test_new_conversation_on_shared_agent_has_no_old_status_trail() -> None:
    config = TrackingConfig(TestConfig("one", "two"))
    agent = Agent("status", config=config, middleware=[progress])

    await agent.ask("start", variables={"step": 9})
    await agent.ask("fresh")

    assert [texts(call.messages) for call in config.calls] == [
        ["start", "progress:9"],
        ["fresh", "progress:0"],
    ]


class JsonStorage(MemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.observation_saves = 0

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        if isinstance(event, StatusObservation):
            self.observation_saves += 1
        restored = type(event).from_dict(json.loads(json.dumps(event.to_dict())))
        await super().save_event(restored, context)


@pytest.mark.asyncio
async def test_storage_round_trip_replays_parts_and_identity_without_resaving() -> None:
    storage = JsonStorage()
    stream = MemoryStream(storage=storage)
    config = TrackingConfig(TestConfig("one", "two"))
    agent = Agent("status", config=config)
    parts = [
        TextInput("saved", metadata={"labels": ["original"]}),
        DataInput({"step": [1]}),
        ImageInput(data=b"png", media_type="image/png"),
    ]

    await agent.ask("start", stream=stream, middleware=[chosen_status], dependencies={"value": parts})
    reloaded = MemoryStream(storage=storage, id=stream.id)
    await agent.ask("resume", stream=reloaded)

    retained = [event for event in await reloaded.history.get_events() if isinstance(event, StatusObservation)]
    assert len(retained) == 1
    assert retained[0].parts == parts
    assert retained[0].source == f"{__name__}.chosen_status"
    assert not is_conversational(retained[0])
    assert [event.parts for event in config.calls[1].messages if isinstance(event, ModelRequest)] == [
        [TextInput("start")],
        parts,
        [TextInput("resume")],
    ]
    assert storage.observation_saves == 1


@pytest.mark.asyncio
async def test_mutation_of_source_containers_cannot_rewrite_stored_observations() -> None:
    config = TrackingConfig(TestConfig("one", "two"))
    agent = Agent("status", config=config, middleware=[chosen_status])
    data = {"steps": ["queued"]}
    part = DataInput(data, metadata={"labels": ["original"]})
    parts = [part]

    reply = await agent.ask("start", dependencies={"value": parts})
    data["steps"].append("done")
    part.metadata["labels"].append("changed")
    parts.append(TextInput("extra"))
    await reply.ask("continue")

    retained = [event for event in await reply.history.get_events() if isinstance(event, StatusObservation)]
    assert [event.parts for event in retained] == [
        [DataInput({"steps": ["queued"]}, metadata={"labels": ["original"]})],
        [DataInput({"steps": ["queued", "done"]}, metadata={"labels": ["original", "changed"]}), TextInput("extra")],
    ]


class MutateProjectedStatus(BaseMiddleware):
    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        for event in events:
            if isinstance(event, ModelRequest) and not is_conversational(event):
                [part] = event.parts
                assert isinstance(part, DataInput)
                part.data["steps"].append("downstream")
        return await call_next(events, context)


@pytest.mark.asyncio
async def test_current_and_historical_provider_views_do_not_alias_stored_payloads() -> None:
    config = TrackingConfig(TestConfig("one", "two"))
    agent = Agent("status", config=config)
    value = DataInput({"steps": ["queued"]})

    reply = await agent.ask("start", dependencies={"value": value}, middleware=[chosen_status, MutateProjectedStatus])
    await reply.ask("continue", middleware=[MutateProjectedStatus])

    retained = [event for event in await reply.history.get_events() if isinstance(event, StatusObservation)]
    assert [event.parts for event in retained] == [[DataInput({"steps": ["queued"]})]]
    assert value == DataInput({"steps": ["queued"]})
    assert [
        event.parts
        for call in config.calls
        for event in call.messages
        if isinstance(event, ModelRequest) and not is_conversational(event)
    ] == [[DataInput({"steps": ["queued", "downstream"]})]] * 2


@pytest.mark.asyncio
async def test_retry_reuses_one_saved_observation_per_logical_call() -> None:
    config = TrackingConfig(TestConfig(TimeoutError("retry"), ToolCallEvent("advance"), "done"))
    agent = Agent(
        "status",
        config=config,
        tools=[advance],
        middleware=[progress, RetryMiddleware(max_retries=1, retry_on=(TimeoutError,))],
    )

    reply = await agent.ask("start")

    assert [texts(call.messages) for call in config.calls] == [
        ["start", "progress:0"],
        ["start", "progress:0"],
        ["start", "progress:0", "progress:1"],
    ]
    assert len([event for event in await reply.history.get_events() if isinstance(event, StatusObservation)]) == 2


@pytest.mark.asyncio
async def test_invalid_later_part_prevents_partial_persistence_and_model_call() -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[chosen_status])
    stream = MemoryStream()

    with pytest.raises(TypeError, match="Input, Sequence"):
        await agent.ask("start", stream=stream, dependencies={"value": [TextInput("valid"), "invalid"]})

    assert config.calls == []
    assert not any(isinstance(event, StatusObservation) for event in await stream.history.get_events())


class ChangingParts(Sequence):
    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int | slice) -> Any:
        if index == 0:
            return TextInput("validated")
        raise IndexError(index)

    def __iter__(self) -> Iterator[Any]:
        self.reads += 1
        return iter([TextInput("validated")] if self.reads == 1 else ["unvalidated"])


@pytest.mark.asyncio
async def test_stateful_sequence_saves_the_exact_parts_that_were_validated() -> None:
    config = TrackingConfig(TestConfig("done"))
    agent = Agent("status", config=config, middleware=[chosen_status])

    reply = await agent.ask("start", dependencies={"value": ChangingParts()})

    retained = [event for event in await reply.history.get_events() if isinstance(event, StatusObservation)]
    assert [event.parts for event in retained] == [[TextInput("validated")]]
    assert texts(config.calls[0].messages) == ["start", "validated"]


class RejectStatusStorage(MemoryStorage):
    async def save_event(self, event: BaseEvent, context: Context) -> None:
        if isinstance(event, StatusObservation):
            raise OSError("storage unavailable")
        await super().save_event(event, context)


@pytest.mark.asyncio
async def test_persistence_failure_prevents_model_call() -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[progress])
    stream = MemoryStream(storage=RejectStatusStorage())

    with pytest.raises(OSError, match="storage unavailable"):
        await agent.ask("start", stream=stream)

    assert config.calls == []
    assert not any(isinstance(event, StatusObservation) for event in await stream.history.get_events())


@pytest.mark.asyncio
async def test_provider_failure_keeps_already_committed_observation() -> None:
    config = TrackingConfig(TestConfig(ValueError("provider failed")))
    agent = Agent("status", config=config, middleware=[progress])
    stream = MemoryStream()

    with pytest.raises(ValueError, match="provider failed"):
        await agent.ask("start", stream=stream)

    retained = [event for event in await stream.history.get_events() if isinstance(event, StatusObservation)]
    assert [event.parts for event in retained] == [[TextInput("progress:0")]]
    assert len(config.calls) == 1


@dataclass
class Gate:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class WaitForModel(BaseMiddleware):
    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        gate = context.dependencies["gate"]
        gate.started.set()
        await gate.release.wait()
        return await call_next(events, context)


@pytest.mark.asyncio
async def test_cancellation_keeps_prepared_observation() -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[progress, WaitForModel])
    stream = MemoryStream()
    gate = Gate()
    task = asyncio.create_task(agent.ask("start", stream=stream, dependencies={"gate": gate}))
    await gate.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    retained = [event for event in await stream.history.get_events() if isinstance(event, StatusObservation)]
    assert [event.parts for event in retained] == [[TextInput("progress:0")]]
    assert config.calls == []


@pytest.mark.asyncio
async def test_status_records_do_not_advance_conversational_compaction_trigger() -> None:
    summary = TrackingConfig(TestConfig("would summarize"))
    agent = Agent(
        "status",
        config=TestConfig("done"),
        middleware=[progress],
        knowledge=KnowledgeConfig(
            store=MemoryKnowledgeStore(),
            compact=SummarizeCompact(target=1, config=summary),
            compact_trigger=CompactTrigger(max_events=2),
        ),
    )

    reply = await agent.ask("start")

    assert summary.calls == []
    assert len([event for event in await reply.history.get_events() if is_conversational(event)]) == 2


@pytest.mark.asyncio
async def test_retention_counts_conversation_and_may_remove_older_status_records() -> None:
    agent = Agent("status", config=TestConfig("one", "two", "three"), middleware=[progress])
    reply = await agent.ask("first")
    reply = await reply.ask("second")
    reply = await reply.ask("third")
    history = list(await reply.history.get_events())

    retained = await TailWindowCompact(target=2).compact(history, reply.context, None)

    assert [event for event in retained if is_conversational(event)] == [
        ModelRequest([TextInput("third")]),
        ModelResponse(ModelMessage("three")),
    ]
    assert len([event for event in retained if isinstance(event, StatusObservation)]) == 1


@pytest.mark.asyncio
async def test_summarization_excludes_retained_status_content() -> None:
    agent = Agent("status", config=TestConfig("one", "two"), middleware=[progress])
    reply = await agent.ask("first user text")
    reply = await reply.ask("latest user text")
    summary = TrackingConfig(TestConfig("summary"))

    await SummarizeCompact(target=2, config=summary).compact(
        list(await reply.history.get_events()), reply.context, None
    )

    [request] = summary.calls[0].messages
    [part] = request.parts
    assert "first user text" in part.content
    assert "progress:" not in part.content


@pytest.mark.asyncio
async def test_status_does_not_create_usage_and_preserves_existing_usage_records() -> None:
    config = TestConfig(
        ModelResponse(ModelMessage("one"), usage=Usage(total_tokens=7)),
        ModelResponse(ModelMessage("two"), usage=Usage(total_tokens=11)),
    )
    agent = Agent("status", config=config, middleware=[progress])
    reply = await agent.ask("start")
    await reply.ask("continue")

    report = await reply.usage()

    assert report.total == Usage(total_tokens=18)
    assert len(report.records) == 2
    assert len([event for event in await reply.history.get_events() if isinstance(event, UsageEvent)]) == 2


@pytest.mark.asyncio
async def test_projected_status_does_not_reset_provider_anchor_relationship_during_reduction() -> None:
    stream = MemoryStream()
    await stream.history.replace([
        DurableReasoning("native anchor"),
        StatusObservation([TextInput("saved")], source="fixture"),
        BuiltinToolCallEvent("web_search", id="hosted"),
        BuiltinToolResultEvent(parent_id="hosted", result=ToolResult("found")),
        ModelResponse(ModelMessage("previous answer")),
    ])
    config = TrackingConfig(TestConfig("next answer"))
    agent = Agent("status", config=config, assembly=[SlidingWindowPolicy(max_events=5)])

    await agent.ask("next", stream=stream)

    assert texts(config.calls[0].messages) == ["saved", "next"]
    assert not any(
        isinstance(event, (DurableReasoning, BuiltinToolCallEvent, BuiltinToolResultEvent))
        for event in config.calls[0].messages
    )


@pytest.mark.asyncio
async def test_projected_status_does_not_hide_missing_provider_turn_state() -> None:
    call = ToolCallEvent("remote", id="call-1")
    stream = MemoryStream()
    await stream.history.replace([
        ProviderTurnState(),
        StatusObservation([TextInput("saved")], source="fixture"),
        ModelResponse(tool_calls=ToolCallsEvent([call])),
        ToolResultsEvent([ToolResultEvent(parent_id="call-1", result=ToolResult("result"))]),
    ])
    config = TrackingConfig(TestConfig("next answer"))
    agent = Agent("status", config=config, assembly=[SlidingWindowPolicy(max_events=4)])

    await agent.ask("next", stream=stream)

    assert texts(config.calls[0].messages) == ["saved", "next"]
    assert not any(
        isinstance(event, (ProviderTurnState, ModelResponse, ToolResultsEvent)) for event in config.calls[0].messages
    )


@pytest.mark.asyncio
async def test_reversed_retry_order_preserves_existing_publication_guard() -> None:
    config = TrackingConfig(TestConfig(TimeoutError("provider"), "unreachable retry"))
    agent = Agent(
        "status", config=config, middleware=[RetryMiddleware(max_retries=1, retry_on=(TimeoutError,)), progress]
    )
    stream = MemoryStream()

    with pytest.raises(TimeoutError, match="provider"):
        await agent.ask("start", stream=stream)

    assert len(config.calls) == 1
    assert len([event for event in await stream.history.get_events() if isinstance(event, StatusObservation)]) == 1
