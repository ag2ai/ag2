# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from uuid import uuid4

import pytest
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.errors import A2ATaskCancelledError, A2ATaskTerminalError
from ag2.a2a.events import A2ATaskArtifactUpdate, A2ATextArtifact
from ag2.a2a.extension import MIME_TOOL_CALL
from ag2.a2a.mappers.parts import data_part
from ag2.a2a.mappers.tools import call_to_payload
from ag2.a2a.tasks import cancel_task, list_tasks
from ag2.a2a.testing import make_test_client_factory
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
)
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

from ._helpers import EventCollector, ScriptedExecutor, a2a_config, make_executor_pair

_URL = "http://test"

_both_modes = pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "polling"])


async def _growing_artifact(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="a")], artifact_id="art")
    await release.wait()
    await updater.add_artifact(parts=[Part(text="b")], artifact_id="art", append=True)
    await updater.add_artifact(parts=[Part(text="c")], artifact_id="art", append=True, last_chunk=True)
    await updater.complete()


async def _resent_artifact(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="a")], artifact_id="art")
    await updater.add_artifact(parts=[Part(text="ab")], artifact_id="art")
    await updater.complete()


async def _answer_in_completion_message(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.complete(message=updater.new_agent_message([Part(text="done")]))


async def _shortened_replacement(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="hello world")], artifact_id="art")
    await updater.add_artifact(parts=[Part(text="hello")], artifact_id="art")
    await updater.complete()


async def _cleared_replacement(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="hello")], artifact_id="art")
    await updater.add_artifact(parts=[Part(text="")], artifact_id="art")
    await updater.complete()


async def _growing_mixed_artifact(updater: TaskUpdater, release: asyncio.Event) -> None:
    image = Part(url="http://test/image.png", media_type="image/png")
    await updater.add_artifact(parts=[Part(text="hello"), image], artifact_id="art")
    await updater.add_artifact(parts=[Part(text="hello world"), image], artifact_id="art")
    await updater.complete()


async def _cancelled_midway(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="partial")], artifact_id="art")
    await updater.cancel()


async def _two_artifacts(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="first")], artifact_id="one", last_chunk=True)
    await updater.add_artifact(parts=[Part(text="-second")], artifact_id="two", last_chunk=True)
    await updater.complete()


class _ChunkGate(EventCollector):
    """Collects chunk events; the executor holds its remaining output until the first one lands."""

    def __init__(self, executor: ScriptedExecutor) -> None:
        super().__init__()
        self._executor = executor

    async def on_event(self, event: BaseEvent) -> None:
        await super().on_event(event)
        self._executor.release.set()


@pytest.mark.asyncio
@_both_modes
async def test_growing_artifact_is_returned_in_full(streaming: bool) -> None:
    executor = ScriptedExecutor(_growing_artifact)
    pair = make_executor_pair(executor, streaming=streaming)
    stream = MemoryStream()
    gate = _ChunkGate(executor)
    stream.where(ModelMessageChunk).subscribe(gate.on_event)
    if streaming:
        # The in-process transport buffers a streamed response until the task ends,
        # so no chunk can arrive early to open the gate.
        executor.release.set()

    reply = await pair.client.ask("go", stream=stream)

    assert reply.response.content == "abc"
    assert "".join(gate.chunks) == "abc"


@pytest.mark.asyncio
async def test_resent_artifact_replaces_instead_of_concatenating() -> None:
    pair = make_executor_pair(ScriptedExecutor(_resent_artifact), streaming=True)

    reply = await pair.client.ask("go")

    assert reply.response.content == "ab"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(_shortened_replacement, "hello", id="shortened"),
        pytest.param(_cleared_replacement, None, id="cleared"),
        pytest.param(_growing_mixed_artifact, "hello world", id="mixed-growing"),
    ],
)
@_both_modes
async def test_full_restatement_replaces_artifact_text(
    script: Callable[[TaskUpdater, asyncio.Event], Awaitable[None]],
    expected: str | None,
    streaming: bool,
) -> None:
    pair = make_executor_pair(ScriptedExecutor(script), streaming=streaming)

    reply = await pair.client.ask("go")

    assert reply.response.content == expected


@pytest.mark.asyncio
@_both_modes
async def test_distinct_artifacts_join_in_first_appearance_order(streaming: bool) -> None:
    pair = make_executor_pair(ScriptedExecutor(_two_artifacts), streaming=streaming)

    reply = await pair.client.ask("go")

    assert reply.response.content == "first-second"


@pytest.mark.asyncio
@_both_modes
async def test_cancelled_task_raises_and_keeps_partial_text(streaming: bool) -> None:
    pair = make_executor_pair(ScriptedExecutor(_cancelled_midway), streaming=streaming)

    with pytest.raises(A2ATaskCancelledError) as exc_info:
        await pair.client.ask("go")

    task = exc_info.value.task
    partial = "".join(part.text for artifact in task.artifacts for part in artifact.parts)
    assert isinstance(exc_info.value, A2ATaskTerminalError)
    assert (task.status.state, partial) == (TaskState.TASK_STATE_CANCELED, "partial")


async def _text_artifact_updates(agent: Agent) -> list[A2ATextArtifact]:
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(A2ATaskArtifactUpdate).subscribe(collector.on_event)

    await agent.ask("go", stream=stream)

    return [e for e in collector.events if isinstance(e, A2ATextArtifact)]


def _client(server_agent: Agent, *, streaming: bool = True, tools: Sequence[Any] = ()) -> Agent:
    server = A2AServer(server_agent)
    return Agent(
        "client",
        config=A2AConfig(
            card_url=_URL,
            httpx_client_factory=make_test_client_factory(server, url=_URL),
            streaming=streaming,
            polling_interval=0.01,
        ),
        tools=list(tools),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [["he", "llo"], ["h", "e", "l", "l", "o"], ["hello"]], ids=str)
async def test_server_closes_streamed_text_artifact_exactly_once(chunks: list[str]) -> None:
    turn = [*(ModelMessageChunk(c) for c in chunks), ModelResponse(message=ModelMessage("".join(chunks)))]
    client = _client(Agent("server", config=TestConfig(*turn)))

    updates = await _text_artifact_updates(client)

    assert len({u.update.artifact.artifact_id for u in updates}) == 1
    assert [u.last_chunk for u in updates] == [False] * (len(updates) - 1) + [True]


@pytest.mark.asyncio
async def test_server_emits_no_closing_marker_when_nothing_streamed() -> None:
    client = _client(Agent("server", config=TestConfig("hi")))

    assert await _text_artifact_updates(client) == []


class _ToolTurnsConfig(ModelConfig):
    """Each turn streams its text (if any); every turn but the last then calls a client tool."""

    def __init__(self, texts: Sequence[str] = ("pre-", "mid-", "end")) -> None:
        self._texts = texts

    def copy(self) -> Self:
        return self

    def create(self) -> "_ToolTurnsClient":
        return _ToolTurnsClient(self._texts)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _ToolTurnsClient(LLMClient):
    def __init__(self, texts: Sequence[str]) -> None:
        self._texts = texts

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **_: Any) -> ModelResponse:
        turn = sum(isinstance(m, ToolResultEvent) for m in messages)
        text = self._texts[turn]
        if text:
            await context.send(ModelMessageChunk(text))
        if turn < len(self._texts) - 1:
            return ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(name="step", arguments="{}")]))
        return ModelResponse(message=ModelMessage(text))


def step() -> str:
    return "done"


@pytest.mark.asyncio
@_both_modes
async def test_tool_round_trips_return_only_the_final_turn_text(streaming: bool) -> None:
    client = _client(Agent("server", config=_ToolTurnsConfig()), streaming=streaming, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "end"


@pytest.mark.asyncio
async def test_tool_only_turn_opens_no_text_artifact() -> None:
    client = _client(Agent("server", config=_ToolTurnsConfig(("", "end"))), tools=[step])

    updates = await _text_artifact_updates(client)

    assert len({u.update.artifact.artifact_id for u in updates}) == 1
    assert [u.last_chunk for u in updates] == [False] * (len(updates) - 1) + [True]


@pytest.mark.asyncio
@_both_modes
async def test_answer_only_in_completion_message_emits_one_chunk(streaming: bool) -> None:
    pair = make_executor_pair(ScriptedExecutor(_answer_in_completion_message), streaming=streaming)
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(ModelMessageChunk).subscribe(collector.on_event)

    reply = await pair.client.ask("go", stream=stream)

    assert reply.response.content == "done"
    assert collector.chunks == ["done"]


class _ReusedArtifactExecutor(AgentExecutor):
    """Opens text artifact ``art`` and asks for a client tool; the continuation reuses ``art`` for the answer."""

    def __init__(self, answer: str = "end", *, append: bool = False) -> None:
        self._answer = answer
        self._append = append

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        continuation = request_context.current_task is not None
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        if not continuation:
            await event_queue.enqueue_event(
                Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        if continuation:
            await updater.add_artifact(
                parts=[Part(text=self._answer)], artifact_id="art", append=self._append, last_chunk=True
            )
            await updater.complete()
            return
        await updater.add_artifact(parts=[Part(text="pre-")], artifact_id="art")
        call = data_part(call_to_payload(ToolCallEvent(name="step", arguments="{}")), media_type=MIME_TOOL_CALL)
        await updater.add_artifact(parts=[call], artifact_id="call")
        await updater.requires_input()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
@_both_modes
async def test_artifact_id_reused_after_tool_round_trip_carries_the_answer(streaming: bool) -> None:
    pair = make_executor_pair(_ReusedArtifactExecutor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "end"


@pytest.mark.asyncio
@_both_modes
async def test_artifact_id_reused_after_tool_round_trip_streams_the_answer(streaming: bool) -> None:
    pair = make_executor_pair(_ReusedArtifactExecutor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(ModelMessageChunk).subscribe(collector.on_event)

    await client.ask("go", stream=stream)

    assert collector.chunks == ["pre-", "end"]


@pytest.mark.asyncio
@_both_modes
async def test_replacement_sharing_the_previous_turn_prefix_is_kept_whole(streaming: bool) -> None:
    pair = make_executor_pair(_ReusedArtifactExecutor("pre-final"), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "pre-final"


@pytest.mark.asyncio
async def test_live_append_to_an_artifact_from_the_previous_turn_adds_only_the_new_text() -> None:
    pair = make_executor_pair(_ReusedArtifactExecutor("end", append=True), streaming=True)
    client = Agent("client", config=pair.client.config, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "end"


class _MixedCallExecutor(AgentExecutor):
    """Opens an artifact of text plus an image, then replaces it with text plus a client-tool call."""

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        continuation = request_context.current_task is not None
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        if not continuation:
            await event_queue.enqueue_event(
                Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        if continuation:
            await updater.complete(message=updater.new_agent_message([Part(text="ok")]))
            return
        image = Part(url="http://test/image.png", media_type="image/png")
        call = data_part(call_to_payload(ToolCallEvent(name="step", arguments="{}")), media_type=MIME_TOOL_CALL)
        await updater.add_artifact(parts=[Part(text="look"), image], artifact_id="art")
        await updater.add_artifact(parts=[Part(text="look"), call], artifact_id="art", last_chunk=True)
        await updater.requires_input()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
@_both_modes
async def test_tool_call_introduced_by_a_mixed_artifact_replacement_runs_once(streaming: bool) -> None:
    pair = make_executor_pair(_MixedCallExecutor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])
    stream = MemoryStream()

    reply = await client.ask("go", stream=stream)

    results = [e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert reply.response.content == "ok"


def step_again() -> str:
    return "again"


class _GrowingCallsExecutor(AgentExecutor):
    """Streams two client-tool calls into one artifact: the first leaves it open, the second closes it."""

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        continuation = request_context.current_task is not None
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        if not continuation:
            await event_queue.enqueue_event(
                Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        if continuation:
            await updater.complete(message=updater.new_agent_message([Part(text="ok")]))
            return
        first = data_part(call_to_payload(ToolCallEvent(name="step", arguments="{}")), media_type=MIME_TOOL_CALL)
        second = data_part(call_to_payload(ToolCallEvent(name="step_again", arguments="{}")), media_type=MIME_TOOL_CALL)
        await updater.add_artifact(parts=[first], artifact_id="calls", last_chunk=False)
        await updater.add_artifact(parts=[second], artifact_id="calls", append=True, last_chunk=True)
        await updater.requires_input()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
@_both_modes
async def test_tool_calls_growing_in_one_artifact_all_run(streaming: bool) -> None:
    pair = make_executor_pair(_GrowingCallsExecutor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step, step_again])
    stream = MemoryStream()

    reply = await client.ask("go", stream=stream)

    results = [e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)]
    assert sorted(r.name for r in results) == ["step", "step_again"]
    assert reply.response.content == "ok"


class _Blocker:
    """A tool that never returns, so a task stays ``working`` until it is cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def hold(self, context: Context) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The model keeps streaming while the cancel is being processed.
            await context.send(ModelMessageChunk("late"))
            raise
        return "unreachable"


@pytest.mark.asyncio
async def test_cancelled_task_closes_the_streamed_text_artifact_for_good() -> None:
    blocker = _Blocker()
    turn = [ModelMessageChunk("partial"), ToolCallEvent(name="hold", arguments="{}"), "unused"]
    client = _client(Agent("server", config=TestConfig(*turn), tools=[blocker.hold]))
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(A2ATaskArtifactUpdate).subscribe(collector.on_event)

    asking = asyncio.create_task(client.ask("go", stream=stream))
    await blocker.started.wait()
    [task] = (await list_tasks(a2a_config(client))).tasks
    await cancel_task(a2a_config(client), task.id)

    with pytest.raises(A2ATaskCancelledError):
        await asking
    updates = [e for e in collector.events if isinstance(e, A2ATextArtifact)]
    assert [u.last_chunk for u in updates] == [False, True]
