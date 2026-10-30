# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Artifact, Message, Part, Role, Task, TaskState, TaskStatus
from a2a.utils.errors import TaskNotCancelableError

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.errors import (
    A2ATaskCancelledError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
    A2ATaskTerminalError,
)
from ag2.a2a.events import A2ATaskArtifactUpdate, A2ATaskStatusUpdate, A2ATextArtifact
from ag2.a2a.extension import MIME_TOOL_CALL
from ag2.a2a.mappers.parts import data_part
from ag2.a2a.mappers.tools import call_to_payload
from ag2.a2a.tasks import cancel_task, get_task, list_tasks
from ag2.a2a.testing import (
    ObservedAgentExecutor,
    TaskStatusRecorder,
    ToolRoundsConfig,
    TurnGate,
    make_test_client_factory,
)
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.observers import observer
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

from ._helpers import EventCollector, ScriptedExecutor, ToolRoundTripExecutor, a2a_config, make_executor_pair

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


async def _answer_in_artifact_and_completion_message(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="done")], artifact_id="art", last_chunk=True)
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


async def _progress_note_then_answer(updater: TaskUpdater, release: asyncio.Event) -> None:
    note = updater.new_agent_message([Part(text="thinking...")])
    await updater.update_status(TaskState.TASK_STATE_WORKING, message=note)
    await updater.add_artifact(parts=[Part(text="answer")], artifact_id="art", last_chunk=True)
    await updater.complete()


async def _held_until_cancelled(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="partial")], artifact_id="art")
    await asyncio.Event().wait()


class _CompletesWhenCancelled(ScriptedExecutor):
    """Its cancel hook lets the task finish instead, so the task ends COMPLETED under the cancel."""

    def __init__(self) -> None:
        super().__init__(self._held_then_completed)
        self.completed = asyncio.Event()

    async def _held_then_completed(self, updater: TaskUpdater, release: asyncio.Event) -> None:
        await updater.add_artifact(parts=[Part(text="done")], artifact_id="art")
        await release.wait()
        await updater.complete()
        self.completed.set()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        self.release.set()
        await self.completed.wait()


class _ArtifactSavedStore(InMemoryTaskStore):
    """Task store that signals once a task carrying an artifact has been persisted."""

    def __init__(self) -> None:
        super().__init__()
        self.artifact_saved = asyncio.Event()

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await super().save(task, context)
        if task.artifacts:
            self.artifact_saved.set()


class _FinishedTaskExecutor(A2AAgentExecutorBase):
    """Answers with a single ``Task`` already in its final state: a task stream with no update events."""

    def __init__(self, state: TaskState, *, status_text: str | None = None) -> None:
        self._state = state
        self._status_text = status_text

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        status = TaskStatus(state=self._state)
        if self._status_text is not None:
            status.message.CopyFrom(
                Message(message_id=uuid4().hex, role=Role.ROLE_AGENT, parts=[Part(text=self._status_text)])
            )
        await event_queue.enqueue_event(
            Task(
                id=msg.task_id or uuid4().hex,
                context_id=msg.context_id or uuid4().hex,
                status=status,
                artifacts=[Artifact(artifact_id="art", parts=[Part(text="partial")])],
            )
        )

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


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
    assert isinstance(exc_info.value, A2ATaskTerminalError)
    assert (task.status.state, _artifact_texts(task)) == (TaskState.TASK_STATE_CANCELED, {"art": "partial"})


def _artifact_texts(task: Task) -> dict[str, str]:
    return {artifact.artifact_id: "".join(part.text for part in artifact.parts) for artifact in task.artifacts}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "error"),
    [
        pytest.param(TaskState.TASK_STATE_CANCELED, A2ATaskCancelledError, id="cancelled"),
        pytest.param(TaskState.TASK_STATE_FAILED, A2ATaskFailedError, id="failed"),
        pytest.param(TaskState.TASK_STATE_REJECTED, A2ATaskRejectedError, id="rejected"),
    ],
)
@_both_modes
async def test_task_that_arrives_already_finished_raises(
    state: TaskState, error: type[A2ATaskTerminalError], streaming: bool
) -> None:
    pair = make_executor_pair(_FinishedTaskExecutor(state), streaming=streaming)

    with pytest.raises(error) as exc_info:
        await pair.client.ask("go")

    task = exc_info.value.task
    assert (task.status.state, _artifact_texts(task)) == (state, {"art": "partial"})


@pytest.mark.asyncio
@_both_modes
async def test_task_that_arrives_already_completed_keeps_its_status_message(streaming: bool) -> None:
    pair = make_executor_pair(
        _FinishedTaskExecutor(TaskState.TASK_STATE_COMPLETED, status_text="-done"), streaming=streaming
    )

    reply = await pair.client.ask("go")

    assert reply.response.content == "partial-done"


@pytest.mark.asyncio
@_both_modes
async def test_progress_note_on_a_working_task_is_not_part_of_the_reply(streaming: bool) -> None:
    pair = make_executor_pair(ScriptedExecutor(_progress_note_then_answer), streaming=streaming)

    reply = await pair.client.ask("go")

    assert reply.response.content == "answer"


@pytest.mark.asyncio
async def test_cancel_that_never_reaches_the_stream_still_raises() -> None:
    # The executor's cancel hook publishes nothing; the SDK records CANCELED in the task
    # store after the stream's queue is gone, so the stream closes without a final status.
    store = _ArtifactSavedStore()
    pair = make_executor_pair(ScriptedExecutor(_held_until_cancelled), streaming=True, task_store=store)

    asking = asyncio.create_task(pair.client.ask("go"))
    await store.artifact_saved.wait()
    [task] = (await list_tasks(a2a_config(pair.client))).tasks
    await cancel_task(a2a_config(pair.client), task.id)

    with pytest.raises(A2ATaskCancelledError) as exc_info:
        await asyncio.wait_for(asking, timeout=5)
    assert _artifact_texts(exc_info.value.task) == {"art": "partial"}


@pytest.mark.asyncio
async def test_cancel_outrun_by_the_task_finishing_is_refused() -> None:
    # A task that ends some other way while the cancel is in flight was not cancelable (spec §3.1.5).
    store = _ArtifactSavedStore()
    pair = make_executor_pair(_CompletesWhenCancelled(), streaming=True, task_store=store)

    asking = asyncio.create_task(pair.client.ask("go"))
    await store.artifact_saved.wait()
    [task] = (await list_tasks(a2a_config(pair.client))).tasks

    with pytest.raises(TaskNotCancelableError):
        await cancel_task(a2a_config(pair.client), task.id)
    reply = await asyncio.wait_for(asking, timeout=5)
    assert reply.body == "done"
    finished = await get_task(a2a_config(pair.client), task.id)
    assert finished.status.state == TaskState.TASK_STATE_COMPLETED


async def _text_artifact_updates(agent: Agent) -> list[A2ATextArtifact]:
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(A2ATaskArtifactUpdate).subscribe(collector.on_event)

    await agent.ask("go", stream=stream)

    return [e for e in collector.events if isinstance(e, A2ATextArtifact)]


def _closing_flags_per_artifact(updates: Sequence[A2ATextArtifact]) -> list[list[bool]]:
    """``last_chunk`` of every update, grouped by artifact in order of first appearance."""
    flags: dict[str, list[bool]] = {}
    for u in updates:
        flags.setdefault(u.update.artifact.artifact_id, []).append(u.last_chunk)
    return list(flags.values())


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

    assert _closing_flags_per_artifact(updates) == [[False] * (len(updates) - 1) + [True]]


@pytest.mark.asyncio
async def test_server_emits_no_closing_marker_when_nothing_streamed() -> None:
    client = _client(Agent("server", config=TestConfig("hi")))

    assert await _text_artifact_updates(client) == []


def step() -> str:
    return "done"


def _call_step() -> ToolCallEvent:
    return ToolCallEvent(name="step", arguments="{}")


@pytest.mark.asyncio
@_both_modes
async def test_tool_round_trips_return_only_the_final_turn_text(streaming: bool) -> None:
    config = ToolRoundsConfig(
        TestConfig(ModelMessageChunk("pre-"), _call_step()),
        TestConfig(ModelMessageChunk("mid-"), _call_step()),
        TestConfig(ModelMessageChunk("end"), ModelResponse(message=ModelMessage("end"))),
    )
    client = _client(Agent("server", config=config), streaming=streaming, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "end"


@pytest.mark.asyncio
async def test_tool_only_turn_opens_no_text_artifact() -> None:
    config = ToolRoundsConfig(
        _call_step(),
        TestConfig(ModelMessageChunk("end"), ModelResponse(message=ModelMessage("end"))),
    )
    client = _client(Agent("server", config=config), tools=[step])

    updates = await _text_artifact_updates(client)

    assert _closing_flags_per_artifact(updates) == [[False] * (len(updates) - 1) + [True]]


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


@pytest.mark.asyncio
@_both_modes
async def test_completion_message_restating_the_artifact_answer_is_not_doubled(streaming: bool) -> None:
    pair = make_executor_pair(ScriptedExecutor(_answer_in_artifact_and_completion_message), streaming=streaming)
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(ModelMessageChunk).subscribe(collector.on_event)

    reply = await pair.client.ask("go", stream=stream)

    assert reply.response.content == "done"
    assert collector.chunks == ["done"]


def _call_part(name: str) -> Part:
    return data_part(call_to_payload(ToolCallEvent(name=name, arguments="{}")), media_type=MIME_TOOL_CALL)


async def _text_then_tool_call(updater: TaskUpdater) -> None:
    await updater.add_artifact(parts=[Part(text="pre-")], artifact_id="art")
    await updater.add_artifact(parts=[_call_part("step")], artifact_id="call")
    await updater.requires_input()


async def _answer_in_reused_artifact(updater: TaskUpdater, *, answer: str, append: bool) -> None:
    await updater.add_artifact(parts=[Part(text=answer)], artifact_id="art", append=append, last_chunk=True)
    await updater.complete()


def _reused_artifact_executor(answer: str = "end", *, append: bool = False) -> ToolRoundTripExecutor:
    """Opens text artifact ``art`` and asks for a client tool; the continuation reuses ``art`` for the answer."""
    return ToolRoundTripExecutor(
        _text_then_tool_call, partial(_answer_in_reused_artifact, answer=answer, append=append)
    )


@pytest.mark.asyncio
@_both_modes
async def test_artifact_id_reused_after_tool_round_trip_carries_the_answer(streaming: bool) -> None:
    pair = make_executor_pair(_reused_artifact_executor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "end"


@pytest.mark.asyncio
@_both_modes
async def test_artifact_id_reused_after_tool_round_trip_streams_the_answer(streaming: bool) -> None:
    pair = make_executor_pair(_reused_artifact_executor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(ModelMessageChunk).subscribe(collector.on_event)

    await client.ask("go", stream=stream)

    assert collector.chunks == ["pre-", "end"]


@pytest.mark.asyncio
@_both_modes
async def test_replacement_sharing_the_previous_turn_prefix_is_kept_whole(streaming: bool) -> None:
    pair = make_executor_pair(_reused_artifact_executor("pre-final"), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "pre-final"


@pytest.mark.asyncio
@_both_modes
async def test_replacement_equal_to_the_previous_turn_text_adds_nothing(streaming: bool) -> None:
    # A non-append update restates the whole artifact; restating it unchanged leaves nothing new to say.
    pair = make_executor_pair(_reused_artifact_executor("pre-"), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(ModelMessageChunk).subscribe(collector.on_event)

    await client.ask("go", stream=stream)

    assert collector.chunks == ["pre-"]


@pytest.mark.asyncio
async def test_live_append_to_an_artifact_from_the_previous_turn_adds_only_the_new_text() -> None:
    pair = make_executor_pair(_reused_artifact_executor("end", append=True), streaming=True)
    client = Agent("client", config=pair.client.config, tools=[step])

    reply = await client.ask("go")

    assert reply.response.content == "end"


async def _mixed_artifact_replaced_by_call(updater: TaskUpdater) -> None:
    image = Part(url="http://test/image.png", media_type="image/png")
    await updater.add_artifact(parts=[Part(text="look"), image], artifact_id="art")
    await updater.add_artifact(parts=[Part(text="look"), _call_part("step")], artifact_id="art", last_chunk=True)
    await updater.requires_input()


def _mixed_call_executor() -> ToolRoundTripExecutor:
    """Opens an artifact of text plus an image, then replaces it with text plus a client-tool call."""
    return ToolRoundTripExecutor(_mixed_artifact_replaced_by_call)


@pytest.mark.asyncio
@_both_modes
async def test_tool_call_introduced_by_a_mixed_artifact_replacement_runs_once(streaming: bool) -> None:
    pair = make_executor_pair(_mixed_call_executor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])
    stream = MemoryStream()

    reply = await client.ask("go", stream=stream)

    results = [e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)]
    assert [str(r.name) for r in results] == ["step"]
    assert reply.response.content == "ok"


def step_again() -> str:
    return "again"


async def _calls_growing_in_one_artifact(updater: TaskUpdater) -> None:
    await updater.add_artifact(parts=[_call_part("step")], artifact_id="calls", last_chunk=False)
    await updater.add_artifact(parts=[_call_part("step_again")], artifact_id="calls", append=True, last_chunk=True)
    await updater.requires_input()


def _growing_calls_executor() -> ToolRoundTripExecutor:
    """Streams two client-tool calls into one artifact: the first leaves it open, the second closes it."""
    return ToolRoundTripExecutor(_calls_growing_in_one_artifact)


@pytest.mark.asyncio
@_both_modes
async def test_tool_calls_growing_in_one_artifact_all_run(streaming: bool) -> None:
    pair = make_executor_pair(_growing_calls_executor(), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step, step_again])
    stream = MemoryStream()

    reply = await client.ask("go", stream=stream)

    results = [e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)]
    assert sorted(str(r.name) for r in results) == ["step", "step_again"]
    assert reply.response.content == "ok"


async def _tool_call_then_complete(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="hi")], artifact_id="art", last_chunk=True)
    await updater.add_artifact(parts=[_call_part("step")], artifact_id="call", last_chunk=True)
    await updater.complete()


async def _mixed_tool_call_then_complete(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="hi"), _call_part("step")], artifact_id="art", last_chunk=True)
    await updater.complete()


@pytest.mark.asyncio
@_both_modes
@pytest.mark.parametrize(
    "script", [_tool_call_then_complete, _mixed_tool_call_then_complete], ids=["call-artifact", "mixed-artifact"]
)
async def test_tool_call_on_a_task_that_completes_is_not_run(
    streaming: bool,
    script: Callable[[TaskUpdater, asyncio.Event], Awaitable[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    pair = make_executor_pair(ScriptedExecutor(script), streaming=streaming)
    client = Agent("client", config=pair.client.config, tools=[step])
    stream = MemoryStream()

    with caplog.at_level(logging.WARNING, logger="ag2.a2a.client"):
        reply = await client.ask("go", stream=stream)

    assert [e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)] == []
    assert reply.response.content == "hi"
    assert [r.getMessage() for r in caplog.records if r.name == "ag2.a2a.client"] == [
        "A2A peer requested client tool calls ['step'] on a task that did not wait for input; ignoring them"
    ]


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
    config = TestConfig(ModelMessageChunk("partial"), ToolCallEvent(name="hold", arguments="{}"), "unused")
    client = _client(Agent("server", config=config, tools=[blocker.hold]))
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


@pytest.mark.asyncio
async def test_cancelling_a_task_waiting_for_a_client_tool_marks_it_cancelled() -> None:
    # No turn runs while the task waits for input, so the executor writes nothing; the SDK closes it.
    blocker = _Blocker()
    server = Agent("server", config=TestConfig(ToolCallEvent(name="hold", arguments="{}"), "unused"))
    client = _client(server, tools=[blocker.hold])

    asking = asyncio.create_task(client.ask("go"))
    await blocker.started.wait()
    [task] = (await list_tasks(a2a_config(client))).tasks
    await cancel_task(a2a_config(client), task.id)
    asking.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asking
    cancelled = await get_task(a2a_config(client), task.id)
    assert cancelled.status.state == TaskState.TASK_STATE_CANCELED


class _FailOn:
    """Lifecycle observer that raises ``error`` when the task reaches ``state``."""

    def __init__(self, state: TaskState, error: BaseException) -> None:
        self.state = state
        self.error = error

    async def on_status(self, event: A2ATaskStatusUpdate) -> None:
        if event.state == self.state:
            raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("observer bug"), asyncio.CancelledError()], ids=["error", "cancelled"])
async def test_failing_observer_does_not_turn_a_cancel_into_a_failure(error: BaseException) -> None:
    # CANCELED is terminal (spec §4.1.3): an observer failing on it must not let the SDK write FAILED after it.
    gate = TurnGate()
    statuses = TaskStatusRecorder()
    server = Agent("server", config=TestConfig(ToolCallEvent(name="hold", arguments="{}"), "unused"), tools=[gate.hold])
    executor = ObservedAgentExecutor(
        server, statuses, observer(A2ATaskStatusUpdate, _FailOn(TaskState.TASK_STATE_CANCELED, error).on_status)
    )
    client = make_executor_pair(executor, streaming=True).client

    asking = asyncio.create_task(client.ask("go"))
    await gate.entered.wait()
    [task] = (await list_tasks(a2a_config(client))).tasks
    cancelled = await cancel_task(a2a_config(client), task.id)

    with pytest.raises(A2ATaskCancelledError):
        await asyncio.wait_for(asking, timeout=5)
    assert cancelled.status.state == TaskState.TASK_STATE_CANCELED
    stored = await get_task(a2a_config(client), task.id)
    assert stored.status.state == TaskState.TASK_STATE_CANCELED
    assert statuses.states == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_CANCELED]


@pytest.mark.asyncio
@_both_modes
async def test_failing_observer_does_not_fail_a_completed_task(streaming: bool) -> None:
    statuses = TaskStatusRecorder()
    server = Agent("server", config=TestConfig("done"))
    executor = ObservedAgentExecutor(
        server,
        statuses,
        observer(A2ATaskStatusUpdate, _FailOn(TaskState.TASK_STATE_COMPLETED, ValueError("observer bug")).on_status),
    )
    client = make_executor_pair(executor, streaming=streaming).client

    reply = await client.ask("go")

    assert reply.response.content == "done"
    [task] = (await list_tasks(a2a_config(client))).tasks
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert statuses.states == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_COMPLETED]


async def _ask_human(updater: TaskUpdater) -> None:
    await updater.requires_input(message=updater.new_agent_message([Part(text="Confirm?")]))


async def _complete_with_tool_call(updater: TaskUpdater) -> None:
    call = data_part(call_to_payload(ToolCallEvent(name="step", arguments="{}")), media_type=MIME_TOOL_CALL)
    await updater.add_artifact([call], artifact_id="call")
    await updater.complete(message=updater.new_agent_message([Part(text="done")]))


@pytest.mark.asyncio
@_both_modes
async def test_completed_hitl_continuation_does_not_execute_client_tools(streaming: bool) -> None:
    calls: list[str] = []

    async def confirm() -> str:
        return "yes"

    def step() -> str:
        calls.append("ran")
        return "result"

    pair = make_executor_pair(
        ToolRoundTripExecutor(_ask_human, _complete_with_tool_call), streaming=streaming, hitl_hook=confirm
    )
    pair.client.tool(step)

    reply = await pair.client.ask("go")

    assert calls == []
    assert reply.response.content == "done"
    assert reply.response.finish_reason == "completed"


async def _ask_human_with_closed_artifact(updater: TaskUpdater) -> None:
    await updater.add_artifact([Part(text="before confirmation")], artifact_id="art", last_chunk=True)
    await _ask_human(updater)


class _ToolAfterHuman:
    def __init__(self) -> None:
        self.turns = 0

    async def __call__(self, updater: TaskUpdater) -> None:
        self.turns += 1
        if self.turns == 1:
            await updater.add_artifact([_call_part("step")], artifact_id="art", last_chunk=True)
            await updater.requires_input()
        else:
            await updater.complete(message=updater.new_agent_message([Part(text="done")]))


@pytest.mark.asyncio
@_both_modes
@pytest.mark.parametrize("with_tool", [False, True], ids=["text", "tool-call"])
async def test_closed_artifact_id_can_be_reused_after_human_input(streaming: bool, with_tool: bool) -> None:
    prompts: list[str] = []

    async def confirm() -> str:
        prompts.append("asked")
        assert len(prompts) == 1, "The continuation should be read instead of asking the human again"
        return "yes"

    continuation = _ToolAfterHuman() if with_tool else partial(_answer_in_reused_artifact, answer="done", append=False)
    pair = make_executor_pair(
        ToolRoundTripExecutor(_ask_human_with_closed_artifact, continuation), streaming=streaming, hitl_hook=confirm
    )
    pair.client.tool(step)
    stream = MemoryStream()

    reply = await pair.client.ask("go", stream=stream)

    assert reply.response.content == "done"
    assert prompts == ["asked"]
    results = [e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)]
    assert [r.name for r in results] == (["step"] if with_tool else [])
