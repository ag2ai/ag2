# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.errors import A2ATaskCancelledError, A2ATaskTerminalError
from ag2.a2a.events import A2ATaskArtifactUpdate, A2ATextArtifact
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

from ._helpers import ScriptedExecutor, make_executor_pair

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


async def _cancelled_midway(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="partial")], artifact_id="art")
    await updater.cancel()


async def _two_artifacts(updater: TaskUpdater, release: asyncio.Event) -> None:
    await updater.add_artifact(parts=[Part(text="first")], artifact_id="one", last_chunk=True)
    await updater.add_artifact(parts=[Part(text="-second")], artifact_id="two", last_chunk=True)
    await updater.complete()


def _release_on_first_chunk(stream: MemoryStream, executor: ScriptedExecutor) -> list[str]:
    """Collect chunk events; the executor holds its remaining output until the first one lands."""
    chunks: list[str] = []

    @stream.where(ModelMessageChunk).subscribe
    async def collect(event: BaseEvent) -> None:
        assert isinstance(event, ModelMessageChunk)
        chunks.append(event.content)
        executor.release.set()

    return chunks


@pytest.mark.asyncio
@_both_modes
async def test_growing_artifact_is_returned_in_full(streaming: bool) -> None:
    executor = ScriptedExecutor(_growing_artifact)
    pair = make_executor_pair(executor, streaming=streaming)
    stream = MemoryStream()
    chunks = _release_on_first_chunk(stream, executor)
    if streaming:
        # The in-process transport buffers a streamed response until the task ends,
        # so no chunk can arrive early to open the gate.
        executor.release.set()

    reply = await pair.client.ask("go", stream=stream)

    assert reply.response.content == "abc"
    assert "".join(chunks) == "abc"


@pytest.mark.asyncio
async def test_resent_artifact_replaces_instead_of_concatenating() -> None:
    pair = make_executor_pair(ScriptedExecutor(_resent_artifact), streaming=True)

    reply = await pair.client.ask("go")

    assert reply.response.content == "ab"


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

    assert isinstance(exc_info.value, A2ATaskTerminalError)
    assert exc_info.value.task.status.state == TaskState.TASK_STATE_CANCELED
    partial = "".join(part.text for artifact in exc_info.value.task.artifacts for part in artifact.parts)
    assert partial == "partial"


class _ChunkingConfig(ModelConfig):
    """Streams each chunk, then answers with their concatenation (a provider that streams)."""

    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = chunks

    def copy(self) -> Self:
        return self

    def create(self) -> "_ChunkingClient":
        return _ChunkingClient(self._chunks)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _ChunkingClient(LLMClient):
    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = chunks

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **_: Any) -> ModelResponse:
        for chunk in self._chunks:
            await context.send(ModelMessageChunk(chunk))
        return ModelResponse(message=ModelMessage("".join(self._chunks)))


async def _text_artifact_updates(agent: Agent) -> list[A2ATextArtifact]:
    stream = MemoryStream()
    updates: list[A2ATextArtifact] = []

    @stream.where(A2ATaskArtifactUpdate).subscribe
    async def collect(event: BaseEvent) -> None:
        if isinstance(event, A2ATextArtifact):
            updates.append(event)

    await agent.ask("go", stream=stream)
    return updates


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
    client = _client(Agent("server", config=_ChunkingConfig(chunks)))

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
    assert updates[-1].last_chunk


@pytest.mark.asyncio
@_both_modes
async def test_answer_only_in_completion_message_emits_one_chunk(streaming: bool) -> None:
    pair = make_executor_pair(ScriptedExecutor(_answer_in_completion_message), streaming=streaming)
    stream = MemoryStream()
    chunks: list[str] = []

    @stream.where(ModelMessageChunk).subscribe
    async def collect(event: BaseEvent) -> None:
        assert isinstance(event, ModelMessageChunk)
        chunks.append(event.content)

    reply = await pair.client.ask("go", stream=stream)

    assert reply.response.content == "done"
    assert chunks == ["done"]
