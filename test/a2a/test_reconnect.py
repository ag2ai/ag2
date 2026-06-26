# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.client.errors import A2AClientError
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Artifact,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from ag2 import Agent, Context
from ag2.a2a.client import A2AClient, A2ADriveState
from ag2.a2a.errors import A2AReconnectError
from ag2.a2a.mappers.messages import build_user_message
from ag2.events import TextInput
from ag2.stream import MemoryStream
from ag2.testing import TestConfig


def _make_agent_card() -> AgentCard:
    return AgentCard(
        name="t",
        description="",
        version="1",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )


def _make_context() -> Context:
    agent = Agent("client", config=TestConfig("unused"))
    return Context(
        stream=MemoryStream(),
        prompt=[],
        dependencies={},
        variables={},
        dependency_provider=agent.dependency_provider,
    )


def _task_event(task_id: str, context_id: str) -> StreamResponse:
    return StreamResponse(
        task=Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_WORKING)),
    )


def _artifact_event(task_id: str, context_id: str, artifact_id: str, text: str, *, last_chunk: bool) -> StreamResponse:
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
            append=False,
            last_chunk=last_chunk,
        ),
    )


def _completed_event(task_id: str, context_id: str) -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        ),
    )


class _ScriptedSdk:
    def __init__(
        self,
        *,
        first_events: list[StreamResponse],
        drop_after: int,
        replay_events: list[StreamResponse],
    ) -> None:
        self._first_events = first_events
        self._drop_after = drop_after
        self._replay_events = replay_events
        self.send_message_calls = 0
        self.subscribe_calls = 0

    def send_message(self, _request: Any) -> AsyncIterator[StreamResponse]:
        self.send_message_calls += 1
        return self._scripted(self._first_events, drop_after=self._drop_after)

    def subscribe(self, _request: Any) -> AsyncIterator[StreamResponse]:
        self.subscribe_calls += 1
        return self._scripted(self._replay_events, drop_after=None)

    async def _scripted(self, events: list[StreamResponse], *, drop_after: int | None) -> AsyncIterator[StreamResponse]:
        for i, ev in enumerate(events):
            yield ev
            if drop_after is not None and i + 1 == drop_after:
                raise A2AClientError("simulated stream drop")


def _attach_mock(client: A2AClient, sdk: _ScriptedSdk) -> None:
    client._agent_card = _make_agent_card()
    client._sdk_client = sdk  # type: ignore[assignment]


@pytest.mark.asyncio
class TestStreamingReconnect:
    async def test_reconnect_after_drop_resumes_stream(self) -> None:
        task_id, context_id = "task-1", "ctx-1"
        first = [
            _task_event(task_id, context_id),
            _artifact_event(task_id, context_id, "art-1", "hello", last_chunk=True),
        ]
        replay = [
            _artifact_event(task_id, context_id, "art-2", " world", last_chunk=True),
            _completed_event(task_id, context_id),
        ]
        sdk = _ScriptedSdk(first_events=first, drop_after=2, replay_events=replay)

        client = A2AClient(card_url="http://test", max_reconnects=3, reconnect_backoff=0.0)
        _attach_mock(client, sdk)

        ctx = _make_context()
        message = build_user_message([TextInput("ping")], task_id=None, context_id=None)
        state = A2ADriveState()
        outcome = await client._consume_streaming(message, ctx, state)

        assert sdk.subscribe_calls == 1
        assert state.accumulated_text == "hello world"
        assert outcome.input_required is False

    async def test_reconnect_dedupes_replayed_artifact(self) -> None:
        task_id, context_id = "task-2", "ctx-2"
        first = [
            _task_event(task_id, context_id),
            _artifact_event(task_id, context_id, "art-1", "hello", last_chunk=True),
        ]
        # Server may resend the same final artifact on resubscribe (per A2A spec).
        replay = [
            _artifact_event(task_id, context_id, "art-1", "hello", last_chunk=True),
            _completed_event(task_id, context_id),
        ]
        sdk = _ScriptedSdk(first_events=first, drop_after=2, replay_events=replay)

        client = A2AClient(card_url="http://test", max_reconnects=3, reconnect_backoff=0.0)
        _attach_mock(client, sdk)

        ctx = _make_context()
        message = build_user_message([TextInput("ping")], task_id=None, context_id=None)
        state = A2ADriveState()
        await client._consume_streaming(message, ctx, state)

        assert state.accumulated_text == "hello"

    async def test_reconnect_exhausted_raises(self) -> None:
        task_id, context_id = "task-3", "ctx-3"
        first = [
            _task_event(task_id, context_id),
            _artifact_event(task_id, context_id, "art-1", "hi", last_chunk=True),
        ]
        sdk = _ScriptedSdk(first_events=first, drop_after=2, replay_events=first)

        async def _failing_subscribe(*args: Any, **kwargs: Any) -> AsyncIterator[StreamResponse]:
            sdk.subscribe_calls += 1
            raise A2AClientError("replay also dropped")
            yield

        sdk.subscribe = _failing_subscribe  # type: ignore[assignment]

        client = A2AClient(card_url="http://test", max_reconnects=2, reconnect_backoff=0.0)
        _attach_mock(client, sdk)

        ctx = _make_context()
        message = build_user_message([TextInput("ping")], task_id=None, context_id=None)
        state = A2ADriveState()
        with pytest.raises(A2AReconnectError):
            await client._consume_streaming(message, ctx, state)

        assert sdk.subscribe_calls >= 1
