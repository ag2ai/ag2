# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import pytest
from a2a.compat.v0_3 import conversions as _v03_conversions
from a2a.compat.v0_3.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.compat.v0_3.types import SendMessageRequest as CompatSendMessageRequest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue

from autogen import ConversableAgent, LLMConfig
from autogen.a2a.agent_executor import AutogenAgentExecutor
from autogen.agentchat.remote import AgentService, ServiceResponse
from autogen.agentchat.remote.protocol import RequestMessage
from autogen.events.client_events import StreamEvent
from autogen.io.base import IOStream
from autogen.testing import TestAgent, ToolCall


def _make_request(text: str = "Say hi") -> RequestMessage:
    return RequestMessage(messages=[{"content": text, "role": "user"}])


async def _collect_responses(agent_service: AgentService, request: RequestMessage) -> list[ServiceResponse]:
    responses: list[ServiceResponse] = []
    async for response in agent_service(request):
        responses.append(response)
    return responses


@pytest.mark.asyncio()
async def test_streaming_chunks_emitted() -> None:
    """Verify that StreamEvent chunks from the LLM are yielded as streaming_text responses."""
    agent = ConversableAgent("test-agent")

    chunks = ["Hello", " ", "world", "!"]

    async def mock_streaming_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        iostream = IOStream.get_default()
        for chunk in chunks:
            iostream.send(StreamEvent(content=chunk))
            await asyncio.sleep(0)  # yield control to allow polling
        return True, "Hello world!"

    agent.a_generate_oai_reply = mock_streaming_reply

    service = AgentService(agent)
    responses = await _collect_responses(service, _make_request())

    streaming_responses = [r for r in responses if r.streaming_text is not None]
    message_responses = [r for r in responses if r.message is not None]

    # Should have streaming chunks
    assert len(streaming_responses) > 0
    streamed_text = "".join(r.streaming_text for r in streaming_responses)
    assert streamed_text == "Hello world!"

    # Should also have the final complete message
    assert len(message_responses) == 1
    assert message_responses[0].message["content"] == "Hello world!"


@pytest.mark.asyncio()
async def test_non_streaming_unchanged() -> None:
    """Verify that when no StreamEvent is produced, no streaming_text responses are yielded."""
    agent = ConversableAgent("test-agent")

    service = AgentService(agent)

    with TestAgent(agent, ["Hi there!"]):
        responses = await _collect_responses(service, _make_request())

    streaming_responses = [r for r in responses if r.streaming_text is not None]
    message_responses = [r for r in responses if r.message is not None]

    assert len(streaming_responses) == 0
    assert len(message_responses) == 1
    assert message_responses[0].message["content"] == "Hi there!"


@pytest.mark.asyncio()
async def test_streaming_through_executor() -> None:
    """Verify streaming text is sent as artifact-update events, not status-update events."""

    agent = ConversableAgent("test-agent")

    chunks = ["Token", "1", " Token", "2"]

    async def mock_streaming_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        iostream = IOStream.get_default()
        for chunk in chunks:
            iostream.send(StreamEvent(content=chunk))
            await asyncio.sleep(0)
        return True, "Token1 Token2"

    agent.a_generate_oai_reply = mock_streaming_reply

    executor = AutogenAgentExecutor(agent)
    event_queue = EventQueue()
    child_queue = await event_queue.tap()

    a2a_message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="test"))],
        message_id="msg-1",
        context_id="ctx-1",
    )

    compat_req = CompatSendMessageRequest(id="req-1", params=MessageSendParams(message=a2a_message))
    core_req = _v03_conversions.to_core_send_message_request(compat_req)
    context = RequestContext(call_context=ServerCallContext(), request=core_req)
    await executor.execute(context, event_queue)

    # Collect all events (1.0 SDK queues protobuf events; convert back for assertions).
    # `dequeue_event` in a2a-sdk 1.0 has no `no_wait` parameter — rely on the
    # outer `wait_for` timeout to bail out once the executor stops emitting.
    raw_events = []
    while True:
        try:
            raw = await asyncio.wait_for(child_queue.dequeue_event(), timeout=0.1)
            raw_events.append(raw)
        except (TimeoutError, asyncio.TimeoutError):
            break

    events: list[Any] = []
    for raw in raw_events:
        cls_name = type(raw).__name__
        if cls_name == "TaskStatusUpdateEvent":
            events.append(_v03_conversions.to_compat_task_status_update_event(raw))
        elif cls_name == "TaskArtifactUpdateEvent":
            events.append(_v03_conversions.to_compat_task_artifact_update_event(raw))
        else:
            events.append(raw)

    artifact_events = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]

    # Streaming chunks should be sent as artifact-update events
    assert len(artifact_events) > 1, f"Expected multiple artifact events, got {len(artifact_events)}"

    # All but the last artifact event should have last_chunk=False
    for ae in artifact_events[:-1]:
        assert ae.last_chunk is not True

    # Last artifact event should have last_chunk=True
    assert artifact_events[-1].last_chunk is True

    # Should have exactly one working status event (the initial one before streaming)
    working_events = [e for e in status_events if e.status.state == TaskState.working]
    assert len(working_events) == 1, f"Expected 1 working status event, got {len(working_events)}"

    # Should have a completed status event
    completed_events = [e for e in status_events if e.status.state == TaskState.completed]
    assert len(completed_events) == 1


@pytest.mark.asyncio()
async def test_streaming_with_tool_calls() -> None:
    """Verify that tool calls work correctly alongside streaming."""
    agent = ConversableAgent(
        "test-agent",
        llm_config=LLMConfig({"model": "gpt-5", "api_key": "wrong-key"}),
    )

    @agent.register_for_llm()
    def get_greeting() -> str:
        return "Hello from tool!"

    service = AgentService(agent)

    # First call returns a tool call (no streaming), second returns streamed text
    call_count = 0

    async def mock_tool_then_stream(*args: Any, **kwargs: Any) -> tuple[bool, str | dict]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Return tool call (no streaming events)
            return True, {
                "role": "assistant",
                "tool_calls": [ToolCall("get_greeting").to_message()["tool_calls"][0]],
            }
        else:
            # Stream the final response
            iostream = IOStream.get_default()
            for chunk in ["Final", " answer"]:
                iostream.send(StreamEvent(content=chunk))
                await asyncio.sleep(0)
            return True, "Final answer"

    agent.a_generate_oai_reply = mock_tool_then_stream

    responses = await _collect_responses(service, _make_request())

    streaming_responses = [r for r in responses if r.streaming_text is not None]
    message_responses = [r for r in responses if r.message is not None]

    # Should have streaming chunks from the second LLM call
    assert len(streaming_responses) > 0
    streamed_text = "".join(r.streaming_text for r in streaming_responses)
    assert streamed_text == "Final answer"

    # Should have message responses: tool call, tool result, and final answer
    assert len(message_responses) >= 2
