# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the AG-UI subscriber's handling of empty model events.

The AG-UI protocol forbids empty deltas (``MinLen(1)`` on
``TextMessageContentEvent.delta`` / ``TextMessageChunkEvent.delta``).
``map_events_to_ag_ui`` must drop empty ``ModelMessageChunk`` and empty
``ModelMessage`` events instead of forwarding them to the AG-UI encoder.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from ag_ui.core import UserMessage
from typing_extensions import Self

from autogen.beta import Agent, Context
from autogen.beta.ag_ui import AGUIStream
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    ToolCallsEvent,
)

from .utils import (
    assert_no_event_type,
    collect_events,
    create_run_input,
    get_events_of_type,
)

pytestmark = pytest.mark.asyncio


class _StreamingClient(LLMClient):
    """LLM client that emits a fixed sequence of ``ModelMessageChunk`` events
    followed by a single ``ModelMessage`` (mirroring how a streaming Gemini
    response surfaces to subscribers)."""

    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        for chunk in self.chunks:
            await context.send(ModelMessageChunk(chunk))

        full = "".join(self.chunks)
        message = ModelMessage(full) if full else None
        if message is not None:
            await context.send(message)

        return ModelResponse(message=message, tool_calls=ToolCallsEvent([]))


class _StreamingConfig(ModelConfig):
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks

    def copy(self) -> Self:
        return self

    def create(self) -> _StreamingClient:
        return _StreamingClient(*self.chunks)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _NonStreamingEmptyClient(LLMClient):
    """Emits a single empty ``ModelMessage`` — exercises the non-streaming
    branch of ``map_events_to_ag_ui``."""

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        message = ModelMessage("")
        await context.send(message)
        return ModelResponse(message=message, tool_calls=ToolCallsEvent([]))


class _NonStreamingEmptyConfig(ModelConfig):
    def copy(self) -> Self:
        return self

    def create(self) -> _NonStreamingEmptyClient:
        return _NonStreamingEmptyClient()

    def create_files_client(self) -> None:
        raise NotImplementedError


class TestEmptyModelMessageChunk:
    async def test_empty_chunk_does_not_open_text_message(self) -> None:
        # Empty chunk arrives first — must not emit TEXT_MESSAGE_START.
        # The non-empty chunk that follows opens the message normally.
        agent = Agent("test_agent", config=_StreamingConfig("", "Hello"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        starts = get_events_of_type(events, "TEXT_MESSAGE_START")
        contents = get_events_of_type(events, "TEXT_MESSAGE_CONTENT")

        # Exactly one start (from the non-empty chunk), one content event.
        assert len(starts) == 1
        assert len(contents) == 1
        assert contents[0]["delta"] == "Hello"

    async def test_empty_chunk_between_real_chunks_dropped(self) -> None:
        agent = Agent("test_agent", config=_StreamingConfig("Hello", "", " world"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        contents = get_events_of_type(events, "TEXT_MESSAGE_CONTENT")
        assert [c["delta"] for c in contents] == ["Hello", " world"]

        # No content event ever carries an empty delta.
        for event in contents:
            assert event["delta"] != ""


class TestEmptyModelMessage:
    async def test_empty_non_streaming_message_emits_no_text_event(self) -> None:
        agent = Agent("test_agent", config=_NonStreamingEmptyConfig())
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert_no_event_type(events, "TEXT_MESSAGE_CONTENT")
        assert_no_event_type(events, "TEXT_MESSAGE_START")
