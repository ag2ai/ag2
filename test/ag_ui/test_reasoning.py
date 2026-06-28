# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import pytest
from ag_ui.core import ReasoningMessage, UserMessage
from dirty_equals import IsPartialDict
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.config import LLMClient, ModelConfig
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelReasoning,
    ModelResponse,
    ToolCallsEvent,
)
from ag2.testing import TestConfig, TrackingConfig

from .utils import (
    assert_event_type,
    assert_no_event_type,
    collect_events,
    create_run_input,
    get_events_of_type,
)

pytestmark = pytest.mark.asyncio


class _ReasoningClient(LLMClient):
    """Emits a fixed sequence of ``ModelReasoning`` events followed by a
    final ``ModelMessage`` — used to drive the outbound subscriber under
    test. ``TestConfig`` cannot inject ``ModelReasoning`` events, so a
    custom client is required (mirrors the streaming pattern in
    ``test_empty_chunks.py``)."""

    def __init__(self, *chunks: str, final: str = "Done") -> None:
        self.chunks = chunks
        self.final = final

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        for chunk in self.chunks:
            await context.send(ModelReasoning(chunk))

        message = ModelMessage(self.final)
        await context.send(message)
        return ModelResponse(message=message, tool_calls=ToolCallsEvent([]))


class _ReasoningConfig(ModelConfig):
    def __init__(self, *chunks: str, final: str = "Done") -> None:
        self.chunks = chunks
        self.final = final

    def copy(self) -> Self:
        return self

    def create(self) -> _ReasoningClient:
        return _ReasoningClient(*self.chunks, final=self.final)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _CapturingClient(LLMClient):
    """Stores the full ``messages`` sequence handed to the LLM so the test
    can assert on the entire pre-LLM history. ``TrackingConfig`` only
    records ``messages[-1]`` and so cannot verify that a particular event
    sits *anywhere* in the list."""

    def __init__(self) -> None:
        self.messages: list[BaseEvent] = []

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        self.messages = list(messages)
        message = ModelMessage("Done")
        await context.send(message)
        return ModelResponse(message=message, tool_calls=ToolCallsEvent([]))


class _CapturingConfig(ModelConfig):
    def __init__(self) -> None:
        self.client = _CapturingClient()

    def copy(self) -> Self:
        return self

    def create(self) -> _CapturingClient:
        return self.client

    def create_files_client(self) -> None:
        raise NotImplementedError


class TestInboundReasoning:
    async def test_reasoning_message_becomes_model_reasoning_event(self) -> None:
        config = _CapturingConfig()
        agent = Agent("test_agent", config=config)
        stream = AGUIStream(agent)

        run_input = create_run_input(
            UserMessage(id="msg_1", content="Hi"),
            ReasoningMessage(id="msg_2", content="user is greeting me"),
        )

        await collect_events(stream, run_input)

        # ReasoningMessage from AG-UI history is restored as a
        # ``ModelReasoning`` event before the agent's turn runs, so the
        # LLM sees it in the messages list.
        assert ModelReasoning("user is greeting me") in config.client.messages

    async def test_empty_reasoning_message_dropped(self) -> None:
        tracking = TrackingConfig(TestConfig("Done"))
        agent = Agent("test_agent", config=tracking)
        stream = AGUIStream(agent)

        run_input = create_run_input(
            UserMessage(id="msg_1", content="Hi"),
            ReasoningMessage(id="msg_2", content=""),
        )

        await collect_events(stream, run_input)

        # Empty reasoning is dropped — last message handed to the LLM is
        # the user's ``ModelRequest``, not a ``ModelReasoning``.
        [(last_msg,)] = [call.args for call in tracking.mock.call_args_list]
        assert not isinstance(last_msg, ModelReasoning)


class TestOutboundReasoning:
    async def test_reasoning_chunks_emit_full_session(self) -> None:
        agent = Agent("test_agent", config=_ReasoningConfig("Thinking", " more"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        start = assert_event_type(events, "REASONING_START")
        message_id = start["messageId"]

        assert assert_event_type(events, "REASONING_MESSAGE_START") == IsPartialDict({
            "messageId": message_id,
            "role": "reasoning",
        })
        assert get_events_of_type(events, "REASONING_MESSAGE_CONTENT") == [
            IsPartialDict({"messageId": message_id, "delta": "Thinking"}),
            IsPartialDict({"messageId": message_id, "delta": " more"}),
        ]
        assert assert_event_type(events, "REASONING_MESSAGE_END") == IsPartialDict({
            "messageId": message_id,
        })
        assert assert_event_type(events, "REASONING_END") == IsPartialDict({
            "messageId": message_id,
        })

    async def test_reasoning_session_closes_before_text_message(self) -> None:
        agent = Agent("test_agent", config=_ReasoningConfig("thinking", final="Final answer"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        types = [e["type"] for e in events if e["type"].startswith(("REASONING_", "TEXT_MESSAGE_"))]
        reasoning_end_idx = types.index("REASONING_END")
        first_text_idx = next(i for i, t in enumerate(types) if t.startswith("TEXT_MESSAGE_"))
        assert reasoning_end_idx < first_text_idx

    async def test_empty_reasoning_chunk_skipped(self) -> None:
        agent = Agent("test_agent", config=_ReasoningConfig("", "real thought"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        assert get_events_of_type(events, "REASONING_MESSAGE_CONTENT") == [
            IsPartialDict({"delta": "real thought"}),
        ]

    async def test_no_reasoning_emits_no_reasoning_events(self) -> None:
        agent = Agent("test_agent", config=_ReasoningConfig(final="Hello"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "REASONING_START")
        assert_no_event_type(events, "REASONING_MESSAGE_START")
        assert_no_event_type(events, "REASONING_MESSAGE_CONTENT")
        assert_no_event_type(events, "REASONING_MESSAGE_END")
        assert_no_event_type(events, "REASONING_END")
