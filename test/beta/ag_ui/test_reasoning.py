# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelReasoning → AG-UI Thinking event forwarding."""

import pytest
from ag_ui.core import UserMessage
from dirty_equals import IsStr

from autogen.beta import Agent
from autogen.beta.ag_ui import AGUIStream
from autogen.beta.events import ModelReasoning
from autogen.beta.testing import TestConfig

from .utils import assert_event_type, assert_no_event_type, collect_events, create_run_input

pytestmark = pytest.mark.asyncio


class TestReasoningForwarding:
    async def test_reasoning_emits_thinking_events(self) -> None:
        """ModelReasoning before a reply produces the full Thinking event sequence."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelReasoning("Let me think about this carefully."),
                "The answer is 42.",
            ),
        )
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="What is the answer?"))

        events = await collect_events(stream, run_input)

        assert_event_type(events, "THINKING_START")
        start = assert_event_type(events, "THINKING_TEXT_MESSAGE_START")
        assert "message_id" in start

        content = assert_event_type(events, "THINKING_TEXT_MESSAGE_CONTENT")
        assert content["delta"] == "Let me think about this carefully."
        assert content["message_id"] == start["message_id"]

        end = assert_event_type(events, "THINKING_TEXT_MESSAGE_END")
        assert end["message_id"] == start["message_id"]

        assert_event_type(events, "THINKING_END")

    async def test_reasoning_followed_by_reply(self) -> None:
        """Both the thinking block and the reply are forwarded."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelReasoning("Thinking..."),
                "My reply.",
            ),
        )
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello"))

        events = await collect_events(stream, run_input)

        assert_event_type(events, "THINKING_START")
        assert_event_type(events, "TEXT_MESSAGE_CHUNK")

    async def test_no_reasoning_emits_no_thinking_events(self) -> None:
        """When the model does not reason, no Thinking events are emitted."""
        agent = Agent("test_agent", config=TestConfig("Just a reply."))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hello"))

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "THINKING_START")
        assert_no_event_type(events, "THINKING_TEXT_MESSAGE_START")
        assert_no_event_type(events, "THINKING_TEXT_MESSAGE_CONTENT")
        assert_no_event_type(events, "THINKING_END")

    async def test_thinking_events_precede_reply(self) -> None:
        """Thinking events appear before the reply in the event stream."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelReasoning("Reasoning first."),
                "Then replying.",
            ),
        )
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Go"))

        events = await collect_events(stream, run_input)
        types = [e["type"] for e in events]

        thinking_start = types.index("THINKING_START")
        # TEXT_MESSAGE_CHUNK or TEXT_MESSAGE_START should come after thinking
        text_idx = next(
            (i for i, t in enumerate(types) if t in ("TEXT_MESSAGE_CHUNK", "TEXT_MESSAGE_START")),
            len(types),
        )
        assert thinking_start < text_idx

    async def test_thinking_message_ids_are_consistent(self) -> None:
        """Start/content/end events all share the same message_id."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelReasoning("My reasoning."),
                "Done.",
            ),
        )
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="Hi"))

        events = await collect_events(stream, run_input)

        start = assert_event_type(events, "THINKING_TEXT_MESSAGE_START")
        content = assert_event_type(events, "THINKING_TEXT_MESSAGE_CONTENT")
        end = assert_event_type(events, "THINKING_TEXT_MESSAGE_END")

        assert start["message_id"] == content["message_id"] == end["message_id"]
        assert start["message_id"] == IsStr()
