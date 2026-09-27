# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from ag_ui.core import ReasoningMessage, UserMessage
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import (
    ModelReasoning,
)
from ag2.testing import TestConfig, TrackingConfig
from test._helpers import LLMCalls

from .utils import (
    assert_event_type,
    assert_no_event_type,
    collect_events,
    create_run_input,
    get_events_of_type,
)

pytestmark = pytest.mark.asyncio


class TestInboundReasoning:
    async def test_reasoning_message_becomes_model_reasoning_event(self) -> None:
        calls = LLMCalls()
        agent = Agent("test_agent", config=TestConfig("Done"), middleware=[calls.middleware()])
        stream = AGUIStream(agent)

        run_input = create_run_input(
            UserMessage(id="msg_1", content="Hi"),
            ReasoningMessage(id="msg_2", content="user is greeting me"),
        )

        await collect_events(stream, run_input)

        # ReasoningMessage from AG-UI history is restored as a
        # ``ModelReasoning`` event before the agent's turn runs, so the
        # LLM sees it in the messages list.
        assert ModelReasoning("user is greeting me") in calls.messages[-1]

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
        agent = Agent("test_agent", config=TestConfig(ModelReasoning("Thinking"), ModelReasoning(" more"), "Done"))
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
        agent = Agent("test_agent", config=TestConfig(ModelReasoning("thinking"), "Final answer"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        types = [e["type"] for e in events if e["type"].startswith(("REASONING_", "TEXT_MESSAGE_"))]
        reasoning_end_idx = types.index("REASONING_END")
        first_text_idx = next(i for i, t in enumerate(types) if t.startswith("TEXT_MESSAGE_"))
        assert reasoning_end_idx < first_text_idx

    async def test_empty_reasoning_chunk_skipped(self) -> None:
        agent = Agent("test_agent", config=TestConfig(ModelReasoning(""), ModelReasoning("real thought"), "Done"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        assert get_events_of_type(events, "REASONING_MESSAGE_CONTENT") == [
            IsPartialDict({"delta": "real thought"}),
        ]

    async def test_no_reasoning_emits_no_reasoning_events(self) -> None:
        agent = Agent("test_agent", config=TestConfig("Hello"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "REASONING_START")
        assert_no_event_type(events, "REASONING_MESSAGE_START")
        assert_no_event_type(events, "REASONING_MESSAGE_CONTENT")
        assert_no_event_type(events, "REASONING_MESSAGE_END")
        assert_no_event_type(events, "REASONING_END")
