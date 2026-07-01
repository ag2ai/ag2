# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.events import ModelRequest, ModelResponse
from ag2.testing import TestConfig


@pytest.mark.asyncio
async def test_repeated_user_message_is_persisted_each_time() -> None:
    # Regression: ``MemoryStorage.save_event`` used to drop value-equal
    # events. A user asking the same question twice (e.g. "как меня
    # зовут?" at turn 4 and turn 7 of demo_a2a_orchestrator_client)
    # would silently lose the second turn, leaving the conversation
    # history ending on an assistant message and breaking Anthropic
    # callers that require the last message to be ``user``.
    agent = Agent(
        "test-agent",
        config=TestConfig("first answer", "second answer", "third answer"),
    )

    reply = await agent.ask("repeating question")
    reply = await reply.ask("different question")
    reply = await reply.ask("repeating question")

    events = list(await reply.context.stream.history.get_events())

    user_inputs = [e.parts[0].content for e in events if isinstance(e, ModelRequest)]
    assert user_inputs == ["repeating question", "different question", "repeating question"]

    assert isinstance(events[-1], ModelResponse)
    assert events[-1].message is not None and events[-1].message.content == "third answer"


@pytest.mark.asyncio
async def test_repeated_assistant_response_is_persisted_each_time() -> None:
    # Mirror case: the LLM may legitimately produce identical text on
    # different turns; the storage must keep both so subsequent
    # provider conversion sees the conversation alternating correctly.
    agent = Agent(
        "test-agent",
        config=TestConfig("same answer", "same answer"),
    )

    reply = await agent.ask("first question")
    reply = await reply.ask("second question")

    events = list(await reply.context.stream.history.get_events())

    assistant_messages = [e for e in events if isinstance(e, ModelResponse)]
    assert len(assistant_messages) == 2
    assert all(m.message is not None and m.message.content == "same answer" for m in assistant_messages)


@pytest.mark.asyncio
async def test_history_preserves_user_assistant_alternation() -> None:
    agent = Agent(
        "test-agent",
        config=TestConfig("a", "b", "c"),
    )

    reply = await agent.ask("q1")
    reply = await reply.ask("q1")  # duplicate
    reply = await reply.ask("q1")  # duplicate again

    events = list(await reply.context.stream.history.get_events())

    roles = [type(e).__name__ for e in events if isinstance(e, (ModelRequest, ModelResponse))]
    assert roles == ["ModelRequest", "ModelResponse"] * 3
