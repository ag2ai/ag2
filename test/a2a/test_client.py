# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock

import pytest
from a2a.types import AgentCapabilities, AgentCard, DataPart, TextPart  # type: ignore

from autogen import ConversableAgent
from autogen.a2a import A2aRemoteAgent, MockClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    (
        pytest.param(
            "Hi, I am mock client!",
            id="str data",
        ),
        pytest.param(
            DataPart(data={"content": "Hi, I am mock client!", "role": "assistant"}),
            id="DataPart data",
        ),
    ),
)
async def test_answer_with_str(data: str | TextPart) -> None:
    # arrange
    remote_agent = A2aRemoteAgent(url="http://fake", name="mock-agent", client=MockClient(data))

    client_agent = AsyncMock(spec=ConversableAgent)
    client_agent.silent = True
    client_agent.name = "original"

    # act
    await remote_agent.a_receive("Hi, I am user! Who are you?", client_agent, request_reply=True)

    # assert message history
    assert remote_agent.chat_messages[client_agent] == [
        {"content": "Hi, I am user! Who are you?", "role": "user", "name": "original"},
        {"content": "Hi, I am mock client!", "role": "assistant", "name": "mock-agent"},
    ]

    # assert correct answer
    assert client_agent.a_receive.call_args[0][0] == {"content": "Hi, I am mock client!", "role": "assistant"}


@pytest.mark.asyncio
async def test_answer_with_text_part() -> None:
    # arrange
    remote_agent = A2aRemoteAgent(
        url="http://fake",
        name="mock-agent",
        client=MockClient(TextPart(text="Hi, I am mock client!")),
    )

    client_agent = AsyncMock(spec=ConversableAgent)
    client_agent.silent = True
    client_agent.name = "original"

    # act
    await remote_agent.a_receive("Hi, I am user! Who are you?", client_agent, request_reply=True)

    # assert message history
    assert remote_agent.chat_messages[client_agent] == [
        {"content": "Hi, I am user! Who are you?", "role": "user", "name": "original"},
        {"content": "Hi, I am mock client!", "role": "assistant", "name": "mock-agent"},
    ]

    # assert correct answer
    assert client_agent.a_receive.call_args[0][0] == {"content": "Hi, I am mock client!"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    (
        pytest.param(
            {"content": "Hi, I am mock client!", "name": "test-agent"},
            id="dict data",
        ),
        pytest.param(
            DataPart(data={"content": "Hi, I am mock client!", "name": "test-agent", "role": "assistant"}),
            id="DataPart data",
        ),
    ),
)
async def test_answer_with_dict(data: dict[str, Any] | DataPart) -> None:
    # arrange
    remote_agent = A2aRemoteAgent(url="http://fake", name="mock-agent", client=MockClient(data))

    client_agent = AsyncMock(spec=ConversableAgent)
    client_agent.silent = True
    client_agent.name = "original"

    # act
    await remote_agent.a_receive("Hi, I am user! Who are you?", client_agent, request_reply=True)

    # assert message history
    assert remote_agent.chat_messages[client_agent] == [
        {"content": "Hi, I am user! Who are you?", "role": "user", "name": "original"},
        {"content": "Hi, I am mock client!", "role": "assistant", "name": "test-agent"},
    ]

    # assert correct answer
    assert client_agent.a_receive.call_args[0][0] == {
        "content": "Hi, I am mock client!",
        "role": "assistant",
        "name": "test-agent",
    }


def test_build_agent_from_card() -> None:
    card = AgentCard(
        name="Test Agent",
        description="A test agent",
        url="http://test.example.com",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
        supports_authenticated_extended_card=False,
    )
    agent = A2aRemoteAgent.from_card(card)

    assert agent.name == "Test Agent"
    assert agent.url == "UNKNOWN"
    assert agent._agent_card == card
