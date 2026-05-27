# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock

import pytest
from a2a.compat.v0_3.types import AgentCapabilities, AgentCard, DataPart, Message, Role, TextPart  # type: ignore

from autogen import ConversableAgent
from autogen.a2a import A2aRemoteAgent, MockClient
from autogen.a2a.errors import A2aClientError


class NoEventClient:
    async def send_message(self, message: Message, **kwargs: Any):
        if False:
            yield

    async def resubscribe(self, params):
        if False:
            yield

    async def get_task(self, params):
        raise AssertionError("get_task should not be called")


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


@pytest.mark.asyncio
async def test_streaming_raises_when_no_task_started() -> None:
    card = AgentCard(
        name="Test Agent",
        description="A test agent",
        url="http://test.example.com",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
        supports_authenticated_extended_card=False,
    )
    agent = A2aRemoteAgent.from_card(card)
    message = Message(message_id="msg_1", role=Role.user, parts=[])

    with pytest.raises(A2aClientError, match="Failed to connect to the agent"):
        async for _ in agent._ask_streaming(NoEventClient(), message):
            pass


@pytest.mark.asyncio
async def test_polling_raises_when_no_task_started() -> None:
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
    message = Message(message_id="msg_2", role=Role.user, parts=[])

    with pytest.raises(A2aClientError, match="Failed to connect to the agent"):
        async for _ in agent._ask_polling(NoEventClient(), message):
            pass


@pytest.mark.asyncio
async def test_streaming_raises_when_no_task_and_no_agent_card() -> None:
    """Test that streaming raises appropriate error when agent_card is None."""
    agent = A2aRemoteAgent(name="test", url="http://test.example.com")
    agent._agent_card = None
    message = Message(message_id="msg_3", role=Role.user, parts=[])

    with pytest.raises(A2aClientError, match="agent card not found"):
        async for _ in agent._ask_streaming(NoEventClient(), message):
            pass


@pytest.mark.asyncio
async def test_polling_raises_when_no_task_and_no_agent_card() -> None:
    """Test that polling raises appropriate error when agent_card is None."""
    agent = A2aRemoteAgent(name="test", url="http://test.example.com")
    agent._agent_card = None
    message = Message(message_id="msg_4", role=Role.user, parts=[])

    with pytest.raises(A2aClientError, match="agent card not found"):
        async for _ in agent._ask_polling(NoEventClient(), message):
            pass


@pytest.mark.asyncio
async def test_get_extended_agent_card_when_advertised() -> None:
    extended = AgentCard(
        name="extended-mock",
        description="extended description",
        url="http://localhost:8000",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
        supports_authenticated_extended_card=True,
    )

    remote_agent = A2aRemoteAgent(
        url="http://fake",
        name="mock-agent",
        client=MockClient("hi", extended_agent_card=extended),
    )

    card = await remote_agent._get_agent_card(auth_http_kwargs={"headers": {"Authorization": "Bearer x"}})

    assert card.name == "extended-mock"
    assert card.description == "extended description"


@pytest.mark.asyncio
async def test_skip_extended_card_when_not_advertised() -> None:
    remote_agent = A2aRemoteAgent(
        url="http://fake",
        name="mock-agent",
        client=MockClient("hi"),
    )

    card = await remote_agent._get_agent_card()

    assert card.name == "mock_agent"
    assert not card.supports_authenticated_extended_card
