# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen import ConversableAgent, LLMConfig
from autogen.testing import TestAgent, ToolCall, tools_message


@pytest.mark.asyncio
async def test_mock_async_client() -> None:
    # arrange
    agent = ConversableAgent(
        name="mock-agent",
        llm_config=LLMConfig({
            "model": "gpt-5",
            "api_key": "wrong-key",
        }),
    )

    client_agent = AsyncMock(spec=ConversableAgent)
    client_agent.silent = True
    client_agent.name = "original"

    # act
    with TestAgent(agent, messages=["Hi, I am mock client!"]):
        await agent.a_receive({"content": "Hi, I am user! Who are you?"}, client_agent, request_reply=True)

    # assert message history
    assert agent.chat_messages[client_agent] == [
        {"content": "Hi, I am user! Who are you?", "role": "user", "name": "original"},
        {"content": "Hi, I am mock client!", "role": "assistant", "name": "mock-agent"},
    ]

    # assert correct answer
    assert client_agent.a_receive.call_args[0][0] == {"content": "Hi, I am mock client!"}


def test_mock_sync_client() -> None:
    # arrange
    agent = ConversableAgent(
        name="mock-agent",
        llm_config=LLMConfig({
            "model": "gpt-5",
            "api_key": "wrong-key",
        }),
    )

    client_agent = AsyncMock(spec=ConversableAgent)
    client_agent.silent = True
    client_agent.name = "original"

    # act
    with TestAgent(agent, messages=[{"content": "Hi, I am mock client!"}]):
        agent.receive("Hi, I am user! Who are you?", client_agent, request_reply=True)

    # assert message history
    assert agent.chat_messages[client_agent] == [
        {"content": "Hi, I am user! Who are you?", "role": "user", "name": "original"},
        {"content": "Hi, I am mock client!", "role": "assistant", "name": "mock-agent"},
    ]

    # assert correct answer
    assert client_agent.receive.call_args[0][0] == {"content": "Hi, I am mock client!", "role": "assistant"}


def test_mock_chat() -> None:
    # arrange
    config = LLMConfig({
        "model": "gpt-5",
        "api_key": "wrong-key",
    })

    agent1 = ConversableAgent(name="speaker1", llm_config=config)
    agent2 = ConversableAgent(name="speaker2", llm_config=config)

    # act
    with (
        TestAgent(agent1, messages=["Hi, I am mock client!", "Nice to meet you!"]),
        TestAgent(agent2, messages=["Hi, I am mock client too!!"]),
    ):
        result = agent2.initiate_chat(agent1, message="Hi agent!", max_turns=2)

    # assert
    assert result.chat_history == [
        {"content": "Hi agent!", "role": "assistant", "name": "speaker2"},
        {"content": "Hi, I am mock client!", "role": "user", "name": "speaker1"},
        {"content": "Hi, I am mock client too!!", "role": "assistant", "name": "speaker2"},
        {"content": "Nice to meet you!", "role": "user", "name": "speaker1"},
    ]


@pytest.mark.asyncio
async def test_mock_async_chat() -> None:
    # arrange
    config = LLMConfig({
        "model": "gpt-5",
        "api_key": "wrong-key",
    })

    agent1 = ConversableAgent(name="speaker1", llm_config=config)
    agent2 = ConversableAgent(name="speaker2", llm_config=config)

    # act
    with (
        TestAgent(agent1, messages=["Hi, I am mock client!", "Nice to meet you!"]),
        TestAgent(agent2, messages=["Hi, I am mock client too!!"]),
    ):
        result = await agent2.a_initiate_chat(agent1, message="Hi agent!", max_turns=2)

    # assert
    assert result.chat_history == [
        {"content": "Hi agent!", "role": "assistant", "name": "speaker2"},
        {"content": "Hi, I am mock client!", "role": "user", "name": "speaker1"},
        {"content": "Hi, I am mock client too!!", "role": "assistant", "name": "speaker2"},
        {"content": "Nice to meet you!", "role": "user", "name": "speaker1"},
    ]


def test_tool_call(mock: MagicMock) -> None:
    # arrange
    config = LLMConfig({
        "model": "gpt-5",
        "api_key": "wrong-key",
    })

    agent1 = ConversableAgent(name="speaker1", llm_config=config)
    agent2 = ConversableAgent(name="speaker2", llm_config=config)

    @agent1.register_for_execution()
    @agent2.register_for_llm()
    def get_weekday(date_str: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
        mock(date_str)
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")

    # act
    with TestAgent(agent1), TestAgent(agent2):
        agent2.initiate_chat(
            agent1,
            message=tools_message(
                ToolCall(
                    "get_weekday",
                    date_str="2025-01-01",
                ),
                ToolCall(
                    "get_weekday",
                    date_str="2025-01-01",
                ),
            ),
            max_turns=1,
        )

    # assert
    mock.assert_called_with("2025-01-01")
    assert mock.call_count == 2
