# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time
from httpx import ASGITransport

from autogen import ConversableAgent
from autogen.a2a import A2aAgentServer, A2aRemoteAgent, HttpxClientFactory
from autogen.testing import TestAgent


@pytest.fixture()
def remote_agent() -> ConversableAgent:
    return ConversableAgent("remote")


@pytest.fixture()
def a2a_client(remote_agent: ConversableAgent) -> HttpxClientFactory:
    a2a_asgi_app = A2aAgentServer(remote_agent).build()
    return HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))


@pytest.mark.asyncio()
async def test_simple_messaging(remote_agent: ConversableAgent, a2a_client: HttpxClientFactory) -> None:
    # arrange
    remote_agent_mirror = A2aRemoteAgent(url="http://memory", name="remote-mirror", client=a2a_client)

    with TestAgent(remote_agent, ["Hi, I am remote agent!"]):
        # act
        _, message = await remote_agent_mirror.a_generate_remote_reply([
            {"content": "Hi, agent!"},
        ])

    # assert
    assert message == {
        "content": "Hi, I am remote agent!",
        "name": "remote",
        "role": "assistant",
    }


@pytest.mark.asyncio()
async def test_empty_message_send(remote_agent: ConversableAgent, a2a_client: HttpxClientFactory) -> None:
    # arrange
    remote_agent_mirror = A2aRemoteAgent(url="http://memory", name="remote-mirror", client=a2a_client)

    with TestAgent(remote_agent, ["Hi, I am remote agent!"]):
        # act
        _, message = await remote_agent_mirror.a_generate_remote_reply([
            {"content": ""},
        ])

    # assert
    assert message == {
        "content": "Hi, I am remote agent!",
        "name": "remote",
        "role": "assistant",
    }


@pytest.mark.asyncio()
async def test_conversation(remote_agent: ConversableAgent, a2a_client: HttpxClientFactory) -> None:
    # arrange
    remote_agent_mirror = A2aRemoteAgent(url="http://memory", name="remote-mirror", client=a2a_client)

    agent = ConversableAgent(name="speaker2")

    # act
    with (
        TestAgent(remote_agent, messages=["Hi, I am mock client!", "Nice to meet you!"]),
        TestAgent(agent, messages=["Hi, I am mock client too!!"]),
    ):
        result = await agent.a_initiate_chat(remote_agent_mirror, message="Hi agent!", max_turns=2)

    # assert
    assert result.chat_history == [
        {"content": "Hi agent!", "role": "assistant", "name": "speaker2"},
        {"content": "Hi, I am mock client!", "role": "user", "name": "remote"},
        {"content": "Hi, I am mock client too!!", "role": "assistant", "name": "speaker2"},
        {"content": "Nice to meet you!", "role": "user", "name": "remote"},
    ]


@pytest.mark.asyncio()
@freeze_time(auto_tick_seconds=5)
async def test_long_living_agent_task(
    remote_agent: ConversableAgent, a2a_client: HttpxClientFactory, mock: MagicMock
) -> None:
    a2a_client.options["timeout"] = 0.1
    # arrange
    remote_agent_mirror = A2aRemoteAgent(url="http://memory", name="remote-mirror", client=a2a_client)

    original_message_generator = remote_agent.a_generate_oai_reply

    async def slow_message(*args, **kwargs):
        await asyncio.sleep(10)
        mock()
        return await original_message_generator(*args, **kwargs)

    remote_agent.a_generate_oai_reply = slow_message

    with TestAgent(remote_agent, ["Hi, I am remote agent!"]):
        # act
        _, message = await remote_agent_mirror.a_generate_remote_reply([
            {"content": "Hi, agent!"},
        ])

    # assert
    assert message == {
        "content": "Hi, I am remote agent!",
        "name": "remote",
        "role": "assistant",
    }

    # ensure that sleep was called
    mock.assert_called_once()
