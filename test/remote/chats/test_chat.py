# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from httpx import ASGITransport

from autogen import ConversableAgent
from autogen.remote import HTTPAgentBus, HTTPRemoteAgent, HttpxClientFactory
from autogen.testing import TestAgent


@pytest.fixture()
def remote_agent() -> ConversableAgent:
    return ConversableAgent("remote")


@pytest.fixture()
def client(remote_agent: ConversableAgent) -> HttpxClientFactory:
    remote_app = HTTPAgentBus(agents=[remote_agent])
    return HttpxClientFactory(transport=ASGITransport(remote_app))


@pytest.mark.asyncio()
async def test_simple_messaging(remote_agent: ConversableAgent, client: HttpxClientFactory) -> None:
    # arrange
    remote_agent_mirror = HTTPRemoteAgent(url="http://memory", name="remote", client=client)

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
async def test_conversation(remote_agent: ConversableAgent, client: HttpxClientFactory) -> None:
    # arrange
    remote_agent_mirror = HTTPRemoteAgent(url="http://memory", name="remote", client=client)

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
