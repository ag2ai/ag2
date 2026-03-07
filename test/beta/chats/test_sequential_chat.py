# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from autogen import ConversableAgent
from autogen.beta import Agent, testing
from autogen.testing import TestAgent


@pytest.mark.asyncio()
async def test_sequential_chat() -> None:
    # arrange remote side
    agent1 = Agent(name="agent1", config=testing.TestConfig("Hi, I am agent one!")).as_conversable()
    agent2 = Agent(name="agent2", config=testing.TestConfig("Hi, I am agent two!")).as_conversable()
    local_agent = ConversableAgent("local")

    # act
    with TestAgent(local_agent, ["Hi, I am local agent!"]):
        chat_results = await local_agent.a_initiate_chats([
            {
                "recipient": agent1,
                "message": "Hi agent!",
                "max_turns": 1,
                "chat_id": "some-chat",
            },
            {
                "recipient": agent2,
                "message": "Hi agent!",
                "max_turns": 1,
                "chat_id": "some-chat2",
            },
        ])

    # assert
    # use IsPartialDict because other fields has no matter for our test
    assert chat_results["some-chat"].chat_history == [
        IsPartialDict({"content": "Hi agent!"}),
        IsPartialDict({"content": "Hi, I am agent one!", "name": "agent1"}),
    ]

    assert chat_results["some-chat2"].chat_history == [
        IsPartialDict({"content": "Hi agent!"}),
        IsPartialDict({"content": "Hi, I am agent two!", "name": "agent2"}),
    ]
