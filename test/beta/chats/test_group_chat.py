# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from autogen import ConversableAgent
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern, RoundRobinPattern
from autogen.beta import Agent, testing
from autogen.testing import TestAgent


@pytest.mark.asyncio()
async def test_round_robin_pattern() -> None:
    # arrange agents
    agent1 = Agent(name="agent1", config=testing.TestConfig("Hi, I am agent one!")).as_conversable()
    agent2 = Agent(name="agent2", config=testing.TestConfig("Hi, I am agent two!")).as_conversable()
    original_agent = ConversableAgent("local")

    pattern = RoundRobinPattern(
        initial_agent=original_agent,
        agents=[original_agent, agent1, agent2],
    )

    # act
    with TestAgent(original_agent, ["Hi, I am local agent!"]):
        result, _, _ = await a_initiate_group_chat(
            pattern=pattern,
            messages="Hi all!",
            max_rounds=4,
        )

    # assert
    # use IsPartialDict because other fields has no matter for our test
    assert result.chat_history == [
        IsPartialDict({"content": "Hi all!"}),
        IsPartialDict({"content": "Hi, I am local agent!", "name": "local"}),
        IsPartialDict({"content": "Hi, I am agent one!", "name": "agent1"}),
        IsPartialDict({"content": "Hi, I am agent two!", "name": "agent2"}),
    ]


@pytest.mark.asyncio()
async def test_handoffs() -> None:
    # arrange agents
    agent1 = Agent(name="agent1", config=testing.TestConfig("Hi, I am agent one!")).as_conversable()
    agent2 = Agent(name="agent2", config=testing.TestConfig("I shouldn't speak...")).as_conversable()
    original_agent = ConversableAgent("local")

    pattern = DefaultPattern(
        initial_agent=original_agent,
        agents=[original_agent, agent1, agent2],
    )

    original_agent.handoffs.set_after_work(AgentTarget(agent1))
    agent1.handoffs.set_after_work(AgentTarget(original_agent))

    # act
    with TestAgent(original_agent, ["Hi, I am local agent!", "Hi agent!"]):
        result, _, _ = await a_initiate_group_chat(
            pattern=pattern,
            messages="Hi all!",
            max_rounds=4,
        )

    # assert
    # use IsPartialDict because other fields has no matter for our test
    assert result.chat_history == [
        IsPartialDict({"content": "Hi all!"}),
        IsPartialDict({"content": "Hi, I am local agent!", "name": "local"}),
        IsPartialDict({"content": "Hi, I am agent one!", "name": "agent1"}),
        IsPartialDict({"content": "Hi agent!", "name": "local"}),
    ]
