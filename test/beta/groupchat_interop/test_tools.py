# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict

from autogen import ConversableAgent
from autogen.agentchat import GroupChat, a_initiate_group_chat
from autogen.agentchat.group import AgentTarget, AskUserTarget, ContextVariables
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.beta import Agent, Context, ToolResult, Variable, events, testing
from autogen.testing import TestAgent


@pytest.mark.asyncio()
async def test_remote_tool_with_context() -> None:
    # arrange agents
    def some_tool(ctx: Context, issue_count: Annotated[int, Variable(default=0)]) -> str:
        ctx.variables["issue_count"] = issue_count + 1
        return "Tool result"

    agent = Agent(
        "agent",
        config=testing.TestConfig(
            events.ToolCallEvent(name="some_tool", arguments="{}"),
            "Hi, I am agent one!",
        ),
        tools=[some_tool],
    ).as_conversable()

    local_agent = ConversableAgent("local")

    # use pattern to check ContextVariables usage
    pattern = RoundRobinPattern(
        initial_agent=local_agent,
        agents=[local_agent, agent],
        context_variables=ContextVariables({"issue_count": 0}),
    )

    # act
    with TestAgent(local_agent, ["Hi, I am local agent!"]):
        result, context, _ = await a_initiate_group_chat(
            pattern=pattern,
            messages="Hi all!",
            max_rounds=3,
        )

    # assert
    # use IsPartialDict because other fields has no matter for our test
    assert result.chat_history == [
        IsPartialDict({"content": "Hi all!"}),
        IsPartialDict({"content": "Hi, I am local agent!", "name": "local"}),
        IsPartialDict({"content": "Hi, I am agent one!", "name": "agent"}),
    ]

    assert context.data == {"issue_count": 1}


@pytest.mark.asyncio()
async def test_tool_agent_handoff(mock: MagicMock) -> None:
    agent2 = Agent(name="agent2", config=testing.TestConfig("Hi, I am agent two!")).as_conversable()
    agent3 = Agent(name="agent3", config=testing.TestConfig("Hi, I am agent three!")).as_conversable()

    def some_tool() -> ToolResult:
        mock()
        return ToolResult(
            "Tool result",
            metadata={"target": AgentTarget(agent3)},
            final=True,
        )

    agent1 = Agent(
        "agent1",
        config=testing.TestConfig(events.ToolCallEvent(name="some_tool", arguments="{}")),
        tools=[some_tool],
    ).as_conversable()

    original_agent = ConversableAgent("local")

    pattern = RoundRobinPattern(
        initial_agent=original_agent,
        agents=[original_agent, agent1, agent2, agent3],
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
    mock.assert_called_once()

    assert result.chat_history == [
        IsPartialDict({"content": "Hi all!"}),
        IsPartialDict({"content": "Hi, I am local agent!", "name": "local"}),
        IsPartialDict({"content": "Tool result", "name": "agent1"}),
        IsPartialDict({"content": "Hi, I am agent three!", "name": "agent3"}),
    ]


@pytest.mark.asyncio()
async def test_user_target_handoff(mock: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    agent2 = Agent(name="agent2", config=testing.TestConfig("Hi, I am agent two!")).as_conversable()
    agent3 = Agent(name="agent3", config=testing.TestConfig("Hi, I am agent three!")).as_conversable()

    def some_tool() -> ToolResult:
        mock()
        return ToolResult(
            "Tool result",
            metadata={"target": AskUserTarget()},
            final=True,
        )

    agent1 = Agent(
        "agent1",
        config=testing.TestConfig(events.ToolCallEvent(name="some_tool", arguments="{}")),
        tools=[some_tool],
    ).as_conversable()

    original_agent = ConversableAgent("local")

    pattern = RoundRobinPattern(
        initial_agent=original_agent,
        agents=[original_agent, agent1, agent2, agent3],
    )

    # act
    with TestAgent(original_agent, ["Hi, I am local agent!"]), monkeypatch.context() as c:
        c.setattr(GroupChat, "manual_select_speaker", lambda *_: agent3)

        result, _, _ = await a_initiate_group_chat(
            pattern=pattern,
            messages="Hi all!",
            max_rounds=4,
        )

    # assert
    # use IsPartialDict because other fields has no matter for our test
    mock.assert_called_once()

    assert result.chat_history == [
        IsPartialDict({"content": "Hi all!"}),
        IsPartialDict({"content": "Hi, I am local agent!", "name": "local"}),
        IsPartialDict({"content": "Tool result", "name": "agent1"}),
        IsPartialDict({"content": "Hi, I am agent three!", "name": "agent3"}),
    ]
