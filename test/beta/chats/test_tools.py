# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest
from dirty_equals import IsPartialDict

from autogen import ConversableAgent
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.beta import Agent, Context, Variable, events, testing
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
