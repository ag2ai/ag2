# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from httpx import ASGITransport

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables, ReplyResult
from autogen.agentchat.group.patterns import DefaultPattern, RoundRobinPattern
from autogen.remote import HTTPAgentBus, HTTPRemoteAgent, HttpxClientFactory
from autogen.testing import TestAgent, ToolCall


@pytest.mark.asyncio()
async def test_round_robin_pattern() -> None:
    # arrange remote side
    remote_agent1 = ConversableAgent("remote-1")
    remote_agent2 = ConversableAgent("remote-2")

    asgi_app = HTTPAgentBus(agents=[remote_agent1, remote_agent2])
    client = HttpxClientFactory(transport=ASGITransport(asgi_app))

    # arrange local side
    remote_agent1_mirror = HTTPRemoteAgent(url="http://memory", name="remote-1", client=client)
    remote_agent2_mirror = HTTPRemoteAgent(url="http://memory", name="remote-2", client=client)
    local_agent = ConversableAgent("local")

    pattern = RoundRobinPattern(
        initial_agent=local_agent,
        agents=[local_agent, remote_agent1_mirror, remote_agent2_mirror],
    )

    # act
    with (
        TestAgent(remote_agent1, ["Hi, I am remote agent one!"]),
        TestAgent(remote_agent2, ["Hi, I am remote agent two!"]),
        TestAgent(local_agent, ["Hi, I am local agent!"]),
    ):
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
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote-1"}),
        IsPartialDict({"content": "Hi, I am remote agent two!", "name": "remote-2"}),
    ]


@pytest.mark.asyncio()
async def test_handoffs() -> None:
    # arrange remote side
    remote_agent1 = ConversableAgent("remote-1")
    remote_agent2 = ConversableAgent("remote-2")

    asgi_app = HTTPAgentBus(agents=[remote_agent1, remote_agent2])
    client = HttpxClientFactory(transport=ASGITransport(asgi_app))

    # arrange local side
    remote_agent1_mirror = HTTPRemoteAgent(url="http://memory", name="remote-1", client=client)
    remote_agent2_mirror = HTTPRemoteAgent(url="http://memory", name="remote-2", client=client)
    local_agent = ConversableAgent("local")

    pattern = DefaultPattern(
        initial_agent=local_agent,
        agents=[local_agent, remote_agent1_mirror, remote_agent2_mirror],
    )

    local_agent.handoffs.set_after_work(AgentTarget(remote_agent1_mirror))
    remote_agent1_mirror.handoffs.set_after_work(AgentTarget(local_agent))

    # act
    with (
        TestAgent(remote_agent1, ["Hi, I am remote agent one!"]),
        TestAgent(remote_agent2, ["I shouldn't speack..."]),
        TestAgent(local_agent, ["Hi, I am local agent!", "Hi remote!"]),
    ):
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
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote-1"}),
        IsPartialDict({"content": "Hi remote!", "name": "local"}),
    ]


@pytest.mark.asyncio()
async def test_remote_tool_with_context() -> None:
    # arrange remote side
    remote_agent = ConversableAgent(
        "remote",
        llm_config=LLMConfig({"model": "gpt-5", "api_key": "wrong-key"}),
    )

    @remote_agent.register_for_llm()
    def some_tool(context_variables: ContextVariables) -> str:
        context_variables.set("issue_count", context_variables.get("issue_count", 0) + 1)
        return ReplyResult(context_variables=context_variables, message="Tool result")

    asgi_app = HTTPAgentBus(agents=[remote_agent])
    client = HttpxClientFactory(transport=ASGITransport(asgi_app))

    # arrange local side
    remote_agent_mirror = HTTPRemoteAgent(url="http://memory", name="remote", client=client)
    local_agent = ConversableAgent("local")

    pattern = RoundRobinPattern(
        initial_agent=local_agent,
        agents=[local_agent, remote_agent_mirror],
        context_variables=ContextVariables({
            "issue_count": 0,
        }),
    )

    # act
    with (
        TestAgent(
            remote_agent,
            [
                ToolCall("some_tool", context_variables={}).to_message(),
                "Hi, I am remote agent one!",
            ],
        ),
        TestAgent(
            local_agent,
            [
                "Hi, I am local agent!",
                "Hi remote!",
            ],
        ),
    ):
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
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote"}),
    ]

    assert context.data == {"issue_count": 1}
