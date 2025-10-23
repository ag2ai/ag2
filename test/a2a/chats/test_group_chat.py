# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from httpx import ASGITransport
from starlette.applications import Starlette
from starlette.routing import Mount

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer, A2aRemoteAgent, HttpxClientFactory
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables, ReplyResult
from autogen.agentchat.group.patterns import DefaultPattern, RoundRobinPattern
from autogen.testing import TestAgent, ToolCall


@pytest.mark.asyncio()
async def test_round_robin_pattern() -> None:
    # arrange remote side
    remote_agent1 = ConversableAgent("remote-1")
    remote_agent2 = ConversableAgent("remote-2")

    a2a_asgi_app = Starlette(
        routes=[
            Mount("/one", A2aAgentServer(remote_agent1, url="http://memory/one/").build()),
            Mount("/two", A2aAgentServer(remote_agent2, url="http://memory/two/").build()),
        ]
    )

    a2a_client = HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))

    # arrange local side
    remote_agent1_mirror = A2aRemoteAgent(url="http://memory/one/", name="remote1-mirror", client=a2a_client)
    remote_agent2_mirror = A2aRemoteAgent(url="http://memory/two/", name="remote2-mirror", client=a2a_client)
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
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote1-mirror"}),
        IsPartialDict({"content": "Hi, I am remote agent two!", "name": "remote2-mirror"}),
    ]


@pytest.mark.asyncio()
async def test_handoffs() -> None:
    # arrange remote side
    remote_agent1 = ConversableAgent("remote-1")
    remote_agent2 = ConversableAgent("remote-2")

    a2a_asgi_app = Starlette(
        routes=[
            Mount("/one", A2aAgentServer(remote_agent1, url="http://memory/one/").build()),
            Mount("/two", A2aAgentServer(remote_agent2, url="http://memory/two/").build()),
        ]
    )

    a2a_client = HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))

    # arrange local side
    remote_agent1_mirror = A2aRemoteAgent(url="http://memory/one/", name="remote1-mirror", client=a2a_client)
    remote_agent2_mirror = A2aRemoteAgent(url="http://memory/two/", name="remote2-mirror", client=a2a_client)
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
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote1-mirror"}),
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

    a2a_asgi_app = A2aAgentServer(remote_agent).build()
    a2a_client = HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))

    # arrange local side
    remote_agent_mirror = A2aRemoteAgent(url="http://memory/", name="remote-mirror", client=a2a_client)
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
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote-mirror"}),
    ]

    assert context.data == {"issue_count": 1}
