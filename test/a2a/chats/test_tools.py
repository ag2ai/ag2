# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Annotated
from unittest.mock import AsyncMock

import pytest
from dirty_equals import IsPartialDict
from httpx import ASGITransport

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer, A2aRemoteAgent, HttpxClientFactory
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import AskUserTarget, ContextVariables, ReplyResult
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.testing import TestAgent, ToolCall


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

    pattern = RoundRobinPattern(  # use pattern to check ContextVariables usage
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
            [ToolCall("some_tool", context_variables={}).to_message(), "Hi, I am remote agent one!"],
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


@pytest.mark.asyncio()
async def test_remote_tool_with_ask_user_target() -> None:
    # arrange remote side
    remote_agent = ConversableAgent(
        "remote",
        llm_config=LLMConfig({"model": "gpt-5", "api_key": "wrong-key"}),
    )

    @remote_agent.register_for_llm()
    def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
        return ReplyResult(
            message=datetime.strptime(date_string, "%Y-%m-%d").strftime("%A"),
            target=AskUserTarget(),
        )

    a2a_asgi_app = A2aAgentServer(remote_agent).build()
    a2a_client = HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))

    # arrange local side
    remote_agent_mirror = A2aRemoteAgent(url="http://memory/", name="remote-mirror", client=a2a_client)
    user_agent = ConversableAgent("user")

    # act
    mock = remote_agent_mirror.a_get_human_input = AsyncMock(return_value="Just tell me the day!")
    with (
        TestAgent(
            remote_agent,
            [ToolCall("get_weekday", date_string="2025-11-07").to_message(), "Friday"],
        ),
        TestAgent(user_agent),
    ):
        result = await user_agent.a_initiate_chat(
            recipient=remote_agent_mirror,
            message="What day is 2025-11-07?",
            max_turns=1,
        )

    # assert
    assert result.chat_history == [
        IsPartialDict({"content": "What day is 2025-11-07?"}),
        IsPartialDict({"content": "Friday"}),
    ]
    mock.assert_awaited_once()
