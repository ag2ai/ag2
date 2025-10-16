# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import ContextVariables, ReplyResult
from autogen.remote.agent_service import AgentService
from autogen.remote.protocol import RequestMessage, ResponseMessage
from autogen.testing import TestAgent, ToolCall, tools_message


@pytest.mark.asyncio
async def test_smoke() -> None:
    agent = ConversableAgent("test-agent")
    service = AgentService(agent)

    with TestAgent(
        agent,
        ["Hi, I am remote agent!"],
    ):
        result = await service(RequestMessage(messages=[{"content": "Hi agent!"}]))

        assert result == ResponseMessage(
            messages=[
                {
                    "content": "Hi, I am remote agent!",
                    "role": "assistant",
                    "name": "test-agent",
                }
            ]
        )


@pytest.mark.asyncio
async def test_remote_tool_call() -> None:
    agent = ConversableAgent(
        "test-agent",
        llm_config=LLMConfig({"model": "gpt-4o-mini", "api_key": "wrong-api-key"}),
    )

    @agent.register_for_execution()
    @agent.register_for_llm()
    def get_time() -> str:
        return "12:00:00"

    service = AgentService(agent)

    with TestAgent(
        agent,
        [
            tools_message(ToolCall("get_time")),
            "Well, good to know!",
        ],
    ):
        result = await service(RequestMessage(messages=[{"content": "Hi agent!"}]))

        assert result == ResponseMessage(
            messages=[
                {
                    "tool_calls": [{"function": {"name": "get_time", "arguments": "{}"}, "type": "function"}],
                    "content": None,
                    "role": "assistant",
                },
                {
                    "content": "12:00:00",
                    "tool_responses": [{"role": "tool", "content": "12:00:00"}],
                    "role": "tool",
                    "name": "test-agent",
                },
                {
                    "content": "Well, good to know!",
                    "role": "assistant",
                    "name": "test-agent",
                },
            ],
        )


@pytest.mark.asyncio
async def test_update_context() -> None:
    agent = ConversableAgent(
        "test-agent",
        llm_config=LLMConfig({"model": "gpt-4o-mini", "api_key": "wrong-api-key"}),
    )

    @agent.register_for_execution()
    @agent.register_for_llm()
    def update_context(context_variables: ContextVariables) -> ReplyResult:
        assert context_variables.data == {
            "time": "00:00:00",
            "another_time": "01:00:00",
        }
        context_variables.set("time", "12:00:00")

        return ReplyResult(message="", context_variables=context_variables)

    service = AgentService(agent)

    with TestAgent(
        agent,
        [
            tools_message(ToolCall("update_context", context_variables={})),
            "Well, good to know!",
        ],
    ):
        result = await service(
            RequestMessage(
                messages=[{"content": "Hi agent!"}],
                context={"time": "00:00:00", "another_time": "01:00:00"},
            ),
        )

        assert result.context == {
            "time": "12:00:00",
            "another_time": "01:00:00",
        }, result
