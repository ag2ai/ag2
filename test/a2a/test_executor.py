# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsStr

from autogen import ConversableAgent, LLMConfig
from autogen.a2a.remote import AgentService, RequestMessage, ServiceResponse
from autogen.agentchat import ContextVariables, ReplyResult
from autogen.agentchat.group.guardrails import Guardrail, GuardrailResult
from autogen.testing import TestAgent, ToolCall


@pytest.mark.asyncio
async def test_smoke() -> None:
    # arrange
    agent = ConversableAgent("test-agent")

    responses = []
    with TestAgent(agent, ["Hi, I am remote agent!"]):
        # act
        service = AgentService(agent)

        async for response in service(RequestMessage(messages=[{"content": "Hi agent!"}])):
            responses.append(response)

    # assert
    assert responses == [
        ServiceResponse(
            message={"content": "Hi, I am remote agent!", "role": "assistant", "name": "test-agent"},
            context=None,
            input_required=None,
        )
    ]


@pytest.mark.asyncio
async def test_remote_tool_call() -> None:
    # arrange
    agent = ConversableAgent(
        "test-agent",
        llm_config=LLMConfig({"model": "gpt-4o-mini", "api_key": "wrong-api-key"}),
    )

    @agent.register_for_execution()
    @agent.register_for_llm()
    def get_time() -> str:
        return "12:00:00"

    responses = []
    with TestAgent(
        agent,
        [
            ToolCall("get_time").to_message(),
            "Well, good to know!",
        ],
    ):
        service = AgentService(agent)

        # act
        async for response in service(RequestMessage(messages=[{"content": "Hi agent!"}])):
            responses.append(response)

    # assert
    assert responses == [
        ServiceResponse(
            message={
                "tool_calls": [
                    {
                        "function": {"name": "get_time", "arguments": "{}"},
                        "type": "function",
                        "id": IsStr(),
                    }
                ],
                "content": None,
                "role": "assistant",
            },
        ),
        ServiceResponse(
            message={
                "content": "12:00:00",
                "tool_responses": [{"role": "tool", "content": "12:00:00", "tool_call_id": IsStr()}],
                "role": "tool",
                "name": "test-agent",
            },
        ),
        ServiceResponse(
            message={
                "content": "Well, good to know!",
                "role": "assistant",
                "name": "test-agent",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_update_context() -> None:
    # arrange
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

    responses = []
    with TestAgent(
        agent,
        [
            ToolCall("update_context", context_variables={}).to_message(),
            "Well, good to know!",
        ],
    ):
        service = AgentService(agent)

        # act
        async for response in service(
            RequestMessage(
                messages=[{"content": "Hi agent!"}],
                context={"time": "00:00:00", "another_time": "01:00:00"},
            ),
        ):
            responses.append(response)

    # assert
    assert responses[-1].context == {
        "time": "12:00:00",
        "another_time": "01:00:00",
    }


@pytest.mark.asyncio
async def test_guardrails() -> None:
    # arrange
    agent = ConversableAgent("test-agent")

    class MockGuardrail(Guardrail):
        def __init__(self) -> None:
            self.mock = MagicMock()

        def check(self, context: str | list[dict[str, Any]]) -> "GuardrailResult":
            self.mock(context)
            return GuardrailResult(activated=False, guardrail=self)

    in_guard = MockGuardrail()
    agent.register_input_guardrail(in_guard)

    out_guard = MockGuardrail()
    agent.register_output_guardrail(out_guard)

    with TestAgent(agent, ["Hi, user!"]):
        service = AgentService(agent)
        # act
        async for _ in service(
            RequestMessage(messages=[{"content": "Hi agent!"}]),
        ):
            pass

    # assert
    in_guard.mock.assert_called_once_with([{"content": "Hi agent!"}])
    out_guard.mock.assert_called_once_with({"content": "Hi, user!"})
