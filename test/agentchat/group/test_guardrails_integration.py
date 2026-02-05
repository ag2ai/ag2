# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for guardrails (LLMGuardrail, ToolCallLLMGuardrail) with real API calls."""

import os

import pytest

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.agentchat.group.guardrails import GuardrailResult, ToolCallLLMGuardrail
from autogen.agentchat.group.targets.transition_target import StayTarget
from autogen.llm_config import LLMConfig


def _openai_llm_config() -> LLMConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return LLMConfig(model="gpt-4o-mini", api_key=api_key, api_type="openai")


@pytest.mark.integration
def test_tool_call_guardrail_group_chat_completes() -> None:
    """Integration test: GroupChat with ToolCallLLMGuardrail (output) completes using OpenAI.

    Patterned after test/twoagent.py: assistant has output guardrail; when the assistant
    replies (text only, no tool_calls), ToolCallLLMGuardrail returns activated=False
    without calling the LLM. Verifies guardrail wiring and that the chat runs to completion.
    """
    llm_config = _openai_llm_config()

    tool_call_guardrail = ToolCallLLMGuardrail(
        name="tool_safety",
        condition="arguments contain harmful or illegal content.",
        target=StayTarget(),
        llm_config=llm_config,
    )

    assistant = AssistantAgent("assistant", llm_config=llm_config)
    assistant.register_output_guardrail(tool_call_guardrail)

    user_proxy = UserProxyAgent(
        "user_proxy",
        code_execution_config={"work_dir": "coding", "use_docker": False},
        human_input_mode="NEVER",
    )

    groupchat = GroupChat(
        agents=[assistant, user_proxy],
        messages=[],
        max_round=2,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_proxy.initiate_chat(manager, message="What is 2+2? Reply in one short sentence.")

    assert len(groupchat.messages) >= 1


@pytest.mark.integration
def test_tool_call_guardrail_check_with_tool_calls_calls_llm() -> None:
    """Integration test: ToolCallLLMGuardrail.check() with context containing tool_calls calls OpenAI.

    Ensures the guardrail uses GuardrailCheckResponse (no Pydantic schema/validation error)
    and returns a valid GuardrailResult.
    """
    llm_config = _openai_llm_config()

    guardrail = ToolCallLLMGuardrail(
        name="tool_safety",
        condition="arguments contain dangerous or illegal content.",
        target=StayTarget(),
        llm_config=llm_config,
    )

    context_with_tool_calls = [
        {"role": "user", "content": "Run this for me."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_code", "arguments": '{"code": "1 + 1"}'},
                }
            ],
        },
    ]

    result = guardrail.check(context=context_with_tool_calls)

    assert isinstance(result, GuardrailResult)
    assert result.guardrail is guardrail
    assert isinstance(result.activated, bool)
    assert isinstance(result.justification, str)
