# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os

import pytest

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.agent_optimizer import AgentOptimizer
from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from test.credentials import Credentials

here = os.path.abspath(os.path.dirname(__file__))


@run_for_optional_imports("openai", "openai")
def test_record_conversation(credentials_all: Credentials):
    problem = "Simplify $\\sqrt[3]{1+8} \\cdot \\sqrt[3]{1+\\sqrt[3]{8}}"

    config_list = credentials_all.config_list
    llm_config = {
        "config_list": config_list,
        "timeout": 60,
        "cache_seed": 42,
    }

    assistant = AssistantAgent("assistant", system_message="You are a helpful assistant.", llm_config=llm_config)
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": f"{here}/test_agent_scripts",
            "use_docker": "python:3",
            "timeout": 60,
        },
        max_consecutive_auto_reply=3,
    )

    user_proxy.initiate_chat(assistant, message=problem)
    optimizer = AgentOptimizer(max_actions_per_step=3, llm_config=llm_config)
    optimizer.record_one_conversation(assistant.chat_messages_for_summary(user_proxy), is_satisfied=True)

    assert len(optimizer._trial_conversations_history) == 1
    assert len(optimizer._trial_conversations_performance) == 1
    assert optimizer._trial_conversations_performance[0]["Conversation 0"] == 1

    optimizer.reset_optimizer()
    assert len(optimizer._trial_conversations_history) == 0
    assert len(optimizer._trial_conversations_performance) == 0


@run_for_optional_imports("openai", "openai")
def test_step(credentials_all: Credentials):
    problem = "Simplify $\\sqrt[3]{1+8} \\cdot \\sqrt[3]{1+\\sqrt[3]{8}}"

    config_list = credentials_all.config_list
    llm_config = {
        "config_list": config_list,
        "timeout": 60,
        "cache_seed": 42,
    }
    assistant = AssistantAgent(
        "assistant",
        system_message="You are a helpful assistant.",
        llm_config=llm_config,
    )
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": f"{here}/test_agent_scripts",
            "use_docker": "python:3",
            "timeout": 60,
        },
        max_consecutive_auto_reply=3,
    )

    optimizer = AgentOptimizer(max_actions_per_step=3, llm_config=llm_config)
    user_proxy.initiate_chat(assistant, message=problem)
    optimizer.record_one_conversation(assistant.chat_messages_for_summary(user_proxy), is_satisfied=True)

    register_for_llm, register_for_exector = optimizer.step()

    print("-------------------------------------")
    print("register_for_llm:")
    print(register_for_llm)
    print("register_for_exector")
    print(register_for_exector)

    for item in register_for_llm:
        assistant.update_function_signature(**item)
    if len(register_for_exector.keys()) > 0:
        user_proxy.register_function(function_map=register_for_exector)

    print("-------------------------------------")
    print("Updated assistant.llm_config:")
    print(assistant.llm_config)
    print("Updated user_proxy._function_map:")
    print(user_proxy._function_map)


@run_for_optional_imports("openai", "openai")
def test_llm_config_current_property(credentials_all: Credentials):
    """Test that AgentOptimizer works when llm_config is explicitly provided."""
    # Create a default LLMConfig
    llm_config = LLMConfig(
        *credentials_all.config_list,
        timeout=60,
        cache_seed=42,
    )

    # Create AgentOptimizer without passing llm_config
    optimizer = AgentOptimizer(max_actions_per_step=3, llm_config=llm_config)

    # Test that it works with record_one_conversation
    conversation = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    optimizer.record_one_conversation(conversation, is_satisfied=True)
    assert len(optimizer._trial_conversations_history) == 1


def test_llm_config_without_context():
    """Test that AgentOptimizer raises ValueError when no llm_config is provided."""
    with pytest.raises(ValueError, match="llm_config is required for AgentOptimizer"):
        AgentOptimizer(max_actions_per_step=3)
