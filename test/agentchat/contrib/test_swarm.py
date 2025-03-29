# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import inspect
import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.swarm_agent import (
    __TOOL_EXECUTOR_NAME__,
    AfterWork,
    AfterWorkOption,
    OnCondition,
    OnContextCondition,
    SwarmResult,
    _change_tool_context_variables_to_depends,
    _cleanup_temp_user_messages,
    _create_nested_chats,
    _determine_next_agent,
    _prepare_swarm_agents,
    _process_initial_messages,
    _run_oncontextconditions,
    _set_to_tool_execution,
    _setup_context_variables,
    _update_conditional_functions,
    a_initiate_swarm_chat,
    initiate_swarm_chat,
    make_remove_function,
    register_hand_off,
)
from autogen.agentchat.conversable_agent import ConversableAgent, UpdateSystemMessage
from autogen.agentchat.group.context_str import ContextStr
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.utils import ContextExpression
from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.tools.tool import Tool

from ...conftest import (
    Credentials,
)

TEST_MESSAGES = [{"role": "user", "content": "Initial message"}]


def invalid_agent(name: str = "invalid_agent") -> Agent:
    @dataclass
    class InvalidAgent:
        name: str

    agent = InvalidAgent(name=name)
    return agent  # type: ignore[return-value]


def test_swarm_result() -> None:
    """Test SwarmResult initialization and string conversion"""
    # Valid initialization
    result = SwarmResult(values="test result")
    assert str(result) == "test result"
    assert result.context_variables == {}

    # Test with context variables
    context = {"key": "value"}
    result = SwarmResult(values="test", context_variables=context)
    assert result.context_variables == context

    # Test with agent
    agent = ConversableAgent("test")
    result = SwarmResult(values="test", agent=agent)
    assert result.agent == agent

    # Test AfterWorkOption
    result = SwarmResult(agent=AfterWorkOption.TERMINATE)
    assert isinstance(result.agent, AfterWorkOption)


def test_swarm_result_serialization() -> None:
    agent = ConversableAgent(name="test_agent", human_input_mode="NEVER")
    result = SwarmResult(
        values="test",
        agent=agent,
        context_variables={"key": "value"},
    )

    serialized = json.loads(result.model_dump_json())
    assert serialized["agent"] == "test_agent"
    assert serialized["values"] == "test"
    assert serialized["context_variables"] == {"key": "value"}

    result = SwarmResult(
        values="test",
        agent="test_agent",
        context_variables={"key": "value"},
    )

    serialized = json.loads(result.model_dump_json())
    assert serialized["agent"] == "test_agent"
    assert serialized["values"] == "test"
    assert serialized["context_variables"] == {"key": "value"}


def test_after_work_initialization() -> None:
    """Test AfterWork initialization with different options"""
    # Test with AfterWorkOption
    after_work = AfterWork(AfterWorkOption.TERMINATE)
    assert after_work.agent == AfterWorkOption.TERMINATE

    # Test with string
    after_work = AfterWork("TERMINATE")
    assert after_work.agent == AfterWorkOption.TERMINATE

    # Test with ConversableAgent
    agent = ConversableAgent("test")
    after_work = AfterWork(agent)
    assert after_work.agent == agent

    # Test with Callable
    def test_callable(s: str) -> ConversableAgent:
        return agent

    after_work = AfterWork(test_callable)  # type: ignore[arg-type]
    assert after_work.agent == test_callable

    # Test with invalid option
    with pytest.raises(ValueError):
        AfterWork("INVALID_OPTION")


def test_on_condition() -> None:
    """Test OnCondition initialization"""

    # Test with a base Agent
    test_conversable_agent = invalid_agent("test_conversable_agent")
    with pytest.raises(ValueError, match="'target' must be a ConversableAgent or a dict"):
        _ = OnCondition(target=test_conversable_agent, condition="test condition")  # type: ignore[arg-type]


def test_receiving_agent() -> None:
    """Test the receiving agent based on various starting messages"""
    # 1. Test with a single message - should always be the initial agent
    messages_one_no_name = [{"role": "user", "content": "Initial message"}]

    test_initial_agent = ConversableAgent("InitialAgent")

    # Test the chat
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=test_initial_agent, messages=messages_one_no_name, agents=[test_initial_agent]
    )

    # Make sure the first speaker (second message) is the initialagent
    assert "name" not in chat_result.chat_history[0]  # _User should not exist
    assert chat_result.chat_history[1].get("name") == "InitialAgent"

    # 2. Test with a single message from an existing agent (should still be initial agent)
    test_second_agent = ConversableAgent("SecondAgent")

    messages_one_w_name = [{"role": "user", "content": "Initial message", "name": "SecondAgent"}]

    # Test the chat
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=test_initial_agent, messages=messages_one_w_name, agents=[test_initial_agent, test_second_agent]
    )

    assert chat_result.chat_history[0].get("name") == "SecondAgent"
    assert chat_result.chat_history[1].get("name") == "InitialAgent"

    # 3. Test with a single message from a user agent, user passed in

    test_user = UserProxyAgent("MyUser")

    messages_one_w_name = [{"role": "user", "content": "Initial message", "name": "MyUser"}]

    # Test the chat
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=test_second_agent,
        user_agent=test_user,
        messages=messages_one_w_name,
        agents=[test_initial_agent, test_second_agent],
    )
    assert context_vars is None
    assert last_speaker

    assert chat_result.chat_history[0].get("name") == "MyUser"  # Should persist
    assert chat_result.chat_history[1].get("name") == "SecondAgent"


def test_resume_speaker() -> None:
    """Tests resumption of chat with multiple messages"""

    test_initial_agent = ConversableAgent("InitialAgent")
    test_second_agent = ConversableAgent("SecondAgent")

    # For multiple messages, last agent initiates the chat
    multiple_messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "name": "InitialAgent", "content": "Second message"},
        {"role": "assistant", "name": "SecondAgent", "content": "Third message"},
    ]

    # Patch initiate_chat on agents so we can monitor which started the conversation
    with (
        patch.object(test_initial_agent, "initiate_chat") as mock_initial_chat,
        patch.object(test_second_agent, "initiate_chat") as mock_second_chat,
    ):
        mock_chat_result = MagicMock()
        mock_chat_result.chat_history = multiple_messages

        # Set up the return value for the mock that will be called
        mock_second_chat.return_value = mock_chat_result

        # Run the function
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=test_initial_agent, messages=multiple_messages, agents=[test_initial_agent, test_second_agent]
        )
        assert chat_result
        assert context_vars is None
        assert last_speaker is None

        # Ensure the second agent initiated the chat
        mock_second_chat.assert_called_once()

        # And it wasn't the initial_agent's agent
        mock_initial_chat.assert_not_called()


def test_after_work_options() -> None:
    """Test different after work options"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    user_agent = UserProxyAgent("test_user")

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    # 1. Test TERMINATE
    agent1._swarm_after_work = AfterWork(AfterWorkOption.TERMINATE)  # type: ignore[attr-defined]
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2]
    )
    assert last_speaker == agent1
    assert context_vars is None

    # 2. Test REVERT_TO_USER
    agent1._swarm_after_work = AfterWork(AfterWorkOption.REVERT_TO_USER)

    test_messages = [
        {"role": "user", "content": "Initial message"},
        {"role": "assistant", "name": "agent1", "content": "Response"},
    ]

    with patch("builtins.input", return_value="continue"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent1, messages=test_messages, agents=[agent1, agent2], user_agent=user_agent, max_rounds=4
        )
        assert context_vars is None
        assert last_speaker

    # Ensure that after agent1 is finished, it goes to user (4th message)
    assert chat_result.chat_history[3]["name"] == "test_user"

    # 3. Test STAY
    agent1._swarm_after_work = AfterWork(AfterWorkOption.STAY)
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=test_messages, agents=[agent1, agent2], max_rounds=4
    )
    assert context_vars is None
    assert last_speaker

    # Stay on agent1
    assert chat_result.chat_history[3]["name"] == "agent1"

    # 4. Test Callable

    # Transfer to agent2
    def test_callable(last_speaker, messages, groupchat):
        return agent2

    agent1._swarm_after_work = AfterWork(test_callable)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=test_messages, agents=[agent1, agent2], max_rounds=4
    )
    assert context_vars is None
    assert last_speaker

    # We should have transferred to agent2 after agent1 has finished
    assert chat_result.chat_history[3]["name"] == "agent2"


@run_for_optional_imports(["openai"], "openai")
def test_on_condition_handoff() -> None:
    """Test OnCondition in handoffs"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)

    register_hand_off(agent1, hand_to=OnCondition(target=agent2, condition="always take me to agent 2"))

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [{"type": "function", "function": {"name": "transfer_agent1_to_agent2"}}],
        }

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2],
        max_rounds=5,
        exclude_transit_message=False,
    )
    assert context_vars is None
    assert last_speaker

    # We should have transferred to agent2 after agent1 has finished
    assert chat_result.chat_history[3]["name"] == "agent2"


def test_temporary_user_proxy() -> None:
    """Test that temporary user proxy agent name is cleared"""
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2]
    )
    assert context_vars is None
    assert last_speaker

    # Verify no message has name "_User"
    for message in chat_result.chat_history:
        assert message.get("name") != "_User"


@run_for_optional_imports(["openai"], "openai")
def test_context_variables_updating_multi_tools() -> None:
    """Test context variables handling in tool calls"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    # Starting context variable, this will increment in the swarm
    test_context_variables = ContextVariables(data={"my_key": 0})

    # Increment the context variable
    def test_func_1(context_variables: dict[str, Any], param1: str) -> SwarmResult:
        context_variables["my_key"] += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    # Increment the context variable
    def test_func_2(context_variables: dict[str, Any], param2: str) -> SwarmResult:
        context_variables["my_key"] += 100
        return SwarmResult(values=f"Test 2 {param2}", context_variables=context_variables, agent=agent1)

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", functions=[test_func_1, test_func_2], llm_config=testing_llm_config)

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [
                {"type": "function", "function": {"name": "test_func_1", "arguments": '{"param1": "test"}'}},
                {"type": "function", "function": {"name": "test_func_2", "arguments": '{"param2": "test"}'}},
            ],
        }

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent2,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2],
        context_variables=test_context_variables,
        max_rounds=3,
    )
    assert chat_result
    assert last_speaker

    # Ensure we've incremented the context variable
    # in both tools, updated values should traverse
    # 0 + 1 (func 1) + 100 (func 2) = 101
    assert context_vars["my_key"] == 101


@run_for_optional_imports(["openai"], "openai")
def test_context_variables_updating_multi_tools_including_pydantic_object() -> None:
    """Test context variables handling in tool calls"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    # Starting pydantic context variable, this will increment in the swarm
    class MyKey(BaseModel):
        key: int = 0

    test_context_variables = ContextVariables(data={"my_key": MyKey(key=0)})

    # Increment the pydantic context variable
    def test_func_1(context_variables: dict[str, Any], param1: str) -> SwarmResult:
        context_variables["my_key"].key += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    # Increment the pydantic context variable
    def test_func_2(context_variables: dict[str, Any], param2: str) -> SwarmResult:
        context_variables["my_key"].key += 100
        return SwarmResult(values=f"Test 2 {param2}", context_variables=context_variables, agent=agent1)

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", functions=[test_func_1, test_func_2], llm_config=testing_llm_config)

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, Union[str, dict[str, Any]]]:
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args: Any, **kwargs: Any) -> tuple[bool, Union[str, dict[str, Any]]]:
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [
                {"type": "function", "function": {"name": "test_func_1", "arguments": '{"param1": "test"}'}},
                {"type": "function", "function": {"name": "test_func_2", "arguments": '{"param2": "test"}'}},
            ],
        }

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent2,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2],
        context_variables=test_context_variables,
        max_rounds=3,
    )
    assert chat_result
    assert last_speaker

    # Ensure we've incremented the context variable
    # in both tools, updated values should traverse
    # 0 + 1 (func 1) + 100 (func 2) = 101
    assert context_vars["my_key"] == MyKey(key=101)


@run_for_optional_imports(["openai"], "openai")
def test_function_transfer() -> None:
    """Tests a function call that has a transfer to agent in the SwarmResult"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    # Starting context variable, this will increment in the swarm
    test_context_variables = ContextVariables(data={"my_key": 0})

    # Increment the context variable
    def test_func_1(context_variables: dict[str, Any], param1: str) -> SwarmResult:
        context_variables["my_key"] += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", functions=[test_func_1], llm_config=testing_llm_config)

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [
                {"type": "function", "function": {"name": "test_func_1", "arguments": '{"param1": "test"}'}},
            ],
        }

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent2,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2],
        context_variables=test_context_variables,
        max_rounds=4,
        exclude_transit_message=False,
    )
    assert context_vars
    assert last_speaker

    assert chat_result.chat_history[3]["name"] == "agent1"


def test_invalid_parameters() -> None:
    """Test various invalid parameter combinations"""
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")

    # Test invalid initial agent type
    with pytest.raises(ValueError, match="initial_agent must be a ConversableAgent"):
        initiate_swarm_chat(initial_agent="not_an_agent", messages=TEST_MESSAGES, agents=[agent1, agent2])  # type: ignore[arg-type]

    # Test invalid agents list
    with pytest.raises(ValueError, match="Agents must be a list of ConversableAgents"):
        initiate_swarm_chat(initial_agent=agent1, messages=TEST_MESSAGES, agents=["not_an_agent", agent2])  # type: ignore[list-item]

    # Test invalid after_work type
    with pytest.raises(ValueError, match="Invalid agent name in after_work: invalid"):
        initiate_swarm_chat(initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2], after_work="invalid")  # type: ignore[arg-type]


def test_non_swarm_in_hand_off() -> None:
    """Test that agents in the group chat are the only agents in hand-offs"""

    agent1 = ConversableAgent("agent1")
    bad_agent = invalid_agent("bad_agent")
    assert not callable(bad_agent)

    with pytest.raises(ValueError, match="Invalid AfterWork agent:"):
        register_hand_off(agent1, hand_to=AfterWork(bad_agent))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="'target' must be a ConversableAgent or a dict"):
        register_hand_off(agent1, hand_to=OnCondition(target=bad_agent, condition="Testing"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hand_to must be a list of OnCondition, OnContextCondition, or AfterWork"):
        register_hand_off(agent1, 0)  # type: ignore[arg-type]


def test_initialization() -> None:
    """Test initiate_swarm_chat"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    agent3 = ConversableAgent("agent3")
    bad_agent = invalid_agent("bad_agent")

    with pytest.raises(ValueError, match="Agents must be a list of ConversableAgent"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent2,
            messages=TEST_MESSAGES,
            agents=[agent1, agent2, bad_agent],  # type: ignore[list-item]
            max_rounds=3,  # type: ignore[list-item]
        )
        assert chat_result
        assert context_vars
        assert last_speaker

    with pytest.raises(ValueError, match="initial_agent must be a ConversableAgent"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=bad_agent,  # type: ignore[arg-type]
            messages=TEST_MESSAGES,
            agents=[agent1, agent2],
            max_rounds=3,  # type: ignore[arg-type]
        )
        assert chat_result
        assert context_vars
        assert last_speaker

    register_hand_off(agent1, hand_to=AfterWork(agent3))

    with pytest.raises(ValueError, match="Agent in hand-off must be in the agents list"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2], max_rounds=3
        )
        assert chat_result
        assert context_vars
        assert last_speaker


def test_update_system_message() -> None:
    """Tests the update_agent_state_before_reply functionality with multiple scenarios"""

    # Test container to capture system messages
    class MessageContainer:
        def __init__(self) -> None:
            self.captured_sys_message = ""

    message_container = MessageContainer()

    # 1. Test with a callable function
    def custom_update_function(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        return f"System message with {agent.context_variables.get('test_var')} and {len(messages)} messages"

    # 2. Test with a string template
    template_message = "Template message with {test_var}"

    # Create agents with different update configurations
    agent1 = ConversableAgent("agent1", update_agent_state_before_reply=UpdateSystemMessage(custom_update_function))

    agent2 = ConversableAgent("agent2", update_agent_state_before_reply=UpdateSystemMessage(template_message))

    # Mock the reply function to capture the system message
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        # Capture the system message for verification
        message_container.captured_sys_message = args[0]._oai_system_message[0]["content"]
        return True, "Mock response"

    # Register mock reply for both agents
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    # Test context and messages
    test_context = ContextVariables(data={"test_var": "test_value"})
    test_messages = [{"role": "user", "content": "Test message"}]

    # Run chat with first agent (using callable function)
    chat_result1, context_vars1, last_speaker1 = initiate_swarm_chat(
        initial_agent=agent1, messages=test_messages, agents=[agent1], context_variables=test_context, max_rounds=2
    )
    assert chat_result1
    assert context_vars1
    assert last_speaker1

    # Verify callable function result
    assert message_container.captured_sys_message == "System message with test_value and 1 messages"

    # Reset captured message
    message_container.captured_sys_message = ""

    # Run chat with second agent (using string template)
    chat_result2, context_vars2, last_speaker2 = initiate_swarm_chat(
        initial_agent=agent2, messages=test_messages, agents=[agent2], context_variables=test_context, max_rounds=2
    )
    assert chat_result2
    assert context_vars2
    assert last_speaker2

    # Verify template result
    assert message_container.captured_sys_message == "Template message with test_value"

    # Test multiple update functions
    def another_update_function(context_variables: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        return "Another update"

    agent6 = ConversableAgent(
        "agent6",
        update_agent_state_before_reply=[
            UpdateSystemMessage(custom_update_function),
            UpdateSystemMessage(another_update_function),
        ],
    )

    agent6.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    chat_result6, context_vars6, last_speaker6 = initiate_swarm_chat(
        initial_agent=agent6, messages=test_messages, agents=[agent6], context_variables=test_context, max_rounds=2
    )
    assert chat_result6
    assert context_vars6
    assert last_speaker6

    # Verify last update function took effect
    assert message_container.captured_sys_message == "Another update"


@run_for_optional_imports(["openai"], "openai")
def test_string_agent_params_for_transfer() -> None:
    """Test that string agent parameters are handled correctly without using real LLMs."""
    # Define test configuration
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    # Define a simple function for testing
    def hello_world(context_variables: dict[str, Any]) -> SwarmResult:
        value = "Hello, World!"
        return SwarmResult(values=value, context_variables=context_variables, agent="agent_2")

    # Create agent instances
    agent_1 = ConversableAgent(
        name="agent_1",
        system_message="Your task is to call hello_world() function.",
        llm_config=testing_llm_config,
        functions=[hello_world],
    )
    agent_2 = ConversableAgent(
        name="agent_2",
        system_message="Your task is to let the user know what the previous agent said.",
        llm_config=testing_llm_config,
    )

    # Mock LLM responses
    def mock_generate_oai_reply_agent1(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {
            "role": "assistant",
            "name": "agent_1",
            "tool_calls": [{"type": "function", "function": {"name": "hello_world", "arguments": "{}"}}],
            "content": "I will call the hello_world function.",
        }

    def mock_generate_oai_reply_agent2(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {
            "role": "assistant",
            "name": "agent_2",
            "content": "The previous agent called hello_world and got: Hello, World!",
        }

    # Register mock responses
    agent_1.register_reply([ConversableAgent, None], mock_generate_oai_reply_agent1)
    agent_2.register_reply([ConversableAgent, None], mock_generate_oai_reply_agent2)

    # Initiate the swarm chat
    chat_result, final_context, last_active_agent = initiate_swarm_chat(
        initial_agent=agent_1,
        agents=[agent_1, agent_2],
        context_variables=ContextVariables(),
        messages="Begin by calling the hello_world() function.",
        after_work=AfterWorkOption.TERMINATE,
        max_rounds=5,
        exclude_transit_message=False,
    )

    # Assertions to verify the behavior
    assert chat_result.chat_history[3]["name"] == "agent_2"
    assert last_active_agent.name == "agent_2"

    # Define a simple function for testing
    def hello_world(context_variables: dict[str, Any]) -> SwarmResult:  # type: ignore[no-redef]
        value = "Hello, World!"
        return SwarmResult(values=value, context_variables=context_variables, agent="agent_unknown")

    agent_1 = ConversableAgent(
        name="agent_1",
        system_message="Your task is to call hello_world() function.",
        llm_config=testing_llm_config,
        functions=[hello_world],
    )
    agent_2 = ConversableAgent(
        name="agent_2",
        system_message="Your task is to let the user know what the previous agent said.",
        llm_config=testing_llm_config,
    )

    # Register mock responses
    agent_1.register_reply([ConversableAgent, None], mock_generate_oai_reply_agent1)
    agent_2.register_reply([ConversableAgent, None], mock_generate_oai_reply_agent2)

    with pytest.raises(
        ValueError, match="No agent found with the name 'agent_unknown'. Ensure the agent exists in the swarm."
    ):
        chat_result, final_context, last_active_agent = initiate_swarm_chat(
            initial_agent=agent_1,
            agents=[agent_1, agent_2],
            context_variables=ContextVariables(),
            messages="Begin by calling the hello_world() function.",
            after_work=AfterWorkOption.TERMINATE,
            max_rounds=5,
        )

        assert final_context


@run_for_optional_imports(["openai"], "openai")
def test_after_work_callable() -> None:
    """Test Callable in an AfterWork handoff"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)
    agent3 = ConversableAgent("agent3", llm_config=testing_llm_config)

    def return_agent(
        last_speaker: ConversableAgent, messages: list[dict[str, Any]], groupchat: GroupChat
    ) -> Union[AfterWorkOption, ConversableAgent, str]:
        return agent2

    def return_agent_str(
        last_speaker: ConversableAgent, messages: list[dict[str, Any]], groupchat: GroupChat
    ) -> Union[AfterWorkOption, ConversableAgent, str]:
        return "agent3"

    def return_after_work_option(
        last_speaker: ConversableAgent, messages: list[dict[str, Any]], groupchat: GroupChat
    ) -> Union[AfterWorkOption, ConversableAgent, str]:
        return AfterWorkOption.TERMINATE

    register_hand_off(
        agent=agent1,
        hand_to=[
            AfterWork(agent=return_agent),
        ],
    )

    register_hand_off(
        agent=agent2,
        hand_to=[
            AfterWork(agent=return_agent_str),
        ],
    )

    register_hand_off(
        agent=agent3,
        hand_to=[
            AfterWork(agent=return_after_work_option),
        ],
    )

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent3.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2, agent3],
        max_rounds=5,
    )

    assert context_vars is None
    assert last_speaker

    # Confirm transitions and it terminated with 4 messages
    assert chat_result.chat_history[1]["name"] == "agent1"
    assert chat_result.chat_history[2]["name"] == "agent2"
    assert chat_result.chat_history[3]["name"] == "agent3"
    assert len(chat_result.chat_history) == 4


@run_for_optional_imports(["openai"], "openai")
def test_on_condition_unique_function_names() -> None:
    """Test that OnCondition in handoffs generate unique function names"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)

    register_hand_off(
        agent=agent1,
        hand_to=[
            OnCondition(target=agent2, condition="always take me to agent 2"),
            OnCondition(target=agent2, condition="sometimes take me there"),
            OnCondition(target=agent2, condition="always take me there"),
        ],
    )

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [{"type": "function", "function": {"name": "transfer_agent1_to_agent2"}}],
        }

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2],
        max_rounds=5,
        exclude_transit_message=False,
    )
    assert chat_result
    assert context_vars is None
    assert last_speaker

    # Check that agent1 has 3 functions and they have unique names
    assert "transfer_agent1_to_agent2" in [t._name for t in agent1.tools]
    assert "transfer_agent1_to_agent2_2" in [t._name for t in agent1.tools]
    assert "transfer_agent1_to_agent2_3" in [t._name for t in agent1.tools]


@run_for_optional_imports(["openai"], "openai")
def test_prepare_swarm_agents() -> None:
    """Test preparation of swarm agents including tool executor setup"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    # Create test agents
    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)
    agent3 = ConversableAgent("agent3", llm_config=testing_llm_config)

    # Add some functions to test tool executor aggregation
    def test_func1() -> None:
        pass

    def test_func2() -> None:
        pass

    agent1._add_single_function(test_func1)
    agent2._add_single_function(test_func2)

    # Add handoffs to test validation
    register_hand_off(agent=agent1, hand_to=AfterWork(agent=agent2))

    # Test valid preparation
    tool_executor, nested_chat_agents = _prepare_swarm_agents(agent1, [agent1, agent2], ContextVariables())

    assert nested_chat_agents == []

    # Verify tool executor setup
    assert tool_executor.name == __TOOL_EXECUTOR_NAME__
    assert "test_func1" in tool_executor._function_map
    assert "test_func2" in tool_executor._function_map

    # Test invalid initial agent type
    with pytest.raises(ValueError):
        _prepare_swarm_agents(invalid_agent("invalid"), [agent1, agent2], ContextVariables())  # type: ignore[arg-type]

    # Test invalid agents list
    with pytest.raises(ValueError):
        _prepare_swarm_agents(agent1, [agent1, invalid_agent("invalid")], ContextVariables())  # type: ignore[list-item]

    # Test missing handoff agent
    register_hand_off(agent=agent3, hand_to=AfterWork(agent=ConversableAgent("missing")))
    with pytest.raises(ValueError):
        _prepare_swarm_agents(agent1, [agent1, agent2, agent3], ContextVariables())


@run_for_optional_imports(["openai"], "openai")
def test_create_nested_chats() -> None:
    """Test creation of nested chat agents and registration of handoffs"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    test_agent = ConversableAgent("test_agent", llm_config=testing_llm_config)
    test_agent_2 = ConversableAgent("test_agent_2", llm_config=testing_llm_config)
    nested_chat_agents: list[Agent] = []

    nested_chat_one = {
        "carryover_config": {"summary_method": "last_msg"},
        "recipient": test_agent_2,
        "message": "Extract the order details",
        "max_turns": 1,
    }

    chat_queue = [nested_chat_one]

    # Register a nested chat handoff
    nested_chat_config = {
        "chat_queue": chat_queue,
        "reply_func_from_nested_chats": "summary_from_nested_chats",
        "config": None,
        "use_async": False,
    }

    register_hand_off(agent=test_agent, hand_to=OnCondition(target=nested_chat_config, condition="test condition"))

    # Create nested chats
    _create_nested_chats(test_agent, nested_chat_agents)  # type: ignore[arg-type]

    # Verify nested chat agent creation
    assert len(nested_chat_agents) == 1
    assert nested_chat_agents[0].name == f"nested_chat_{test_agent.name}_1"

    # Verify nested chat configuration
    # The nested chat agent should have a handoff back to the passed in agent
    assert nested_chat_agents[0]._swarm_after_work.agent == test_agent  # type: ignore[attr-defined]


def test_process_initial_messages() -> None:
    """Test processing of initial messages in different scenarios"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    nested_agent = ConversableAgent("nested_chat_agent1_1")
    user_agent = UserProxyAgent("test_user")

    # Test single string message
    messages: Union[str, list[dict[str, Any]]] = "Initial message"
    processed_messages, last_agent, agent_names, temp_users = _process_initial_messages(
        messages, None, [agent1, agent2], [nested_agent]
    )

    assert len(processed_messages) == 1
    assert processed_messages[0]["content"] == "Initial message"
    assert len(temp_users) == 1  # Should create temporary user
    assert temp_users[0].name == "_User"

    # Test message with existing agent name
    messages = [{"role": "user", "content": "Test", "name": "agent1"}]
    processed_messages, last_agent, agent_names, temp_users = _process_initial_messages(
        messages, user_agent, [agent1, agent2], [nested_agent]
    )

    assert last_agent == agent1
    assert len(temp_users) == 0  # Should not create temp user

    # Test message with user agent name
    messages = [{"role": "user", "content": "Test", "name": "test_user"}]
    processed_messages, last_agent, agent_names, temp_users = _process_initial_messages(
        messages, user_agent, [agent1, agent2], [nested_agent]
    )

    assert last_agent == user_agent
    assert len(temp_users) == 0
    assert agent_names

    # Test invalid agent name
    messages = [{"role": "user", "content": "Test", "name": "invalid_agent"}]
    with pytest.raises(ValueError):
        _process_initial_messages(messages, user_agent, [agent1, agent2], [nested_agent])


def test_setup_context_variables() -> None:
    """Test setup of context variables across agents"""

    tool_execution = ConversableAgent(__TOOL_EXECUTOR_NAME__)
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")

    groupchat = GroupChat(agents=[tool_execution, agent1, agent2], messages=[])
    manager = GroupChatManager(groupchat)

    test_context = ContextVariables(data={"test_key": "test_value"})

    _setup_context_variables(tool_execution, [agent1, agent2], manager, test_context)

    # Verify all agents share the same context_variables reference
    assert tool_execution.context_variables is test_context
    assert agent1.context_variables is test_context
    assert agent2.context_variables is test_context
    assert manager.context_variables is test_context


def test_cleanup_temp_user_messages() -> None:
    """Test cleanup of temporary user messages"""
    chat_result = MagicMock()
    chat_result.chat_history = [
        {"role": "user", "name": "_User", "content": "Test 1"},
        {"role": "assistant", "name": "agent1", "content": "Response 1"},
        {"role": "user", "name": "_User", "content": "Test 2"},
    ]

    _cleanup_temp_user_messages(chat_result)

    # Verify _User names are removed
    for message in chat_result.chat_history:
        if message["role"] == "user":
            assert "name" not in message


@pytest.mark.asyncio
async def test_a_initiate_swarm_chat() -> None:
    """Test async swarm chat"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    user_agent = UserProxyAgent("test_user")

    # Mock async reply function
    async def mock_a_generate_oai_reply(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "This is a mock response from the agent."

    # Register mock replies
    agent1.register_reply([ConversableAgent, None], mock_a_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_a_generate_oai_reply)

    # Test with string message
    chat_result, context_vars, last_speaker = await a_initiate_swarm_chat(
        initial_agent=agent1, messages="Test message", agents=[agent1, agent2], user_agent=user_agent, max_rounds=3
    )

    assert len(chat_result.chat_history) > 0

    # Test with message list which should include call a_resume
    messages = [{"role": "user", "content": "Test"}, {"role": "assistant", "name": "agent1", "content": "Response"}]

    chat_result, context_vars, last_speaker = await a_initiate_swarm_chat(
        initial_agent=agent1, messages=messages, agents=[agent1, agent2], user_agent=user_agent, max_rounds=3
    )

    assert len(chat_result.chat_history) > 1

    # Test context variables
    test_context = ContextVariables(data={"test_key": "test_value"})
    chat_result, context_vars, last_speaker = await a_initiate_swarm_chat(
        initial_agent=agent1, messages="Test", agents=[agent1, agent2], context_variables=test_context, max_rounds=3
    )

    assert context_vars == test_context.to_dict()
    assert last_speaker is None


def test_swarmresult_afterworkoption() -> None:
    """Tests processing of the return of an AfterWorkOption in a SwarmResult. This is put in the tool executors _next_agent attribute."""

    def call_determine_next_agent(
        next_agent_afterworkoption: AfterWorkOption, swarm_afterworkoption: AfterWorkOption
    ) -> Optional[Union[Agent, Literal["auto"]]]:
        last_speaker_agent = ConversableAgent("dummy_1")
        tool_executor, _ = _prepare_swarm_agents(last_speaker_agent, [last_speaker_agent], ContextVariables())
        user = UserProxyAgent("User")
        groupchat = GroupChat(
            agents=[last_speaker_agent],
            messages=[
                {"tool_calls": "", "role": "tool", "content": "Test message"},
                {"role": "tool", "content": "Test message 2", "name": "dummy_1"},
                {"role": "assistant", "content": "Test message 3", "name": "dummy_1"},
            ],
        )

        last_speaker_agent._swarm_after_work = next_agent_afterworkoption  # type: ignore[attr-defined]

        return _determine_next_agent(
            last_speaker=last_speaker_agent,
            groupchat=groupchat,
            initial_agent=last_speaker_agent,
            use_initial_agent=False,
            tool_execution=tool_executor,
            swarm_agent_names=["dummy_1"],
            user_agent=user,
            swarm_after_work=swarm_afterworkoption,
        )

    next_speaker = call_determine_next_agent(AfterWorkOption.TERMINATE, AfterWorkOption.STAY)
    assert next_speaker is None, "Expected None as the next speaker for AfterWorkOption.TERMINATE"

    next_speaker = call_determine_next_agent(AfterWorkOption.STAY, AfterWorkOption.TERMINATE)
    assert isinstance(next_speaker, ConversableAgent), (
        "Expected the last speaker as the next speaker for AfterWorkOption.STAY"
    )
    assert next_speaker.name == "dummy_1", "Expected the last speaker as the next speaker for AfterWorkOption.TERMINATE"

    next_speaker = call_determine_next_agent(AfterWorkOption.REVERT_TO_USER, AfterWorkOption.TERMINATE)
    assert isinstance(next_speaker, ConversableAgent), (
        "Expected the last speaker as the next speaker for AfterWorkOption.STAY"
    )
    assert next_speaker.name == "User", "Expected the user agent as the next speaker for AfterWorkOption.REVERT_TO_USER"

    next_speaker = call_determine_next_agent(AfterWorkOption.SWARM_MANAGER, AfterWorkOption.TERMINATE)
    assert next_speaker == "auto", "Expected the auto speaker selection mode for AfterWorkOption.SWARM_MANAGER"


@run_for_optional_imports(["openai"], "openai")
def test_update_on_condition_str() -> None:
    """Test UpdateOnConditionStr updates condition strings properly for handoffs"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)

    # Test container to capture condition
    class ConditionContainer:
        def __init__(self) -> None:
            self.captured_condition = None

    condition_container = ConditionContainer()

    # Test with string template
    register_hand_off(
        agent1,
        hand_to=OnCondition(target=agent2, condition=ContextStr(template="Transfer when {test_var} is active")),
    )

    # Mock LLM responses
    def mock_generate_oai_reply_tool_1_2(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        # Get the function description (condition) from the agent's function map
        func_name = "transfer_agent1_to_agent2"
        # Store the condition for verification by accessing the function's description
        func = args[0].tools[0]._func
        condition_container.captured_condition = func._description
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [{"type": "function", "function": {"name": func_name}}],
        }

    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool_1_2)
    agent2.register_reply([ConversableAgent, None], lambda *args, **kwargs: (True, "Response from agent2"))

    # Test string template substitution
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1,
        messages=TEST_MESSAGES,
        agents=[agent1, agent2],
        context_variables=ContextVariables({"test_var": "condition1"}),
        max_rounds=3,
    )

    assert condition_container.captured_condition == "Transfer when condition1 is active"

    agent3 = ConversableAgent("agent3", llm_config=testing_llm_config)
    register_hand_off(
        agent2, hand_to=OnCondition(target=agent3, condition=ContextStr(template="Transfer based on {test_var}"))
    )

    # Reset condition container
    condition_container.captured_condition = None

    def mock_generate_oai_reply_tool_2_3(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
        # Get the function description (condition) from the agent's function map
        func_name = "transfer_agent2_to_agent3"
        # Store the condition for verification by accessing the function's description
        func = args[0].tools[0]._func
        condition_container.captured_condition = func._description
        return True, {
            "role": "assistant",
            "name": "agent1",
            "tool_calls": [{"type": "function", "function": {"name": func_name}}],
        }

    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply_tool_2_3)
    agent3.register_reply([ConversableAgent, None], lambda *args, **kwargs: (True, "Response from agent3"))

    # Test callable function update
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent2,
        messages=TEST_MESSAGES,
        agents=[agent2, agent3],
        context_variables=ContextVariables({"test_var": "condition2"}),
        max_rounds=3,
    )
    assert chat_result is not None
    assert context_vars is not None
    assert last_speaker is not None

    assert condition_container.captured_condition == "Transfer based on condition2"


@run_for_optional_imports(["openai"], "openai")
def test_agent_tool_registration_for_execution(mock_credentials: Credentials) -> None:
    """Tests that an agent's tools property is used for registering tools for execution with the internal tool executor."""

    agent = ConversableAgent(
        name="my_agent",
        llm_config=mock_credentials.llm_config,
    )

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    # Create the mock tool and add it to the agent's _tools property
    mock_tool = Tool(name="test_tool", description="A test tool", func_or_tool=sample_tool_func)
    agent.register_for_llm()(mock_tool)

    # Prepare swarm agents, this is where the tool will be registered for execution
    # with the internal tool executor agent
    tool_execution, _ = _prepare_swarm_agents(agent, [agent], ContextVariables())

    # Check that the tool is register for execution with the tool_execution agent
    assert "test_tool" in tool_execution._function_map


def test_compress_message_func() -> None:
    # test make_remove_function, which is the core to enable `exclude_transit_message` passed to `initiate_swarm_chat`
    message_processor = make_remove_function([
        "transfer_Agent_1_to_Agent_2"  # remove the function call
    ])

    messages: list[dict[str, Any]] = [
        {"content": "start", "name": "_User", "role": "user"},
        {
            "content": "None",
            "tool_calls": [
                {
                    "id": "call_Lkt8SsSwatkffvkPIgXjBNO3",
                    "function": {"arguments": "{}", "name": "transfer_Agent_1_to_Agent_2"},
                    "type": "function",
                }
            ],
            "name": "Agent_1",
            "role": "assistant",
        },
        {
            "content": "Swarm agent --> Agent_2",
            "tool_responses": [
                {"tool_call_id": "call_Lkt8SsSwatkffvkPIgXjBNO3", "role": "tool", "content": "Swarm agent --> Agent_2"}
            ],
            "name": "_Swarm_Tool_Executor",
            "role": "tool",
        },
        {
            "content": "None",
            "tool_calls": [
                {"id": "mock_call_id", "function": {"arguments": "{}", "name": "custom_func"}, "type": "function"}
            ],
            "name": "Agent_2",
            "role": "assistant",
        },
        {
            "content": "N/A",
            "tool_responses": [{"tool_call_id": "mock_call_id", "role": "tool", "content": "N/A"}],
            "name": "_Swarm_Tool_Executor",
            "role": "tool",
        },
    ]
    modified = message_processor(messages)
    assert len(modified) == 3 and modified[-1]["tool_responses"][0]["tool_call_id"] == "mock_call_id", (
        f"Wrong message processing: {modified}"
    )


def test_swarmresult_afterworkoption_tool_swarmresult() -> None:
    """Tests processing of the return of an AfterWorkOption in a SwarmResult. This is put in the tool executors _next_agent attribute."""

    def call_determine_next_agent_from_tool_execution(
        last_speaker_agent: ConversableAgent,
        tool_execution_swarm_result: Union[ConversableAgent, AfterWorkOption, str],
        next_agent_afterworkoption: AfterWorkOption,
        swarm_afterworkoption: AfterWorkOption,
    ) -> Optional[Union[Agent, Literal["auto"]]]:
        another_agent = ConversableAgent(name="another_agent")
        tool_executor, _ = _prepare_swarm_agents(
            last_speaker_agent, [last_speaker_agent, another_agent], ContextVariables()
        )
        tool_executor._swarm_next_agent = tool_execution_swarm_result  # type: ignore[attr-defined]
        user = UserProxyAgent("User")
        groupchat = GroupChat(
            agents=[last_speaker_agent],
            messages=[
                {"tool_calls": "", "role": "tool", "content": "Test message"},
                {"role": "tool", "content": "Test message 2", "name": "dummy_1"},
                {"role": "assistant", "content": "Test message 3", "name": "dummy_1"},
            ],
        )

        last_speaker_agent._swarm_after_work = next_agent_afterworkoption  # type: ignore[attr-defined]

        return _determine_next_agent(
            last_speaker=last_speaker_agent,
            groupchat=groupchat,
            initial_agent=last_speaker_agent,
            use_initial_agent=False,
            tool_execution=tool_executor,
            swarm_agent_names=["dummy_1"],
            user_agent=user,
            swarm_after_work=swarm_afterworkoption,
        )

    dummy_agent = ConversableAgent("dummy_1")
    next_speaker = call_determine_next_agent_from_tool_execution(
        dummy_agent, AfterWorkOption.TERMINATE, AfterWorkOption.STAY, AfterWorkOption.STAY
    )
    assert next_speaker is None, "Expected None as the next speaker for AfterWorkOption.TERMINATE"

    dummy_agent = ConversableAgent("dummy_1")
    next_speaker = call_determine_next_agent_from_tool_execution(
        dummy_agent, AfterWorkOption.STAY, AfterWorkOption.TERMINATE, AfterWorkOption.TERMINATE
    )
    assert isinstance(next_speaker, ConversableAgent)
    assert next_speaker.name == "dummy_1", "Expected the last speaker as the next speaker for AfterWorkOption.TERMINATE"

    dummy_agent = ConversableAgent("dummy_1")
    next_speaker = call_determine_next_agent_from_tool_execution(
        dummy_agent, AfterWorkOption.REVERT_TO_USER, AfterWorkOption.TERMINATE, AfterWorkOption.TERMINATE
    )
    assert isinstance(next_speaker, ConversableAgent)
    assert next_speaker.name == "User", "Expected the user agent as the next speaker for AfterWorkOption.REVERT_TO_USER"

    dummy_agent = ConversableAgent("dummy_1")
    next_speaker = call_determine_next_agent_from_tool_execution(
        dummy_agent, AfterWorkOption.SWARM_MANAGER, AfterWorkOption.TERMINATE, AfterWorkOption.TERMINATE
    )
    assert next_speaker == "auto", "Expected the auto speaker selection mode for AfterWorkOption.SWARM_MANAGER"

    dummy_agent = ConversableAgent("dummy_1")
    next_speaker = call_determine_next_agent_from_tool_execution(
        dummy_agent, "dummy_1", AfterWorkOption.TERMINATE, AfterWorkOption.TERMINATE
    )
    assert next_speaker == dummy_agent, "Expected the auto speaker selection mode for AfterWorkOption.SWARM_MANAGER"

    dummy_agent = ConversableAgent("dummy_1")
    next_speaker = call_determine_next_agent_from_tool_execution(
        dummy_agent, dummy_agent, AfterWorkOption.TERMINATE, AfterWorkOption.TERMINATE
    )
    assert next_speaker == dummy_agent, "Expected the auto speaker selection mode for AfterWorkOption.SWARM_MANAGER"


@run_for_optional_imports(["openai"], "openai")
def test_on_condition_available() -> None:
    """Test OnCondition's available parameter"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)

    # Test container to capture condition
    class ConditionContainer:
        def __init__(self) -> None:
            self.captured_condition = None

    # 1. Test with no available parameter
    register_hand_off(
        agent1,
        hand_to=OnCondition(target=agent2, condition="my_condition_is_true"),
    )

    # Evaluate hand-offs
    _update_conditional_functions(agent=agent1, messages=[{"role": "user", "content": "Test"}])

    assert agent1.llm_config is not False and isinstance(agent1.llm_config, (dict, LLMConfig))
    assert len(agent1.llm_config["tools"]) == 1  # Is available

    # 2. Test with an available parameter that equates to True
    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent1.context_variables.set("context_var_is_true", True)

    register_hand_off(
        agent1,
        hand_to=OnCondition(target=agent2, condition="my_condition_is_true", available="context_var_is_true"),
    )

    # Evaluate hand-offs
    _update_conditional_functions(agent=agent1, messages=[{"role": "user", "content": "Test"}])

    assert agent1.llm_config is not False and isinstance(agent1.llm_config, (dict, LLMConfig))
    assert len(agent1.llm_config["tools"]) == 1  # Is available

    # 3. Test with an available parameter that equates to False
    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent1.context_variables.set("context_var_is_false", False)

    register_hand_off(
        agent1,
        hand_to=OnCondition(target=agent2, condition="my_condition_is_true", available="context_var_is_false"),
    )

    # Evaluate hand-offs
    _update_conditional_functions(agent=agent1, messages=[{"role": "user", "content": "Test"}])

    assert agent1.llm_config is not False and isinstance(agent1.llm_config, (dict, LLMConfig))
    assert "tools" not in agent1.llm_config  # Is not available

    # 4. Test with an available parameter that equates to True using NOT operator "!"
    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent1.context_variables.set("context_var_is_false", False)

    register_hand_off(
        agent1,
        hand_to=OnCondition(
            target=agent2, condition="my_condition_is_true", available=ContextExpression("not(${context_var_is_false})")
        ),
    )

    # Evaluate hand-offs
    _update_conditional_functions(agent=agent1, messages=[{"role": "user", "content": "Test"}])

    assert agent1.llm_config is not False and isinstance(agent1.llm_config, (dict, LLMConfig))
    assert len(agent1.llm_config["tools"]) == 1  # Is available (Not False)

    # 5. Test with an available parameter using a Callable
    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent1._oai_messages[agent2].append({"role": "user", "content": "Test"})

    def is_available(agent: ConversableAgent, messages: list[dict[str, Any]]) -> bool:
        return True

    register_hand_off(
        agent1,
        hand_to=OnCondition(target=agent2, condition="my_condition_is_true", available=is_available),
    )

    # Evaluate hand-offs
    _update_conditional_functions(agent=agent1, messages=[{"role": "user", "content": "Test"}])

    assert agent1.llm_config is not False and isinstance(agent1.llm_config, (dict, LLMConfig))
    assert len(agent1.llm_config["tools"]) == 1  # Is available


def test_on_context_condition() -> None:
    """Test OnContextCondition initialisation and validation."""

    # Test valid initialisation with a string condition
    test_conversable_agent = ConversableAgent("test_agent")
    on_context_condition = OnContextCondition(target=test_conversable_agent, condition="is_valid")

    # Check that the condition was converted to a ContextExpression
    assert isinstance(on_context_condition._context_condition, ContextExpression)

    # Test valid initialisation with a ContextExpression condition
    context_expression = ContextExpression("${is_valid} and ${is_ready}")
    on_context_condition = OnContextCondition(target=test_conversable_agent, condition=context_expression)

    # Check that the condition was stored correctly
    assert on_context_condition._context_condition == context_expression

    # Test invalid target
    test_invalid_agent = invalid_agent("invalid_agent")
    with pytest.raises(ValueError, match="'target' must be a ConversableAgent or a dict"):
        OnContextCondition(target=test_invalid_agent, condition="is_valid")  # type: ignore[arg-type]

    # Test invalid condition type
    with pytest.raises(ValueError, match="'condition' must be a string on ContextExpression"):
        OnContextCondition(target=test_conversable_agent, condition=123)  # type: ignore[arg-type]

    # Test empty string condition
    with pytest.raises(ValueError, match="'condition' must be a non-empty string"):
        OnContextCondition(target=test_conversable_agent, condition="")

    # Test invalid available parameter
    with pytest.raises(ValueError, match="'available' must be a callable, a string, or a ContextExpression"):
        OnContextCondition(target=test_conversable_agent, condition="is_valid", available=123)  # type: ignore[arg-type]


def test_register_hand_off_on_context_condition() -> None:
    """Test registering OnContextCondition with register_hand_off."""

    # Create test agents
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")

    # Register an OnContextCondition handoff
    context_expr = ContextExpression("${is_valid}")
    on_context_condition = OnContextCondition(target=agent2, condition=context_expr)
    register_hand_off(agent1, hand_to=on_context_condition)

    # Check that the OnContextCondition was added to the agent's _swarm_oncontextconditions
    assert len(agent1._swarm_oncontextconditions) == 1  # type: ignore[attr-defined]
    assert agent1._swarm_oncontextconditions[0] == on_context_condition  # type: ignore[attr-defined]


def test_on_context_condition_run() -> None:
    """Test the _run_oncontextconditions function directly."""

    # Create test agents
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    tool_executor = ConversableAgent(__TOOL_EXECUTOR_NAME__)
    _set_to_tool_execution(tool_executor)

    # Create a group chat with these agents
    groupchat = GroupChat(agents=[agent1, agent2, tool_executor], messages=[])
    manager = GroupChatManager(groupchat)

    # Link agent1 to the swarm manager
    agent1._swarm_manager = manager  # type: ignore[attr-defined]

    # Add an OnContextCondition to agent1
    agent1._swarm_oncontextconditions = [OnContextCondition(target=agent2, condition="transfer_to_agent2")]  # type: ignore[attr-defined]

    # Set up context variables for agent1
    agent1.context_variables = ContextVariables({"transfer_to_agent2": True})

    # Call _run_oncontextconditions
    result, message = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])

    # Check that the function returns True and a message
    assert result is True
    assert message == "[Handing off to agent2]"

    # Check that the tool executor's _swarm_next_agent attribute is set to agent2
    assert tool_executor._swarm_next_agent == agent2  # type: ignore[attr-defined]

    # Check that the function returns False when the condition is not met
    agent1.context_variables = ContextVariables({"transfer_to_agent2": False})
    tool_executor._swarm_next_agent = None  # type: ignore[attr-defined]

    result, message = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])

    assert result is False
    assert message is None
    assert tool_executor._swarm_next_agent is None  # type: ignore[attr-defined]

    # Test with a nested chat target
    nested_chat_config = {
        "chat_queue": [],
        "reply_func_from_nested_chats": "summary_from_nested_chats",
        "config": None,
        "use_async": False,
    }

    agent1._swarm_oncontextconditions = [OnContextCondition(target=nested_chat_config, condition="transfer_to_nested")]  # type: ignore[attr-defined]
    agent1.context_variables = ContextVariables({"transfer_to_nested": True})

    result, message = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])

    assert result is True
    assert message == "[Handing off to a nested chat]"
    assert tool_executor._swarm_next_agent == nested_chat_config  # type: ignore[attr-defined]


@run_for_optional_imports(["openai"], "openai")
def test_on_context_condition_available() -> None:
    """Test OnContextCondition's available parameter."""

    # Create test agents
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    tool_executor = ConversableAgent(__TOOL_EXECUTOR_NAME__)
    _set_to_tool_execution(tool_executor)

    # Create a group chat with these agents
    groupchat = GroupChat(agents=[agent1, agent2, tool_executor], messages=[])
    manager = GroupChatManager(groupchat)

    # Link agent1 to the swarm manager
    agent1._swarm_manager = manager  # type: ignore[attr-defined]

    # 1. Test with no available parameter (should be available)
    agent1._swarm_oncontextconditions = [OnContextCondition(target=agent2, condition="transfer_condition")]  # type: ignore[attr-defined]

    agent1.context_variables = ContextVariables({"transfer_condition": True})
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is True

    # 2. Test with string available parameter that's True
    agent1._swarm_oncontextconditions = [  # type: ignore[attr-defined]
        OnContextCondition(target=agent2, condition="transfer_condition", available="is_available")
    ]

    agent1.context_variables = ContextVariables({"transfer_condition": True, "is_available": True})
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is True

    # 3. Test with string available parameter that's False
    agent1.context_variables = ContextVariables({"transfer_condition": True, "is_available": False})
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is False

    # 4. Test with ContextExpression available parameter that's True
    expr = ContextExpression("${feature_enabled} and ${user_authorized}")
    agent1._swarm_oncontextconditions = [  # type: ignore[attr-defined]
        OnContextCondition(target=agent2, condition="transfer_condition", available=expr)
    ]

    agent1.context_variables = ContextVariables({
        "transfer_condition": True,
        "feature_enabled": True,
        "user_authorized": True,
    })
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is True

    # 5. Test with ContextExpression available parameter that's False
    agent1.context_variables = ContextVariables({
        "transfer_condition": True,
        "feature_enabled": True,
        "user_authorized": False,
    })
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is False

    # 6. Test with callable available parameter
    def is_available(agent: ConversableAgent, messages: list[dict[str, Any]]) -> bool:
        return agent.context_variables.get("dynamic_availability", False)  # type: ignore[return-value]

    agent1._swarm_oncontextconditions = [  # type: ignore[attr-defined]
        OnContextCondition(target=agent2, condition="transfer_condition", available=is_available)
    ]

    agent1.context_variables = ContextVariables({"transfer_condition": True, "dynamic_availability": True})
    agent1._oai_messages[agent2].append({"role": "user", "content": "Test"})
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is True

    agent1.context_variables = ContextVariables({"transfer_condition": True, "dynamic_availability": False})
    result, _ = _run_oncontextconditions(agent1, messages=[{"role": "user", "content": "Test"}])
    assert result is False


@run_for_optional_imports(["openai"], "openai")
def test_change_tool_context_variables_to_depends() -> None:
    """
    Test that _change_tool_context_variables_to_depends correctly modifies a tool
    that has a context_variables parameter to use dependency injection.
    """

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent = ConversableAgent(name="test_agent", llm_config=testing_llm_config)

    # Create a test function that has a context_variables parameter
    def test_func_with_context_vars(param1: str, context_variables: dict[str, Any]) -> str:
        """Test function with context variables."""
        return f"param1: {param1}, context_var: {context_variables.get('test_key', 'not_found')}"

    # Create a test function that doesn't have a context_variables parameter
    def test_func_without_context_vars(param1: str, param2: int) -> str:
        """Test function without context variables."""
        return f"param1: {param1}, param2: {param2}"

    # Create tools from these functions
    tool_with_context = Tool(
        name="tool_with_context",
        description="A tool with context_variables parameter",
        func_or_tool=test_func_with_context_vars,
    )

    tool_without_context = Tool(
        name="tool_without_context",
        description="A tool without context_variables parameter",
        func_or_tool=test_func_without_context_vars,
    )

    # Register the tools with the agent
    agent.register_for_llm()(tool_with_context)
    agent.register_for_llm()(tool_without_context)

    # Verify that the tools are registered before the change
    assert "tool_with_context" in [tool._name for tool in agent.tools]
    assert "tool_without_context" in [tool._name for tool in agent.tools]

    # Verify that the tool has context_variables parameter before the change
    assert "context_variables" in tool_with_context.tool_schema["function"]["parameters"]["properties"]

    # Test context variables
    context_variables = ContextVariables({"test_key": "test_value"})

    # Keep track of the number of tools before the change
    tools_count_before = len(agent.tools)

    # Case 1: Tool with context_variables parameter
    _change_tool_context_variables_to_depends(agent, tool_with_context, context_variables)

    # Verify that the number of tools remains the same (one removed, one added)
    assert len(agent.tools) == tools_count_before

    # Find the tool with the same name after the change
    modified_tool = None
    for tool in agent.tools:
        if tool._name == "tool_with_context":
            modified_tool = tool
            break

    assert modified_tool is not None, "Modified tool should still be registered"

    # Case 2: Tool without context_variables parameter
    _change_tool_context_variables_to_depends(agent, tool_without_context, context_variables)

    # Verify that the tool without context_variables is still registered
    assert "tool_without_context" in [tool._name for tool in agent.tools]


@run_for_optional_imports(["openai"], "openai")
def test_change_tool_context_variables_dependency_injection() -> None:
    """
    Test that the modified tool correctly uses dependency injection for the context_variables parameter.
    This test verifies that after modification, the tool can access the context variables without
    explicitly passing them.
    """

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent = ConversableAgent(name="test_agent", llm_config=testing_llm_config)

    # Context variables to be injected
    context_variables = ContextVariables({"test_key": "injected_value"})
    agent.context_variables = context_variables

    # Create a test function that has a context_variables parameter
    def test_func_with_context_vars(param1: str, context_variables: dict[str, Any]) -> str:
        """Test function with context variables."""
        return f"param1: {param1}, context_var: {context_variables.get('test_key', 'not_found')}"

    # Create a tool from this function
    original_tool = Tool(
        name="tool_with_context",
        description="A tool with context_variables parameter",
        func_or_tool=test_func_with_context_vars,
    )

    # Register the tool with the agent
    agent.register_for_llm()(original_tool)

    # Call the function to modify the tool
    _change_tool_context_variables_to_depends(agent, original_tool, context_variables)

    # Find the modified tool
    modified_tool = None
    for tool in agent.tools:
        if tool._name == "tool_with_context":
            modified_tool = tool
            break

    assert modified_tool is not None, "Modified tool should be registered"

    # Call the original tool with explicitly passed context_variables
    original_result = original_tool(param1="test_param", context_variables=context_variables)

    # Call the modified tool without explicitly passing context_variables
    # The context_variables should be automatically injected
    modified_result = modified_tool(param1="test_param")

    # Both results should be the same
    assert original_result == modified_result
    assert "param1: test_param" in modified_result
    assert "context_var: injected_value" in modified_result


@run_for_optional_imports(["openai"], "openai")
def test_change_tool_context_variables_function_signature() -> None:
    """
    Test that specifically checks if the function signature is properly updated
    after modifying the tool to use dependency injection.
    """
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
                "api_type": "openai",
            }
        ]
    }

    agent = ConversableAgent(name="test_agent", llm_config=testing_llm_config)

    # Create a test function that has a context_variables parameter
    def test_func_with_context_vars(param1: str, context_variables: dict[str, Any]) -> str:
        """Test function with context variables."""
        return f"param1: {param1}, context_var: {context_variables.get('test_key', 'not_found')}"

    # Create a tool from this function
    original_tool = Tool(
        name="tool_with_context",
        description="A tool with context_variables parameter",
        func_or_tool=test_func_with_context_vars,
    )

    # Register the tool with the agent
    agent.register_for_llm()(original_tool)

    # Get the original function's signature
    original_sig = inspect.signature(original_tool.func)

    # Verify the original signature has context_variables
    assert "context_variables" in original_sig.parameters
    orig_param = original_sig.parameters["context_variables"]

    # Store the original annotation for later comparison
    orig_annotation = orig_param.annotation
    orig_default = orig_param.default

    # Context variables
    context_variables = ContextVariables({"test_key": "test_value"})

    # Call the function to modify the tool
    _change_tool_context_variables_to_depends(agent, original_tool, context_variables)

    # Find the modified tool
    modified_tool = None
    for tool in agent.tools:
        if tool._name == "tool_with_context":
            modified_tool = tool
            break

    assert modified_tool is not None, "Modified tool should be registered"

    # Get the modified function's signature
    modified_sig = inspect.signature(modified_tool.func)

    # Test the specific changes to the signature
    if "context_variables" in modified_sig.parameters:
        # If context_variables is still in the parameters, its annotation or default should have changed
        mod_param = modified_sig.parameters["context_variables"]

        # Check that something about the parameter has changed
        sig_changed = (
            mod_param.annotation != orig_annotation
            or mod_param.default != orig_default
            or hasattr(mod_param.annotation, "__metadata__")
        )

        assert sig_changed, "The signature of context_variables parameter should be modified"

        # If the function uses Depends, the annotation will typically have __metadata__
        if hasattr(mod_param.annotation, "__metadata__"):
            # __metadata__ should contain at least one item
            assert len(mod_param.annotation.__metadata__) > 0
    else:
        # If context_variables is completely removed, that's also an acceptable change
        assert "context_variables" not in modified_sig.parameters


if __name__ == "__main__":
    pytest.main([__file__])
