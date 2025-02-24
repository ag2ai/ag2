# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.swarm_agent import (
    __TOOL_EXECUTOR_NAME__,
    AfterWork,
    AfterWorkOption,
    ContextStr,
    OnCondition,
    SwarmResult,
    _cleanup_temp_user_messages,
    _create_nested_chats,
    _determine_next_agent,
    _prepare_swarm_agents,
    _process_initial_messages,
    _setup_context_variables,
    a_initiate_swarm_chat,
    initiate_swarm_chat,
    make_remove_function,
    register_hand_off,
)
from autogen.agentchat.conversable_agent import ConversableAgent, UpdateSystemMessage
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.tools.tool import Tool

from ...conftest import (
    Credentials,
)

TEST_MESSAGES = [{"role": "user", "content": "Initial message"}]


class NotConversableAgent(Agent):
    """Dummy class to test invalid agent types"""

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)


def test_swarm_result():
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


def test_swarm_result_serialization():
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


def test_after_work_initialization():
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
    def test_callable(x: int) -> ConversableAgent:
        return agent

    after_work = AfterWork(test_callable)
    assert after_work.agent == test_callable

    # Test with invalid option
    with pytest.raises(ValueError):
        AfterWork("INVALID_OPTION")


def test_on_condition():
    """Test OnCondition initialization"""

    # Test with a base Agent
    test_conversable_agent = NotConversableAgent("test_conversable_agent")
    with pytest.raises(AssertionError, match="'target' must be a ConversableAgent or a dict"):
        _ = OnCondition(target=test_conversable_agent, condition="test condition")


def test_receiving_agent():
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


def test_resume_speaker():
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


def test_after_work_options():
    """Test different after work options"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    user_agent = UserProxyAgent("test_user")

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args, **kwargs):
        return True, "This is a mock response from the agent."

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    # 1. Test TERMINATE
    agent1._swarm_after_work = AfterWork(AfterWorkOption.TERMINATE)
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


def test_on_condition_handoff():
    """Test OnCondition in handoffs"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)

    register_hand_off(agent1, hand_to=OnCondition(target=agent2, condition="always take me to agent 2"))

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args, **kwargs):
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args, **kwargs):
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
    )
    assert context_vars is None
    assert last_speaker

    # We should have transferred to agent2 after agent1 has finished
    assert chat_result.chat_history[3]["name"] == "agent2"


def test_temporary_user_proxy():
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


def test_context_variables_updating_multi_tools():
    """Test context variables handling in tool calls"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    # Starting context variable, this will increment in the swarm
    test_context_variables = {"my_key": 0}

    # Increment the context variable
    def test_func_1(context_variables: dict[str, Any], param1: str) -> str:
        context_variables["my_key"] += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    # Increment the context variable
    def test_func_2(context_variables: dict[str, Any], param2: str) -> str:
        context_variables["my_key"] += 100
        return SwarmResult(values=f"Test 2 {param2}", context_variables=context_variables, agent=agent1)

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", functions=[test_func_1, test_func_2], llm_config=testing_llm_config)

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args, **kwargs):
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args, **kwargs):
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


def test_function_transfer():
    """Tests a function call that has a transfer to agent in the SwarmResult"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    # Starting context variable, this will increment in the swarm
    test_context_variables = {"my_key": 0}

    # Increment the context variable
    def test_func_1(context_variables: dict[str, Any], param1: str) -> str:
        context_variables["my_key"] += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", functions=[test_func_1], llm_config=testing_llm_config)

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args, **kwargs):
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args, **kwargs):
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
    )
    assert context_vars
    assert last_speaker

    assert chat_result.chat_history[3]["name"] == "agent1"


def test_invalid_parameters():
    """Test various invalid parameter combinations"""
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")

    # Test invalid initial agent type
    with pytest.raises(AssertionError):
        initiate_swarm_chat(initial_agent="not_an_agent", messages=TEST_MESSAGES, agents=[agent1, agent2])

    # Test invalid agents list
    with pytest.raises(AssertionError):
        initiate_swarm_chat(initial_agent=agent1, messages=TEST_MESSAGES, agents=["not_an_agent", agent2])

    # Test invalid after_work type
    with pytest.raises(ValueError):
        initiate_swarm_chat(initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2], after_work="invalid")


def test_non_swarm_in_hand_off():
    """Test that agents in the group chat are the only agents in hand-offs"""

    agent1 = ConversableAgent("agent1")
    bad_agent = NotConversableAgent("bad_agent")

    with pytest.raises(AssertionError, match="Invalid After Work value"):
        register_hand_off(agent1, hand_to=AfterWork(bad_agent))

    with pytest.raises(AssertionError, match="'target' must be a ConversableAgent or a dict"):
        register_hand_off(agent1, hand_to=OnCondition(target=bad_agent, condition="Testing"))

    with pytest.raises(ValueError, match="hand_to must be a list of OnCondition or AfterWork"):
        register_hand_off(agent1, 0)


def test_initialization():
    """Test initiate_swarm_chat"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    agent3 = ConversableAgent("agent3")
    bad_agent = NotConversableAgent("bad_agent")

    with pytest.raises(AssertionError, match="Agents must be a list of ConversableAgent"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent2, messages=TEST_MESSAGES, agents=[agent1, agent2, bad_agent], max_rounds=3
        )
        assert chat_result
        assert context_vars
        assert last_speaker

    with pytest.raises(AssertionError, match="initial_agent must be a ConversableAgent"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=bad_agent, messages=TEST_MESSAGES, agents=[agent1, agent2], max_rounds=3
        )
        assert chat_result
        assert context_vars
        assert last_speaker

    register_hand_off(agent1, hand_to=AfterWork(agent3))

    with pytest.raises(AssertionError, match="Agent in hand-off must be in the agents list"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2], max_rounds=3
        )
        assert chat_result
        assert context_vars
        assert last_speaker


def test_update_system_message():
    """Tests the update_agent_state_before_reply functionality with multiple scenarios"""

    # Test container to capture system messages
    class MessageContainer:
        def __init__(self):
            self.captured_sys_message = ""

    message_container = MessageContainer()

    # 1. Test with a callable function
    def custom_update_function(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
        return f"System message with {agent.get_context('test_var')} and {len(messages)} messages"

    # 2. Test with a string template
    template_message = "Template message with {test_var}"

    # Create agents with different update configurations
    agent1 = ConversableAgent("agent1", update_agent_state_before_reply=UpdateSystemMessage(custom_update_function))

    agent2 = ConversableAgent("agent2", update_agent_state_before_reply=UpdateSystemMessage(template_message))

    # Mock the reply function to capture the system message
    def mock_generate_oai_reply(*args, **kwargs):
        # Capture the system message for verification
        message_container.captured_sys_message = args[0]._oai_system_message[0]["content"]
        return True, "Mock response"

    # Register mock reply for both agents
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    # Test context and messages
    test_context = {"test_var": "test_value"}
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


def test_string_agent_params_for_transfer():
    """Test that string agent parameters are handled correctly without using real LLMs."""
    # Define test configuration
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    # Define a simple function for testing
    def hello_world(context_variables: dict) -> SwarmResult:
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
    def mock_generate_oai_reply_agent1(*args, **kwargs):
        return True, {
            "role": "assistant",
            "name": "agent_1",
            "tool_calls": [{"type": "function", "function": {"name": "hello_world", "arguments": "{}"}}],
            "content": "I will call the hello_world function.",
        }

    def mock_generate_oai_reply_agent2(*args, **kwargs):
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
        context_variables={},
        messages="Begin by calling the hello_world() function.",
        after_work=AfterWork(AfterWorkOption.TERMINATE),
        max_rounds=5,
    )

    # Assertions to verify the behavior
    assert chat_result.chat_history[3]["name"] == "agent_2"
    assert last_active_agent.name == "agent_2"

    # Define a simple function for testing
    def hello_world(context_variables: dict) -> SwarmResult:
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
            context_variables={},
            messages="Begin by calling the hello_world() function.",
            after_work=AfterWork(AfterWorkOption.TERMINATE),
            max_rounds=5,
        )

        assert final_context


def test_after_work_callable():
    """Test Callable in an AfterWork handoff"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
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
    def mock_generate_oai_reply(*args, **kwargs):
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


def test_on_condition_unique_function_names():
    """Test that OnCondition in handoffs generate unique function names"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
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
    def mock_generate_oai_reply(*args, **kwargs):
        return True, "This is a mock response from the agent."

    # Fake generate_oai_reply
    def mock_generate_oai_reply_tool(*args, **kwargs):
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
    )
    assert chat_result
    assert context_vars is None
    assert last_speaker

    # Check that agent1 has 3 functions and they have unique names
    assert "transfer_agent1_to_agent2" in agent1._function_map
    assert "transfer_agent1_to_agent2_2" in agent1._function_map
    assert "transfer_agent1_to_agent2_3" in agent1._function_map


def test_prepare_swarm_agents():
    """Test preparation of swarm agents including tool executor setup"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    # Create test agents
    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)
    agent3 = ConversableAgent("agent3", llm_config=testing_llm_config)

    # Add some functions to test tool executor aggregation
    def test_func1():
        pass

    def test_func2():
        pass

    agent1._add_single_function(test_func1)
    agent2._add_single_function(test_func2)

    # Add handoffs to test validation
    register_hand_off(agent=agent1, hand_to=AfterWork(agent=agent2))

    # Test valid preparation
    tool_executor, nested_chat_agents = _prepare_swarm_agents(agent1, [agent1, agent2])

    assert nested_chat_agents == []

    # Verify tool executor setup
    assert tool_executor.name == __TOOL_EXECUTOR_NAME__
    assert "test_func1" in tool_executor._function_map
    assert "test_func2" in tool_executor._function_map

    # Test invalid initial agent type
    with pytest.raises(AssertionError):
        _prepare_swarm_agents(NotConversableAgent("invalid"), [agent1, agent2])

    # Test invalid agents list
    with pytest.raises(AssertionError):
        _prepare_swarm_agents(agent1, [agent1, NotConversableAgent("invalid")])

    # Test missing handoff agent
    register_hand_off(agent=agent3, hand_to=AfterWork(agent=ConversableAgent("missing")))
    with pytest.raises(AssertionError):
        _prepare_swarm_agents(agent1, [agent1, agent2, agent3])


def test_create_nested_chats():
    """Test creation of nested chat agents and registration of handoffs"""
    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    test_agent = ConversableAgent("test_agent", llm_config=testing_llm_config)
    test_agent_2 = ConversableAgent("test_agent_2", llm_config=testing_llm_config)
    nested_chat_agents = []

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
    _create_nested_chats(test_agent, nested_chat_agents)

    # Verify nested chat agent creation
    assert len(nested_chat_agents) == 1
    assert nested_chat_agents[0].name == f"nested_chat_{test_agent.name}_1"

    # Verify nested chat configuration
    # The nested chat agent should have a handoff back to the passed in agent
    assert nested_chat_agents[0]._swarm_after_work.agent == test_agent


def test_process_initial_messages():
    """Test processing of initial messages in different scenarios"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    nested_agent = ConversableAgent("nested_chat_agent1_1")
    user_agent = UserProxyAgent("test_user")

    # Test single string message
    messages = "Initial message"
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


def test_setup_context_variables():
    """Test setup of context variables across agents"""

    tool_execution = ConversableAgent(__TOOL_EXECUTOR_NAME__)
    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")

    groupchat = GroupChat(agents=[tool_execution, agent1, agent2], messages=[])
    manager = GroupChatManager(groupchat)

    test_context = {"test_key": "test_value"}

    _setup_context_variables(tool_execution, [agent1, agent2], manager, test_context)

    # Verify all agents share the same context_variables reference
    assert tool_execution._context_variables is test_context
    assert agent1._context_variables is test_context
    assert agent2._context_variables is test_context
    assert manager._context_variables is test_context


def test_cleanup_temp_user_messages():
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
async def test_a_initiate_swarm_chat():
    """Test async swarm chat"""

    agent1 = ConversableAgent("agent1")
    agent2 = ConversableAgent("agent2")
    user_agent = UserProxyAgent("test_user")

    # Mock async reply function
    async def mock_a_generate_oai_reply(*args, **kwargs):
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
    test_context = {"test_key": "test_value"}
    chat_result, context_vars, last_speaker = await a_initiate_swarm_chat(
        initial_agent=agent1, messages="Test", agents=[agent1, agent2], context_variables=test_context, max_rounds=3
    )

    assert context_vars == test_context
    assert last_speaker is None


def test_swarmresult_afterworkoption():
    """Tests processing of the return of an AfterWorkOption in a SwarmResult. This is put in the tool executors _next_agent attribute."""

    def call_determine_next_agent(
        next_agent_afterworkoption: AfterWorkOption, swarm_afterworkoption: AfterWorkOption
    ) -> Optional[Agent]:
        last_speaker_agent = ConversableAgent("dummy_1")
        tool_executor, _ = _prepare_swarm_agents(last_speaker_agent, [last_speaker_agent])
        user = UserProxyAgent("User")
        groupchat = GroupChat(
            agents=[last_speaker_agent],
            messages=[
                {"tool_calls": "", "role": "tool", "content": "Test message"},
                {"role": "tool", "content": "Test message 2", "name": "dummy_1"},
                {"role": "assistant", "content": "Test message 3", "name": "dummy_1"},
            ],
        )

        last_speaker_agent._swarm_after_work = next_agent_afterworkoption

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
    assert next_speaker.name == "dummy_1", "Expected the last speaker as the next speaker for AfterWorkOption.TERMINATE"

    next_speaker = call_determine_next_agent(AfterWorkOption.REVERT_TO_USER, AfterWorkOption.TERMINATE)
    assert next_speaker.name == "User", "Expected the user agent as the next speaker for AfterWorkOption.REVERT_TO_USER"

    next_speaker = call_determine_next_agent(AfterWorkOption.SWARM_MANAGER, AfterWorkOption.TERMINATE)
    assert next_speaker == "auto", "Expected the auto speaker selection mode for AfterWorkOption.SWARM_MANAGER"


def test_update_on_condition_str():
    """Test UpdateOnConditionStr updates condition strings properly for handoffs"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    agent1 = ConversableAgent("agent1", llm_config=testing_llm_config)
    agent2 = ConversableAgent("agent2", llm_config=testing_llm_config)

    # Test container to capture condition
    class ConditionContainer:
        def __init__(self):
            self.captured_condition = None

    condition_container = ConditionContainer()

    # Test with string template
    register_hand_off(
        agent1,
        hand_to=OnCondition(target=agent2, condition=ContextStr(template="Transfer when {test_var} is active")),
    )

    # Mock LLM responses
    def mock_generate_oai_reply_tool_1_2(*args, **kwargs):
        # Get the function description (condition) from the agent's function map
        func_name = "transfer_agent1_to_agent2"
        # Store the condition for verification by accessing the function's description
        func = args[0]._function_map[func_name]
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
        context_variables={"test_var": "condition1"},
        max_rounds=3,
    )

    assert condition_container.captured_condition == "Transfer when condition1 is active"

    agent3 = ConversableAgent("agent3", llm_config=testing_llm_config)
    register_hand_off(
        agent2, hand_to=OnCondition(target=agent3, condition=ContextStr(template="Transfer based on {test_var}"))
    )

    # Reset condition container
    condition_container.captured_condition = None

    def mock_generate_oai_reply_tool_2_3(*args, **kwargs):
        # Get the function description (condition) from the agent's function map
        func_name = "transfer_agent2_to_agent3"
        # Store the condition for verification by accessing the function's description
        func = args[0]._function_map[func_name]
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
        context_variables={"test_var": "condition2"},
        max_rounds=3,
    )
    assert chat_result is not None
    assert context_vars is not None
    assert last_speaker is not None

    assert condition_container.captured_condition == "Transfer based on condition2"


def test_agent_tool_registration_for_execution(mock_credentials: Credentials):
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
    tool_execution, _ = _prepare_swarm_agents(agent, [agent])

    # Check that the tool is register for execution with the tool_execution agent
    assert "test_tool" in tool_execution._function_map


def test_compress_message_func():
    # test make_remove_function, which is the core to enable `exclude_transit_message` passed to `initiate_swarm_chat`
    message_processor = make_remove_function([
        "transfer_Agent_1_to_Agent_2"  # remove the function call
    ])

    messages = [
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


if __name__ == "__main__":
    pytest.main([__file__])
