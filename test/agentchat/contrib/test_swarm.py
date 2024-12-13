# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.contrib.swarm_agent import (
    __CONTEXT_VARIABLES_PARAM_NAME__,
    AFTER_WORK,
    ON_CONDITION,
    AfterWorkOption,
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
)
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

TEST_MESSAGES = [{"role": "user", "content": "Initial message"}]


def test_swarm_agent_initialization():
    """Test SwarmAgent initialization with valid and invalid parameters"""

    # Invalid functions parameter
    with pytest.raises(TypeError):
        SwarmAgent("test_agent", functions="invalid")


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
    agent = SwarmAgent("test")
    result = SwarmResult(values="test", agent=agent)
    assert result.agent == agent


def test_after_work_initialization():
    """Test AFTER_WORK initialization with different options"""
    # Test with AfterWorkOption
    after_work = AFTER_WORK(AfterWorkOption.TERMINATE)
    assert after_work.agent == AfterWorkOption.TERMINATE

    # Test with string
    after_work = AFTER_WORK("TERMINATE")
    assert after_work.agent == AfterWorkOption.TERMINATE

    # Test with SwarmAgent
    agent = SwarmAgent("test")
    after_work = AFTER_WORK(agent)
    assert after_work.agent == agent

    # Test with Callable
    def test_callable(x: int) -> SwarmAgent:
        return agent

    after_work = AFTER_WORK(test_callable)
    assert after_work.agent == test_callable

    # Test with invalid option
    with pytest.raises(ValueError):
        AFTER_WORK("INVALID_OPTION")


def test_on_condition():
    """Test ON_CONDITION initialization"""

    # Test with a ConversableAgent
    test_conversable_agent = ConversableAgent("test_conversable_agent")
    with pytest.raises(AssertionError, match="'target' must be a SwarmAgent or a Dict"):
        _ = ON_CONDITION(target=test_conversable_agent, condition="test condition")


def test_receiving_agent():
    """Test the receiving agent based on various starting messages"""

    # 1. Test with a single message - should always be the initial agent
    messages_one_no_name = [{"role": "user", "content": "Initial message"}]

    test_initial_agent = SwarmAgent("InitialAgent")

    # Test the chat
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=test_initial_agent, messages=messages_one_no_name, agents=[test_initial_agent]
    )

    # Make sure the first speaker (second message) is the initialagent
    assert "name" not in chat_result.chat_history[0]  # _User should not exist
    assert chat_result.chat_history[1].get("name") == "InitialAgent"

    # 2. Test with a single message from an existing agent (should still be initial agent)
    test_second_agent = SwarmAgent("SecondAgent")

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

    assert chat_result.chat_history[0].get("name") == "MyUser"  # Should persist
    assert chat_result.chat_history[1].get("name") == "SecondAgent"


def test_resume_speaker():
    """Tests resumption of chat with multiple messages"""

    test_initial_agent = SwarmAgent("InitialAgent")
    test_second_agent = SwarmAgent("SecondAgent")

    # For multiple messages, last agent initiates the chat
    multiple_messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "name": "InitialAgent", "content": "Second message"},
        {"role": "assistant", "name": "SecondAgent", "content": "Third message"},
    ]

    # Patch initiate_chat on agents so we can monitor which started the conversation
    with patch.object(test_initial_agent, "initiate_chat") as mock_initial_chat, patch.object(
        test_second_agent, "initiate_chat"
    ) as mock_second_chat:

        mock_chat_result = MagicMock()
        mock_chat_result.chat_history = multiple_messages

        # Set up the return value for the mock that will be called
        mock_second_chat.return_value = mock_chat_result

        # Run the function
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=test_initial_agent, messages=multiple_messages, agents=[test_initial_agent, test_second_agent]
        )

        # Ensure the second agent initiated the chat
        mock_second_chat.assert_called_once()

        # And it wasn't the initial_agent's agent
        mock_initial_chat.assert_not_called()


def test_after_work_options():
    """Test different after work options"""

    agent1 = SwarmAgent("agent1")
    agent2 = SwarmAgent("agent2")
    user_agent = UserProxyAgent("test_user")

    # Fake generate_oai_reply
    def mock_generate_oai_reply(*args, **kwargs):
        return True, "This is a mock response from the agent."

    # Mock LLM responses
    agent1.register_reply([ConversableAgent, None], mock_generate_oai_reply)
    agent2.register_reply([ConversableAgent, None], mock_generate_oai_reply)

    # 1. Test TERMINATE
    agent1.after_work = AFTER_WORK(AfterWorkOption.TERMINATE)
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2]
    )
    assert last_speaker == agent1

    # 2. Test REVERT_TO_USER
    agent1.after_work = AFTER_WORK(AfterWorkOption.REVERT_TO_USER)

    test_messages = [
        {"role": "user", "content": "Initial message"},
        {"role": "assistant", "name": "agent1", "content": "Response"},
    ]

    with patch("builtins.input", return_value="continue"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent1, messages=test_messages, agents=[agent1, agent2], user_agent=user_agent, max_rounds=4
        )

    # Ensure that after agent1 is finished, it goes to user (4th message)
    assert chat_result.chat_history[3]["name"] == "test_user"

    # 3. Test STAY
    agent1.after_work = AFTER_WORK(AfterWorkOption.STAY)
    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=test_messages, agents=[agent1, agent2], max_rounds=4
    )

    # Stay on agent1
    assert chat_result.chat_history[3]["name"] == "agent1"

    # 4. Test Callable

    # Transfer to agent2
    def test_callable(last_speaker, messages, groupchat):
        return agent2

    agent1.after_work = AFTER_WORK(test_callable)

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=test_messages, agents=[agent1, agent2], max_rounds=4
    )

    # We should have transferred to agent2 after agent1 has finished
    assert chat_result.chat_history[3]["name"] == "agent2"


def test_on_condition_handoff():
    """Test ON_CONDITION in handoffs"""

    testing_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "SAMPLE_API_KEY",
            }
        ]
    }

    agent1 = SwarmAgent("agent1", llm_config=testing_llm_config)
    agent2 = SwarmAgent("agent2", llm_config=testing_llm_config)

    agent1.register_hand_off(hand_to=ON_CONDITION(target=agent2, condition="always take me to agent 2"))

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

    # We should have transferred to agent2 after agent1 has finished
    assert chat_result.chat_history[3]["name"] == "agent2"


def test_temporary_user_proxy():
    """Test that temporary user proxy agent name is cleared"""
    agent1 = SwarmAgent("agent1")
    agent2 = SwarmAgent("agent2")

    chat_result, context_vars, last_speaker = initiate_swarm_chat(
        initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2]
    )

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
    def test_func_1(context_variables: Dict[str, Any], param1: str) -> str:
        context_variables["my_key"] += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    # Increment the context variable
    def test_func_2(context_variables: Dict[str, Any], param2: str) -> str:
        context_variables["my_key"] += 100
        return SwarmResult(values=f"Test 2 {param2}", context_variables=context_variables, agent=agent1)

    agent1 = SwarmAgent("agent1", llm_config=testing_llm_config)
    agent2 = SwarmAgent("agent2", functions=[test_func_1, test_func_2], llm_config=testing_llm_config)

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
    def test_func_1(context_variables: Dict[str, Any], param1: str) -> str:
        context_variables["my_key"] += 1
        return SwarmResult(values=f"Test 1 {param1}", context_variables=context_variables, agent=agent1)

    agent1 = SwarmAgent("agent1", llm_config=testing_llm_config)
    agent2 = SwarmAgent("agent2", functions=[test_func_1], llm_config=testing_llm_config)

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

    assert chat_result.chat_history[3]["name"] == "agent1"


def test_invalid_parameters():
    """Test various invalid parameter combinations"""
    agent1 = SwarmAgent("agent1")
    agent2 = SwarmAgent("agent2")

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
    """Test that SwarmAgents in the group chat are the only agents in hand-offs"""

    agent1 = SwarmAgent("agent1")
    bad_agent = ConversableAgent("bad_agent")

    with pytest.raises(AssertionError, match="Invalid After Work value"):
        agent1.register_hand_off(hand_to=AFTER_WORK(bad_agent))

    with pytest.raises(AssertionError, match="'target' must be a SwarmAgent or a Dict"):
        agent1.register_hand_off(hand_to=ON_CONDITION(target=bad_agent, condition="Testing"))

    with pytest.raises(ValueError, match="hand_to must be a list of ON_CONDITION or AFTER_WORK"):
        agent1.register_hand_off(0)


def test_initialization():
    """Test initiate_swarm_chat"""

    agent1 = SwarmAgent("agent1")
    agent2 = SwarmAgent("agent2")
    agent3 = SwarmAgent("agent3")
    bad_agent = ConversableAgent("bad_agent")

    with pytest.raises(AssertionError, match="Agents must be a list of SwarmAgents"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent2, messages=TEST_MESSAGES, agents=[agent1, agent2, bad_agent], max_rounds=3
        )

    with pytest.raises(AssertionError, match="initial_agent must be a SwarmAgent"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=bad_agent, messages=TEST_MESSAGES, agents=[agent1, agent2], max_rounds=3
        )

    agent1.register_hand_off(hand_to=AFTER_WORK(agent3))

    with pytest.raises(AssertionError, match="Agent in hand-off must be in the agents list"):
        chat_result, context_vars, last_speaker = initiate_swarm_chat(
            initial_agent=agent1, messages=TEST_MESSAGES, agents=[agent1, agent2], max_rounds=3
        )


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

    # Create SwarmAgent instances
    agent_1 = SwarmAgent(
        name="agent_1",
        system_message="Your task is to call hello_world() function.",
        llm_config=testing_llm_config,
        functions=[hello_world],
    )
    agent_2 = SwarmAgent(
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
        after_work=AFTER_WORK(AfterWorkOption.TERMINATE),
        max_rounds=5,
    )

    # Assertions to verify the behavior
    assert chat_result.chat_history[3]["name"] == "agent_2"
    assert last_active_agent.name == "agent_2"

    # Define a simple function for testing
    def hello_world(context_variables: dict) -> SwarmResult:
        value = "Hello, World!"
        return SwarmResult(values=value, context_variables=context_variables, agent="agent_unknown")

    agent_1 = SwarmAgent(
        name="agent_1",
        system_message="Your task is to call hello_world() function.",
        llm_config=testing_llm_config,
        functions=[hello_world],
    )
    agent_2 = SwarmAgent(
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
            after_work=AFTER_WORK(AfterWorkOption.TERMINATE),
            max_rounds=5,
        )


if __name__ == "__main__":
    pytest.main([__file__])
