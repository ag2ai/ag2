# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat import ConversableAgent
from autogen.agentchat.conversableAgent.types import UpdateSystemMessage
from autogen.tools import Tool
from test.credentials import Credentials


# ============================================================================
# register_hook Tests
# ============================================================================


def test_register_hook_valid_method():
    """Test register_hook registers hook for valid hookable method."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_hook(messages):
        return messages

    agent.register_hook("process_all_messages_before_reply", test_hook)
    assert test_hook in agent.hook_lists["process_all_messages_before_reply"]


def test_register_hook_invalid_method():
    """Test register_hook raises AssertionError for invalid hookable method."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_hook(messages):
        return messages

    with pytest.raises(AssertionError, match="is not a hookable method"):
        agent.register_hook("invalid_method", test_hook)


def test_register_hook_duplicate():
    """Test register_hook raises AssertionError when registering duplicate hook."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_hook(messages):
        return messages

    agent.register_hook("process_all_messages_before_reply", test_hook)
    with pytest.raises(AssertionError, match="is already registered as a hook"):
        agent.register_hook("process_all_messages_before_reply", test_hook)


# ============================================================================
# update_agent_state_before_reply Tests
# ============================================================================


def test_update_agent_state_before_reply_calls_hooks():
    """Test update_agent_state_before_reply calls registered hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(agent, messages):
        call_order.append(1)

    def hook2(agent, messages):
        call_order.append(2)

    agent.register_hook("update_agent_state", hook1)
    agent.register_hook("update_agent_state", hook2)

    messages = [{"role": "user", "content": "test"}]
    agent.update_agent_state_before_reply(messages)

    assert call_order == [1, 2]


def test_update_agent_state_before_reply_modifies_messages():
    """Test update_agent_state_before_reply allows hooks to modify messages."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(agent, messages):
        messages.append({"role": "assistant", "content": "modified"})

    agent.register_hook("update_agent_state", hook)

    messages = [{"role": "user", "content": "test"}]
    agent.update_agent_state_before_reply(messages)

    assert len(messages) == 2
    assert messages[1]["content"] == "modified"


# ============================================================================
# process_all_messages_before_reply Tests
# ============================================================================


def test_process_all_messages_before_reply_no_hooks():
    """Test process_all_messages_before_reply returns original messages when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [{"role": "user", "content": "test"}]

    result = agent.process_all_messages_before_reply(messages)
    assert result == messages


def test_process_all_messages_before_reply_none_messages():
    """Test process_all_messages_before_reply returns None when messages is None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = agent.process_all_messages_before_reply(None)
    assert result is None


def test_process_all_messages_before_reply_calls_hooks():
    """Test process_all_messages_before_reply calls hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(messages):
        call_order.append(1)
        return messages

    def hook2(messages):
        call_order.append(2)
        return messages

    agent.register_hook("process_all_messages_before_reply", hook1)
    agent.register_hook("process_all_messages_before_reply", hook2)

    messages = [{"role": "user", "content": "test"}]
    agent.process_all_messages_before_reply(messages)

    assert call_order == [1, 2]


def test_process_all_messages_before_reply_modifies_messages():
    """Test process_all_messages_before_reply allows hooks to modify messages."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(messages):
        messages = messages.copy()
        messages.append({"role": "assistant", "content": "modified"})
        return messages

    agent.register_hook("process_all_messages_before_reply", hook)

    messages = [{"role": "user", "content": "test"}]
    result = agent.process_all_messages_before_reply(messages)

    assert len(result) == 2
    assert result[1]["content"] == "modified"


# ============================================================================
# process_last_received_message Tests
# ============================================================================


def test_process_last_received_message_no_hooks():
    """Test process_last_received_message returns original when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [{"role": "user", "content": "test"}]

    result = agent.process_last_received_message(messages)
    assert result == messages


def test_process_last_received_message_none():
    """Test process_last_received_message returns None when messages is None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = agent.process_last_received_message(None)
    assert result is None


def test_process_last_received_message_empty():
    """Test process_last_received_message returns empty list when messages is empty."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = agent.process_last_received_message([])
    assert result == []


def test_process_last_received_message_function_call():
    """Test process_last_received_message returns original when last message has function_call."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(content):
        return "modified"

    agent.register_hook("process_last_received_message", hook)

    messages = [{"role": "assistant", "function_call": {"name": "test", "arguments": "{}"}}]
    result = agent.process_last_received_message(messages)

    assert result == messages
    assert result[0]["content"] != "modified"


def test_process_last_received_message_context_key():
    """Test process_last_received_message returns original when last message has context key."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(content):
        return "modified"

    agent.register_hook("process_last_received_message", hook)

    messages = [{"role": "user", "content": "test", "context": {"key": "value"}}]
    result = agent.process_last_received_message(messages)

    assert result == messages


def test_process_last_received_message_no_content():
    """Test process_last_received_message returns original when no content."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(content):
        return "modified"

    agent.register_hook("process_last_received_message", hook)

    messages = [{"role": "user"}]
    result = agent.process_last_received_message(messages)

    assert result == messages


def test_process_last_received_message_exit_command():
    """Test process_last_received_message returns original when content is 'exit'."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(content):
        return "modified"

    agent.register_hook("process_last_received_message", hook)

    messages = [{"role": "user", "content": "exit"}]
    result = agent.process_last_received_message(messages)

    assert result == messages


def test_process_last_received_message_modifies_content():
    """Test process_last_received_message modifies content when hook changes it."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(content):
        return f"modified: {content}"

    agent.register_hook("process_last_received_message", hook)

    messages = [{"role": "user", "content": "test"}]
    result = agent.process_last_received_message(messages)

    assert result[0]["content"] == "modified: test"


def test_process_last_received_message_multimodal_list():
    """Test process_last_received_message handles list content (multimodal)."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(content):
        return content  # Should handle list

    agent.register_hook("process_last_received_message", hook)

    messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
    result = agent.process_last_received_message(messages)

    assert result == messages


# ============================================================================
# _add_functions Tests
# ============================================================================


def test__add_functions_registers_multiple(mock_credentials: Credentials):
    """Test _add_functions registers multiple functions."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def func1(x: str) -> str:
        return x

    def func2(x: str) -> str:
        return x

    with patch.object(agent, "_add_single_function") as mock_add:
        agent._add_functions([func1, func2])
        assert mock_add.call_count == 2


def test__add_functions_calls_add_single(mock_credentials: Credentials):
    """Test _add_functions calls _add_single_function for each function."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def func1(x: str) -> str:
        return x

    def func2(x: str) -> str:
        return x

    with patch.object(agent, "_add_single_function") as mock_add:
        agent._add_functions([func1, func2])
        mock_add.assert_any_call(func1)
        mock_add.assert_any_call(func2)


# ============================================================================
# _add_single_function Tests
# ============================================================================


def test__add_single_function_with_name(mock_credentials: Credentials):
    """Test _add_single_function uses provided name."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    with patch.object(agent, "register_for_llm") as mock_register:
        agent._add_single_function(test_func, name="custom_name", description="test")
        mock_register.assert_called_once()
        # Check that name was set
        assert hasattr(test_func, "_name")
        assert test_func._name == "custom_name"


def test__add_single_function_without_name(mock_credentials: Credentials):
    """Test _add_single_function uses function __name__ when name not provided."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    with patch.object(agent, "register_for_llm") as mock_register:
        agent._add_single_function(test_func, description="test")
        assert hasattr(test_func, "_name")
        assert test_func._name == "test_func"


def test__add_single_function_with_description(mock_credentials: Credentials):
    """Test _add_single_function uses provided description."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    agent._add_single_function(test_func, description="custom description")
    assert hasattr(test_func, "_description")
    assert test_func._description == "custom description"


def test__add_single_function_with_docstring(mock_credentials: Credentials):
    """Test _add_single_function uses function docstring when description not provided."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        """This is a test function docstring."""
        return x

    agent._add_single_function(test_func)
    assert hasattr(test_func, "_description")
    assert test_func._description == "This is a test function docstring."


def test__add_single_function_preserves_existing_description(mock_credentials: Credentials):
    """Test _add_single_function preserves existing _description."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    test_func._description = "existing description"
    agent._add_single_function(test_func)
    assert test_func._description == "existing description"


def test__add_single_function_calls_register_for_llm(mock_credentials: Credentials):
    """Test _add_single_function calls register_for_llm."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    with patch.object(agent, "register_for_llm") as mock_register:
        agent._add_single_function(test_func, name="test", description="desc")
        mock_register.assert_called_once_with(name="test", description="desc", silent_override=True)


# ============================================================================
# _register_update_agent_state_before_reply Tests
# ============================================================================


def test__register_update_agent_state_none():
    """Test _register_update_agent_state_before_reply returns early when functions is None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    # Should not raise any error
    agent._register_update_agent_state_before_reply(None)
    assert len(agent.hook_lists["update_agent_state"]) == 0


def test__register_update_agent_state_single_callable():
    """Test _register_update_agent_state_before_reply registers single callable."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def update_func(agent, messages):
        pass

    agent._register_update_agent_state_before_reply(update_func)
    assert len(agent.hook_lists["update_agent_state"]) == 1
    assert agent.hook_lists["update_agent_state"][0] == update_func


def test__register_update_agent_state_list_callables():
    """Test _register_update_agent_state_before_reply registers list of callables."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def update_func1(agent, messages):
        pass

    def update_func2(agent, messages):
        pass

    agent._register_update_agent_state_before_reply([update_func1, update_func2])
    assert len(agent.hook_lists["update_agent_state"]) == 2


def test__register_update_agent_state_update_system_message_string():
    """Test _register_update_agent_state_before_reply handles UpdateSystemMessage with string template."""
    agent = ConversableAgent(name="agent", llm_config=False)
    agent.context_variables = MagicMock()
    agent.context_variables.to_dict.return_value = {"key": "value"}

    update_msg = UpdateSystemMessage("Template with {key}")
    agent._register_update_agent_state_before_reply(update_msg)

    assert len(agent.hook_lists["update_agent_state"]) == 1

    # Test that the hook works
    with patch.object(agent, "update_system_message") as mock_update:
        messages = [{"role": "user", "content": "test"}]
        hook = agent.hook_lists["update_agent_state"][0]
        hook(agent, messages)
        mock_update.assert_called_once()


def test__register_update_agent_state_update_system_message_callable():
    """Test _register_update_agent_state_before_reply handles UpdateSystemMessage with callable."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def update_func(agent, messages):
        return "Updated system message"

    update_msg = UpdateSystemMessage(update_func)
    agent._register_update_agent_state_before_reply(update_msg)

    assert len(agent.hook_lists["update_agent_state"]) == 1

    # Test that the hook works
    with patch.object(agent, "update_system_message") as mock_update:
        messages = [{"role": "user", "content": "test"}]
        hook = agent.hook_lists["update_agent_state"][0]
        hook(agent, messages)
        mock_update.assert_called_once_with("Updated system message")


def test__register_update_agent_state_invalid_type():
    """Test _register_update_agent_state_before_reply raises ValueError for invalid type."""
    agent = ConversableAgent(name="agent", llm_config=False)

    with pytest.raises(ValueError, match="functions must be a list of callables"):
        agent._register_update_agent_state_before_reply("not a callable")


# ============================================================================
# _unset_previous_ui_tools Tests
# ============================================================================


def test__unset_previous_ui_tools_removes_from_llm(mock_credentials: Credentials):
    """Test _unset_previous_ui_tools removes tools from LLM config."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def tool_func(x: str) -> str:
        return x

    tool1 = Tool(name="tool1", description="test", func_or_tool=tool_func)
    tool2 = Tool(name="tool2", description="test", func_or_tool=tool_func)

    agent.set_ui_tools([tool1, tool2])
    assert len(agent.llm_config.get("tools", [])) == 2

    agent._unset_previous_ui_tools()
    assert len(agent.llm_config.get("tools", [])) == 0


def test__unset_previous_ui_tools_removes_from_tools_list(mock_credentials: Credentials):
    """Test _unset_previous_ui_tools removes from _tools list."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def tool_func(x: str) -> str:
        return x

    tool1 = Tool(name="tool1", description="test", func_or_tool=tool_func)
    tool2 = Tool(name="tool2", description="test", func_or_tool=tool_func)

    agent.set_ui_tools([tool1, tool2])
    assert len(agent._tools) >= 2

    agent._unset_previous_ui_tools()
    # Tools should be removed from _tools list
    assert tool1 not in agent._tools or tool2 not in agent._tools


def test__unset_previous_ui_tools_removes_from_function_map(mock_credentials: Credentials):
    """Test _unset_previous_ui_tools removes from function_map."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def tool_func(x: str) -> str:
        return x

    tool1 = Tool(name="tool1", description="test", func_or_tool=tool_func)

    agent.set_ui_tools([tool1])
    assert "tool1" in agent._function_map

    agent._unset_previous_ui_tools()
    assert "tool1" not in agent._function_map


# ============================================================================
# register_model_client Tests
# ============================================================================


def test_register_model_client_calls_client_method():
    """Test register_model_client calls client.register_model_client."""
    agent = ConversableAgent(name="agent", llm_config=False)

    class MockModelClient:
        pass

    with patch.object(agent.client, "register_model_client") as mock_register:
        agent.register_model_client(MockModelClient, param1="value1", param2="value2")
        mock_register.assert_called_once_with(MockModelClient, param1="value1", param2="value2")


def test_register_model_client_passes_kwargs():
    """Test register_model_client passes kwargs to client."""
    agent = ConversableAgent(name="agent", llm_config=False)

    class MockModelClient:
        pass

    with patch.object(agent.client, "register_model_client") as mock_register:
        agent.register_model_client(MockModelClient, api_key="test", base_url="http://test")
        mock_register.assert_called_once()
        assert mock_register.call_args[1]["api_key"] == "test"
        assert mock_register.call_args[1]["base_url"] == "http://test"


# ============================================================================
# _process_tool_input Tests
# ============================================================================


def test__process_tool_input_no_hooks():
    """Test _process_tool_input returns original when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    tool_input = {"name": "test_func", "arguments": {"x": "value"}}

    result = agent._process_tool_input(tool_input)
    assert result == tool_input


def test__process_tool_input_calls_hooks():
    """Test _process_tool_input calls hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(input_dict):
        call_order.append(1)
        return input_dict

    def hook2(input_dict):
        call_order.append(2)
        return input_dict

    agent.register_hook("safeguard_tool_inputs", hook1)
    agent.register_hook("safeguard_tool_inputs", hook2)

    tool_input = {"name": "test_func", "arguments": {"x": "value"}}
    agent._process_tool_input(tool_input)

    assert call_order == [1, 2]


def test__process_tool_input_hook_returns_none():
    """Test _process_tool_input returns None when hook returns None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(input_dict):
        return None

    agent.register_hook("safeguard_tool_inputs", hook)

    tool_input = {"name": "test_func", "arguments": {"x": "value"}}
    result = agent._process_tool_input(tool_input)

    assert result is None


def test__process_tool_input_modifies_input():
    """Test _process_tool_input allows hooks to modify input."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(input_dict):
        input_dict = input_dict.copy()
        input_dict["arguments"]["x"] = "modified"
        return input_dict

    agent.register_hook("safeguard_tool_inputs", hook)

    tool_input = {"name": "test_func", "arguments": {"x": "value"}}
    result = agent._process_tool_input(tool_input)

    assert result["arguments"]["x"] == "modified"


# ============================================================================
# _process_tool_output Tests
# ============================================================================


def test__process_tool_output_no_hooks():
    """Test _process_tool_output returns original when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    tool_output = {"tool_call_id": "1", "content": "result"}

    result = agent._process_tool_output(tool_output)
    assert result == tool_output


def test__process_tool_output_calls_hooks():
    """Test _process_tool_output calls hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(response):
        call_order.append(1)
        return response

    def hook2(response):
        call_order.append(2)
        return response

    agent.register_hook("safeguard_tool_outputs", hook1)
    agent.register_hook("safeguard_tool_outputs", hook2)

    tool_output = {"tool_call_id": "1", "content": "result"}
    agent._process_tool_output(tool_output)

    assert call_order == [1, 2]


def test__process_tool_output_modifies_output():
    """Test _process_tool_output allows hooks to modify output."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(response):
        response = response.copy()
        response["content"] = "modified"
        return response

    agent.register_hook("safeguard_tool_outputs", hook)

    tool_output = {"tool_call_id": "1", "content": "result"}
    result = agent._process_tool_output(tool_output)

    assert result["content"] == "modified"


# ============================================================================
# _process_llm_input Tests
# ============================================================================


def test__process_llm_input_no_hooks():
    """Test _process_llm_input returns original when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [{"role": "user", "content": "test"}]

    result = agent._process_llm_input(messages)
    assert result == messages


def test__process_llm_input_calls_hooks():
    """Test _process_llm_input calls hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(messages):
        call_order.append(1)
        return messages

    def hook2(messages):
        call_order.append(2)
        return messages

    agent.register_hook("safeguard_llm_inputs", hook1)
    agent.register_hook("safeguard_llm_inputs", hook2)

    messages = [{"role": "user", "content": "test"}]
    agent._process_llm_input(messages)

    assert call_order == [1, 2]


def test__process_llm_input_hook_returns_none():
    """Test _process_llm_input returns None when hook returns None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(messages):
        return None

    agent.register_hook("safeguard_llm_inputs", hook)

    messages = [{"role": "user", "content": "test"}]
    result = agent._process_llm_input(messages)

    assert result is None


def test__process_llm_input_modifies_messages():
    """Test _process_llm_input allows hooks to modify messages."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(messages):
        messages = messages.copy()
        messages.append({"role": "system", "content": "modified"})
        return messages

    agent.register_hook("safeguard_llm_inputs", hook)

    messages = [{"role": "user", "content": "test"}]
    result = agent._process_llm_input(messages)

    assert len(result) == 2
    assert result[1]["content"] == "modified"


# ============================================================================
# _process_llm_output Tests
# ============================================================================


def test__process_llm_output_no_hooks():
    """Test _process_llm_output returns original when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    response = "test response"

    result = agent._process_llm_output(response)
    assert result == response


def test__process_llm_output_calls_hooks():
    """Test _process_llm_output calls hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(response):
        call_order.append(1)
        return response

    def hook2(response):
        call_order.append(2)
        return response

    agent.register_hook("safeguard_llm_outputs", hook1)
    agent.register_hook("safeguard_llm_outputs", hook2)

    response = "test response"
    agent._process_llm_output(response)

    assert call_order == [1, 2]


def test__process_llm_output_modifies_output_string():
    """Test _process_llm_output allows hooks to modify string output."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(response):
        return f"modified: {response}"

    agent.register_hook("safeguard_llm_outputs", hook)

    response = "test response"
    result = agent._process_llm_output(response)

    assert result == "modified: test response"


def test__process_llm_output_modifies_output_dict():
    """Test _process_llm_output allows hooks to modify dict output."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(response):
        if isinstance(response, dict):
            response = response.copy()
            response["content"] = "modified"
        return response

    agent.register_hook("safeguard_llm_outputs", hook)

    response = {"role": "assistant", "content": "test"}
    result = agent._process_llm_output(response)

    assert result["content"] == "modified"


# ============================================================================
# _process_human_input Tests
# ============================================================================


def test__process_human_input_no_hooks():
    """Test _process_human_input returns original when no hooks."""
    agent = ConversableAgent(name="agent", llm_config=False)
    human_input = "test input"

    result = agent._process_human_input(human_input)
    assert result == human_input


def test__process_human_input_calls_hooks():
    """Test _process_human_input calls hooks in order."""
    agent = ConversableAgent(name="agent", llm_config=False)

    call_order = []

    def hook1(input_str):
        call_order.append(1)
        return input_str

    def hook2(input_str):
        call_order.append(2)
        return input_str

    agent.register_hook("safeguard_human_inputs", hook1)
    agent.register_hook("safeguard_human_inputs", hook2)

    human_input = "test input"
    agent._process_human_input(human_input)

    assert call_order == [1, 2]


def test__process_human_input_hook_returns_none():
    """Test _process_human_input returns None when hook returns None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(input_str):
        return None

    agent.register_hook("safeguard_human_inputs", hook)

    human_input = "test input"
    result = agent._process_human_input(human_input)

    assert result is None


def test__process_human_input_modifies_input():
    """Test _process_human_input allows hooks to modify input."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def hook(input_str):
        return f"modified: {input_str}"

    agent.register_hook("safeguard_human_inputs", hook)

    human_input = "test input"
    result = agent._process_human_input(human_input)

    assert result == "modified: test input"
