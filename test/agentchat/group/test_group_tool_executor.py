# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import json

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.code_utils import content_str
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import __TOOL_EXECUTOR_NAME__, GroupToolExecutor
from autogen.agentchat.group.reply_result import ReplyResult
from autogen.agentchat.group.targets.transition_target import TransitionTarget
from autogen.tools import Tool


class TestGroupToolExecutor:
    @pytest.fixture
    def mock_agent(self) -> "ConversableAgent":
        """Create a mock ConversableAgent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "MockAgent"
        agent.tools = []
        agent.remove_tool_for_llm = MagicMock()
        agent.register_for_llm = MagicMock()
        return agent

    @pytest.fixture
    def executor(self) -> GroupToolExecutor:
        """Create a GroupToolExecutor for testing."""
        return GroupToolExecutor()

    def test_initialisation(self, executor: GroupToolExecutor) -> None:
        """Test that the executor initialises with the correct name and defaults."""
        assert executor.name == __TOOL_EXECUTOR_NAME__
        assert executor._group_next_target is None
        assert executor.system_message == "Tool Execution, do not use this agent directly."
        assert executor.human_input_mode == "NEVER"
        assert not executor._code_execution_config

    def test_next_target_management(self, executor: GroupToolExecutor) -> None:
        """Test setting, getting, checking and clearing the next target."""
        # Initially, there should be no next target
        with pytest.raises(ValueError):
            assert executor.get_next_target() is None  # Raises an error if we try to get it without one

        assert not executor.has_next_target()

        # Set a next target
        mock_target = MagicMock(spec=TransitionTarget)
        executor.set_next_target(mock_target)

        # Check the target is set correctly
        assert executor.get_next_target() is mock_target
        assert executor.has_next_target()

        # Clear the target
        executor.clear_next_target()
        with pytest.raises(ValueError):
            assert executor.get_next_target() is None
        assert not executor.has_next_target()

    @patch("autogen.tools.Depends")
    @patch("inspect.signature")
    def test_modify_context_variables_param(
        self, mock_signature: MagicMock, mock_depends: MagicMock, executor: GroupToolExecutor
    ) -> None:
        """Test modifying function parameters to use dependency injection."""
        # Create mock function and signature
        mock_func = MagicMock()
        mock_param = MagicMock()

        # Mock signature with context_variables parameter
        mock_signature.return_value.parameters = {"context_variables": mock_param, "other_param": "other_value"}

        # Mock the new signature
        mock_new_sig = MagicMock()
        mock_signature.return_value.replace.return_value = mock_new_sig

        # Run the function
        context_vars = ContextVariables()
        result = executor._modify_context_variables_param(mock_func, context_vars)

        # Assert the parameter was replaced
        mock_param.replace.assert_called_once()
        mock_signature.return_value.replace.assert_called_once()

        # The function's signature should be updated
        assert hasattr(result, "__signature__")
        assert result.__signature__ == mock_new_sig

    @patch("autogen.agentchat.group.group_tool_executor.inject_params")
    def test_change_tool_context_variables_to_depends(
        self, mock_inject_params: MagicMock, executor: GroupToolExecutor, mock_agent: MagicMock
    ) -> None:
        """Test changing a tool's context_variables parameter to use dependency injection."""

        # Define a real function for the tool to use
        def test_tool_func(arg1: str, context_variables: ContextVariables | None = None) -> str:
            return f"Result: {arg1}, {context_variables}"

        # Create a mock tool with context_variables in its schema
        mock_tool = MagicMock(spec=Tool)
        mock_tool._name = "test_tool"
        mock_tool._description = "Test tool description"
        mock_tool._func = test_tool_func
        mock_tool.tool_schema = {"function": {"parameters": {"properties": {"context_variables": {"type": "object"}}}}}

        # Set up mocks
        mock_agent.tools = [mock_tool]
        mock_agent.remove_tool_for_llm.return_value = None

        # Mock the inject_params to return the original function
        mock_inject_params.return_value = test_tool_func

        # Make the internal _create_tool_if_needed method return a mock tool
        with patch.object(ConversableAgent, "_create_tool_if_needed", return_value=mock_tool):
            # Run the function
            context_vars = ContextVariables()
            executor._change_tool_context_variables_to_depends(mock_agent, mock_tool, context_vars)

            # Verify the right methods were called
            mock_agent.remove_tool_for_llm.assert_called_once_with(mock_tool)
            mock_inject_params.assert_called_once()
            mock_agent.register_for_llm.assert_called_once()

            # Check the tool was removed and re-registered
            mock_agent.remove_tool_for_llm.assert_called_once_with(mock_tool)
            mock_agent.register_for_llm.assert_called_once()

    def test_register_agents_functions(self, executor: GroupToolExecutor) -> None:
        """Test registering functions from multiple agents."""
        # Create mock agents with tools and function_map
        mock_agent1 = MagicMock(spec=ConversableAgent)
        mock_agent1._function_map = {"func1": lambda: "result1"}

        mock_agent2 = MagicMock(spec=ConversableAgent)
        mock_agent2._function_map = {"func2": lambda: "result2"}

        # Create tools that don't have context_variables in their schema
        mock_tool1 = MagicMock(spec=Tool)
        mock_tool1._name = "tool1"
        mock_tool1.tool_schema = {
            "function": {
                "parameters": {
                    "properties": {}  # No context_variables
                }
            }
        }

        mock_tool2 = MagicMock(spec=Tool)
        mock_tool2._name = "tool2"
        mock_tool2.tool_schema = {
            "function": {
                "parameters": {
                    "properties": {}  # No context_variables
                }
            }
        }

        mock_agent1.tools = [mock_tool1]
        mock_agent2.tools = [mock_tool2]

        # Patch _change_tool_context_variables_to_depends to do nothing
        # Patch register_for_execution
        with (
            patch.object(executor, "_change_tool_context_variables_to_depends") as mock_change,
            patch.object(executor, "register_for_execution", return_value=lambda x: x) as mock_register,
        ):
            # Run the function
            context_vars = ContextVariables()
            executor.register_agents_functions([mock_agent1, mock_agent2], context_vars)

            # Check that function maps were merged
            assert "func1" in executor._function_map
            assert "func2" in executor._function_map

            # Check that _change_tool_context_variables_to_depends was called for each tool
            assert mock_change.call_count == 2

            # Check the tools were registered
            assert mock_register.call_count == 2
            mock_register.assert_has_calls(
                [call(serialize=False, silent_override=True), call(serialize=False, silent_override=True)],
                any_order=True,
            )

    def test_generate_group_tool_reply_with_no_tool_calls(self, executor: GroupToolExecutor) -> None:
        """Test _generate_group_tool_reply with a message without tool_calls."""
        # Create message without tool_calls
        message = {"role": "user", "content": "Hello"}
        messages = [message]

        # Run the function
        success, result = executor._generate_group_tool_reply(agent=executor, messages=messages)

        # Should return False and None
        assert success is False
        assert result is None

    def test_generate_group_tool_reply_with_tool_calls(self, executor: GroupToolExecutor) -> None:
        """Test _generate_group_tool_reply with a message with tool_calls."""

        # Create a mock function to execute
        def mock_func(arg1: str, arg2: str) -> str:
            return f"Result: {arg1}, {arg2}"

        # Add the function to the executor's function map
        executor._function_map = {"test_function": mock_func}

        # Create message with tool_calls
        tool_call = {
            "id": "call1",
            "function": {"name": "test_function", "arguments": '{"arg1": "value1", "arg2": "value2"}'},
        }
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        # Mock the generate_tool_calls_reply method
        mock_tool_response = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call1", "role": "tool", "content": "Result: value1, value2"}],
            "content": "Result: value1, value2",
        }

        with patch.object(
            executor, "generate_tool_calls_reply", return_value=(True, mock_tool_response)
        ) as mock_generate:
            # Run the function
            success, result = executor._generate_group_tool_reply(agent=executor, messages=messages)

            # Should call generate_tool_calls_reply with the message containing only the first tool call
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert len(call_args) == 1
            assert "tool_calls" in call_args[0]
            assert len(call_args[0]["tool_calls"]) == 1

            # Should return True and the tool response
            assert success is True
            assert result == mock_tool_response

    def test_generate_group_tool_reply_with_reply_result(self, executor: GroupToolExecutor) -> None:
        """Test _generate_group_tool_reply handling a ReplyResult response."""
        # Create a ReplyResult to be returned by the tool
        result = ReplyResult(
            message="Tool executed successfully",
            target=MagicMock(spec=TransitionTarget),
            context_variables=ContextVariables(data={"new_var": "new_value"}),
        )

        # Create a mock agent with context variables
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables()

        # Create message with tool_calls
        tool_call = {"id": "call1", "function": {"name": "test_function", "arguments": "{}"}}
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        # Mock the generate_tool_calls_reply method to return a ReplyResult
        mock_tool_response = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call1", "role": "tool", "content": result}],
            "content": str(result),
        }

        with patch.object(mock_agent, "generate_tool_calls_reply", return_value=(True, mock_tool_response)):
            # Run the function
            success, response = executor._generate_group_tool_reply(agent=mock_agent, messages=messages)

            # Context variables should be updated
            assert mock_agent.context_variables.get("new_var") == "new_value"

            # Next target should be set
            assert executor._group_next_target == result.target

            # Response content should be converted to string
            assert success is True
            assert response is not None
            assert "content" in response
            assert response["content"] == str(result)

    def test_generate_group_tool_reply_with_multiple_tools(self, executor: GroupToolExecutor) -> None:
        """Test _generate_group_tool_reply with multiple tool calls."""
        # Create a ReplyResult to be returned by the first tool
        result1 = ReplyResult(
            message="Tool 1 executed", target=None, context_variables=ContextVariables(data={"var1": "value1"})
        )

        # Create a ReplyResult to be returned by the second tool
        result2 = ReplyResult(
            message="Tool 2 executed",
            target=MagicMock(spec=TransitionTarget),
            context_variables=ContextVariables(data={"var2": "value2"}),
        )

        # Create a mock agent with context variables
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables()

        # Create message with multiple tool_calls
        tool_call1 = {"id": "call1", "function": {"name": "function1", "arguments": "{}"}}
        tool_call2 = {"id": "call2", "function": {"name": "function2", "arguments": "{}"}}
        message = {"role": "user", "content": "Execute tools", "tool_calls": [tool_call1, tool_call2]}
        messages = [message]

        # Mock the generate_tool_calls_reply method for each tool call
        mock_response1 = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call1", "role": "tool", "content": result1}],
            "content": str(result1),
        }

        mock_response2 = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call2", "role": "tool", "content": result2}],
            "content": str(result2),
        }

        def side_effect(messages: list[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
            if len(messages) == 0:
                return False, None

            if messages[0]["tool_calls"][0]["id"] == "call1":
                return True, mock_response1
            else:
                return True, mock_response2

        with patch.object(mock_agent, "generate_tool_calls_reply", side_effect=side_effect):
            # Run the function
            success, response = executor._generate_group_tool_reply(agent=mock_agent, messages=messages)

            # Context variables should be updated with all values
            assert mock_agent.context_variables.get("var1") == "value1"
            assert mock_agent.context_variables.get("var2") == "value2"

            # Next target should be set to the last non-None target
            assert executor._group_next_target == result2.target

            # Response should contain concatenated content
            assert success is True
            assert response is not None  # Ensure response is not None before indexing
            assert "content" in response
            assert str(result1) in response["content"]
            assert str(result2) in response["content"]

    def test_error_handling(self, executor: GroupToolExecutor) -> None:
        """Test error handling in _generate_group_tool_reply."""
        # Create message with tool_calls
        tool_call = {"id": "call1", "function": {"name": "test_function", "arguments": "{}"}}
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        # Mock generate_tool_calls_reply to return None for tool_message
        with patch.object(executor, "generate_tool_calls_reply", return_value=(True, None)):
            # Run the function - should raise ValueError
            with pytest.raises(ValueError) as excinfo:
                executor._generate_group_tool_reply(agent=executor, messages=messages)

            assert "Tool call did not return a message" in str(excinfo.value)

    def test_function_is_agent_llm_handoff(self, executor: GroupToolExecutor) -> None:
        """Test function_is_agent_llm_handoff method."""
        # Setup mock group manager and groupchat
        mock_group_manager = MagicMock()
        mock_groupchat = MagicMock()
        mock_group_manager.groupchat = mock_groupchat
        executor._group_manager = mock_group_manager

        # Test case 1: Agent not found
        mock_groupchat.agent_by_name.return_value = None
        result = executor.function_is_agent_llm_handoff("NonExistentAgent", "some_function")
        assert result is False
        mock_groupchat.agent_by_name.assert_called_with("NonExistentAgent")

        # Test case 2: Agent found but not a ConversableAgent
        mock_agent = MagicMock()
        mock_agent.__class__.__name__ = "NotConversableAgent"
        mock_groupchat.agent_by_name.return_value = mock_agent
        result = executor.function_is_agent_llm_handoff("SomeAgent", "some_function")
        assert result is False

        # Test case 3: ConversableAgent but no handoffs attribute
        mock_conversable = MagicMock(spec=ConversableAgent)
        del mock_conversable.handoffs  # Remove handoffs attribute
        mock_groupchat.agent_by_name.return_value = mock_conversable
        result = executor.function_is_agent_llm_handoff("SomeAgent", "some_function")
        assert result is False

        # Test case 4: Agent with handoffs but function not found
        mock_conversable = MagicMock(spec=ConversableAgent)
        mock_handoffs = MagicMock()
        mock_condition1 = MagicMock()
        mock_condition1.llm_function_name = "other_function"
        mock_condition2 = MagicMock()
        mock_condition2.llm_function_name = "another_function"
        mock_handoffs.llm_conditions = [mock_condition1, mock_condition2]
        mock_conversable.handoffs = mock_handoffs
        mock_groupchat.agent_by_name.return_value = mock_conversable
        result = executor.function_is_agent_llm_handoff("SomeAgent", "target_function")
        assert result is False

        # Test case 5: Function found in handoffs
        mock_condition3 = MagicMock()
        mock_condition3.llm_function_name = "target_function"
        mock_handoffs.llm_conditions.append(mock_condition3)
        result = executor.function_is_agent_llm_handoff("SomeAgent", "target_function")
        assert result is True

    def test_get_sender_agent_for_message(self, executor: GroupToolExecutor) -> None:
        """Test get_sender_agent_for_message method."""
        # Setup mock group manager and groupchat
        mock_group_manager = MagicMock()
        mock_groupchat = MagicMock()
        mock_group_manager.groupchat = mock_groupchat
        executor._group_manager = mock_group_manager

        # Test case 1: No 'name' in message
        message = {"content": "some content"}
        result = executor.get_sender_agent_for_message(message)
        assert result is None
        mock_groupchat.agent_by_name.assert_not_called()

        # Test case 2: No group manager
        executor._group_manager = None
        message = {"name": "SomeAgent", "content": "some content"}
        result = executor.get_sender_agent_for_message(message)
        assert result is None

        # Test case 3: Agent found successfully
        executor._group_manager = mock_group_manager
        mock_agent = MagicMock()
        mock_agent.name = "SomeAgent"
        mock_groupchat.agent_by_name.return_value = mock_agent
        message = {"name": "SomeAgent", "content": "some content"}
        result = executor.get_sender_agent_for_message(message)
        assert result == mock_agent
        mock_groupchat.agent_by_name.assert_called_with("SomeAgent")

        # Test case 4: Agent not found in groupchat
        mock_groupchat.agent_by_name.return_value = None
        message = {"name": "NonExistentAgent", "content": "some content"}
        result = executor.get_sender_agent_for_message(message)
        assert result is None

    def test_is_handoff_function(self, executor: GroupToolExecutor) -> None:
        """Test is_handoff_function method."""
        # Setup mock group manager and groupchat
        mock_group_manager = MagicMock()
        mock_groupchat = MagicMock()
        mock_group_manager.groupchat = mock_groupchat
        executor._group_manager = mock_group_manager

        # Test case 1: No 'name' in message
        message = {"content": "some content"}
        result = executor.is_handoff_function(message)
        assert result is False

        # Test case 2: No tool_calls in message
        message = {"name": "SomeAgent", "content": "some content"}
        result = executor.is_handoff_function(message)
        assert result is False

        # Test case 3: No group manager
        executor._group_manager = None
        message3: dict[str, Any] = {"name": "SomeAgent", "tool_calls": [{"function": {"name": "some_function"}}]}
        result = executor.is_handoff_function(message3)
        assert result is False

        # Test case 4: Tool call without function key
        executor._group_manager = mock_group_manager
        message4: dict[str, Any] = {"name": "SomeAgent", "tool_calls": [{"id": "call1"}]}
        result = executor.is_handoff_function(message4)
        assert result is False

        # Test case 5: Tool call without function name
        message5: dict[str, Any] = {"name": "SomeAgent", "tool_calls": [{"function": {"arguments": "{}"}}]}
        result = executor.is_handoff_function(message5)
        assert result is False

        # Test case 6: Valid tool call but not a handoff function
        with patch.object(executor, "function_is_agent_llm_handoff", return_value=False) as mock_check:
            message6: dict[str, Any] = {"name": "SomeAgent", "tool_calls": [{"function": {"name": "regular_function"}}]}
            result = executor.is_handoff_function(message6)
            assert result is False
            mock_check.assert_called_with("SomeAgent", "regular_function")

        # Test case 7: Valid handoff function
        with patch.object(executor, "function_is_agent_llm_handoff", return_value=True) as mock_check:
            message7: dict[str, Any] = {"name": "SomeAgent", "tool_calls": [{"function": {"name": "handoff_function"}}]}
            result = executor.is_handoff_function(message7)
            assert result is True
            mock_check.assert_called_with("SomeAgent", "handoff_function")

        # Test case 8: Multiple tool calls, one is a handoff
        with patch.object(executor, "function_is_agent_llm_handoff") as mock_check:
            mock_check.side_effect = [False, True]  # First returns False, second returns True
            message8: dict[str, Any] = {
                "name": "SomeAgent",
                "tool_calls": [{"function": {"name": "regular_function"}}, {"function": {"name": "handoff_function"}}],
            }
            result = executor.is_handoff_function(message8)
            assert result is True
            assert mock_check.call_count == 2

    def test_normalize_tool_content_none(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with None."""
        result = executor._normalize_tool_content(None)
        assert result == ""

    def test_normalize_tool_content_string(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a string."""
        result = executor._normalize_tool_content("hello world")
        assert result == "hello world"

    def test_normalize_tool_content_plain_list(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a plain Python list."""
        content = [5, 3, 10]
        result = executor._normalize_tool_content(content)
        assert result == json.dumps(content)

    def test_normalize_tool_content_empty_list(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with an empty list."""
        result = executor._normalize_tool_content([])
        assert result == json.dumps([])

    def test_normalize_tool_content_openai_format(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with OpenAI message format (list of dicts with 'type' keys)."""
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "test.jpg"}},
        ]
        result = executor._normalize_tool_content(content)
        assert result == content_str(content)

    def test_normalize_tool_content_list_of_dicts_no_type(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a list of dicts without 'type' keys."""
        content = [{"key": "value"}, {"another": "dict"}]
        result = executor._normalize_tool_content(content)
        assert result == json.dumps(content)

    def test_normalize_tool_content_tuple(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a tuple."""
        content = (5, 3, 10)
        result = executor._normalize_tool_content(content)
        assert result == json.dumps(content)

    def test_normalize_tool_content_dict(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a dictionary."""
        content = {"key": "value", "nested": {"a": 1}}
        result = executor._normalize_tool_content(content)
        assert result == json.dumps(content)

    def test_normalize_tool_content_int(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with an integer."""
        result = executor._normalize_tool_content(42)
        assert result == json.dumps(42)

    def test_normalize_tool_content_float(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a float."""
        result = executor._normalize_tool_content(3.14)
        assert result == json.dumps(3.14)

    def test_normalize_tool_content_bool(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a boolean."""
        result = executor._normalize_tool_content(True)
        assert result == json.dumps(True)

    def test_normalize_tool_content_non_json_serializable(self, executor: GroupToolExecutor) -> None:
        """Test _normalize_tool_content with a non-JSON-serializable object."""

        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        content = CustomObject()
        result = executor._normalize_tool_content(content)
        assert result == str(content)


class TestGroupToolExecutorAsync:
    """Tests for the async _a_generate_group_tool_reply method."""

    @pytest.fixture
    def executor(self) -> GroupToolExecutor:
        """Create a GroupToolExecutor for testing."""
        return GroupToolExecutor()

    def test_async_reply_handler_registered(self, executor: GroupToolExecutor) -> None:
        """Test that the async reply handler is registered during __init__."""
        async_handler_found = False
        for reply_func_tuple in executor._reply_func_list:
            func = reply_func_tuple["reply_func"]
            func_name = getattr(func, "__name__", "") or getattr(func, "__func__", lambda: None).__name__
            if func_name == "_a_generate_group_tool_reply":
                assert reply_func_tuple.get("ignore_async_in_sync_chat"), (
                    "Async handler should have ignore_async_in_sync_chat=True"
                )
                async_handler_found = True
                break
        assert async_handler_found, "_a_generate_group_tool_reply should be registered as a reply function"

    @pytest.mark.asyncio
    async def test_a_generate_group_tool_reply_with_no_tool_calls(self, executor: GroupToolExecutor) -> None:
        """Test async handler with a message without tool_calls."""
        message = {"role": "user", "content": "Hello"}
        messages = [message]

        success, result = await executor._a_generate_group_tool_reply(agent=executor, messages=messages)

        assert success is False
        assert result is None

    @pytest.mark.asyncio
    async def test_a_generate_group_tool_reply_with_tool_calls(self, executor: GroupToolExecutor) -> None:
        """Test async handler with a message with tool_calls."""
        tool_call = {
            "id": "call1",
            "function": {"name": "test_function", "arguments": '{"arg1": "value1"}'},
        }
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        mock_tool_response = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call1", "role": "tool", "content": "Result: value1"}],
            "content": "Result: value1",
        }

        with patch.object(
            executor, "a_generate_tool_calls_reply", new_callable=AsyncMock, return_value=(True, mock_tool_response)
        ) as mock_generate:
            success, result = await executor._a_generate_group_tool_reply(agent=executor, messages=messages)

            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert len(call_args) == 1
            assert "tool_calls" in call_args[0]
            assert len(call_args[0]["tool_calls"]) == 1

            assert success is True
            assert result == mock_tool_response

    @pytest.mark.asyncio
    async def test_a_generate_group_tool_reply_with_reply_result(self, executor: GroupToolExecutor) -> None:
        """Test async handler handling a ReplyResult response."""
        result = ReplyResult(
            message="Tool executed successfully",
            target=MagicMock(spec=TransitionTarget),
            context_variables=ContextVariables(data={"new_var": "new_value"}),
        )

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables()

        tool_call = {"id": "call1", "function": {"name": "test_function", "arguments": "{}"}}
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        mock_tool_response = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call1", "role": "tool", "content": result}],
            "content": str(result),
        }

        mock_agent.a_generate_tool_calls_reply = AsyncMock(return_value=(True, mock_tool_response))

        success, response = await executor._a_generate_group_tool_reply(agent=mock_agent, messages=messages)

        assert mock_agent.context_variables.get("new_var") == "new_value"
        assert executor._group_next_target == result.target
        assert success is True
        assert response is not None
        assert "content" in response
        assert response["content"] == str(result)

    @pytest.mark.asyncio
    async def test_a_generate_group_tool_reply_with_multiple_tools(self, executor: GroupToolExecutor) -> None:
        """Test async handler with multiple tool calls."""
        result1 = ReplyResult(
            message="Tool 1 executed", target=None, context_variables=ContextVariables(data={"var1": "value1"})
        )
        result2 = ReplyResult(
            message="Tool 2 executed",
            target=MagicMock(spec=TransitionTarget),
            context_variables=ContextVariables(data={"var2": "value2"}),
        )

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables()

        tool_call1 = {"id": "call1", "function": {"name": "function1", "arguments": "{}"}}
        tool_call2 = {"id": "call2", "function": {"name": "function2", "arguments": "{}"}}
        message = {"role": "user", "content": "Execute tools", "tool_calls": [tool_call1, tool_call2]}
        messages = [message]

        mock_response1 = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call1", "role": "tool", "content": result1}],
            "content": str(result1),
        }
        mock_response2 = {
            "role": "tool",
            "tool_responses": [{"tool_call_id": "call2", "role": "tool", "content": result2}],
            "content": str(result2),
        }

        async def side_effect(messages: list[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
            if len(messages) == 0:
                return False, None
            if messages[0]["tool_calls"][0]["id"] == "call1":
                return True, mock_response1
            else:
                return True, mock_response2

        mock_agent.a_generate_tool_calls_reply = AsyncMock(side_effect=side_effect)

        success, response = await executor._a_generate_group_tool_reply(agent=mock_agent, messages=messages)

        assert mock_agent.context_variables.get("var1") == "value1"
        assert mock_agent.context_variables.get("var2") == "value2"
        assert executor._group_next_target == result2.target
        assert success is True
        assert response is not None
        assert "content" in response
        assert str(result1) in response["content"]
        assert str(result2) in response["content"]

    @pytest.mark.asyncio
    async def test_a_generate_group_tool_reply_error_handling(self, executor: GroupToolExecutor) -> None:
        """Test error handling in async handler."""
        tool_call = {"id": "call1", "function": {"name": "test_function", "arguments": "{}"}}
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        with (
            patch.object(executor, "a_generate_tool_calls_reply", new_callable=AsyncMock, return_value=(True, None)),
            pytest.raises(ValueError, match="Tool call did not return a message"),
        ):
            await executor._a_generate_group_tool_reply(agent=executor, messages=messages)

    @pytest.mark.asyncio
    async def test_a_generate_group_tool_reply_structured_output(self, executor: GroupToolExecutor) -> None:
        """Test async handler returns structured output directly."""
        expected_args = {"key": "value", "nested": {"a": 1}}
        tool_call = {
            "id": "call1",
            "function": {"name": "__structured_output", "arguments": expected_args},
        }
        message = {"role": "user", "content": "Execute tool", "tool_calls": [tool_call]}
        messages = [message]

        success, result = await executor._a_generate_group_tool_reply(agent=executor, messages=messages)

        assert success is True
        assert result == expected_args
