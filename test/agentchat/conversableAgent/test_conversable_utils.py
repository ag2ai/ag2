# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat import ConversableAgent
from autogen.exception_utils import InvalidCarryOverTypeError
from autogen.tools import Tool
from test.credentials import Credentials

# ============================================================================
# _normalize_name Tests
# ============================================================================


def test__normalize_name():
    """Test _normalize_name replaces invalid chars and truncates."""
    assert ConversableAgent._normalize_name("valid-name_123") == "valid-name_123"
    assert ConversableAgent._normalize_name("invalid name!@#") == "invalid_name___"
    long_name = "a" * 70
    assert len(ConversableAgent._normalize_name(long_name)) == 64


def test__normalize_name_special_chars():
    """Test _normalize_name handles various special characters."""
    assert ConversableAgent._normalize_name("test.name") == "test_name"
    assert ConversableAgent._normalize_name("test$name") == "test_name"
    assert ConversableAgent._normalize_name("test name") == "test_name"


# ============================================================================
# _assert_valid_name Tests
# ============================================================================


def test__assert_valid_name_valid():
    """Test _assert_valid_name with valid names."""
    assert ConversableAgent._assert_valid_name("valid_name") == "valid_name"
    assert ConversableAgent._assert_valid_name("Valid-Name_123") == "Valid-Name_123"


def test__assert_valid_name_invalid_characters():
    """Test _assert_valid_name with invalid characters raises ValueError."""
    with pytest.raises(ValueError, match="Only letters, numbers, '_' and '-' are allowed"):
        ConversableAgent._assert_valid_name("invalid name")


def test__assert_valid_name_too_long():
    """Test _assert_valid_name with name > 64 chars raises ValueError."""
    long_name = "a" * 65
    with pytest.raises(ValueError, match="Name must be less than 64 characters"):
        ConversableAgent._assert_valid_name(long_name)


def test__assert_valid_name_edge_cases():
    """Test _assert_valid_name edge cases."""
    # Exactly 64 characters should be valid
    valid_long = "a" * 64
    assert ConversableAgent._assert_valid_name(valid_long) == valid_long


# ============================================================================
# _is_silent Tests
# ============================================================================


def test__is_silent_agent_silent():
    """Test _is_silent when agent.silent is set."""
    agent = ConversableAgent(name="agent", llm_config=False)
    agent.silent = True
    assert ConversableAgent._is_silent(agent) is True

    agent.silent = False
    assert ConversableAgent._is_silent(agent) is False


def test__is_silent_default():
    """Test _is_silent uses default when agent.silent is None."""
    agent = ConversableAgent(name="agent", llm_config=False)
    agent.silent = None
    assert ConversableAgent._is_silent(agent, silent=True) is True
    assert ConversableAgent._is_silent(agent, silent=False) is False


# ============================================================================
# _should_terminate_chat Tests
# ============================================================================


def test__should_terminate_chat_not_conversable_agent():
    """Test _should_terminate_chat returns False when recipient is not ConversableAgentBase."""
    agent = ConversableAgent(name="agent", llm_config=False)
    mock_recipient = MagicMock()

    result = agent._should_terminate_chat(mock_recipient, {"content": "TERMINATE"})
    assert result is False


def test__should_terminate_chat_no_content():
    """Test _should_terminate_chat returns False when content is None."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(
        name="recipient", llm_config=False, is_termination_msg=lambda x: x.get("content") == "TERMINATE"
    )

    result = agent._should_terminate_chat(recipient, {"content": None})
    assert result is False


# ============================================================================
# _check_chat_queue_for_sender Tests
# ============================================================================


def test__check_chat_queue_for_sender_adds_sender():
    """Test _check_chat_queue_for_sender adds sender when missing."""
    agent = ConversableAgent(name="agent", llm_config=False)
    chat_queue = [{"message": "test"}, {"message": "test2"}]

    result = agent._check_chat_queue_for_sender(chat_queue)

    assert result[0]["sender"] == agent
    assert result[1]["sender"] == agent


def test__check_chat_queue_for_sender_preserves_existing():
    """Test _check_chat_queue_for_sender preserves existing sender."""
    agent = ConversableAgent(name="agent", llm_config=False)
    other_agent = ConversableAgent(name="other", llm_config=False)
    chat_queue = [{"sender": other_agent, "message": "test"}]

    result = agent._check_chat_queue_for_sender(chat_queue)

    assert result[0]["sender"] == other_agent


def test__check_chat_queue_for_sender_empty_queue():
    """Test _check_chat_queue_for_sender handles empty queue."""
    agent = ConversableAgent(name="agent", llm_config=False)
    result = agent._check_chat_queue_for_sender([])
    assert result == []


# ============================================================================
# _run_async_in_thread Tests
# ============================================================================


def test__run_async_in_thread_basic():
    """Test _run_async_in_thread runs coroutine successfully."""
    agent = ConversableAgent(name="agent", llm_config=False)

    async def test_coro():
        await asyncio.sleep(0.01)
        return "success"

    result = agent._run_async_in_thread(test_coro())
    assert result == "success"


def test__run_async_in_thread_exception():
    """Test _run_async_in_thread handles exceptions."""
    agent = ConversableAgent(name="agent", llm_config=False)

    async def failing_coro():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        agent._run_async_in_thread(failing_coro())


# ============================================================================
# _str_for_tool_response Tests
# ============================================================================


def test__str_for_tool_response_with_content():
    """Test _str_for_tool_response returns content string."""
    agent = ConversableAgent(name="agent", llm_config=False)
    tool_response = {"content": "test result"}

    result = agent._str_for_tool_response(tool_response)
    assert result == "test result"


def test__str_for_tool_response_no_content():
    """Test _str_for_tool_response returns empty string when no content."""
    agent = ConversableAgent(name="agent", llm_config=False)
    tool_response = {}

    result = agent._str_for_tool_response(tool_response)
    assert result == ""


# ============================================================================
# _process_carryover Tests
# ============================================================================


def test_process_carryover_string():
    """Test _process_carryover with string carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)
    content = "Hello"
    kwargs = {"carryover": "Context info"}

    result = agent._process_carryover(content, kwargs)
    assert "Context: \n" in result
    assert "Context info" in result


def test_process_carryover_list():
    """Test _process_carryover with list carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)
    content = "Hello"
    kwargs = {"carryover": ["Context 1", "Context 2"]}

    result = agent._process_carryover(content, kwargs)
    assert "Context: \n" in result


def test_process_carryover_invalid_type():
    """Test _process_carryover with invalid type raises error."""
    agent = ConversableAgent(name="agent", llm_config=False)
    content = "Hello"
    kwargs = {"carryover": 123}

    with pytest.raises(InvalidCarryOverTypeError):
        agent._process_carryover(content, kwargs)


def test_process_carryover_none():
    """Test _process_carryover with no carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)
    content = "Hello"
    kwargs = {}

    result = agent._process_carryover(content, kwargs)
    assert result == content


# ============================================================================
# _process_multimodal_carryover Tests
# ============================================================================


def test__process_multimodal_carryover_with_carryover():
    """Test _process_multimodal_carryover prepends text with carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)
    content = [{"type": "text", "text": "Hello"}]
    kwargs = {"carryover": "Context"}

    result = agent._process_multimodal_carryover(content, kwargs)

    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert "Context" in result[0]["text"]


def test__process_multimodal_carryover_no_carryover():
    """Test _process_multimodal_carryover returns content unchanged."""
    agent = ConversableAgent(name="agent", llm_config=False)
    content = [{"type": "text", "text": "Hello"}]
    kwargs = {}

    result = agent._process_multimodal_carryover(content, kwargs)
    assert result == content


# ============================================================================
# generate_init_message Tests
# ============================================================================


def test_generate_init_message_with_message():
    """Test generate_init_message processes provided message."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = agent.generate_init_message("Hello")
    assert result == "Hello"


def test_generate_init_message_none_calls_input():
    """Test generate_init_message calls get_human_input when None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    with patch.object(agent, "get_human_input", return_value="user input") as mock_input:
        result = agent.generate_init_message(None)

    assert result == "user input"
    mock_input.assert_called_once_with(">")


def test_generate_init_message_with_carryover():
    """Test generate_init_message handles carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = agent.generate_init_message("Hello", carryover="Context")
    assert "Context" in result


def test_generate_init_message_dict_message():
    """Test generate_init_message handles dict message."""
    agent = ConversableAgent(name="agent", llm_config=False)
    message = {"content": "Hello", "role": "user"}

    result = agent.generate_init_message(message, carryover="Context")
    assert isinstance(result, dict)
    assert "Context" in result["content"]


# ============================================================================
# a_generate_init_message Tests
# ============================================================================


@pytest.mark.asyncio
async def test_a_generate_init_message_with_message():
    """Test a_generate_init_message processes provided message."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = await agent.a_generate_init_message("Hello")
    assert result == "Hello"


@pytest.mark.asyncio
async def test_a_generate_init_message_none_calls_async_input():
    """Test a_generate_init_message calls a_get_human_input when None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    with patch.object(agent, "a_get_human_input", return_value="user input") as mock_input:
        result = await agent.a_generate_init_message(None)

    assert result == "user input"
    mock_input.assert_called_once_with(">")


@pytest.mark.asyncio
async def test_a_generate_init_message_with_carryover():
    """Test a_generate_init_message handles carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)

    result = await agent.a_generate_init_message("Hello", carryover="Context")
    assert "Context" in result


# ============================================================================
# _handle_carryover Tests
# ============================================================================


def test_handle_carryover_dict_str_content():
    """Test _handle_carryover with dict containing str content."""
    agent = ConversableAgent(name="agent", llm_config=False)
    message = {"content": "Hello", "role": "user"}

    result = agent._handle_carryover(message, {"carryover": "Context"})

    assert isinstance(result, dict)
    assert "Context" in result["content"]
    # Original should not be mutated
    assert message["content"] == "Hello"


def test_handle_carryover_dict_list_content():
    """Test _handle_carryover with dict containing list content."""
    agent = ConversableAgent(name="agent", llm_config=False)
    message = {"content": [{"type": "text", "text": "Hello"}], "role": "user"}

    result = agent._handle_carryover(message, {"carryover": "Context"})

    assert isinstance(result, dict)
    assert len(result["content"]) == 2


def test_handle_carryover_no_carryover():
    """Test _handle_carryover returns message when no carryover."""
    agent = ConversableAgent(name="agent", llm_config=False)
    message = "Hello"

    result = agent._handle_carryover(message, {})
    assert result == message


def test_handle_carryover_invalid_message_type():
    """Test _handle_carryover raises error for invalid message type."""
    agent = ConversableAgent(name="agent", llm_config=False)

    with pytest.raises(InvalidCarryOverTypeError):
        agent._handle_carryover(123, {"carryover": "Context"})


# ============================================================================
# _create_tool_if_needed Tests
# ============================================================================


def test__create_tool_if_needed_with_tool():
    """Test _create_tool_if_needed returns Tool when Tool passed."""

    def test_func():
        pass

    tool = Tool(name="test", description="test", func_or_tool=test_func)
    result = ConversableAgent._create_tool_if_needed(tool, None, None)

    assert isinstance(result, Tool)
    assert result == tool


def test__create_tool_if_needed_with_tool_name_override():
    """Test _create_tool_if_needed creates new Tool with name override."""

    def test_func():
        pass

    tool = Tool(name="old_name", description="test", func_or_tool=test_func)
    result = ConversableAgent._create_tool_if_needed(tool, "new_name", None)

    assert isinstance(result, Tool)
    assert result.name == "new_name"


def test__create_tool_if_needed_with_function():
    """Test _create_tool_if_needed creates Tool from function."""

    def test_func(x: str) -> str:
        return x

    result = ConversableAgent._create_tool_if_needed(test_func, "func_name", "description")

    assert isinstance(result, Tool)
    assert result.name == "func_name"


def test__create_tool_if_needed_invalid_type():
    """Test _create_tool_if_needed raises TypeError for invalid type."""
    with pytest.raises(TypeError, match="must be a function or a Tool object"):
        ConversableAgent._create_tool_if_needed("not a tool", None, None)


# ============================================================================
# _create_or_get_executor Tests
# ============================================================================


def test__create_or_get_executor_creates_new(mock_credentials: Credentials):
    """Test _create_or_get_executor creates executor when none exists."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    assert not hasattr(agent, "run_executor") or agent.run_executor is None

    with agent._create_or_get_executor() as executor:
        assert executor is not None
        assert isinstance(executor, ConversableAgent)
        assert executor.name == "executor"


def test__create_or_get_executor_reuses_existing(mock_credentials: Credentials):
    """Test _create_or_get_executor reuses existing executor."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with agent._create_or_get_executor() as executor1:
        executor_id = id(executor1)

    with agent._create_or_get_executor() as executor2:
        assert id(executor2) == executor_id


def test__create_or_get_executor_registers_tools(mock_credentials: Credentials):
    """Test _create_or_get_executor registers passed tools."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    tool = Tool(name="test_tool", description="test", func_or_tool=test_func)

    with agent._create_or_get_executor(tools=[tool]) as executor:
        assert "test_tool" in executor.function_map
        assert "test_tool" in [t["function"]["name"] for t in agent.llm_config.get("tools", [])]


def test__create_or_get_executor_combines_agent_tools(mock_credentials: Credentials):
    """Test _create_or_get_executor combines agent and passed tools."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def agent_func(x: str) -> str:
        return x

    def passed_func(x: str) -> str:
        return x

    agent_tool = Tool(name="agent_tool", description="test", func_or_tool=agent_func)
    passed_tool = Tool(name="passed_tool", description="test", func_or_tool=passed_func)

    agent.register_for_llm()(agent_tool)

    with agent._create_or_get_executor(tools=[passed_tool]) as executor:
        assert "agent_tool" in executor.function_map
        assert "passed_tool" in executor.function_map


def test__create_or_get_executor_cleanup(mock_credentials: Credentials):
    """Test _create_or_get_executor cleans up passed tools on exit."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    tool = Tool(name="test_tool", description="test", func_or_tool=test_func)

    with agent._create_or_get_executor(tools=[tool]):
        pass

    # Tool should be removed from LLM config after context exit
    tool_names = [t["function"]["name"] for t in agent.llm_config.get("tools", [])]
    assert "test_tool" not in tool_names


def test__create_or_get_executor_single_tool(mock_credentials: Credentials):
    """Test _create_or_get_executor handles single Tool (not iterable)."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def test_func(x: str) -> str:
        return x

    tool = Tool(name="test_tool", description="test", func_or_tool=test_func)

    with agent._create_or_get_executor(tools=tool) as executor:
        assert "test_tool" in executor.function_map


def test__create_or_get_executor_default_termination(mock_credentials: Credentials):
    """Test _create_or_get_executor sets default termination message."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with agent._create_or_get_executor() as executor:
        assert executor.is_termination_msg({"content": "TERMINATE"}) is True
        assert executor.is_termination_msg({"content": "Hello"}) is False


# ============================================================================
# _deprecated_run Tests
# ============================================================================


def test__deprecated_run_msg_to_agent(mock_credentials: Credentials):
    """Test _deprecated_run with msg_to='agent'."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config, human_input_mode="NEVER")

    with patch.object(agent, "_create_or_get_executor") as mock_executor:
        mock_exec = MagicMock()
        mock_exec.initiate_chat.return_value = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_exec

        agent._deprecated_run("Hello", msg_to="agent", max_turns=1)

        mock_exec.initiate_chat.assert_called_once()
        assert mock_exec.initiate_chat.call_args[0][0] == agent


def test__deprecated_run_msg_to_user(mock_credentials: Credentials):
    """Test _deprecated_run with msg_to='user'."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config, human_input_mode="NEVER")

    with patch.object(agent, "initiate_chat") as mock_initiate:
        mock_initiate.return_value = MagicMock()
        with patch.object(agent, "_create_or_get_executor") as mock_executor:
            mock_exec = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_exec

            agent._deprecated_run("Hello", msg_to="user", max_turns=1)

            mock_initiate.assert_called_once()


def test__deprecated_run_user_input_true(mock_credentials: Credentials):
    """Test _deprecated_run sets human_input_mode ALWAYS when user_input=True."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with patch.object(agent, "_create_or_get_executor") as mock_executor:
        agent._deprecated_run("Hello", user_input=True, max_turns=1)

        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["agent_human_input_mode"] == "ALWAYS"


# ============================================================================
# _deprecated_a_run Tests
# ============================================================================


@pytest.mark.asyncio
async def test__deprecated_a_run_msg_to_agent(mock_credentials: Credentials):
    """Test _deprecated_a_run with msg_to='agent'."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config, human_input_mode="NEVER")

    with patch.object(agent, "_create_or_get_executor") as mock_executor:
        mock_exec = MagicMock()
        mock_exec.a_initiate_chat = MagicMock(return_value=MagicMock())
        mock_executor.return_value.__aenter__ = MagicMock(return_value=mock_exec)

        await agent._deprecated_a_run("Hello", msg_to="agent", max_turns=1)

        mock_exec.a_initiate_chat.assert_called_once()


@pytest.mark.asyncio
async def test__deprecated_a_run_user_input_false(mock_credentials: Credentials):
    """Test _deprecated_a_run sets human_input_mode NEVER when user_input=False."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with patch.object(agent, "_create_or_get_executor") as mock_executor:
        await agent._deprecated_a_run("Hello", user_input=False, max_turns=1)

        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["agent_human_input_mode"] == "NEVER"
