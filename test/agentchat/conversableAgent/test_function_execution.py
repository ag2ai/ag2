# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat import ConversableAgent
from autogen.events.agent_events import ExecuteFunctionEvent, ExecutedFunctionEvent
from autogen.tools import Tool
from test.credentials import Credentials


# ============================================================================
# _format_json_str Tests
# ============================================================================


def test__format_json_str_escaped_quotes():
    """Test _format_json_str handles escaped quotes correctly."""
    agent = ConversableAgent(name="agent", llm_config=False)
    jstr = '{"code": "a=\\"hello\\""}'
    result = agent._format_json_str(jstr)
    assert result == '{"code": "a=\\"hello\\""}'


def test__format_json_str_complex_nested():
    """Test _format_json_str handles complex nested JSON."""
    agent = ConversableAgent(name="agent", llm_config=False)
    jstr = '{\n"outer": {\n"inner": "value"\n}\n}'
    result = agent._format_json_str(jstr)
    assert "\n" not in result or result.count("\n") < jstr.count("\n")


def test__format_json_str_edge_cases():
    """Test _format_json_str edge cases."""
    agent = ConversableAgent(name="agent", llm_config=False)
    assert agent._format_json_str("") == ""
    assert agent._format_json_str('{"key": "value"}') == '{"key": "value"}'


# ============================================================================
# execute_function Tests
# ============================================================================


def test_execute_function_emits_events():
    """Test execute_function emits ExecuteFunctionEvent and ExecutedFunctionEvent."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func(x: str) -> str:
        return x

    agent.register_function({"test_func": test_func})
    func_call = {"name": "test_func", "arguments": json.dumps({"x": "test"})}

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.function_execution.IOStream.get_default", return_value=mock_iostream):
        success, result = agent.execute_function(func_call)

    assert success is True
    assert len(mock_iostream.send.call_args_list) == 2
    assert isinstance(mock_iostream.send.call_args_list[0][0][0], ExecuteFunctionEvent)
    assert isinstance(mock_iostream.send.call_args_list[1][0][0], ExecutedFunctionEvent)


def test_execute_function_awaitable_result():
    """Test execute_function handles awaitable return values."""
    agent = ConversableAgent(name="agent", llm_config=False)

    async def async_func(x: str) -> str:
        await asyncio.sleep(0)
        return x

    agent.register_function({"async_func": async_func})
    func_call = {"name": "async_func", "arguments": json.dumps({"x": "test"})}

    success, result = agent.execute_function(func_call)
    assert success is True
    assert result["content"] == "test"


def test_execute_function_exception_handling():
    """Test execute_function handles exceptions during execution."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def failing_func():
        raise ValueError("test error")

    agent.register_function({"failing_func": failing_func})
    func_call = {"name": "failing_func", "arguments": "{}"}

    success, result = agent.execute_function(func_call)
    assert success is False
    assert "Error: test error" in result["content"]


def test_execute_function_no_call_id():
    """Test execute_function handles None call_id."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> str:
        return "success"

    agent.register_function({"test_func": test_func})
    func_call = {"name": "test_func", "arguments": "{}"}

    success, result = agent.execute_function(func_call, call_id=None)
    assert success is True


def test_execute_function_empty_arguments():
    """Test execute_function handles empty arguments."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> str:
        return "success"

    agent.register_function({"test_func": test_func})
    func_call = {"name": "test_func", "arguments": "{}"}

    success, result = agent.execute_function(func_call)
    assert success is True
    assert result["content"] == "success"


# ============================================================================
# a_execute_function Tests
# ============================================================================


@pytest.mark.asyncio
async def test_a_execute_function_async_func():
    """Test a_execute_function executes async function."""
    agent = ConversableAgent(name="agent", llm_config=False)

    async def async_func(x: str) -> str:
        await asyncio.sleep(0)
        return x

    agent.register_function({"async_func": async_func})
    func_call = {"name": "async_func", "arguments": json.dumps({"x": "test"})}

    success, result = await agent.a_execute_function(func_call)
    assert success is True
    assert result["content"] == "test"


@pytest.mark.asyncio
async def test_a_execute_function_sync_func():
    """Test a_execute_function falls back to sync function."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def sync_func(x: str) -> str:
        return x

    agent.register_function({"sync_func": sync_func})
    func_call = {"name": "sync_func", "arguments": json.dumps({"x": "test"})}

    success, result = await agent.a_execute_function(func_call)
    assert success is True
    assert result["content"] == "test"


@pytest.mark.asyncio
async def test_a_execute_function_emits_events():
    """Test a_execute_function emits events."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> str:
        return "success"

    agent.register_function({"test_func": test_func})
    func_call = {"name": "test_func", "arguments": "{}"}

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.function_execution.IOStream.get_default", return_value=mock_iostream):
        await agent.a_execute_function(func_call)

    assert mock_iostream.send.call_count == 2


# ============================================================================
# generate_function_call_reply Tests
# ============================================================================


def test_generate_function_call_reply_with_function_call():
    """Test generate_function_call_reply handles function_call in message."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func(x: str) -> str:
        return x

    agent.register_function({"test_func": test_func})
    messages = [{"function_call": {"name": "test_func", "arguments": json.dumps({"x": "test"})}}]

    success, result = agent.generate_function_call_reply(messages)
    assert success is True
    assert result["name"] == "test_func"


def test_generate_function_call_reply_async_function():
    """Test generate_function_call_reply handles async function via _run_async_in_thread."""
    agent = ConversableAgent(name="agent", llm_config=False)

    async def async_func(x: str) -> str:
        await asyncio.sleep(0)
        return x

    agent.register_function({"async_func": async_func})
    messages = [{"function_call": {"name": "async_func", "arguments": json.dumps({"x": "test"})}}]

    success, result = agent.generate_function_call_reply(messages)
    assert success is True
    assert result["content"] == "test"


def test_generate_function_call_reply_no_function_call():
    """Test generate_function_call_reply returns False when no function_call."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [{"content": "hello", "role": "user"}]

    success, result = agent.generate_function_call_reply(messages)
    assert success is False
    assert result is None


def test_generate_function_call_reply_with_call_id():
    """Test generate_function_call_reply handles call_id correctly."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> str:
        return "success"

    agent.register_function({"test_func": test_func})
    messages = [{"id": "call-123", "function_call": {"name": "test_func", "arguments": "{}"}}]

    success, result = agent.generate_function_call_reply(messages)
    assert success is True


def test_generate_function_call_reply_messages_from_sender():
    """Test generate_function_call_reply retrieves messages from sender."""
    agent = ConversableAgent(name="agent", llm_config=False)
    sender = ConversableAgent(name="sender", llm_config=False)

    def test_func() -> str:
        return "success"

    agent.register_function({"test_func": test_func})
    agent._oai_messages[sender] = [{"function_call": {"name": "test_func", "arguments": "{}"}}]

    success, result = agent.generate_function_call_reply(sender=sender, messages=None)
    assert success is True


# ============================================================================
# a_generate_function_call_reply Tests
# ============================================================================


@pytest.mark.asyncio
async def test_a_generate_function_call_reply_with_function_call():
    """Test a_generate_function_call_reply handles function_call."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func(x: str) -> str:
        return x

    agent.register_function({"test_func": test_func})
    messages = [{"function_call": {"name": "test_func", "arguments": json.dumps({"x": "test"})}}]

    success, result = await agent.a_generate_function_call_reply(messages)
    assert success is True
    assert result["name"] == "test_func"


@pytest.mark.asyncio
async def test_a_generate_function_call_reply_async_function():
    """Test a_generate_function_call_reply awaits async function."""
    agent = ConversableAgent(name="agent", llm_config=False)

    async def async_func(x: str) -> str:
        await asyncio.sleep(0)
        return x

    agent.register_function({"async_func": async_func})
    messages = [{"function_call": {"name": "async_func", "arguments": json.dumps({"x": "test"})}}]

    success, result = await agent.a_generate_function_call_reply(messages)
    assert success is True
    assert result["content"] == "test"


@pytest.mark.asyncio
async def test_a_generate_function_call_reply_no_function_call():
    """Test a_generate_function_call_reply returns False when no function_call."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [{"content": "hello", "role": "user"}]

    success, result = await agent.a_generate_function_call_reply(messages)
    assert success is False
    assert result is None


@pytest.mark.asyncio
async def test_a_generate_function_call_reply_sync_function():
    """Test a_generate_function_call_reply handles sync function."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def sync_func() -> str:
        return "success"

    agent.register_function({"sync_func": sync_func})
    messages = [{"function_call": {"name": "sync_func", "arguments": "{}"}}]

    success, result = await agent.a_generate_function_call_reply(messages)
    assert success is True
    assert result["content"] == "success"


# ============================================================================
# generate_tool_calls_reply Tests
# ============================================================================


def test_generate_tool_calls_reply_structured_output():
    """Test generate_tool_calls_reply handles __structured_output special case."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [
        {
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "__structured_output", "arguments": json.dumps({"key": "value"})},
                }
            ]
        }
    ]

    success, result = agent.generate_tool_calls_reply(messages)
    assert success is True
    assert result == {"key": "value"}


def test_generate_tool_calls_reply_process_tool_input_hook():
    """Test generate_tool_calls_reply processes tool input hook."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func(x: str) -> str:
        return x

    agent.register_function({"test_func": test_func})

    def process_input(func_call):
        func_call["name"] = "test_func"
        func_call["arguments"] = json.dumps({"x": "modified"})
        return func_call

    agent._process_tool_input = process_input
    messages = [
        {
            "tool_calls": [
                {"id": "call-1", "function": {"name": "test_func", "arguments": json.dumps({"x": "original"})}}
            ]
        }
    ]

    success, result = agent.generate_tool_calls_reply(messages)
    assert success is True
    assert "modified" in result["content"]


def test_generate_tool_calls_reply_no_tool_call_id():
    """Test generate_tool_calls_reply handles missing tool_call_id (Mistral compatibility)."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> str:
        return "success"

    agent.register_function({"test_func": test_func})
    messages = [{"tool_calls": [{"function": {"name": "test_func", "arguments": "{}"}}]}]

    success, result = agent.generate_tool_calls_reply(messages)
    assert success is True
    assert "tool_call_id" not in result["tool_responses"][0]


def test_generate_tool_calls_reply_content_none():
    """Test generate_tool_calls_reply handles None content."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> None:
        return None

    agent.register_function({"test_func": test_func})
    messages = [{"tool_calls": [{"id": "call-1", "function": {"name": "test_func", "arguments": "{}"}}]}]

    success, result = agent.generate_tool_calls_reply(messages)
    assert success is True
    assert result["tool_responses"][0]["content"] == ""


def test_generate_tool_calls_reply_hook_returns_none():
    """Test generate_tool_calls_reply raises ValueError when hook returns None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def return_none(func_call):
        return None

    agent._process_tool_input = return_none
    messages = [{"tool_calls": [{"id": "call-1", "function": {"name": "test", "arguments": "{}"}}]}]

    with pytest.raises(ValueError, match="safeguard_tool_inputs hook returned None"):
        agent.generate_tool_calls_reply(messages)


# ============================================================================
# _a_execute_tool_call Tests
# ============================================================================


@pytest.mark.asyncio
async def test__a_execute_tool_call_structured_output():
    """Test _a_execute_tool_call handles __structured_output."""
    agent = ConversableAgent(name="agent", llm_config=False)
    tool_call = {
        "id": "call-1",
        "function": {"name": "__structured_output", "arguments": json.dumps({"key": "value"})},
    }

    result = await agent._a_execute_tool_call(tool_call)
    assert result["tool_call_id"] == "call-1"
    assert result["content"] == {"key": "value"}


@pytest.mark.asyncio
async def test__a_execute_tool_call_normal_execution():
    """Test _a_execute_tool_call normal tool execution."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func(x: str) -> str:
        return x

    agent.register_function({"test_func": test_func})
    tool_call = {"id": "call-1", "function": {"name": "test_func", "arguments": json.dumps({"x": "test"})}}

    result = await agent._a_execute_tool_call(tool_call)
    assert result["tool_call_id"] == "call-1"
    assert result["content"] == "test"


@pytest.mark.asyncio
async def test__a_execute_tool_call_hook_returns_none():
    """Test _a_execute_tool_call raises ValueError when hook returns None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def return_none(func_call):
        return None

    agent._process_tool_input = return_none
    tool_call = {"id": "call-1", "function": {"name": "test", "arguments": "{}"}}

    with pytest.raises(ValueError, match="safeguard_tool_inputs hook returned None"):
        await agent._a_execute_tool_call(tool_call)


# ============================================================================
# a_generate_tool_calls_reply Tests
# ============================================================================


@pytest.mark.asyncio
async def test_a_generate_tool_calls_reply_multiple_parallel():
    """Test a_generate_tool_calls_reply executes multiple tool calls in parallel."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1() -> str:
        return "result1"

    def func2() -> str:
        return "result2"

    agent.register_function({"func1": func1, "func2": func2})
    messages = [
        {
            "tool_calls": [
                {"id": "call-1", "function": {"name": "func1", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "func2", "arguments": "{}"}},
            ]
        }
    ]

    success, result = await agent.a_generate_tool_calls_reply(messages)
    assert success is True
    assert len(result["tool_responses"]) == 2


@pytest.mark.asyncio
async def test_a_generate_tool_calls_reply_no_tool_calls():
    """Test a_generate_tool_calls_reply returns False when no tool_calls."""
    agent = ConversableAgent(name="agent", llm_config=False)
    messages = [{"content": "hello", "role": "user"}]

    success, result = await agent.a_generate_tool_calls_reply(messages)
    assert success is False
    assert result is None


# ============================================================================
# register_function Tests
# ============================================================================


def test_register_function_warning_override():
    """Test register_function warns on override."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1():
        pass

    def func2():
        pass

    agent.register_function({"test_func": func1})
    with pytest.warns(UserWarning, match="Function 'test_func' is being overridden"):
        agent.register_function({"test_func": func2})


def test_register_function_silent_override():
    """Test register_function silent_override suppresses warning."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1():
        pass

    def func2():
        pass

    agent.register_function({"test_func": func1})
    with pytest.warns(UserWarning):
        # Should not warn
        agent.register_function({"test_func": func2}, silent_override=True)


def test_register_function_removes_none():
    """Test register_function removes functions when value is None."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1():
        pass

    agent.register_function({"func1": func1, "func2": func1})
    agent.register_function({"func2": None})

    assert "func1" in agent._function_map
    assert "func2" not in agent._function_map


def test_register_function_invalid_name():
    """Test register_function raises ValueError for invalid name."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func():
        pass

    with pytest.raises(ValueError, match="Invalid name"):
        agent.register_function({"invalid name": func})


# ============================================================================
# update_function_signature Tests
# ============================================================================


def test_update_function_signature_no_llm_config():
    """Test update_function_signature raises AssertionError when no llm_config."""
    agent = ConversableAgent(name="agent", llm_config=False)

    with pytest.raises(AssertionError, match="must have an llm_config"):
        agent.update_function_signature({"name": "test"})


def test_update_function_signature_remove_not_found(mock_credentials: Credentials):
    """Test update_function_signature raises AssertionError when removing non-existent."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with pytest.raises(AssertionError, match="doesn't have function"):
        agent.update_function_signature("nonexistent", is_remove=True)


def test_update_function_signature_not_dict(mock_credentials: Credentials):
    """Test update_function_signature raises ValueError when func_sig is not dict."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with pytest.raises(ValueError, match="must be of the type dict"):
        agent.update_function_signature("not a dict")


def test_update_function_signature_no_name_key(mock_credentials: Credentials):
    """Test update_function_signature raises ValueError when no 'name' key."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with pytest.raises(ValueError, match="must have a 'name' key"):
        agent.update_function_signature({"description": "test"})


def test_update_function_signature_warning_override(mock_credentials: Credentials):
    """Test update_function_signature warns on override."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    func_sig = {"name": "test_func", "description": "test"}
    agent.update_function_signature(func_sig)

    with pytest.warns(UserWarning, match="Function 'test_func' is being overridden"):
        agent.update_function_signature(func_sig)


def test_update_function_signature_removes_empty(mock_credentials: Credentials):
    """Test update_function_signature removes 'functions' key when empty."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    func_sig = {"name": "test_func", "description": "test"}
    agent.update_function_signature(func_sig)
    agent.update_function_signature("test_func", is_remove=True)

    assert "functions" not in agent.llm_config


# ============================================================================
# update_tool_signature Tests
# ============================================================================


def test_update_tool_signature_add_tool(mock_credentials: Credentials):
    """Test update_tool_signature adds tool signature."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    tool_sig = {
        "type": "function",
        "function": {"name": "test_tool", "description": "test", "parameters": {"type": "object", "properties": {}}},
    }

    agent.update_tool_signature(tool_sig, is_remove=False)
    assert len(agent.llm_config["tools"]) == 1
    assert agent.llm_config["tools"][0]["function"]["name"] == "test_tool"


def test_update_tool_signature_no_llm_config():
    """Test update_tool_signature raises AssertionError when no llm_config."""
    agent = ConversableAgent(name="agent", llm_config=False)

    with pytest.raises(AssertionError, match="must have an llm_config"):
        agent.update_tool_signature({"type": "function", "function": {"name": "test"}}, is_remove=False)


def test_update_tool_signature_remove_by_name(mock_credentials: Credentials):
    """Test update_tool_signature removes tool by name string."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    tool_sig = {
        "type": "function",
        "function": {"name": "test_tool", "description": "test", "parameters": {"type": "object", "properties": {}}},
    }

    agent.update_tool_signature(tool_sig, is_remove=False)
    agent.update_tool_signature("test_tool", is_remove=True)

    assert len(agent.llm_config.get("tools", [])) == 0


def test_update_tool_signature_remove_not_found(mock_credentials: Credentials):
    """Test update_tool_signature raises AssertionError when removing non-existent."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with pytest.raises(AssertionError, match="doesn't have tool"):
        agent.update_tool_signature("nonexistent", is_remove=True)


def test_update_tool_signature_not_dict(mock_credentials: Credentials):
    """Test update_tool_signature raises ValueError when tool_sig is not dict."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    with pytest.raises(ValueError, match="must be of the type dict"):
        agent.update_tool_signature("not a dict", is_remove=False)


def test_update_tool_signature_warning_override(mock_credentials: Credentials):
    """Test update_tool_signature warns on override."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    tool_sig = {
        "type": "function",
        "function": {"name": "test_tool", "description": "test", "parameters": {"type": "object", "properties": {}}},
    }

    agent.update_tool_signature(tool_sig, is_remove=False)

    with pytest.warns(UserWarning, match="Function 'test_tool' is being overridden"):
        agent.update_tool_signature(tool_sig, is_remove=False)


def test_update_tool_signature_removes_empty(mock_credentials: Credentials):
    """Test update_tool_signature removes 'tools' key when empty."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    tool_sig = {
        "type": "function",
        "function": {"name": "test_tool", "description": "test", "parameters": {"type": "object", "properties": {}}},
    }

    agent.update_tool_signature(tool_sig, is_remove=False)
    agent.update_tool_signature("test_tool", is_remove=True)

    assert "tools" not in agent.llm_config


# ============================================================================
# can_execute_function Tests
# ============================================================================


def test_can_execute_function_single():
    """Test can_execute_function returns True/False for single function name."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1():
        pass

    agent.register_function({"func1": func1})

    assert agent.can_execute_function("func1") is True
    assert agent.can_execute_function("func2") is False


def test_can_execute_function_list_all_exist():
    """Test can_execute_function returns True when all in list exist."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1():
        pass

    def func2():
        pass

    agent.register_function({"func1": func1, "func2": func2})

    assert agent.can_execute_function(["func1", "func2"]) is True


def test_can_execute_function_list_some_missing():
    """Test can_execute_function returns False when any missing."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def func1():
        pass

    agent.register_function({"func1": func1})

    assert agent.can_execute_function(["func1", "func2"]) is False


# ============================================================================
# _wrap_function Tests
# ============================================================================


def test__wrap_function_inject_params():
    """Test _wrap_function injects chat context parameters."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func(x: str, chat_context) -> str:
        return f"{x}_{chat_context}"

    wrapped = agent._wrap_function(test_func, inject_params={"chat_context": "ctx"})
    result = wrapped("test")
    assert "ctx" in result


def test__wrap_function_serialize_false():
    """Test _wrap_function serialize=False returns original value."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func() -> dict:
        return {"key": "value"}

    wrapped = agent._wrap_function(test_func, serialize=False)
    result = wrapped()
    assert isinstance(result, dict)
    assert result == {"key": "value"}


def test__wrap_function_preserves_origin():
    """Test _wrap_function preserves _origin attribute."""
    agent = ConversableAgent(name="agent", llm_config=False)

    def test_func():
        pass

    wrapped = agent._wrap_function(test_func)
    assert hasattr(wrapped, "_origin")
    assert wrapped._origin == test_func


# ============================================================================
# register_for_llm Tests
# ============================================================================


def test_register_for_llm_decorator(mock_credentials: Credentials):
    """Test register_for_llm decorator registers function."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    @agent.register_for_llm()
    def test_func(x: str) -> str:
        return x

    assert "test_func" in [t["function"]["name"] for t in agent.llm_config.get("tools", [])]


def test_register_for_llm_with_name(mock_credentials: Credentials):
    """Test register_for_llm with custom name parameter."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    @agent.register_for_llm(name="custom_name")
    def test_func(x: str) -> str:
        return x

    assert "custom_name" in [t["function"]["name"] for t in agent.llm_config.get("tools", [])]


def test_register_for_llm_api_style_function(mock_credentials: Credentials):
    """Test register_for_llm with api_style='function'."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    @agent.register_for_llm(api_style="function")
    def test_func(x: str) -> str:
        return x

    assert "test_func" in [f["name"] for f in agent.llm_config.get("functions", [])]


def test_register_for_llm_adds_to_tools(mock_credentials: Credentials):
    """Test register_for_llm adds tool to _tools list."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    @agent.register_for_llm()
    def test_func(x: str) -> str:
        return x

    assert len(agent._tools) == 1
    assert agent._tools[0].name == "test_func"


# ============================================================================
# _register_for_llm Tests
# ============================================================================


def test__register_for_llm_no_llm_config():
    """Test _register_for_llm raises RuntimeError when llm_config is None."""
    agent = ConversableAgent(name="agent", llm_config=None)
    tool = Tool(name="test", description="test", func_or_tool=lambda x: x)

    with pytest.raises(RuntimeError, match="LLM config must be setup"):
        agent._register_for_llm(tool, "tool")


def test__register_for_llm_api_style_function(mock_credentials: Credentials):
    """Test _register_for_llm calls update_function_signature for 'function' style."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    tool = Tool(name="test", description="test", func_or_tool=lambda x: x)

    with patch.object(agent, "update_function_signature") as mock_update:
        agent._register_for_llm(tool, "function")
        mock_update.assert_called_once()


def test__register_for_llm_api_style_tool(mock_credentials: Credentials):
    """Test _register_for_llm calls update_tool_signature for 'tool' style."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    tool = Tool(name="test", description="test", func_or_tool=lambda x: x)

    with patch.object(agent, "update_tool_signature") as mock_update:
        agent._register_for_llm(tool, "tool")
        mock_update.assert_called_once()


def test__register_for_llm_invalid_api_style(mock_credentials: Credentials):
    """Test _register_for_llm raises ValueError for invalid api_style."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    tool = Tool(name="test", description="test", func_or_tool=lambda x: x)

    with pytest.raises(ValueError, match="Unsupported API style"):
        agent._register_for_llm(tool, "invalid")


# ============================================================================
# register_for_execution Tests
# ============================================================================


def test_register_for_execution_decorator():
    """Test register_for_execution decorator registers function."""
    agent = ConversableAgent(name="agent", llm_config=False)

    @agent.register_for_execution()
    def test_func(x: str) -> str:
        return x

    assert "test_func" in agent.function_map


def test_register_for_execution_injects_context():
    """Test register_for_execution injects ChatContext parameters."""
    agent = ConversableAgent(name="agent", llm_config=False)

    @agent.register_for_execution()
    def test_func(x: str, chat_context) -> str:
        assert chat_context is not None
        return x

    # Function should be wrapped and ready to receive chat_context
    assert "test_func" in agent.function_map


def test_register_for_execution_serialize_false():
    """Test register_for_execution serialize=False parameter."""
    agent = ConversableAgent(name="agent", llm_config=False)

    @agent.register_for_execution(serialize=False)
    def test_func() -> dict:
        return {"key": "value"}

    result = agent.function_map["test_func"]()
    assert isinstance(result, dict)


def test_register_for_execution_wraps_function():
    """Test register_for_execution wraps function with _wrap_function."""
    agent = ConversableAgent(name="agent", llm_config=False)

    @agent.register_for_execution()
    def test_func() -> str:
        return "success"

    wrapped_func = agent.function_map["test_func"]
    assert hasattr(wrapped_func, "_origin")
    assert wrapped_func._origin == test_func