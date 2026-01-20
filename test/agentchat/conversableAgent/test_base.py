# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import defaultdict

import pytest

from autogen.agentchat import ConversableAgent
from autogen.agentchat.group import ContextVariables
from autogen.llm_config import LLMConfig
from autogen.oai.client import OpenAILLMConfigEntry
from test.credentials import Credentials

# Initialization & Setup Tests


def test__init__basic():
    """Test basic initialization with minimal parameters."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert agent.name == "test_agent"
    assert agent.llm_config is False
    assert isinstance(agent.chat_messages, dict)
    assert agent.context_variables.to_dict() == {}


def test__init__code_execution_config_false():
    """Test initialization with code_execution_config=False."""
    agent = ConversableAgent(name="test_agent", code_execution_config=False, llm_config=False)
    assert agent._code_execution_config is False
    assert agent.use_docker is None


def test__init__code_execution_config_dict():
    """Test initialization with code_execution_config as dict."""
    agent = ConversableAgent(name="test_agent", code_execution_config={"use_docker": False}, llm_config=False)
    assert isinstance(agent._code_execution_config, dict)
    assert "use_docker" in agent._code_execution_config


def test__init__code_execution_config_none_deprecation():
    """Test initialization with code_execution_config=None raises deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent = ConversableAgent(name="test_agent", code_execution_config=None, llm_config=False)
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()
        assert isinstance(agent._code_execution_config, dict)


def test__init__functions_list():
    """Test initialization with functions as list."""

    def func1():
        pass

    def func2():
        pass

    agent = ConversableAgent(name="test_agent", functions=[func1, func2], llm_config=False)
    assert len(agent.function_map) == 2


def test__init__functions_single():
    """Test initialization with single function."""

    def func():
        pass

    agent = ConversableAgent(name="test_agent", functions=func, llm_config=False)
    assert len(agent.function_map) == 1


def test__init__functions_invalid_type():
    """Test initialization with invalid functions type raises TypeError."""
    with pytest.raises(TypeError, match="Functions must be a callable or a list of callables"):
        ConversableAgent(name="test_agent", functions="invalid", llm_config=False)


def test__init__functions_list_with_non_callable():
    """Test initialization with list containing non-callable raises TypeError."""
    with pytest.raises(TypeError, match="All elements in the functions list must be callable"):
        ConversableAgent(name="test_agent", functions=[lambda x: x, "not_callable"], llm_config=False)


def test__init__context_variables():
    """Test initialization with context_variables."""
    ctx = ContextVariables(data={"key": "value"})
    agent = ConversableAgent(name="test_agent", context_variables=ctx, llm_config=False)
    assert agent.context_variables.get("key") == "value"


def test__init__context_variables_none():
    """Test initialization without context_variables creates empty ContextVariables."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert isinstance(agent.context_variables, ContextVariables)
    assert agent.context_variables.to_dict() == {}


def test__init__chat_messages():
    """Test initialization with chat_messages."""
    messages = defaultdict(list)
    messages[ConversableAgent("other", llm_config=False)] = [{"content": "hello", "role": "user"}]

    agent = ConversableAgent(name="test_agent", chat_messages=messages, llm_config=False)
    assert len(agent.chat_messages) == 1


# Code Execution Setup Tests


def test__setup_code_execution_false():
    """Test _setup_code_execution with False."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent._setup_code_execution(False)
    assert agent._code_execution_config is False


def test__setup_code_execution_dict():
    """Test _setup_code_execution with dict."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    config = {"use_docker": False}
    agent._setup_code_execution(config)
    assert isinstance(agent._code_execution_config, dict)


def test__setup_code_execution_none():
    """Test _setup_code_execution with None (deprecated)."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent._setup_code_execution(None)
        assert len(w) == 1
        assert isinstance(agent._code_execution_config, dict)


def test__setup_code_execution_invalid_type():
    """Test _setup_code_execution with invalid type raises ValueError."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    with pytest.raises(ValueError, match="code_execution_config must be a dict or False"):
        agent._setup_code_execution("invalid")


def test__setup_code_execution_executor_conflicts():
    """Test _setup_code_execution with executor conflicts raises ValueError."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    config = {"executor": "some_executor", "use_docker": True}

    with pytest.raises(ValueError, match="'use_docker' in code_execution_config is not valid"):
        agent._setup_code_execution(config)


# Name Validation Tests


def test__validate_name_no_llm_config():
    """Test _validate_name when llm_config is False (should pass)."""
    agent = ConversableAgent(name="agent with spaces", llm_config=False)
    # Should not raise
    agent._validate_name("agent with spaces")


def test__validate_name_openai_with_whitespace(mock_credentials: Credentials):
    """Test _validate_name with OpenAI config and whitespace raises ValueError."""
    llm_config = mock_credentials.llm_config
    # Only test if it's OpenAI
    if any(entry.api_type == "openai" for entry in llm_config.config_list):
        agent = ConversableAgent(name="valid_name", llm_config=llm_config)
        with pytest.raises(ValueError, match="cannot contain any whitespace"):
            agent._validate_name("agent with spaces")


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


def test__normalize_name():
    """Test _normalize_name replaces invalid chars and truncates."""
    assert ConversableAgent._normalize_name("valid-name_123") == "valid-name_123"
    assert ConversableAgent._normalize_name("invalid name!@#") == "invalid_name___"
    long_name = "a" * 70
    assert len(ConversableAgent._normalize_name(long_name)) == 64


# LLM Config Tests


def test__validate_llm_config_none():
    """Test _validate_llm_config with None."""
    result = ConversableAgent._validate_llm_config(None)
    # Should return DEFAULT_CONFIG or current config
    assert result in (False, None) or isinstance(result, LLMConfig)


def test__validate_llm_config_false():
    """Test _validate_llm_config with False."""
    result = ConversableAgent._validate_llm_config(False)
    assert result is False


def test__validate_llm_config_dict():
    """Test _validate_llm_config with dict."""
    config = {"config_list": [{"model": "gpt-3", "api_key": "test"}]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = ConversableAgent._validate_llm_config(config)
        assert isinstance(result, LLMConfig)


def test__validate_llm_config_llmconfig():
    """Test _validate_llm_config with LLMConfig object."""
    config = LLMConfig(OpenAILLMConfigEntry(model="gpt-3"))
    result = ConversableAgent._validate_llm_config(config)
    assert isinstance(result, LLMConfig)


def test__create_client_false():
    """Test _create_client with False returns None."""
    result = ConversableAgent._create_client(False)
    assert result is None


def test__create_client_llmconfig():
    """Test _create_client with LLMConfig returns OpenAIWrapper."""
    config = LLMConfig(OpenAILLMConfigEntry(model="gpt-3"))
    result = ConversableAgent._create_client(config)
    assert result is not None


# Property Tests


def test_name_property():
    """Test name property."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert agent.name == "test_agent"


def test_description_property_getter():
    """Test description property getter."""
    agent = ConversableAgent(name="test_agent", description="Test description", llm_config=False)
    assert agent.description == "Test description"


def test_description_property_setter():
    """Test description property setter."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent.description = "New description"
    assert agent.description == "New description"


def test_description_defaults_to_system_message():
    """Test description defaults to system_message when not provided."""
    agent = ConversableAgent(name="test_agent", system_message="Custom system message", llm_config=False)
    assert agent.description == "Custom system message"


def test_system_message_property():
    """Test system_message property."""
    agent = ConversableAgent(name="test_agent", system_message="Custom message", llm_config=False)
    assert agent.system_message == "Custom message"


def test_code_executor_property_disabled():
    """Test code_executor property when code execution is disabled."""
    agent = ConversableAgent(name="test_agent", code_execution_config=False, llm_config=False)
    assert agent.code_executor is None


def test_chat_messages_property():
    """Test chat_messages property."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert isinstance(agent.chat_messages, dict)
    assert len(agent.chat_messages) == 0


def test_use_docker_property_disabled():
    """Test use_docker property when code execution is disabled."""
    agent = ConversableAgent(name="test_agent", code_execution_config=False, llm_config=False)
    assert agent.use_docker is None


def test_use_docker_property_enabled():
    """Test use_docker property when code execution is enabled."""
    agent = ConversableAgent(name="test_agent", code_execution_config={"use_docker": False}, llm_config=False)
    assert agent.use_docker is False


def test_tools_property():
    """Test tools property returns copy."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    tools = agent.tools
    assert isinstance(tools, list)
    tools.append("test")  # Modify copy
    assert len(agent.tools) == 0  # Original unchanged


def test_function_map_property():
    """Test function_map property."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert isinstance(agent.function_map, dict)
    assert len(agent.function_map) == 0


# Method Tests


def test_update_system_message():
    """Test update_system_message."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent.update_system_message("New system message")
    assert agent.system_message == "New system message"


def test_update_max_consecutive_auto_reply_global():
    """Test update_max_consecutive_auto_reply without sender."""
    agent = ConversableAgent(name="test_agent", max_consecutive_auto_reply=5, llm_config=False)
    agent.update_max_consecutive_auto_reply(10)
    assert agent.max_consecutive_auto_reply() == 10


def test_update_max_consecutive_auto_reply_sender():
    """Test update_max_consecutive_auto_reply with sender."""
    agent = ConversableAgent(name="test_agent", max_consecutive_auto_reply=5, llm_config=False)
    sender = ConversableAgent(name="sender", llm_config=False)
    agent.update_max_consecutive_auto_reply(10, sender=sender)
    assert agent.max_consecutive_auto_reply(sender=sender) == 10
    assert agent.max_consecutive_auto_reply() == 5  # Global unchanged


def test_max_consecutive_auto_reply_global():
    """Test max_consecutive_auto_reply without sender."""
    agent = ConversableAgent(name="test_agent", max_consecutive_auto_reply=5, llm_config=False)
    assert agent.max_consecutive_auto_reply() == 5


def test_max_consecutive_auto_reply_sender():
    """Test max_consecutive_auto_reply with sender."""
    agent = ConversableAgent(name="test_agent", max_consecutive_auto_reply=5, llm_config=False)
    sender = ConversableAgent(name="sender", llm_config=False)
    agent.update_max_consecutive_auto_reply(10, sender=sender)
    assert agent.max_consecutive_auto_reply(sender=sender) == 10


def test_chat_messages_for_summary():
    """Test chat_messages_for_summary."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    other_agent = ConversableAgent(name="other", llm_config=False)

    messages = [{"content": "hello", "role": "user"}]
    agent.chat_messages[other_agent] = messages

    assert agent.chat_messages_for_summary(other_agent) == messages


def test_last_message_none_agent():
    """Test last_message with None agent - single conversation."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    other_agent = ConversableAgent(name="other", llm_config=False)

    messages = [{"content": "hello", "role": "user"}]
    agent.chat_messages[other_agent] = messages

    assert agent.last_message() == messages[-1]


def test_last_message_none_agent_no_conversations():
    """Test last_message with None agent and no conversations returns None."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert agent.last_message() is None


def test_last_message_none_agent_multiple_conversations():
    """Test last_message with None agent and multiple conversations raises ValueError."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    agent1 = ConversableAgent(name="agent1", llm_config=False)
    agent2 = ConversableAgent(name="agent2", llm_config=False)

    agent.chat_messages[agent1] = [{"content": "hello", "role": "user"}]
    agent.chat_messages[agent2] = [{"content": "hi", "role": "user"}]

    with pytest.raises(ValueError, match="More than one conversation is found"):
        agent.last_message()


def test_last_message_specific_agent():
    """Test last_message with specific agent."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    other_agent = ConversableAgent(name="other", llm_config=False)

    messages = [{"content": "hello", "role": "user"}, {"content": "hi", "role": "assistant"}]
    agent.chat_messages[other_agent] = messages

    assert agent.last_message(other_agent) == messages[-1]


def test_last_message_agent_not_found():
    """Test last_message with agent not in conversations raises KeyError."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    other_agent = ConversableAgent(name="other", llm_config=False)

    with pytest.raises(KeyError, match="is not present in any conversation"):
        agent.last_message(other_agent)


def test__get_display_name():
    """Test _get_display_name."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert agent._get_display_name() == "test_agent"


def test__str__():
    """Test __str__ method."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert str(agent) == "test_agent"


def test__is_silent_agent_silent():
    """Test _is_silent when agent has silent attribute."""
    agent = ConversableAgent(name="test_agent", silent=True, llm_config=False)
    assert ConversableAgent._is_silent(agent) is True


def test__is_silent_default():
    """Test _is_silent with default parameter."""
    agent = ConversableAgent(name="test_agent", silent=None, llm_config=False)
    assert ConversableAgent._is_silent(agent, silent=True) is True
    assert ConversableAgent._is_silent(agent, silent=False) is False


def test__content_str_string():
    """Test _content_str with string."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    assert agent._content_str("hello") == "hello"


def test__content_str_dict():
    """Test _content_str with dict."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    content = {"type": "text", "text": "hello"}
    result = agent._content_str(content)
    assert isinstance(result, str)


def test__content_str_list():
    """Test _content_str with list."""
    agent = ConversableAgent(name="test_agent", llm_config=False)
    content = [{"type": "text", "text": "hello"}]
    result = agent._content_str(content)
    assert isinstance(result, str)


# Error Case Tests


def test__init__code_execution_config_executor_conflicts():
    """Test __init__ with executor conflicts raises ValueError."""
    config = {"executor": "some_executor", "use_docker": True}

    with pytest.raises(ValueError, match="'use_docker' in code_execution_config is not valid"):
        ConversableAgent(name="test_agent", code_execution_config=config, llm_config=False)
