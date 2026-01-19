# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.agentchat import ConversableAgent
from autogen.agentchat.chat import ChatResult
from autogen.cache.cache import Cache
from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials


# Chat Initiation Tests


def test_prepare_chat():
    """Test _prepare_chat prepares chat correctly."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Add some messages first
    agent.chat_messages[recipient] = [{"content": "old message", "role": "user"}]
    agent._consecutive_auto_reply_counter[recipient] = 5

    # Clear history
    agent._prepare_chat(recipient, clear_history=True, prepare_recipient=False)
    assert len(agent.chat_messages[recipient]) == 0
    assert agent._consecutive_auto_reply_counter[recipient] == 0
    assert agent.reply_at_receive[recipient] is True


def test_prepare_chat_without_clear_history():
    """Test _prepare_chat without clearing history."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    messages = [{"content": "message", "role": "user"}]
    agent.chat_messages[recipient] = messages.copy()

    agent._prepare_chat(recipient, clear_history=False, prepare_recipient=False)
    assert len(agent.chat_messages[recipient]) == 1
    assert agent.reply_at_receive[recipient] is True


def test_initiate_chat_with_max_turns():
    """Test initiate_chat with max_turns parameter."""
    agent = ConversableAgent(
        name="agent", llm_config=False, human_input_mode="NEVER", default_auto_reply="Hello"
    )
    recipient = ConversableAgent(
        name="recipient", llm_config=False, human_input_mode="NEVER", default_auto_reply="Hi"
    )

    result = agent.initiate_chat(recipient, message="Start", max_turns=2, clear_history=True)
    assert isinstance(result, ChatResult)
    assert len(result.chat_history) <= 4  # max 2 turns * 2 agents


def test_initiate_chat_without_max_turns():
    """Test initiate_chat without max_turns terminates normally."""
    agent = ConversableAgent(
        name="agent",
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="TERMINATE",
        max_consecutive_auto_reply=1,
    )
    recipient = ConversableAgent(
        name="recipient", llm_config=False, human_input_mode="NEVER", default_auto_reply="TERMINATE"
    )

    result = agent.initiate_chat(recipient, message="Start", clear_history=True)
    assert isinstance(result, ChatResult)


def test_initiate_chat_with_callable_message():
    """Test initiate_chat with callable message."""
    agent = ConversableAgent(name="agent", llm_config=False, human_input_mode="NEVER")
    recipient = ConversableAgent(name="recipient", llm_config=False, human_input_mode="NEVER")

    def message_func(sender, recipient, context):
        return "Dynamic message"

    result = agent.initiate_chat(recipient, message=message_func, max_turns=1, clear_history=True)
    assert isinstance(result, ChatResult)


def test_initiate_chat_clear_history():
    """Test initiate_chat with clear_history parameter."""
    agent = ConversableAgent(name="agent", llm_config=False, human_input_mode="NEVER")
    recipient = ConversableAgent(name="recipient", llm_config=False, human_input_mode="NEVER")

    # Add some messages
    agent.chat_messages[recipient] = [{"content": "old", "role": "user"}]

    result = agent.initiate_chat(recipient, message="New", max_turns=1, clear_history=True)
    assert len(result.chat_history) >= 1
    # History should be cleared, so first message should be "New"
    assert result.chat_history[0]["content"] == "New"


@pytest.mark.asyncio
async def test_a_initiate_chat():
    """Test async initiate_chat."""
    agent = ConversableAgent(
        name="agent", llm_config=False, human_input_mode="NEVER", default_auto_reply="Hello"
    )
    recipient = ConversableAgent(
        name="recipient", llm_config=False, human_input_mode="NEVER", default_auto_reply="Hi"
    )

    result = await agent.a_initiate_chat(recipient, message="Start", max_turns=1, clear_history=True)
    assert isinstance(result, ChatResult)


@pytest.mark.asyncio
async def test_a_initiate_chat_with_callable_message():
    """Test async initiate_chat with callable message."""
    agent = ConversableAgent(name="agent", llm_config=False, human_input_mode="NEVER")
    recipient = ConversableAgent(name="recipient", llm_config=False, human_input_mode="NEVER")

    def message_func(sender, recipient, context):
        return "Async dynamic message"

    result = await agent.a_initiate_chat(recipient, message=message_func, max_turns=1, clear_history=True)
    assert isinstance(result, ChatResult)


# Summary Tests


def test_summarize_chat_last_msg():
    """Test _summarize_chat with last_msg method."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    agent.chat_messages[recipient] = [
        {"content": "Hello", "role": "user"},
        {"content": "Hi there", "role": "assistant"},
    ]

    summary = agent._summarize_chat("last_msg", {}, recipient)
    assert isinstance(summary, str)
    assert "Hi there" in summary or summary == "Hi there"


def test_summarize_chat_custom_method():
    """Test _summarize_chat with custom callable method."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    def custom_summary(sender, recipient, summary_args):
        return f"Custom summary: {summary_args.get('key', 'default')}"

    summary = agent._summarize_chat(custom_summary, {"key": "value"}, recipient)
    assert "Custom summary: value" in summary


def test_summarize_chat_dict_summary():
    """Test _summarize_chat with dict return value."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    def dict_summary(sender, recipient, summary_args):
        return {"content": "Summary content"}

    summary = agent._summarize_chat(dict_summary, {}, recipient)
    assert summary == "Summary content"


def test_last_msg_as_summary():
    """Test _last_msg_as_summary static method."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    recipient.chat_messages[agent] = [
        {"content": "Hello", "role": "user"},
        {"content": "HiTERMINATE", "role": "assistant"},
    ]

    summary = ConversableAgent._last_msg_as_summary(agent, recipient, {})
    assert isinstance(summary, str)
    assert "TERMINATE" not in summary


def test_last_msg_as_summary_no_messages():
    """Test _last_msg_as_summary with no messages."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # No messages
    summary = ConversableAgent._last_msg_as_summary(agent, recipient, {})
    assert summary == ""


# Carryover Tests


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
    from autogen.exception_utils import InvalidCarryOverTypeError

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


# Chat Results Tests


def test_get_chat_results():
    """Test get_chat_results returns chat results."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Simulate finished chats
    chat_result = ChatResult(chat_history=[], summary="Test", cost={}, human_input=[])
    agent._finished_chats = [chat_result]

    results = agent.get_chat_results()
    assert len(results) == 1
    assert results[0].summary == "Test"


def test_get_chat_results_with_index():
    """Test get_chat_results with specific index."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    chat_result1 = ChatResult(chat_history=[], summary="First", cost={}, human_input=[])
    chat_result2 = ChatResult(chat_history=[], summary="Second", cost={}, human_input=[])
    agent._finished_chats = [chat_result1, chat_result2]

    result = agent.get_chat_results(0)
    assert isinstance(result, ChatResult)
    assert result.summary == "First"

    result = agent.get_chat_results(1)
    assert result.summary == "Second"


def test_get_chat_results_empty():
    """Test get_chat_results with no finished chats."""
    agent = ConversableAgent(name="agent", llm_config=False)
    results = agent.get_chat_results()
    assert results == []


# Chat Queue Tests


def test_check_chat_queue_for_sender():
    """Test _check_chat_queue_for_sender adds sender if missing."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    chat_queue = [{"recipient": recipient, "message": "Hello"}]
    result = agent._check_chat_queue_for_sender(chat_queue)

    assert len(result) == 1
    assert result[0]["sender"] == agent


def test_check_chat_queue_for_sender_with_sender():
    """Test _check_chat_queue_for_sender doesn't override existing sender."""
    agent = ConversableAgent(name="agent", llm_config=False)
    other_sender = ConversableAgent(name="other", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    chat_queue = [{"sender": other_sender, "recipient": recipient, "message": "Hello"}]
    result = agent._check_chat_queue_for_sender(chat_queue)

    assert result[0]["sender"] == other_sender


def test_get_chats_to_run():
    """Test _get_chats_to_run extracts valid chats."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    messages = [{"content": "Last message", "role": "user"}]
    chat_queue = [
        {"sender": agent, "recipient": recipient, "message": "Chat 1"},
        {"sender": agent, "recipient": recipient, "message": None},  # Should be skipped
        {"sender": agent, "recipient": recipient, "message": "Chat 2"},
    ]

    result = ConversableAgent._get_chats_to_run(chat_queue, recipient, messages, agent, None)
    assert len(result) == 2
    assert result[0]["message"] == "Chat 1"
    assert result[1]["message"] == "Chat 2"


def test_get_chats_to_run_with_callable_message():
    """Test _get_chats_to_run with callable message."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    messages = [{"content": "Last message", "role": "user"}]

    def message_func(recipient, messages, sender, config):
        return "Dynamic message"

    chat_queue = [{"sender": agent, "recipient": recipient, "message": message_func}]
    result = ConversableAgent._get_chats_to_run(chat_queue, recipient, messages, agent, None)

    assert len(result) == 1
    assert result[0]["message"] == "Dynamic message"


def test_get_chats_to_run_defaults_first_message():
    """Test _get_chats_to_run uses last message for first chat if message is None."""
    agent = ConversableAgent(name="agent", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    messages = [{"content": "Last message", "role": "user"}]
    chat_queue = [{"sender": agent, "recipient": recipient, "message": None}]

    result = ConversableAgent._get_chats_to_run(chat_queue, recipient, messages, agent, None)
    assert len(result) == 1
    assert result[0]["message"] == "Last message"


# Cache Tests


@run_for_optional_imports("openai", "openai")
def test_initiate_chat_with_cache(credentials_gpt_4o_mini: Credentials):
    """Test initiate_chat uses cache when provided."""
    agent = ConversableAgent(
        name="agent",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        default_auto_reply="Hello",
    )
    recipient = ConversableAgent(
        name="recipient",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        default_auto_reply="Hi",
    )

    with Cache.disk(cache_seed=42, cache_path_root=".cache"):
        result = agent.initiate_chat(recipient, message="Test", max_turns=1, clear_history=True)
        assert isinstance(result, ChatResult)


@pytest.mark.asyncio
@run_for_optional_imports("openai", "openai")
async def test_a_initiate_chat_with_cache(credentials_gpt_4o_mini: Credentials):
    """Test async initiate_chat uses cache when provided."""
    agent = ConversableAgent(
        name="agent",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        default_auto_reply="Hello",
    )
    recipient = ConversableAgent(
        name="recipient",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        default_auto_reply="Hi",
    )

    with Cache.disk(cache_seed=42, cache_path_root=".cache"):
        result = await agent.a_initiate_chat(recipient, message="Test", max_turns=1, clear_history=True)
        assert isinstance(result, ChatResult)
