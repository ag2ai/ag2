# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import asyncio
import copy
import os
import threading
import time
import unittest
from collections.abc import Callable
from typing import Annotated, Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

import autogen
from autogen.agentchat import AssistantAgent, ConversableAgent, UpdateSystemMessage, UserProxyAgent
from autogen.agentchat.conversable_agent import register_function
from autogen.agentchat.group import ContextVariables
from autogen.cache.cache import Cache
from autogen.exception_utils import InvalidCarryOverTypeError, SenderRequiredError
from autogen.fast_depends.utils import is_coroutine_callable
from autogen.import_utils import run_for_optional_imports, skip_on_missing_imports
from autogen.llm_config import LLMConfig
from autogen.oai.client import OpenAILLMConfigEntry
from autogen.tools import tool
from autogen.tools.tool import Tool
from test.credentials import Credentials
from test.marks import credentials_all_llms
from test.utils import suppress_gemini_resource_exhausted, suppress_json_decoder_error

here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def conversable_agent():
    return ConversableAgent(
        "conversable_agent_0",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        llm_config=False,
        human_input_mode="NEVER",
    )


@pytest.mark.parametrize("name", ["agent name", "agent_name ", " agent\nname", " agent\tname"])
def test_conversable_agent_name_with_white_space(
    name: str,
    mock_credentials: Credentials,
) -> None:
    agent = ConversableAgent(name=name)
    assert agent.name == name

    llm_config = mock_credentials.llm_config
    with pytest.raises(
        ValueError,
        match=f"The name of the agent cannot contain any whitespace. The name provided is: '{name}'",
    ):
        ConversableAgent(name=name, llm_config=llm_config)


def test_sync_trigger():
    agent = ConversableAgent("a0", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent.register_reply(agent1, lambda recipient, messages, sender, config: (True, "hello"))
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello"
    agent.register_reply("a1", lambda recipient, messages, sender, config: (True, "hello a1"))
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello a1"
    agent.register_reply(
        ConversableAgent, lambda recipient, messages, sender, config: (True, "hello conversable agent")
    )
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello conversable agent"
    agent.register_reply(
        lambda sender: sender.name.startswith("a"), lambda recipient, messages, sender, config: (True, "hello a")
    )
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello a"
    agent.register_reply(
        lambda sender: sender.name.startswith("b"), lambda recipient, messages, sender, config: (True, "hello b")
    )
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello a"
    agent.register_reply(
        ["agent2", agent1], lambda recipient, messages, sender, config: (True, "hello agent2 or agent1")
    )
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello agent2 or agent1"
    agent.register_reply(
        ["agent2", "agent3"], lambda recipient, messages, sender, config: (True, "hello agent2 or agent3")
    )
    agent1.initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello agent2 or agent1"
    pytest.raises(ValueError, agent.register_reply, 1, lambda recipient, messages, sender, config: (True, "hi"))
    pytest.raises(ValueError, agent._match_trigger, 1, agent1)


@pytest.mark.asyncio
async def test_async_trigger():
    agent = ConversableAgent("a0", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")

    def a_reply(recipient, messages, sender, config):
        print("hello from a_reply")
        return (True, "hello")

    agent.register_reply(agent1, a_reply)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello"

    def a_reply_a1(recipient, messages, sender, config):
        print("hello from a_reply_a1")
        return (True, "hello a1")

    agent.register_reply("a1", a_reply_a1)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello a1"

    def a_reply_conversable_agent(recipient, messages, sender, config):
        print("hello from a_reply_conversable_agent")
        return (True, "hello conversable agent")

    agent.register_reply(ConversableAgent, a_reply_conversable_agent)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello conversable agent"

    def a_reply_a(recipient, messages, sender, config):
        print("hello from a_reply_a")
        return (True, "hello a")

    agent.register_reply(lambda sender: sender.name.startswith("a"), a_reply_a)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello a"

    def a_reply_b(recipient, messages, sender, config):
        print("hello from a_reply_b")
        return (True, "hello b")

    agent.register_reply(lambda sender: sender.name.startswith("b"), a_reply_b)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello a"

    def a_reply_agent2_or_agent1(recipient, messages, sender, config):
        print("hello from a_reply_agent2_or_agent1")
        return (True, "hello agent2 or agent1")

    agent.register_reply(["agent2", agent1], a_reply_agent2_or_agent1)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello agent2 or agent1"

    def a_reply_agent2_or_agent3(recipient, messages, sender, config):
        print("hello from a_reply_agent2_or_agent3")
        return (True, "hello agent2 or agent3")

    agent.register_reply(["agent2", "agent3"], a_reply_agent2_or_agent3)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello agent2 or agent1"

    with pytest.raises(ValueError):
        agent.register_reply(1, a_reply)

    with pytest.raises(ValueError):
        agent._match_trigger(1, agent1)


def test_async_trigger_in_sync_chat():
    agent = ConversableAgent("a0", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent("a2", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")

    reply_mock = unittest.mock.MagicMock()

    async def a_reply(recipient, messages, sender, config):
        reply_mock()
        print("hello from a_reply")
        return (True, "hello from reply function")

    agent.register_reply(agent1, a_reply)

    with pytest.raises(RuntimeError) as e:
        agent1.initiate_chat(agent, message="hi")

    assert (
        e.value.args[0] == "Async reply functions can only be used with ConversableAgent.a_initiate_chat(). "
        "The following async reply functions are found: a_reply"
    )

    agent2.register_reply(agent1, a_reply, ignore_async_in_sync_chat=True)
    reply_mock.assert_not_called()


@pytest.mark.asyncio
async def test_sync_trigger_in_async_chat():
    agent = ConversableAgent("a0", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")

    def a_reply(recipient, messages, sender, config):
        print("hello from a_reply")
        return (True, "hello from reply function")

    agent.register_reply(agent1, a_reply)
    await agent1.a_initiate_chat(agent, message="hi")
    assert agent1.last_message(agent)["content"] == "hello from reply function"


def test_context():
    agent = ConversableAgent("a0", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    agent1.send(
        {
            "content": "hello {name}",
            "context": {
                "name": "there",
            },
        },
        agent,
    )
    # expect hello {name} to be printed
    agent1.send(
        {
            "content": lambda context: f"hello {context['name']}",
            "context": {
                "name": "there",
            },
        },
        agent,
    )
    # expect hello there to be printed
    agent.llm_config = {"allow_format_str_template": True}
    agent1.send(
        {
            "content": "hello {name}",
            "context": {
                "name": "there",
            },
        },
        agent,
    )
    # expect hello there to be printed


def test_generate_code_execution_reply():
    agent = ConversableAgent(
        "a0", max_consecutive_auto_reply=10, code_execution_config=False, llm_config=False, human_input_mode="NEVER"
    )

    dummy_messages = [
        {
            "content": "no code block",
            "role": "user",
        },
        {
            "content": "no code block",
            "role": "user",
        },
    ]

    code_message = {
        "content": '```python\nprint("hello world")\n```',
        "role": "user",
    }

    # scenario 1: if code_execution_config is not provided, the code execution should return false, none
    assert agent.generate_code_execution_reply(dummy_messages, config=False) == (False, None)

    # scenario 2: if code_execution_config is provided, but no code block is found, the code execution should return false, none
    assert agent.generate_code_execution_reply(dummy_messages, config={}) == (False, None)

    # scenario 3: if code_execution_config is provided, and code block is found, but it's not within the range of last_n_messages, the code execution should return false, none
    assert agent.generate_code_execution_reply([code_message] + dummy_messages, config={"last_n_messages": 1}) == (
        False,
        None,
    )

    # scenario 4: if code_execution_config is provided, and code block is found, and it's within the range of last_n_messages, the code execution should return true, code block
    agent._code_execution_config = {"last_n_messages": 3, "use_docker": False}
    assert agent.generate_code_execution_reply([code_message] + dummy_messages) == (
        True,
        "exitcode: 0 (execution succeeded)\nCode output: \nhello world\n",
    )
    assert agent._code_execution_config["last_n_messages"] == 3

    # scenario 5: if last_n_messages is set to 'auto' and no code is found, then nothing breaks both when an assistant message is and isn't present
    assistant_message_for_auto = {
        "content": "This is me! The assistant!",
        "role": "assistant",
    }

    dummy_messages_for_auto = []
    for i in range(3):
        dummy_messages_for_auto.append({
            "content": "no code block",
            "role": "user",
        })

        # Without an assistant present
        agent._code_execution_config = {"last_n_messages": "auto", "use_docker": False}
        assert agent.generate_code_execution_reply(dummy_messages_for_auto) == (
            False,
            None,
        )

        # With an assistant message present
        agent._code_execution_config = {"last_n_messages": "auto", "use_docker": False}
        assert agent.generate_code_execution_reply([assistant_message_for_auto] + dummy_messages_for_auto) == (
            False,
            None,
        )

    # scenario 6: if last_n_messages is set to 'auto' and code is found, then we execute it correctly
    dummy_messages_for_auto = []
    for i in range(4):
        # Without an assistant present
        agent._code_execution_config = {"last_n_messages": "auto", "use_docker": False}
        assert agent.generate_code_execution_reply([code_message] + dummy_messages_for_auto) == (
            True,
            "exitcode: 0 (execution succeeded)\nCode output: \nhello world\n",
        )

        # With an assistant message present
        agent._code_execution_config = {"last_n_messages": "auto", "use_docker": False}
        assert agent.generate_code_execution_reply(
            [assistant_message_for_auto] + [code_message] + dummy_messages_for_auto
        ) == (
            True,
            "exitcode: 0 (execution succeeded)\nCode output: \nhello world\n",
        )

        dummy_messages_for_auto.append({
            "content": "no code block",
            "role": "user",
        })

    # scenario 7: if last_n_messages is set to 'auto' and code is present, but not before an assistant message, then nothing happens
    agent._code_execution_config = {"last_n_messages": "auto", "use_docker": False}
    assert agent.generate_code_execution_reply(
        [code_message] + [assistant_message_for_auto] + dummy_messages_for_auto
    ) == (
        False,
        None,
    )
    assert agent._code_execution_config["last_n_messages"] == "auto"

    # scenario 8: if last_n_messages is misconfigures, we expect to see an error
    with pytest.raises(ValueError):
        agent._code_execution_config = {"last_n_messages": -1, "use_docker": False}
        agent.generate_code_execution_reply([code_message])

    with pytest.raises(ValueError):
        agent._code_execution_config = {"last_n_messages": "hello world", "use_docker": False}
        agent.generate_code_execution_reply([code_message])


def test_max_consecutive_auto_reply():
    agent = ConversableAgent("a0", max_consecutive_auto_reply=2, llm_config=False, human_input_mode="NEVER")
    agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    assert agent.max_consecutive_auto_reply() == agent.max_consecutive_auto_reply(agent1) == 2
    agent.update_max_consecutive_auto_reply(1)
    assert agent.max_consecutive_auto_reply() == agent.max_consecutive_auto_reply(agent1) == 1

    agent1.initiate_chat(agent, message="hello")
    assert agent._consecutive_auto_reply_counter[agent1] == 1
    agent1.initiate_chat(agent, message="hello again")
    # with auto reply because the counter is reset
    assert agent1.last_message(agent)["role"] == "user"
    assert len(agent1.chat_messages[agent]) == 2
    assert len(agent.chat_messages[agent1]) == 2

    assert agent._consecutive_auto_reply_counter[agent1] == 1
    agent1.send(message="bye", recipient=agent)
    # no auto reply
    assert agent1.last_message(agent)["role"] == "assistant"

    agent1.initiate_chat(agent, clear_history=False, message="hi")
    assert len(agent1.chat_messages[agent]) > 2
    assert len(agent.chat_messages[agent1]) > 2

    assert agent1.reply_at_receive[agent] == agent.reply_at_receive[agent1] is True
    agent1.stop_reply_at_receive(agent)
    assert agent1.reply_at_receive[agent] is False and agent.reply_at_receive[agent1] is True


def test_max_consecutive_auto_reply_with_max_turns(capsys: pytest.CaptureFixture[str]):
    agent1 = ConversableAgent("agent1", max_consecutive_auto_reply=1, llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent("agent2", max_consecutive_auto_reply=100, llm_config=False, human_input_mode="NEVER")

    # max_consecutive_auto_reply parameter on the agent that initiates chat
    agent1.initiate_chat(agent2, message="hello", max_turns=50)
    assert len(agent2.chat_messages[agent1]) == 4
    assert len(agent1.chat_messages[agent2]) == 4
    # checking captured output
    captured = capsys.readouterr()
    assert "TERMINATING RUN" in captured.out
    assert "Maximum number of consecutive auto-replies reached" in captured.out

    _ = capsys.readouterr()  # Explicitly clear buffer

    # max_consecutive_auto_reply parameter on the recipient agent
    agent2.initiate_chat(agent1, message="hello", max_turns=50)
    assert len(agent1.chat_messages[agent2]) == 3
    assert len(agent2.chat_messages[agent1]) == 3
    # checking captured output
    captured = capsys.readouterr()
    assert "TERMINATING RUN" in captured.out
    assert "Maximum number of consecutive auto-replies reached" in captured.out


def test_conversable_agent():
    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="ALWAYS")
    dummy_agent_2 = ConversableAgent(name="dummy_agent_2", llm_config=False, human_input_mode="TERMINATE")

    # monkeypatch.setattr(sys, "stdin", StringIO("exit"))
    dummy_agent_1.receive([{"role": "user", "content": "hello"}], dummy_agent_2)
    # monkeypatch.setattr(sys, "stdin", StringIO("TERMINATE\n\n"))
    dummy_agent_1.receive(
        [
            {
                "role": "user",
                "content": "hello {name}",
                "context": {
                    "name": "dummy_agent_2",
                },
            }
        ],
        dummy_agent_2,
    )
    assert "context" in dummy_agent_1.chat_messages[dummy_agent_2][-1]
    # receive dict without openai fields to be printed, such as "content", 'function_call'. There should be no error raised.
    pre_len = len(dummy_agent_1.chat_messages[dummy_agent_2])
    with pytest.raises(ValueError):
        dummy_agent_1.receive([{"message": "hello"}], dummy_agent_2)
    assert pre_len == len(dummy_agent_1.chat_messages[dummy_agent_2]), (
        "When the message is not an valid openai message, it should not be appended to the oai conversation."
    )

    # monkeypatch.setattr(sys, "stdin", StringIO("exit"))
    dummy_agent_1.send("TERMINATE", dummy_agent_2)  # send a str
    # monkeypatch.setattr(sys, "stdin", StringIO("exit"))
    dummy_agent_1.send(
        {
            "content": "TERMINATE",
        },
        dummy_agent_2,
    )  # send a dict

    # send dict with no openai fields
    pre_len = len(dummy_agent_1.chat_messages[dummy_agent_2])
    with pytest.raises(ValueError):
        dummy_agent_1.send({"message": "hello"}, dummy_agent_2)

    assert pre_len == len(dummy_agent_1.chat_messages[dummy_agent_2]), (
        "When the message is not a valid openai message, it should not be appended to the oai conversation."
    )

    # update system message
    dummy_agent_1.update_system_message("new system message")
    assert dummy_agent_1.system_message == "new system message"

    dummy_agent_3 = ConversableAgent(name="dummy_agent_3", llm_config=False, human_input_mode="TERMINATE")
    with pytest.raises(KeyError):
        dummy_agent_1.last_message(dummy_agent_3)

    # Check the description field
    assert dummy_agent_1.description != dummy_agent_1.system_message
    assert dummy_agent_2.description == dummy_agent_2.system_message

    dummy_agent_4 = ConversableAgent(
        name="dummy_agent_4",
        system_message="The fourth dummy agent used for testing.",
        llm_config=False,
        human_input_mode="TERMINATE",
    )
    assert dummy_agent_4.description == "The fourth dummy agent used for testing."  # Same as system message

    dummy_agent_5 = ConversableAgent(
        name="dummy_agent_5",
        system_message="",
        description="The fifth dummy agent used for testing.",
        llm_config=False,
        human_input_mode="TERMINATE",
    )
    assert dummy_agent_5.description == "The fifth dummy agent used for testing."  # Same as system message


def test_terminate_chat_true():
    """Test _terminate_chat returns True for a termination message."""
    agent = ConversableAgent("agent", llm_config=False)
    recipient = ConversableAgent(
        "recipient",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") == "TERMINATE",
    )
    message = {"content": "TERMINATE"}
    assert agent._should_terminate_chat(recipient, message) is True


def test_terminate_chat_false_non_termination_content():
    """Test _terminate_chat returns False for a non-termination message."""
    agent = ConversableAgent("agent", llm_config=False)
    recipient = ConversableAgent(
        "recipient",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") == "TERMINATE",
    )
    message = {"content": "Hello"}
    assert agent._should_terminate_chat(recipient, message) is False


def test_terminate_chat_false_non_string_content():
    """Test _terminate_chat returns False if content is not a string."""
    agent = ConversableAgent("agent", llm_config=False)
    recipient = ConversableAgent(
        "recipient",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") == "TERMINATE",
    )
    message = {"content": None}
    assert agent._should_terminate_chat(recipient, message) is False


@pytest.mark.asyncio
async def test_a_initiate_chat_triggers_terminate_chat(monkeypatch):
    agent = ConversableAgent("agent", llm_config=False, human_input_mode="NEVER")
    recipient = ConversableAgent(
        "recipient",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") == "TERMINATE",
        human_input_mode="NEVER",
    )

    async def fake_a_generate_init_message(self, message, **kwargs):
        return {"content": "TERMINATE"}

    monkeypatch.setattr(ConversableAgent, "a_generate_init_message", fake_a_generate_init_message)

    async def fake_a_get_human_input(self, prompt):
        return ""

    monkeypatch.setattr(ConversableAgent, "a_get_human_input", fake_a_get_human_input)
    result = await agent.a_initiate_chat(recipient, message="irrelevant", max_turns=2)
    assert len(result.chat_history) == 1
    assert result.chat_history[0]["content"] == "TERMINATE"


def test_generate_reply():
    def add_num(num_to_be_added):
        given_num = 10
        return num_to_be_added + given_num

    dummy_agent_2 = ConversableAgent(
        name="user_proxy", llm_config=False, human_input_mode="TERMINATE", function_map={"add_num": add_num}
    )
    messages = [{"function_call": {"name": "add_num", "arguments": '{ "num_to_be_added": 5 }'}, "role": "assistant"}]

    # when sender is None, messages is provided
    assert dummy_agent_2.generate_reply(messages=messages, sender=None)["content"] == 15, (
        "generate_reply not working when sender is None"
    )

    # when sender is provided, messages is None
    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="ALWAYS")
    dummy_agent_2._oai_messages[dummy_agent_1] = messages
    assert dummy_agent_2.generate_reply(messages=None, sender=dummy_agent_1)["content"] == 15, (
        "generate_reply not working when messages is None"
    )

    dummy_agent_2.register_reply(["str", None], ConversableAgent.generate_oai_reply)
    with pytest.raises(SenderRequiredError):
        dummy_agent_2.generate_reply(messages=messages, sender=None)


def test_generate_reply_raises_on_messages_and_sender_none(conversable_agent):
    with pytest.raises(AssertionError):
        conversable_agent.generate_reply(messages=None, sender=None)


@pytest.mark.asyncio
async def test_a_generate_reply_raises_on_messages_and_sender_none(conversable_agent):
    with pytest.raises(AssertionError):
        await conversable_agent.a_generate_reply(messages=None, sender=None)


def test_generate_reply_with_messages_and_sender_none(conversable_agent):
    messages = [{"role": "user", "content": "hello"}]
    try:
        response = conversable_agent.generate_reply(messages=messages, sender=None)
        assert response is not None, "Response should not be None"
    except AssertionError as e:
        pytest.fail(f"Unexpected AssertionError: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


@pytest.mark.asyncio
async def test_a_generate_reply_with_messages_and_sender_none(conversable_agent):
    messages = [{"role": "user", "content": "hello"}]
    try:
        response = await conversable_agent.a_generate_reply(messages=messages, sender=None)
        assert response is not None, "Response should not be None"
    except AssertionError as e:
        pytest.fail(f"Unexpected AssertionError: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_generate_reply_with_single_dict_message(conversable_agent):
    """Test generate_reply with a single message of type dict[str, Any]."""
    single_message = {"role": "user", "content": "Hello, this is a single message"}

    try:
        response = conversable_agent.generate_reply(messages=single_message, sender=None)
        assert response is not None, "Response should not be None"
        # Since conversable_agent has no LLM config, it should return the default auto reply
        assert response == conversable_agent._default_auto_reply
    except AssertionError as e:
        pytest.fail(f"Unexpected AssertionError: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


@pytest.mark.asyncio
async def test_a_generate_reply_with_single_dict_message(conversable_agent):
    """Test a_generate_reply with a single message of type dict[str, Any]."""
    single_message = {"role": "user", "content": "Hello, this is a single message"}

    try:
        response = await conversable_agent.a_generate_reply(messages=single_message, sender=None)
        assert response is not None, "Response should not be None"
        # Since conversable_agent has no LLM config, it should return the default auto reply
        assert response == conversable_agent._default_auto_reply
    except AssertionError as e:
        pytest.fail(f"Unexpected AssertionError: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


@pytest.mark.asyncio
@patch("builtins.input")
async def test_a_get_human_input_console_io(mock_input) -> None:
    from autogen.io.base import IOStream
    from autogen.io.console import IOConsole

    mock_input.return_value = "test input"
    with IOStream.set_default(IOConsole()):
        agent = ConversableAgent(name="agent", llm_config=False, human_input_mode="ALWAYS")
        assert agent.human_input_mode == "ALWAYS"
        result = await agent.a_get_human_input("Please enter your input: ")
        assert result == "test input"


@pytest.mark.asyncio
async def test_a_get_human_input_thread_stream() -> None:
    from autogen.io.base import IOStream
    from autogen.io.thread_io_stream import ThreadIOStream

    agent = ConversableAgent(name="agent", llm_config=False, human_input_mode="ALWAYS")
    ts = ThreadIOStream()

    def responder():
        _evt = ts.input_stream.get()
        time.sleep(0.01)
        ts._output_stream.put("thread-ok")

    t = threading.Thread(target=responder, daemon=True)
    t.start()

    with IOStream.set_default(ts):
        result = await agent.a_get_human_input("Enter:")
    assert result == "thread-ok"
    assert agent._human_input[-1] == "thread-ok"
    t.join(timeout=1)


@pytest.mark.asyncio
async def test_a_get_human_input_async_thread_stream() -> None:
    from autogen.io.base import IOStream
    from autogen.io.thread_io_stream import AsyncThreadIOStream

    agent = ConversableAgent(name="agent", llm_config=False, human_input_mode="ALWAYS")
    ats = AsyncThreadIOStream()

    async def responder():
        _evt = await ats.input_stream.get()
        await asyncio.sleep(0)
        outq = ats.output_stream if hasattr(ats, "output_stream") else ats._output_stream
        await outq.put("async-thread-ok")

    task = asyncio.create_task(responder())
    try:
        with IOStream.set_default(ats):
            result = await agent.a_get_human_input("Enter:")
    finally:
        task.cancel()

    assert result == "async-thread-ok"
    assert agent._human_input[-1] == "async-thread-ok"


def test_update_function_signature_and_register_functions(mock_credentials: Credentials) -> None:
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def exec_python(cell: str) -> None:
        pass

    def exec_sh(script: str) -> None:
        pass

    agent.update_function_signature(
        {
            "name": "python",
            "description": "run cell in ipython and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Valid Python cell to execute.",
                    }
                },
                "required": ["cell"],
            },
        },
        is_remove=False,
    )

    functions = agent.llm_config["functions"]
    assert {f["name"] for f in functions} == {"python"}

    agent.update_function_signature(
        {
            "name": "sh",
            "description": "run a shell script and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Valid shell script to execute.",
                    }
                },
                "required": ["script"],
            },
        },
        is_remove=False,
    )

    functions = agent.llm_config["functions"]
    assert {f["name"] for f in functions} == {"python", "sh"}

    # register the functions
    agent.register_function(
        function_map={
            "python": exec_python,
            "sh": exec_sh,
        }
    )
    assert set(agent.function_map.keys()) == {"python", "sh"}
    assert agent.function_map["python"] == exec_python
    assert agent.function_map["sh"] == exec_sh

    # remove the functions
    agent.register_function(
        function_map={
            "python": None,
        }
    )

    assert set(agent.function_map.keys()) == {"sh"}
    assert agent.function_map["sh"] == exec_sh


class TestWrapFunction:
    def test__wrap_function_sync(self):
        CurrencySymbol = Literal["USD", "EUR"]  # noqa: N806

        class Currency(BaseModel):
            currency: CurrencySymbol = Field(description="Currency code")
            amount: float = Field(default=100.0, description="Amount of money in the currency")

        Currency(currency="USD", amount=100.0)

        def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
            if base_currency == quote_currency:
                return 1.0
            elif base_currency == "USD" and quote_currency == "EUR":
                return 1 / 1.1
            elif base_currency == "EUR" and quote_currency == "USD":
                return 1.1
            else:
                raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")

        agent = ConversableAgent(name="agent", llm_config=False)

        @agent._wrap_function
        def currency_calculator(
            base: Annotated[Currency, "Base currency"],
            quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
        ) -> Currency:
            quote_amount = exchange_rate(base.currency, quote_currency) * base.amount
            return Currency(amount=quote_amount, currency=quote_currency)

        assert (
            currency_calculator(base={"currency": "USD", "amount": 110.11}, quote_currency="EUR")
            == '{"currency":"EUR","amount":100.1}'
        )

        assert not is_coroutine_callable(currency_calculator)

    @pytest.mark.skip(reason="Not implemented yet")
    def test__wrap_function_list(self) -> None:
        class Point(BaseModel):
            x: float
            y: float

        agent = ConversableAgent(name="agent", llm_config=False)

        @agent._wrap_function
        def f(xs: list[tuple[float, float]], ys: list[Point]) -> list[Point]:
            return [Point(x=x, y=y) for (x, y) in xs] + ys

        assert f([(1.0, 2.0), (3.0, 4.0)], [Point(x=5.0, y=6.0)]) == [
            Point(x=x, y=y) for (x, y) in [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        ]

    @pytest.mark.asyncio
    async def test__wrap_function_async(self):
        CurrencySymbol = Literal["USD", "EUR"]  # noqa: N806

        class Currency(BaseModel):
            currency: CurrencySymbol = Field(description="Currency code")
            amount: float = Field(default=100.0, description="Amount of money in the currency")

        Currency(currency="USD", amount=100.0)

        def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
            if base_currency == quote_currency:
                return 1.0
            elif base_currency == "USD" and quote_currency == "EUR":
                return 1 / 1.1
            elif base_currency == "EUR" and quote_currency == "USD":
                return 1.1
            else:
                raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")

        agent = ConversableAgent(name="agent", llm_config=False)

        @agent._wrap_function
        async def currency_calculator(
            base: Annotated[Currency, "Base currency"],
            quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
        ) -> Currency:
            quote_amount = exchange_rate(base.currency, quote_currency) * base.amount
            return Currency(amount=quote_amount, currency=quote_currency)

        assert (
            await currency_calculator(base={"currency": "USD", "amount": 110.11}, quote_currency="EUR")
            == '{"currency":"EUR","amount":100.1}'
        )

        assert is_coroutine_callable(currency_calculator)


def get_origin(d: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    return {k: v._origin for k, v in d.items()}


def test_register_for_llm(mock_credentials: Credentials) -> None:
    agent3 = ConversableAgent(name="agent3", llm_config=mock_credentials.llm_config)
    agent2 = ConversableAgent(name="agent2", llm_config=mock_credentials.llm_config)
    agent1 = ConversableAgent(name="agent1", llm_config=mock_credentials.llm_config)

    @agent3.register_for_llm()
    @agent2.register_for_llm(name="python")
    @agent1.register_for_llm(description="run cell in ipython and return the execution result.")
    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
        pass

    expected1 = [
        {
            "type": "function",
            "function": {
                "description": "run cell in ipython and return the execution result.",
                "name": "exec_python",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python cell to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        }
    ]
    expected2 = copy.deepcopy(expected1)
    expected2[0]["function"]["name"] = "python"
    expected3 = expected2

    assert agent1.llm_config["tools"] == expected1
    assert agent2.llm_config["tools"] == expected2
    assert agent3.llm_config["tools"] == expected3

    @agent3.register_for_llm()
    @agent2.register_for_llm()
    @agent1.register_for_llm(name="sh", description="run a shell script and return the execution result.")
    async def exec_sh(script: Annotated[str, "Valid shell script to execute."]) -> str:
        pass

    expected1 = expected1 + [
        {
            "type": "function",
            "function": {
                "name": "sh",
                "description": "run a shell script and return the execution result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Valid shell script to execute.",
                        }
                    },
                    "required": ["script"],
                },
            },
        }
    ]
    expected2 = expected2 + [expected1[1]]
    expected3 = expected3 + [expected1[1]]

    assert agent1.llm_config["tools"] == expected1
    assert agent2.llm_config["tools"] == expected2
    assert agent3.llm_config["tools"] == expected3


def test_register_for_llm_api_style_function(mock_credentials: Credentials):
    agent3 = ConversableAgent(name="agent3", llm_config=mock_credentials.llm_config)
    agent2 = ConversableAgent(name="agent2", llm_config=mock_credentials.llm_config)
    agent1 = ConversableAgent(name="agent1", llm_config=mock_credentials.llm_config)

    @agent3.register_for_llm(api_style="function")
    @agent2.register_for_llm(name="python", api_style="function")
    @agent1.register_for_llm(description="run cell in ipython and return the execution result.", api_style="function")
    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
        pass

    expected1 = [
        {
            "description": "run cell in ipython and return the execution result.",
            "name": "exec_python",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Valid Python cell to execute.",
                    }
                },
                "required": ["cell"],
            },
        }
    ]
    expected2 = copy.deepcopy(expected1)
    expected2[0]["name"] = "python"
    expected3 = expected2

    assert agent1.llm_config["functions"] == expected1
    assert agent2.llm_config["functions"] == expected2
    assert agent3.llm_config["functions"] == expected3

    @agent3.register_for_llm(api_style="function")
    @agent2.register_for_llm(api_style="function")
    @agent1.register_for_llm(
        name="sh", description="run a shell script and return the execution result.", api_style="function"
    )
    async def exec_sh(script: Annotated[str, "Valid shell script to execute."]) -> str:
        pass

    expected1 = expected1 + [
        {
            "name": "sh",
            "description": "run a shell script and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Valid shell script to execute.",
                    }
                },
                "required": ["script"],
            },
        }
    ]
    expected2 = expected2 + [expected1[1]]
    expected3 = expected3 + [expected1[1]]

    assert agent1.llm_config["functions"] == expected1
    assert agent2.llm_config["functions"] == expected2
    assert agent3.llm_config["functions"] == expected3


def test_register_for_llm_without_description(mock_credentials: Credentials):
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    @agent.register_for_llm()
    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
        pass

    assert exec_python.description == ""


def test_register_for_llm_with_docstring(mock_credentials: Credentials):
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    @agent.register_for_llm()
    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
        """Execute a Python cell."""
        pass

    assert exec_python.description == "Execute a Python cell."


def test_register_for_llm_without_LLM():  # noqa: N802
    agent = ConversableAgent(name="agent", llm_config=None)
    with pytest.raises(
        AssertionError,
        match="To update a tool signature, agent must have an llm_config",
    ):

        @agent.register_for_llm(description="do things.")
        def do_stuff(s: str) -> str:
            return f"{s} done"


def test_register_for_llm_without_configuration():
    with pytest.raises(
        ValueError,
        match="List should have at least 1 item after validation, not 0",
    ):
        ConversableAgent(name="agent", llm_config={"config_list": []})


def test_register_for_llm_without_model_name():
    with pytest.raises(
        ValueError,
        match="String should have at least 1 character",
    ):
        ConversableAgent(name="agent", llm_config={"config_list": [{"model": ""}]})


def test_register_for_execution(mock_credentials: Credentials):
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    user_proxy_1 = UserProxyAgent(name="user_proxy_1")
    user_proxy_2 = UserProxyAgent(name="user_proxy_2")

    @user_proxy_2.register_for_execution(name="python")
    @agent.register_for_execution()
    @agent.register_for_llm(description="run cell in ipython and return the execution result.")
    @user_proxy_1.register_for_execution()
    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]):
        pass

    expected_function_map_1 = {"exec_python": exec_python.func}
    assert get_origin(agent.function_map) == expected_function_map_1
    assert get_origin(user_proxy_1.function_map) == expected_function_map_1

    expected_function_map_2 = {"python": exec_python.func}
    assert get_origin(user_proxy_2.function_map) == expected_function_map_2

    @agent.register_for_execution()
    @agent.register_for_llm(description="run a shell script and return the execution result.")
    @user_proxy_1.register_for_execution(name="sh")
    async def exec_sh(script: Annotated[str, "Valid shell script to execute."]):
        pass

    expected_function_map = {
        "exec_python": exec_python.func,
        "sh": exec_sh.func,
    }
    assert get_origin(agent.function_map) == expected_function_map
    assert get_origin(user_proxy_1.function_map) == expected_function_map


def test_register_functions(mock_credentials: Credentials):
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    user_proxy = UserProxyAgent(name="user_proxy")

    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
        pass

    register_function(
        exec_python,
        caller=agent,
        executor=user_proxy,
        description="run cell in ipython and return the execution result.",
    )

    expected_function_map = {"exec_python": exec_python}
    assert get_origin(user_proxy.function_map).keys() == expected_function_map.keys()

    expected = [
        {
            "type": "function",
            "function": {
                "description": "run cell in ipython and return the execution result.",
                "name": "exec_python",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python cell to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        }
    ]
    assert agent.llm_config["tools"] == expected


@run_for_optional_imports("openai", "openai")
def test_function_registration_e2e_sync(credentials_gpt_4o_mini: Credentials) -> None:
    llm_config = credentials_gpt_4o_mini.llm_config

    coder = autogen.AssistantAgent(
        name="chatbot",
        system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
        llm_config=llm_config,
    )

    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        system_message="A proxy for the user for executing code.",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding"},
        llm_config=llm_config,
    )

    # define functions according to the function description
    timer_mock = unittest.mock.MagicMock()
    stopwatch_mock = unittest.mock.MagicMock()

    # An example async function registered using decorators
    @user_proxy.register_for_execution()
    @coder.register_for_llm(description="create a timer for N seconds")
    def timer(num_seconds: Annotated[str, "Number of seconds in the timer."]) -> str:
        print("timer is running")
        for i in range(int(num_seconds)):
            print(".", end="")
            time.sleep(0.01)
        print()

        timer_mock(num_seconds=num_seconds)
        return "Timer is done!"

    # An example sync function registered using register_function
    def stopwatch(num_seconds: Annotated[str, "Number of seconds in the stopwatch."]) -> str:
        print("stopwatch is running")
        # assert False, "stopwatch's alive!"
        for i in range(int(num_seconds)):
            print(".", end="")
            time.sleep(0.01)
        print()

        stopwatch_mock(num_seconds=num_seconds)
        return "Stopwatch is done!"

    register_function(stopwatch, caller=coder, executor=user_proxy, description="create a stopwatch for N seconds")

    # start the conversation
    # 'await' is used to pause and resume code execution for async IO operations.
    # Without 'await', an async function returns a coroutine object but doesn't execute the function.
    # With 'await', the async function is executed and the current function is paused until the awaited function returns a result.
    user_proxy.initiate_chat(
        coder,
        message="Create a timer for 1 second and then a stopwatch for 2 seconds.",
    )

    timer_mock.assert_called_once_with(num_seconds="1")
    stopwatch_mock.assert_called_once_with(num_seconds="2")


async def _test_function_registration_e2e_async(credentials: Credentials) -> None:
    coder = autogen.AssistantAgent(
        name="chatbot",
        system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
        llm_config=credentials.llm_config,
    )

    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        system_message="A proxy for the user for executing code.",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding"},
    )

    # define functions according to the function description
    timer_mock = unittest.mock.MagicMock()
    stopwatch_mock = unittest.mock.MagicMock()

    # An example async function registered using register_function
    async def timer(num_seconds: Annotated[str, "Number of seconds in the timer."]) -> str:
        print("timer is running")
        for i in range(int(num_seconds)):
            print(".", end="")
            await asyncio.sleep(0.01)
        print()

        timer_mock(num_seconds=num_seconds)
        return "Timer is done!"

    register_function(timer, caller=coder, executor=user_proxy, description="create a timer for N seconds")

    # An example sync function registered using decorators
    @user_proxy.register_for_execution()
    @coder.register_for_llm(description="create a stopwatch for N seconds")
    def stopwatch(num_seconds: Annotated[str, "Number of seconds in the stopwatch."]) -> str:
        print("stopwatch is running")
        # assert False, "stopwatch's alive!"
        for i in range(int(num_seconds)):
            print(".", end="")
            time.sleep(0.01)
        print()

        stopwatch_mock(num_seconds=num_seconds)
        return "Stopwatch is done!"

    # start the conversation
    # 'await' is used to pause and resume code execution for async IO operations.
    # Without 'await', an async function returns a coroutine object but doesn't execute the function.
    # With 'await', the async function is executed and the current function is paused until the awaited function returns a result.
    await user_proxy.a_initiate_chat(
        coder,
        message="Create a timer for 1 second and then a stopwatch for 2 seconds.",
    )

    timer_mock.assert_called_once_with(num_seconds="1")
    stopwatch_mock.assert_called_once_with(num_seconds="2")


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
@pytest.mark.asyncio
async def test_function_registration_e2e_async(
    credentials_from_test_param: Credentials,
) -> None:
    if credentials_from_test_param.api_type == "google":
        pytest.skip("This test currently fails with gemini flash model")
    await _test_function_registration_e2e_async(credentials_from_test_param)


@run_for_optional_imports("openai", "openai")
def test_max_turn(credentials_gpt_4o_mini: Credentials) -> None:
    # create an AssistantAgent instance named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=10,
        llm_config=credentials_gpt_4o_mini.llm_config,
    )

    user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="ALWAYS", code_execution_config=False)

    # Use MagicMock to create a mock get_human_input function
    user_proxy.get_human_input = MagicMock(return_value="Not funny. Try again.")
    res = user_proxy.initiate_chat(assistant, clear_history=True, max_turns=3, message="Hello, make a joke about AI.")
    print("Result summary:", res.summary)
    print("Human input:", res.human_input)
    print("history", res.chat_history)
    assert len(res.chat_history) <= 6


@run_for_optional_imports("openai", "openai")
def test_message_func(credentials_gpt_4o_mini: Credentials):
    import random

    class Function:
        call_count = 0

        def get_random_number(self):
            self.call_count += 1
            return str(random.randint(0, 100))

    def my_message_play(sender, recipient, context):
        final_msg = {}
        final_msg["content"] = "Let's play a game."
        final_msg["function_call"] = {"name": "get_random_number", "arguments": "{}"}
        return final_msg

    func = Function()
    # autogen.ChatCompletion.start_logging()
    user = UserProxyAgent(
        "user",
        code_execution_config={
            "work_dir": here,
            "use_docker": False,
        },
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
    )
    player = autogen.AssistantAgent(
        name="Player",
        system_message="You will use function `get_random_number` to get a random number. Stop only when you get at least 1 even number and 1 odd number. Reply TERMINATE to stop.",
        description="A player that makes function_calls.",
        llm_config=credentials_gpt_4o_mini.llm_config,
        function_map={"get_random_number": func.get_random_number},
    )

    chat_res_play = user.initiate_chat(
        player,
        message={"content": "Let's play a game.", "function_call": {"name": "get_random_number", "arguments": "{}"}},
        max_turns=1,
    )
    print(chat_res_play.summary)

    chat_res_play = user.initiate_chat(
        player,
        message=my_message_play,
        max_turns=1,
    )
    print(chat_res_play.summary)


@run_for_optional_imports("openai", "openai")
def test_summary(credentials_gpt_4o_mini: Credentials):
    import random

    class Function:
        call_count = 0

        def get_random_number(self):
            self.call_count += 1
            return str(random.randint(0, 100))

    def my_message_play(sender, recipient, context):
        final_msg = {}
        final_msg["content"] = "Let's play a game."
        final_msg["function_call"] = {"name": "get_random_number", "arguments": "{}"}
        return final_msg

    def my_summary(sender, recipient, summary_args):
        prefix = summary_args.get("prefix", "Summary:")
        return prefix + recipient.chat_messages[sender][-1].get("content", "")

    func = Function()
    # autogen.ChatCompletion.start_logging()
    user = UserProxyAgent(
        "user",
        code_execution_config={
            "work_dir": here,
            "use_docker": False,
        },
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
    )
    player = autogen.AssistantAgent(
        name="Player",
        system_message="You will use function `get_random_number` to get a random number. Stop only when you get at least 1 even number and 1 odd number. Reply TERMINATE to stop.",
        description="A player that makes function_calls.",
        llm_config=credentials_gpt_4o_mini.llm_config,
        function_map={"get_random_number": func.get_random_number},
    )

    chat_res_play = user.initiate_chat(
        player,
        message=my_message_play,
        # message="Make a joke about AI",
        max_turns=1,
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Summarize the conversation into less than five words."},
    )
    print(chat_res_play.summary)

    chat_res_play = user.initiate_chat(
        player,
        # message=my_message_play,
        message="Make a joke about AI",
        max_turns=1,
        summary_method=my_summary,
        summary_args={"prefix": "This is the last message:"},
    )
    print(chat_res_play.summary)

    chat_res_play = user.initiate_chat(
        player,
        message={"content": "Let's play a game.", "function_call": {"name": "get_random_number", "arguments": "{}"}},
        max_turns=1,
        summary_method=my_summary,
        summary_args={"prefix": "This is the last message:"},
    )
    print(chat_res_play.summary)


def test_summarize_chat_with_dict_summary():
    user = UserProxyAgent(name="user", human_input_mode="NEVER", default_auto_reply="Hello.", llm_config=False)
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=False,
        default_auto_reply="This is a test assistant.",
    )

    def my_summary(sender, recipient, summary_args):
        return {"content": "This is a summary of the conversation."}

    chat_res = user.initiate_chat(
        assistant,
        message="Hello, how are you?",
        max_turns=1,
        summary_method=my_summary,
        summary_args={"summary_prompt": "Summarize the conversation."},
    )
    assert chat_res.summary == "This is a summary of the conversation."


def test_process_before_send():
    print_mock = unittest.mock.MagicMock()

    # Updated to include sender parameter
    def send_to_frontend(sender, message, recipient, silent):
        assert sender.name == "dummy_agent_1", "Sender is not the expected agent"
        if not silent:
            print(f"Message sent from {sender.name} to {recipient.name}: {message}")
            print_mock(message=message)
        return message

    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="NEVER")
    dummy_agent_2 = ConversableAgent(name="dummy_agent_2", llm_config=False, human_input_mode="NEVER")
    dummy_agent_1.register_hook("process_message_before_send", send_to_frontend)
    dummy_agent_1.send("hello", dummy_agent_2)
    # Updated expectation: hooks now receive normalized dict format
    print_mock.assert_called_once_with(message={"content": "hello", "role": "user"})
    dummy_agent_1.send("silent hello", dummy_agent_2, silent=True)
    # Still only called once (silent message doesn't trigger print)
    print_mock.assert_called_once_with(message={"content": "hello", "role": "user"})


def test_messages_with_carryover():
    agent1 = autogen.ConversableAgent(
        "alice",
        max_consecutive_auto_reply=10,
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice speaking.",
    )
    context = {"message": "hello", "carryover": "Testing carryover."}
    generated_message = agent1.generate_init_message(**context)
    assert isinstance(generated_message, str)

    context = {"message": "hello", "carryover": ["Testing carryover.", "This should pass"]}
    generated_message = agent1.generate_init_message(**context)
    assert isinstance(generated_message, str)

    context = {"message": "hello", "carryover": 3}
    with pytest.raises(InvalidCarryOverTypeError):
        agent1.generate_init_message(**context)

    # Test multimodal messages
    mm_content = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "goodbye"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
    ]
    mm_message = {"content": mm_content}
    context = {
        "message": mm_message,
        "carryover": "Testing carryover.",
    }
    generated_message = agent1.generate_init_message(**context)
    assert isinstance(generated_message, dict)
    assert len(generated_message["content"]) == 4

    context = {"message": mm_message, "carryover": ["Testing carryover.", "This should pass"]}
    generated_message = agent1.generate_init_message(**context)
    assert isinstance(generated_message, dict)
    assert len(generated_message["content"]) == 4

    context = {"message": mm_message, "carryover": 3}
    with pytest.raises(InvalidCarryOverTypeError):
        agent1.generate_init_message(**context)

    # Test without carryover
    print(mm_message)
    context = {"message": mm_message}
    generated_message = agent1.generate_init_message(**context)
    assert isinstance(generated_message, dict)
    assert len(generated_message["content"]) == 3

    # Test without text in multimodal message
    mm_content = [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    mm_message = {"content": mm_content}
    context = {"message": mm_message}
    generated_message = agent1.generate_init_message(**context)
    assert isinstance(generated_message, dict)
    assert len(generated_message["content"]) == 1

    generated_message = agent1.generate_init_message(**context, carryover="Testing carryover.")
    assert isinstance(generated_message, dict)
    assert len(generated_message["content"]) == 2


def test_chat_history():
    alice = autogen.ConversableAgent(
        "alice",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is alice speaking.",
    )

    charlie = autogen.ConversableAgent(
        "charlie",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is charlie speaking.",
    )

    max_turns = 2

    def bob_initiate_chat(agent: ConversableAgent, text: Literal["past", "future"]):
        _ = agent.initiate_chat(
            alice,
            message=f"This is bob from the {text} speaking.",
            max_turns=max_turns,
            clear_history=False,
            silent=True,
        )
        _ = agent.initiate_chat(
            charlie,
            message=f"This is bob from the {text} speaking.",
            max_turns=max_turns,
            clear_history=False,
            silent=True,
        )

    bob = autogen.ConversableAgent(
        "bob",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob from the past speaking.",
    )
    bob_initiate_chat(bob, "past")
    context = bob.chat_messages

    del bob

    # Test agent with chat history
    bob = autogen.ConversableAgent(
        "bob",
        human_input_mode="NEVER",
        llm_config=False,
        default_auto_reply="This is bob from the future speaking.",
        chat_messages=context,
    )

    assert bool(bob.chat_messages)
    assert bob.chat_messages == context

    # two times the max turns due to bob replies
    assert len(bob.chat_messages[alice]) == 2 * max_turns
    assert len(bob.chat_messages[charlie]) == 2 * max_turns

    bob_initiate_chat(bob, "future")
    assert len(bob.chat_messages[alice]) == 4 * max_turns
    assert len(bob.chat_messages[charlie]) == 4 * max_turns

    assert bob.chat_messages[alice][0]["content"] == "This is bob from the past speaking."
    assert bob.chat_messages[charlie][0]["content"] == "This is bob from the past speaking."

    assert bob.chat_messages[alice][-2]["content"] == "This is bob from the future speaking."
    assert bob.chat_messages[charlie][-2]["content"] == "This is bob from the future speaking."


def test_http_client():
    import httpx

    with pytest.raises(TypeError):
        config_list = [
            {
                "model": "my-gpt-4-deployment",
                "api_key": "",
                "http_client": httpx.Client(),
            }
        ]

        autogen.ConversableAgent(
            "test_agent",
            human_input_mode="NEVER",
            llm_config={"config_list": config_list},
            default_auto_reply="This is alice speaking.",
        )


def test_adding_duplicate_function_warning():
    config_base = [{"base_url": "http://0.0.0.0:8000", "api_key": "NULL", "model": "gpt-4"}]

    agent = autogen.ConversableAgent(
        "jtoy",
        llm_config={"config_list": config_base},
    )

    def sample_function():
        pass

    agent.register_function(
        function_map={
            "sample_function": sample_function,
        }
    )
    agent.update_function_signature(
        {
            "name": "foo",
        },
        is_remove=False,
    )
    agent.update_tool_signature(
        {
            "type": "function",
            "function": {
                "name": "yo",
            },
        },
        is_remove=False,
    )

    with pytest.warns(UserWarning, match="Function 'sample_function' is being overridden."):
        agent.register_function(
            function_map={
                "sample_function": sample_function,
            }
        )
    with pytest.warns(UserWarning, match="Function 'foo' is being overridden."):
        agent.update_function_signature(
            {
                "name": "foo",
            },
            is_remove=False,
        )
    with pytest.warns(UserWarning, match="Function 'yo' is being overridden."):
        agent.update_tool_signature(
            {
                "type": "function",
                "function": {
                    "name": "yo",
                },
            },
            is_remove=False,
        )


def test_process_gemini_carryover():
    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="NEVER")
    content = "I am your assistant."
    carryover_content = "How can I help you?"
    gemini_kwargs = {"carryover": [{"content": carryover_content}]}
    proc_content = dummy_agent_1._process_carryover(content=content, kwargs=gemini_kwargs)
    assert proc_content == content + "\nContext: \n" + carryover_content, "Incorrect carryover processing"


def test_process_carryover():
    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="NEVER")
    content = "I am your assistant."
    carryover = "How can I help you?"
    kwargs = {"carryover": carryover}
    proc_content = dummy_agent_1._process_carryover(content=content, kwargs=kwargs)
    assert proc_content == content + "\nContext: \n" + carryover, "Incorrect carryover processing"

    carryover_l = ["How can I help you?"]
    kwargs = {"carryover": carryover_l}
    proc_content = dummy_agent_1._process_carryover(content=content, kwargs=kwargs)
    assert proc_content == content + "\nContext: \n" + carryover, "Incorrect carryover processing"

    proc_content_empty_carryover = dummy_agent_1._process_carryover(content=content, kwargs={"carryover": None})
    assert proc_content_empty_carryover == content, "Incorrect carryover processing"


def test_handle_gemini_carryover():
    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="NEVER")
    content = "I am your assistant"
    carryover_content = "How can I help you?"
    gemini_kwargs = {"carryover": [{"content": carryover_content}]}
    proc_content = dummy_agent_1._handle_carryover(message=content, kwargs=gemini_kwargs)
    assert proc_content == content + "\nContext: \n" + carryover_content, "Incorrect carryover processing"


def test_handle_carryover():
    dummy_agent_1 = ConversableAgent(name="dummy_agent_1", llm_config=False, human_input_mode="NEVER")
    content = "I am your assistant."
    carryover = "How can I help you?"
    kwargs = {"carryover": carryover}
    proc_content = dummy_agent_1._handle_carryover(message=content, kwargs=kwargs)
    assert proc_content == content + "\nContext: \n" + carryover, "Incorrect carryover processing"

    carryover_l = ["How can I help you?"]
    kwargs = {"carryover": carryover_l}
    proc_content = dummy_agent_1._handle_carryover(message=content, kwargs=kwargs)
    assert proc_content == content + "\nContext: \n" + carryover, "Incorrect carryover processing"

    proc_content_empty_carryover = dummy_agent_1._handle_carryover(message=content, kwargs={"carryover": None})
    assert proc_content_empty_carryover == content, "Incorrect carryover processing"


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
def test_conversable_agent_with_whitespaces_in_name_end2end(
    credentials_from_test_param: Credentials,
    request: pytest.FixtureRequest,
) -> None:
    agent = ConversableAgent(
        name="first_agent",
        llm_config=credentials_from_test_param.llm_config,
    )

    user_proxy = UserProxyAgent(
        name="user proxy",
        human_input_mode="NEVER",
    )

    # Get the parameter name request node
    current_llm = request.node.callspec.id
    if "gpt_4" in current_llm:
        with pytest.raises(
            ValueError,
            match="This error typically occurs when the agent name contains invalid characters, such as spaces or special symbols.",
        ):
            user_proxy.initiate_chat(agent, message="Hello, how are you?", max_turns=2)
    # anthropic and gemini will not raise an error if agent name contains whitespaces
    else:
        user_proxy.initiate_chat(agent, message="Hello, how are you?", max_turns=2)


@run_for_optional_imports("openai", "openai")
def test_context_variables():
    # Test initialization with context_variables
    initial_context = ContextVariables(data={"test_key": "test_value", "number": 42, "nested": {"inner": "value"}})
    agent = ConversableAgent(name="context_test_agent", llm_config=False, context_variables=initial_context)

    # Check that context was properly initialized
    assert agent.context_variables.to_dict() == initial_context.to_dict()

    # Test initialization without context_variables
    agent_no_context = ConversableAgent(name="no_context_agent", llm_config=False)
    assert agent_no_context.context_variables.to_dict() == {}

    # Test get_context
    assert agent.context_variables.get("test_key") == "test_value"
    assert agent.context_variables.get("number") == 42
    assert agent.context_variables.get("nested") == {"inner": "value"}
    assert agent.context_variables.get("non_existent") is None
    assert agent.context_variables.get("non_existent", default="default") == "default"

    # Test set_context
    agent.context_variables.set("new_key", "new_value")
    assert agent.context_variables.get("new_key") == "new_value"

    # Test overwriting existing value
    agent.context_variables.set("test_key", "updated_value")
    assert agent.context_variables.get("test_key") == "updated_value"

    # Test update_context
    new_values = {"bulk_key1": "bulk_value1", "bulk_key2": "bulk_value2", "test_key": "bulk_updated_value"}
    agent.context_variables.update(new_values)
    assert agent.context_variables.get("bulk_key1") == "bulk_value1"
    assert agent.context_variables.get("bulk_key2") == "bulk_value2"
    assert agent.context_variables.get("test_key") == "bulk_updated_value"


@pytest.mark.skip(reason="'anyOf' parameters works with vertexai setup but it is not supported in google.genai")
@pytest.mark.gemini
@suppress_gemini_resource_exhausted
def test_gemini_with_tools_parameters_set_to_is_annotated_with_none_as_default_value(
    credentials_gemini_flash: Credentials,
) -> None:
    agent = ConversableAgent(name="agent", llm_config=credentials_gemini_flash.llm_config)

    user_proxy = UserProxyAgent(
        name="user_proxy_1",
        human_input_mode="NEVER",
    )

    mock = MagicMock()

    @user_proxy.register_for_execution()
    @agent.register_for_llm(description="Login function")
    def login(
        additional_notes: Annotated[str | None, "Additional notes"] = None,
    ) -> str:
        mock()
        return "Login successful."

    user_proxy.initiate_chat(agent, message="Please login", max_turns=2)

    mock.assert_called_once()


@pytest.mark.deepseek
@suppress_json_decoder_error
@skip_on_missing_imports(["openai"], "openai")
def test_conversable_agent_with_deepseek_reasoner(
    credentials_deepseek_reasoner: Credentials,
) -> None:
    agent = ConversableAgent(
        name="agent",
        llm_config=credentials_deepseek_reasoner.llm_config,
    )

    user_proxy = UserProxyAgent(
        name="user_proxy_1",
        human_input_mode="NEVER",
    )

    result = user_proxy.initiate_chat(
        agent, message="Hello, how are you?", summary_method="reflection_with_llm", max_turns=2
    )
    assert isinstance(result.summary, str)


def test_invalid_functions_parameter():
    """Test initialization with valid and invalid parameters"""

    # Invalid functions parameter
    with pytest.raises(TypeError):
        ConversableAgent("test_agent", functions="invalid")


def test_update_system_message():
    """Tests the update_agent_state_before_reply functionality with multiple scenarios"""

    # Test invalid update function
    with pytest.raises(ValueError, match="The update function must be either a string or a callable"):
        ConversableAgent("agent3", update_agent_state_before_reply=UpdateSystemMessage(123))

    # Test invalid callable (wrong number of parameters)
    def invalid_update_function(context_variables):
        return "Invalid function"

    with pytest.raises(ValueError, match="The update function must accept two parameters"):
        ConversableAgent("agent4", update_agent_state_before_reply=UpdateSystemMessage(invalid_update_function))

    # Test invalid callable (wrong return type)
    def invalid_return_function(context_variables, messages) -> dict:
        return {}

    with pytest.raises(ValueError, match="The update function must return a string"):
        ConversableAgent("agent5", update_agent_state_before_reply=UpdateSystemMessage(invalid_return_function))


def test_tools_property(conversable_agent):
    """Test the tools property returns a copy of the tools list."""
    # Initial tools list should be empty
    assert len(conversable_agent.tools) == 0

    # Create a mock tool
    mock_tool = Tool(name="test_tool", description="A test tool", func_or_tool=lambda x: x)

    # Add tool directly to internal list
    conversable_agent._tools.append(mock_tool)

    # Get tools list
    tools = conversable_agent.tools
    assert len(tools) == 1
    assert tools[0] == mock_tool

    # Verify we got a copy by modifying the returned list
    tools.clear()
    assert len(conversable_agent._tools) == 1  # Original list should be unchanged


def test_add_tool_for_llm(mock_credentials: Credentials):
    """Test adding a tool for LLM use."""

    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    mock_tool = Tool(name="test_tool", description="A test tool", func_or_tool=sample_tool_func)

    # Add the tool
    agent.register_for_llm()(mock_tool)

    # Verify tool was added to internal list
    assert len(agent._tools) == 1
    assert agent._tools[0] == mock_tool

    # Verify tool was registered with LLM
    tool_schemas = [tool["function"]["name"] for tool in agent.llm_config.get("tools", [])]
    assert mock_tool.name in tool_schemas


def test_add_tool_for_llm_invalid_type(conversable_agent):
    """Test adding an invalid tool type raises TypeError."""
    with pytest.raises(
        TypeError, match="'func_or_tool' must be a function or a Tool object, got '<class 'str'>' instead."
    ):
        conversable_agent.register_for_llm()("not a tool")


def test_remove_tool_for_llm(mock_credentials: Credentials):
    """Test removing a tool."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    mock_tool = Tool(name="test_tool", description="A test tool", func_or_tool=sample_tool_func)

    agent.register_for_llm()(mock_tool)

    # Remove the tool
    agent.remove_tool_for_llm(mock_tool)

    # Verify tool was unregistered from LLM
    tool_schemas = [tool["function"]["name"] for tool in agent.llm_config.tools]
    print(mock_tool.name)
    print(tool_schemas)
    assert mock_tool.name not in tool_schemas


def test_remove_tool_by_name_for_llm(mock_credentials: Credentials):
    """Test removing a tool."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    mock_tool = Tool(name="test_tool", description="A test tool", func_or_tool=sample_tool_func)

    agent.register_for_llm()(mock_tool)

    # Remove the tool by name
    agent.update_tool_signature(tool_sig="test_tool", is_remove=True)

    # Verify tool was unregistered from LLM
    tool_schemas = [tool["function"]["name"] for tool in agent.llm_config.tools]
    print(mock_tool.name)
    print(tool_schemas)
    assert mock_tool.name not in tool_schemas


def test_remove_tool_for_llm_not_found(mock_credentials: Credentials):
    """Test removing a non-existent tool raises ValueError."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    mock_tool = Tool(name="test_tool", description="A test tool", func_or_tool=sample_tool_func)

    with pytest.raises(AssertionError, match="The agent config doesn't have tool"):
        agent.remove_tool_for_llm(mock_tool)


def test_tool_integration(mock_credentials: Credentials):
    """Test full tool integration workflow."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    tool1 = Tool(name="tool1", description="A test tool", func_or_tool=sample_tool_func)

    tool2 = Tool(name="tool2", description="A test tool", func_or_tool=sample_tool_func)

    # Add tools
    agent.register_for_llm()(tool1)
    agent.register_for_llm()(tool2)

    # Verify both tools are in the list
    assert len(agent.tools) == 2
    tool_names = {t.name for t in agent.tools}
    assert tool_names == {"tool1", "tool2"}

    # Remove one tool
    agent.remove_tool_for_llm(tool1)

    # Verify only one tool remains
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "tool2"

    # Verify LLM config was updated
    tool_schemas = [tool["function"]["name"] for tool in agent.llm_config.get("tools", [])]
    assert "tool1" not in tool_schemas
    assert "tool2" in tool_schemas


def test_create_or_get_executor(mock_credentials: Credentials):
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
    executor_agent = None

    def log_result(result: str):
        return "I have logged the result."

    tool = Tool(
        name="log_result",
        description="Logs the result of the task.",
        func_or_tool=log_result,
    )

    expected_tools = [
        {
            "type": "function",
            "function": {
                "description": "Logs the result of the task.",
                "name": "log_result",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "string", "description": "result"}},
                    "required": ["result"],
                },
            },
        }
    ]
    executor_agent = None
    for _ in range(2):
        with agent._create_or_get_executor(
            tools=[tool],
        ) as executor:
            if not executor_agent:
                executor_agent = executor
            else:
                assert executor_agent == executor
            assert isinstance(executor_agent, ConversableAgent)
            # Runtime tools should be available to the LLM during execution
            assert agent.llm_config["tools"] == expected_tools
            # And should be available for execution in the executor
            assert len(executor_agent.function_map.keys()) == 1


@pytest.mark.parametrize(
    "llm_config, expected",
    [
        (None, False),
        (False, False),
        pytest.param(
            {"config_list": [{"model": "gpt-3", "api_key": "whatever"}]},
            LLMConfig(OpenAILLMConfigEntry(model="gpt-3", api_key="whatever")),
            id="deprecated (remove in 0.11): legacy dict format with config_list",
            marks=pytest.mark.filterwarnings("ignore::DeprecationWarning"),
        ),
        pytest.param(
            {"model": "gpt-3", "api_key": "whatever"},
            LLMConfig(OpenAILLMConfigEntry(model="gpt-3", api_key="whatever")),
            id="deprecated (remove in 0.11): legacy dict format",
            marks=pytest.mark.filterwarnings("ignore::DeprecationWarning"),
        ),
        pytest.param(
            LLMConfig(OpenAILLMConfigEntry(model="gpt-3")),
            LLMConfig(OpenAILLMConfigEntry(model="gpt-3")),
            id="LLMConfig passed",
        ),
    ],
)
def test_validate_llm_config(
    llm_config: LLMConfig | dict[str, Any] | Literal[False] | None, expected: LLMConfig | Literal[False]
):
    actual = ConversableAgent._validate_llm_config(llm_config)
    assert actual == expected, f"{actual} != {expected}"


@run_for_optional_imports("openai", "openai")
def test_cache_context(credentials_gpt_4o_mini: Credentials) -> None:
    message = "Hello, make a joke about AI."

    assistant = autogen.AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=10,
        llm_config=credentials_gpt_4o_mini.llm_config,
    )

    user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="ALWAYS", code_execution_config=False)

    # Use MagicMock to create a mock get_human_input function
    user_proxy.get_human_input = MagicMock(return_value="Not funny. Try again.")

    # Test without cache
    start_time = time.time()
    res = user_proxy.initiate_chat(assistant, clear_history=True, max_turns=3, message=message, cache=None)
    end_time = time.time()
    duration_without_cache = end_time - start_time
    assert len(res.chat_history) <= 6
    assert user_proxy.client_cache is None

    # Test with cache context
    with Cache.disk(cache_seed=39, cache_path_root=".cache"):
        # Test with cold cache
        start_time = time.time()
        # initiate_chat should use the cache context automatically
        res = user_proxy.initiate_chat(
            assistant,
            clear_history=True,
            max_turns=3,
            message=message,
        )
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time
        assert len(res.chat_history) <= 6

        # Test with warm cache
        start_time = time.time()
        # initiate_chat should use the cache context automatically
        res = user_proxy.initiate_chat(
            assistant,
            clear_history=True,
            max_turns=3,
            message=message,
        )
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert len(res.chat_history) <= 6

    # Test that warm cache is faster than cold cache and no cache.
    assert duration_with_warm_cache < duration_with_cold_cache
    assert duration_with_warm_cache < duration_without_cache


def test_set_ui_tools(mock_credentials: Credentials):
    """Test setting UI tools."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    for i in range(3):
        mock_tool = Tool(name=f"test_ui_tool_{i}", description="A test UI tool", func_or_tool=sample_tool_func)
        agent.set_ui_tools([mock_tool])

        # Verify tool was added to llm_config
        assert len(agent.llm_config.get("tools", [])) == 1
        tool_schemas = [tool["function"]["name"] for tool in agent.llm_config.get("tools", [])]
        assert mock_tool.name in tool_schemas
        assert f"test_ui_tool_{i - 1}" not in tool_schemas

        # Verify tool was registered for execution
        expected_function_map = {mock_tool.name: mock_tool.func}
        assert get_origin(agent.function_map) == expected_function_map


def test_unset_ui_tools(mock_credentials: Credentials):
    """Test unsetting UI tools."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def sample_tool_func(my_prop: str) -> str:
        return my_prop * 2

    mock_tool = Tool(name="test_ui_tool", description="A test UI tool", func_or_tool=sample_tool_func)

    # Set the tool first
    agent.set_ui_tools([mock_tool])
    assert len(agent.llm_config.get("tools", [])) == 1

    # Unset the tool
    agent.unset_ui_tools([mock_tool])

    # Verify tool was removed from llm_config
    assert len(agent.llm_config.get("tools", [])) == 0


def test_run_method_no_double_tool_registration(mock_credentials: Credentials):
    """Test that tools from agent's self._tools aren't double-registered for LLM in run method."""
    agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)

    def pre_registered_tool(message: str) -> str:
        return f"Pre-registered: {message}"

    def runtime_tool(message: str) -> str:
        return f"Runtime: {message}"

    # Create tools
    pre_tool = Tool(name="pre_tool", description="Pre-registered tool", func_or_tool=pre_registered_tool)
    runtime_tool_obj = Tool(name="runtime_tool", description="Runtime tool", func_or_tool=runtime_tool)

    # Pre-register tool with agent (simulating functions parameter during init)
    agent.register_for_llm()(pre_tool)
    initial_tools_count = len(agent.llm_config.get("tools", []))
    assert initial_tools_count == 1

    # Create executor manually to test tool registration
    with agent._create_or_get_executor(tools=[runtime_tool_obj]) as executor:
        # Check that pre-registered tool is not double-registered
        tools_after_executor = len(agent.llm_config.get("tools", []))
        assert tools_after_executor == 2  # pre_tool + runtime_tool

        # Verify tool names in LLM config
        tool_names = [tool["function"]["name"] for tool in agent.llm_config.get("tools", [])]
        assert "pre_tool" in tool_names
        assert "runtime_tool" in tool_names
        assert tool_names.count("pre_tool") == 1  # Should not be duplicated

        # Verify executor has both tools
        assert len(executor.function_map) == 2
        assert "pre_tool" in executor.function_map
        assert "runtime_tool" in executor.function_map


def test_send_with_list_of_messages(mock_credentials: Credentials):
    """Test send() method with a list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list of messages
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Second message"},
        {"role": "user", "content": "Third message"},
    ]

    # Send the list of messages
    sender.send(messages, recipient, request_reply=False, silent=True)

    # Verify that all messages were processed and sent
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "First message"
    assert recipient.chat_messages[sender][1]["content"] == "Second message"
    assert recipient.chat_messages[sender][2]["content"] == "Third message"


def test_send_with_empty_list_of_messages(mock_credentials: Credentials):
    """Test send() method with an empty list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Send an empty list of messages
    sender.send([], recipient, request_reply=False, silent=True)

    # Verify that no messages were added
    assert len(recipient.chat_messages[sender]) == 0


@pytest.mark.asyncio
async def test_a_send_with_list_of_messages(mock_credentials: Credentials):
    """Test a_send() method with a list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list of messages
    messages = [
        {"role": "user", "content": "Async first message"},
        {"role": "assistant", "content": "Async second message"},
        {"role": "user", "content": "Async third message"},
    ]

    # Send the list of messages asynchronously
    await sender.a_send(messages, recipient, request_reply=False, silent=True)

    # Verify that all messages were processed and sent
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "Async first message"
    assert recipient.chat_messages[sender][1]["content"] == "Async second message"
    assert recipient.chat_messages[sender][2]["content"] == "Async third message"


@pytest.mark.asyncio
async def test_a_send_with_mixed_content_list(mock_credentials: Credentials):
    """Test a_send() method with a list containing different message types."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list with mixed content types
    messages = [
        {"role": "user", "content": "Text message"},
        {"role": "assistant", "content": ""},  # Avoid None content to keep strict validation
        {"role": "function", "name": "test_function", "content": "Function result"},
    ]

    # Send the list of messages asynchronously
    await sender.a_send(messages, recipient, request_reply=False, silent=True)

    # Verify that all messages were processed and sent
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "Text message"
    assert recipient.chat_messages[sender][1]["content"] == ""
    assert recipient.chat_messages[sender][2]["content"] == "Function result"


def test_receive_with_list_of_messages(mock_credentials: Credentials):
    """Test receive() method with a list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list of messages
    messages = [
        {"role": "user", "content": "Received first message"},
        {"role": "assistant", "content": "Received second message"},
        {"role": "user", "content": "Received third message"},
    ]

    # Receive the list of messages
    recipient.receive(messages, sender, request_reply=False, silent=True)

    # Verify that all messages were processed and received
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "Received first message"
    assert recipient.chat_messages[sender][1]["content"] == "Received second message"
    assert recipient.chat_messages[sender][2]["content"] == "Received third message"


def test_receive_with_list_containing_function_calls(mock_credentials: Credentials):
    """Test receive() method with a list containing function call messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list with function call messages
    messages = [
        {"role": "user", "content": "Please call a function"},
        {"role": "assistant", "function_call": {"name": "test_func", "arguments": '{"arg": "value"}'}},
        {"role": "function", "name": "test_func", "content": "Function executed successfully"},
    ]

    # Receive the list of messages
    recipient.receive(messages, sender, request_reply=False, silent=True)

    # Verify that all messages were processed and received
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "Please call a function"
    assert "function_call" in recipient.chat_messages[sender][1]
    assert recipient.chat_messages[sender][2]["content"] == "Function executed successfully"


@pytest.mark.asyncio
async def test_a_receive_with_list_of_messages(mock_credentials: Credentials):
    """Test a_receive() method with a list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list of messages
    messages = [
        {"role": "user", "content": "Async received first message"},
        {"role": "assistant", "content": "Async received second message"},
        {"role": "user", "content": "Async received third message"},
    ]

    # Receive the list of messages asynchronously
    await recipient.a_receive(messages, sender, request_reply=False, silent=True)

    # Verify that all messages were processed and received
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "Async received first message"
    assert recipient.chat_messages[sender][1]["content"] == "Async received second message"
    assert recipient.chat_messages[sender][2]["content"] == "Async received third message"


@pytest.mark.asyncio
async def test_a_receive_with_list_containing_tool_calls(mock_credentials: Credentials):
    """Test a_receive() method with a list containing tool call messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list with tool call messages
    messages = [
        {"role": "user", "content": "Please use a tool"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Tool executed successfully"},
    ]

    # Receive the list of messages asynchronously
    await recipient.a_receive(messages, sender, request_reply=False, silent=True)

    # Verify that all messages were processed and received
    assert len(recipient.chat_messages[sender]) == 3
    assert recipient.chat_messages[sender][0]["content"] == "Please use a tool"
    assert "tool_calls" in recipient.chat_messages[sender][1]
    assert recipient.chat_messages[sender][2]["content"] == "Tool executed successfully"


def test_process_message_before_send_with_list_of_messages(mock_credentials: Credentials):
    """Test _process_message_before_send() method with a list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Create a list of messages
    messages = [
        {"role": "user", "content": "Process first message"},
        {"role": "assistant", "content": "Process second message"},
        {"role": "user", "content": "Process third message"},
    ]

    # Process the list of messages
    processed_messages = sender._process_message_before_send(messages, recipient, silent=True)

    # Verify that the processed messages are returned as a list
    assert isinstance(processed_messages, list)
    assert len(processed_messages) == 3
    assert processed_messages[0]["content"] == "Process first message"
    assert processed_messages[1]["content"] == "Process second message"
    assert processed_messages[2]["content"] == "Process third message"


def test_process_message_before_send_with_empty_list(mock_credentials: Credentials):
    """Test _process_message_before_send() method with an empty list of messages."""
    sender = ConversableAgent(name="sender", llm_config=False)
    recipient = ConversableAgent(name="recipient", llm_config=False)

    # Process an empty list of messages
    processed_messages = sender._process_message_before_send([], recipient, silent=True)

    # Verify that an empty list is returned
    assert isinstance(processed_messages, list)
    assert len(processed_messages) == 0


# ============================================================================
# Test 1: Hook Behavior with List Messages
# ============================================================================


def test_hook_processes_multiple_messages():
    """Test that hooks receive and can modify each message in a list"""
    # Track how many times the hook is called
    hook_calls = []

    def tracking_hook(sender, message, recipient, silent):
        # Verify message is always a dict
        assert isinstance(message, dict), f"Hook received non-dict: {type(message)}"

        # Track this call
        hook_calls.append({
            "sender": sender.name,
            "content": message.get("content"),
            "recipient": recipient.name,
        })

        # Modify the message
        message["modified"] = True
        message["hook_processed"] = len(hook_calls)
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", tracking_hook)

    # Send multiple messages at once
    messages = [
        {"content": "First message", "role": "user"},
        {"content": "Second message", "role": "user"},
        {"content": "Third message", "role": "user"},
    ]
    agent1.send(messages, agent2)

    # Verify hook was called exactly 3 times (once per message)
    assert len(hook_calls) == 3, f"Expected 3 hook calls, got {len(hook_calls)}"

    # Verify each call had correct content
    assert hook_calls[0]["content"] == "First message"
    assert hook_calls[1]["content"] == "Second message"
    assert hook_calls[2]["content"] == "Third message"

    # Verify all messages were modified
    received_messages = agent2.chat_messages[agent1]
    assert len(received_messages) == 3
    for i, msg in enumerate(received_messages):
        assert msg.get("modified") is True, f"Message {i} was not modified"
        assert msg.get("hook_processed") == i + 1, f"Message {i} has wrong processing order"


def test_hook_receives_dict_for_string_input():
    """Test that hooks receive normalized dict even when string is sent"""
    received_messages = []

    def capturing_hook(sender, message, recipient, silent):
        received_messages.append(message)
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", capturing_hook)

    # Send a string message
    agent1.send("Hello world", agent2)

    # Hook should have received a dict, not a string
    assert len(received_messages) == 1
    assert isinstance(received_messages[0], dict)
    assert received_messages[0]["content"] == "Hello world"
    assert received_messages[0]["role"] == "user"


def test_multiple_hooks_process_list_messages():
    """Test that multiple hooks all process each message in a list"""
    hook1_calls = []
    hook2_calls = []

    def hook1(sender, message, recipient, silent):
        hook1_calls.append(message.get("content"))
        message["hook1_processed"] = True
        return message

    def hook2(sender, message, recipient, silent):
        # Verify hook1 already processed it
        assert message.get("hook1_processed") is True
        hook2_calls.append(message.get("content"))
        message["hook2_processed"] = True
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", hook1)
    agent1.register_hook("process_message_before_send", hook2)

    messages = [
        {"content": "Message 1"},
        {"content": "Message 2"},
    ]
    agent1.send(messages, agent2)

    # Both hooks should have been called for both messages
    assert hook1_calls == ["Message 1", "Message 2"]
    assert hook2_calls == ["Message 1", "Message 2"]

    # Verify both hooks processed all messages
    received = agent2.chat_messages[agent1]
    for msg in received:
        assert msg.get("hook1_processed") is True
        assert msg.get("hook2_processed") is True


def test_hook_return_validation_with_list():
    """Test that hook return value validation works for list messages"""

    def bad_hook_returns_none(sender, message, recipient, silent):
        return None  # Invalid!

    def bad_hook_returns_string(sender, message, recipient, silent):
        return "invalid"  # Invalid!

    def bad_hook_returns_list(sender, message, recipient, silent):
        return [message]  # Invalid! Should return dict, not list

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    # Test None return
    agent1.register_hook("process_message_before_send", bad_hook_returns_none)
    with pytest.raises(TypeError, match="Hook must return a dict"):
        agent1.send([{"content": "test"}], agent2)

    # Clear hooks and test string return
    agent1.hook_lists["process_message_before_send"].clear()
    agent1.register_hook("process_message_before_send", bad_hook_returns_string)
    with pytest.raises(TypeError, match="Hook must return a dict"):
        agent1.send([{"content": "test"}], agent2)

    # Clear hooks and test list return
    agent1.hook_lists["process_message_before_send"].clear()
    agent1.register_hook("process_message_before_send", bad_hook_returns_list)
    with pytest.raises(TypeError, match="Hook must return a dict"):
        agent1.send([{"content": "test"}], agent2)


# ============================================================================
# Test 2: Empty Message Edge Cases
# ============================================================================


def test_send_empty_list_shows_warning():
    """Test that sending an empty message list triggers a warning"""
    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    # Send empty list should trigger warning
    with pytest.warns(UserWarning, match="Attempted to send empty message list"):
        agent1.send([], agent2)

    # Verify no messages were actually sent
    assert agent1 not in agent2.chat_messages or len(agent2.chat_messages.get(agent1, [])) == 0


def test_hook_cannot_create_empty_list():
    """Test behavior when hook would create an empty message scenario"""
    # Note: Currently hooks process individual messages, so they can't create an empty list
    # But they could return an empty dict which should be validated

    def hook_returns_empty_dict(sender, message, recipient, silent):
        return {}  # Empty dict - technically valid but might be problematic

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", hook_returns_empty_dict)

    # This should work but might fail validation at _append_oai_message
    # because empty dict has no content/function_call/tool_calls
    with pytest.raises(ValueError, match="can't be converted into a valid ChatCompletion message"):
        agent1.send({"content": "test"}, agent2)


def test_empty_string_message():
    """Test behavior with empty string message"""
    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    # Empty string should still be sent (it's not an empty list)
    agent1.send("", agent2)

    # Should have created a message with empty content
    received = agent2.chat_messages[agent1]
    assert len(received) == 1
    assert received[0]["content"] == ""


def test_message_with_none_content():
    """Test sending message with None content (valid for function/tool calls)"""
    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    # Message with None content but with tool_calls should be valid
    message = {
        "content": None,
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_func", "arguments": "{}"},
            }
        ],
    }

    agent1.send(message, agent2)

    # Should have been sent successfully
    received = agent2.chat_messages[agent1]
    assert len(received) == 1
    assert received[0]["content"] is None
    assert received[0]["tool_calls"] is not None


def test_mixed_valid_and_empty_messages():
    """Test sending a list with mix of valid and questionable messages"""
    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    # Mix of valid messages
    messages = [
        {"content": "Valid message 1"},
        {"content": ""},  # Empty content but still valid
        {"content": "Valid message 2"},
    ]

    agent1.send(messages, agent2)

    # All three should be received
    received = agent2.chat_messages[agent1]
    assert len(received) == 3
    assert received[0]["content"] == "Valid message 1"
    assert received[1]["content"] == ""
    assert received[2]["content"] == "Valid message 2"


def test_hook_modifies_content_to_empty():
    """Test hook that modifies content to empty string"""

    def empty_content_hook(sender, message, recipient, silent):
        message["content"] = ""
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", empty_content_hook)
    agent1.send("Original content", agent2)

    # Should receive message with empty content
    received = agent2.chat_messages[agent1]
    assert len(received) == 1
    assert received[0]["content"] == ""


# ============================================================================
# Test 3: Async versions of the above tests
# ============================================================================


@pytest.mark.asyncio
async def test_hook_processes_multiple_messages_async():
    """Async version: Test that hooks receive and can modify each message in a list"""
    hook_calls = []

    def tracking_hook(sender, message, recipient, silent):
        assert isinstance(message, dict)
        hook_calls.append(message.get("content"))
        message["modified"] = True
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", tracking_hook)

    messages = [
        {"content": "First async"},
        {"content": "Second async"},
        {"content": "Third async"},
    ]
    await agent1.a_send(messages, agent2)

    assert len(hook_calls) == 3
    assert hook_calls == ["First async", "Second async", "Third async"]

    received = agent2.chat_messages[agent1]
    assert len(received) == 3
    for msg in received:
        assert msg.get("modified") is True


@pytest.mark.asyncio
async def test_send_empty_list_shows_warning_async():
    """Async version: Test that sending empty list triggers warning"""
    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    with pytest.warns(UserWarning, match="Attempted to send empty message list"):
        await agent1.a_send([], agent2)

    assert agent1 not in agent2.chat_messages or len(agent2.chat_messages.get(agent1, [])) == 0


# ============================================================================
# Test 4: Integration tests combining hooks with list messages
# ============================================================================


def test_hook_adds_metadata_to_list_messages():
    """Test realistic scenario: hook adds metadata/tracking to all messages"""
    import time

    def add_timestamp_hook(sender, message, recipient, silent):
        message["timestamp"] = time.time()
        message["sender_name"] = sender.name
        message["recipient_name"] = recipient.name
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", add_timestamp_hook)

    messages = [{"content": f"Message {i}"} for i in range(5)]
    agent1.send(messages, agent2)

    received = agent2.chat_messages[agent1]
    assert len(received) == 5

    for msg in received:
        assert "timestamp" in msg
        assert msg["sender_name"] == "agent1"
        assert msg["recipient_name"] == "agent2"
        assert isinstance(msg["timestamp"], float)


def test_hook_filters_sensitive_content():
    """Test realistic scenario: hook sanitizes messages"""

    def sanitize_hook(sender, message, recipient, silent):
        content = message.get("content", "")
        if isinstance(content, str):
            # Remove sensitive patterns
            content = content.replace("SECRET", "[REDACTED]")
            content = content.replace("PASSWORD", "[REDACTED]")
            message["content"] = content
        return message

    agent1 = ConversableAgent(name="agent1", llm_config=False, human_input_mode="NEVER")
    agent2 = ConversableAgent(name="agent2", llm_config=False, human_input_mode="NEVER")

    agent1.register_hook("process_message_before_send", sanitize_hook)

    messages = [
        {"content": "This is a SECRET message"},
        {"content": "My PASSWORD is 12345"},
        {"content": "Normal message"},
    ]
    agent1.send(messages, agent2)

    received = agent2.chat_messages[agent1]
    assert received[0]["content"] == "This is a [REDACTED] message"
    assert received[1]["content"] == "My [REDACTED] is 12345"
    assert received[2]["content"] == "Normal message"


# Add at the end of the file


@tool(name="calculator", description="Basic calculator for math operations")
def calculator(operation: str, a: float, b: float) -> float:
    """Basic calculator function"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else 0
    return 0


@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
def test_single_agent_generate_reply_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Integration test: single agent generate_reply with OpenAI."""
    assistant = autogen.AssistantAgent(
        "assistant",
        llm_config=credentials_gpt_4o_mini.llm_config,
        system_message="You are a helpful assistant. Keep responses concise.",
    )

    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = assistant.generate_reply(messages)

    assert response is not None
    assert isinstance(response, (str, dict))


@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
def test_two_agent_chat_with_calculator_tool_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Integration test: two agent chat with calculator tool."""
    llm_config = credentials_gpt_4o_mini.llm_config

    math_agent = ConversableAgent(
        "math_agent",
        llm_config=llm_config,
        system_message="You are a math expert. Use calculator tool for calculations.",
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    user_proxy.register_function(function_map={"calculator": calculator})

    user_proxy.initiate_chat(math_agent, message="Calculate 15 * 7 using your calculator tool", max_turns=2)

    assert len(user_proxy.chat_messages[math_agent]) > 0


@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
def test_agent_run_method_single_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Integration test: run() method with single agent."""
    assistant = autogen.AssistantAgent(
        "assistant",
        llm_config=credentials_gpt_4o_mini.llm_config,
    )

    message = [{"content": "What is the capital of France?", "role": "user"}]
    response = assistant.run(message=message, max_turns=1)
    response.process()

    assert response.summary is not None or len(response.messages) > 0
    assert response.last_speaker is not None


@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
def test_agent_run_method_two_agents_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Integration test: run() method with two agents."""
    math_agent = ConversableAgent(
        "math_agent",
        llm_config=credentials_gpt_4o_mini.llm_config,
        system_message="You are a math expert.",
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    user_proxy.register_function(function_map={"calculator": calculator})

    message = [{"content": "Calculate 10 + 5", "role": "user"}]
    response = user_proxy.run(recipient=math_agent, message=message, max_turns=3)

    # Consume events
    for event in response.events:
        pass

    assert len(response.messages) > 0


@pytest.mark.asyncio
@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
async def test_single_agent_generate_reply_async_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Async integration test: single agent generate_reply with OpenAI."""
    assistant = AssistantAgent(
        "assistant",
        llm_config=credentials_gpt_4o_mini.llm_config,
        system_message="You are a helpful assistant. Keep responses concise.",
    )

    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = await assistant.a_generate_reply(messages)

    assert response is not None
    assert isinstance(response, (str, dict))


@pytest.mark.asyncio
@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
async def test_two_agent_chat_with_calculator_tool_async_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Async integration test: two agent chat with calculator tool."""
    llm_config = credentials_gpt_4o_mini.llm_config

    math_agent = ConversableAgent(
        "math_agent",
        llm_config=llm_config,
        system_message="You are a math expert. Use calculator tool for calculations.",
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    user_proxy.register_function(function_map={"calculator": calculator})

    await user_proxy.a_initiate_chat(math_agent, message="Calculate 15 * 7 using your calculator tool", max_turns=3)

    assert len(user_proxy.chat_messages[math_agent]) > 0


@pytest.mark.asyncio
@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
async def test_agent_a_run_method_single_async_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Async integration test: a_run() method with single agent."""
    assistant = AssistantAgent(
        "assistant",
        llm_config=credentials_gpt_4o_mini.llm_config,
    )

    message = [{"content": "What is the capital of France?", "role": "user"}]
    response = await assistant.a_run(message=message, max_turns=1)
    await response.process()

    assert response.summary is not None or len(response.messages) > 0
    assert response.last_speaker is not None


@pytest.mark.asyncio
@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
async def test_agent_a_run_method_two_agents_async_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Async integration test: a_run() method with two agents."""
    math_agent = ConversableAgent(
        "math_agent",
        llm_config=credentials_gpt_4o_mini.llm_config,
        system_message="You are a math expert.",
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    user_proxy.register_function(function_map={"calculator": calculator})

    message = [{"content": "Calculate 10 + 5", "role": "user"}]
    response = await user_proxy.a_run(recipient=math_agent, message=message, max_turns=3)

    # Consume events
    async for event in response.events:
        pass

    assert len(await response.messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    # test_trigger()
    # test_context()
    # test_handle_carryover():
    # test_max_turn()
    # test_process_before_send()
    # test_message_func()
    # test_summary()
    # test_adding_duplicate_function_warning()
    # test_function_registration_e2e_sync()
    # test_process_gemini_carryover()
    # test_process_carryover()
    # test_context_variables()
    # test_max_consecutive_auto_reply_with_max_turns()
    test_invalid_functions_parameter()
