# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

import pytest
import termcolor.termcolor

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.coding.base import CodeBlock
from autogen.events.agent_events import (
    ClearAgentsHistoryEvent,
    ClearConversableAgentHistoryEvent,
    ClearConversableAgentHistoryWarningEvent,
    ConversableAgentUsageSummaryEvent,
    ConversableAgentUsageSummaryNoCostIncurredEvent,
    ExecuteCodeBlockEvent,
    ExecuteFunctionEvent,
    ExecutedFunctionEvent,
    FunctionCallEvent,
    FunctionResponseEvent,
    GenerateCodeExecutionReplyEvent,
    GroupChatResumeEvent,
    GroupChatRunChatEvent,
    PostCarryoverProcessingEvent,
    SelectSpeakerEvent,
    SelectSpeakerInvalidInputEvent,
    SelectSpeakerTryCountExceededEvent,
    SpeakerAttemptFailedMultipleAgentsEvent,
    SpeakerAttemptFailedNoAgentsEvent,
    SpeakerAttemptSuccessfulEvent,
    TerminationAndHumanReplyNoInputEvent,
    TerminationEvent,
    TextEvent,
    ToolCallEvent,
    ToolResponseEvent,
    UsingAutoReplyEvent,
)
from autogen.import_utils import optional_import_block
from autogen.messages.agent_messages import (
    ClearAgentsHistoryMessage,
    ClearConversableAgentHistoryMessage,
    ClearConversableAgentHistoryWarningMessage,
    ConversableAgentUsageSummaryMessage,
    ConversableAgentUsageSummaryNoCostIncurredMessage,
    ExecuteCodeBlockMessage,
    ExecuteFunctionMessage,
    ExecutedFunctionMessage,
    GenerateCodeExecutionReplyMessage,
    GroupChatResumeMessage,
    GroupChatRunChatMessage,
    PostCarryoverProcessingMessage,
    SelectSpeakerInvalidInputMessage,
    SelectSpeakerMessage,
    SelectSpeakerTryCountExceededMessage,
    SpeakerAttemptFailedMultipleAgentsMessage,
    SpeakerAttemptFailedNoAgentsMessage,
    SpeakerAttemptSuccessfulMessage,
    TerminationAndHumanReplyNoInputMessage,
    TerminationMessage,
    UsingAutoReplyMessage,
    create_received_message_model,
)

with optional_import_block():
    pass


@pytest.fixture(autouse=True)
def enable_color_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_can_do_colour(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(termcolor.termcolor, "_can_do_colour", mock_can_do_colour)


@pytest.fixture
def sender() -> ConversableAgent:
    return ConversableAgent("sender", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")


@pytest.fixture
def recipient() -> ConversableAgent:
    return ConversableAgent("recipient", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")


class TestToolResponseMessage:
    def test_print(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        message = {
            "role": "tool",
            "tool_responses": [
                {"tool_call_id": "call_rJfVpHU3MXuPRR2OAdssVqUV", "role": "tool", "content": "Timer is done!"},
                {"tool_call_id": "call_zFZVYovdsklFYgqxttcOHwlr", "role": "tool", "content": "Stopwatch is done!"},
            ],
            "content": "Timer is done!\\n\\nStopwatch is done!",
        }
        actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)
        assert isinstance(actual, ToolResponseEvent)


class TestFunctionResponseMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        message = {"name": "get_random_number", "role": "function", "content": "76"}
        actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)
        assert isinstance(actual, FunctionResponseEvent)


class TestFunctionCallMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        fc_message = {
            "content": "Let's play a game.",
            "function_call": {"name": "get_random_number", "arguments": "{}"},
        }

        message = create_received_message_model(uuid=uuid, message=fc_message, sender=sender, recipient=recipient)

        assert isinstance(message, FunctionCallEvent)


class TestToolCallMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        message = {
            "content": None,
            "refusal": None,
            "role": None,
            "audio": None,
            "function_call": None,
            "tool_calls": [
                {
                    "id": "call_rJfVpHU3MXuPRR2OAdssVqUV",
                    "function": {"arguments": '{"num_seconds": "1"}', "name": "timer"},
                    "type": "function",
                },
                {
                    "id": "call_zFZVYovdsklFYgqxttcOHwlr",
                    "function": {"arguments": '{"num_seconds": "2"}', "name": "stopwatch"},
                    "type": "function",
                },
            ],
        }

        actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)
        assert isinstance(actual, ToolCallEvent)


class TestTextMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        message = {
            "content": lambda context: f"hello {context['name']}",
            "context": {
                "name": "there",
            },
        }

        actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

        assert isinstance(actual, TextEvent)


class TestPostCarryoverProcessingMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        chat_info = {
            "carryover": ["This is a test message 1", "This is a test message 2"],
            "message": "Start chat",
            "verbose": True,
            "sender": sender,
            "recipient": recipient,
            "summary_method": "last_msg",
            "max_turns": 5,
        }

        actual = PostCarryoverProcessingMessage(uuid=uuid, chat_info=chat_info)
        assert isinstance(actual, PostCarryoverProcessingEvent)


class TestClearAgentsHistoryMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        actual = ClearAgentsHistoryMessage(uuid=uuid, agent=None, nr_messages_to_preserve=None)
        assert isinstance(actual, ClearAgentsHistoryEvent)


class TestSpeakerAttemptSuccessfulMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        attempt = 1
        attempts_left = 2
        verbose = True
        mentions = {"agent_1": 1}

        actual = SpeakerAttemptSuccessfulMessage(
            uuid=uuid,
            mentions=mentions,
            attempt=attempt,
            attempts_left=attempts_left,
            select_speaker_auto_verbose=verbose,
        )
        assert isinstance(actual, SpeakerAttemptSuccessfulEvent)


class TestSpeakerAttemptFailedMultipleAgentsMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        attempt = 1
        attempts_left = 2
        verbose = True
        mentions = {"agent_1": 1, "agent_2": 2}

        actual = SpeakerAttemptFailedMultipleAgentsMessage(
            uuid=uuid,
            mentions=mentions,
            attempt=attempt,
            attempts_left=attempts_left,
            select_speaker_auto_verbose=verbose,
        )
        assert isinstance(actual, SpeakerAttemptFailedMultipleAgentsEvent)


class TestSpeakerAttemptFailedNoAgentsMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        mentions: dict[str, int] = {}
        attempt = 1
        attempts_left = 2
        verbose = True

        actual = SpeakerAttemptFailedNoAgentsMessage(
            uuid=uuid,
            mentions=mentions,
            attempt=attempt,
            attempts_left=attempts_left,
            select_speaker_auto_verbose=verbose,
        )
        assert isinstance(actual, SpeakerAttemptFailedNoAgentsEvent)


class TestGroupChatResumeMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        last_speaker_name = "Coder"
        messages = [
            {"content": "You are an expert at coding.", "role": "system", "name": "chat_manager"},
            {"content": "Let's get coding, should I use Python?", "name": "Coder", "role": "assistant"},
        ]
        silent = False

        actual = GroupChatResumeMessage(
            uuid=uuid, last_speaker_name=last_speaker_name, messages=messages, silent=silent
        )
        assert isinstance(actual, GroupChatResumeEvent)


class TestGroupChatRunChatMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        speaker = ConversableAgent(
            "assistant_uno", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"
        )
        silent = False

        actual = GroupChatRunChatMessage(uuid=uuid, speaker=speaker, silent=silent)
        assert isinstance(actual, GroupChatRunChatEvent)


class TestTerminationAndHumanReplyMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        no_human_input_msg = "NO HUMAN INPUT RECEIVED."

        actual = TerminationAndHumanReplyNoInputMessage(
            uuid=uuid,
            no_human_input_msg=no_human_input_msg,
            sender=sender,
            recipient=recipient,
        )
        assert isinstance(actual, TerminationAndHumanReplyNoInputEvent)


class TestTerminationMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        termination_reason = "User requested to end the conversation."

        actual = TerminationMessage(
            uuid=uuid,
            termination_reason=termination_reason,
        )
        assert isinstance(actual, TerminationEvent)


class TestUsingAutoReplyMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        human_input_mode = "ALWAYS"

        actual = UsingAutoReplyMessage(
            uuid=uuid,
            human_input_mode=human_input_mode,
            sender=sender,
            recipient=recipient,
        )
        assert isinstance(actual, UsingAutoReplyEvent)


class TestExecuteCodeBlockMessage:
    def test_deprecated(self, uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
        code = """print("hello world")"""
        language = "python"
        code_block_count = 0

        actual = ExecuteCodeBlockMessage(
            uuid=uuid, code=code, language=language, code_block_count=code_block_count, recipient=recipient
        )
        assert isinstance(actual, ExecuteCodeBlockEvent)


class TestExecuteFunctionMessage:
    def test_deprecated(self, uuid: UUID, recipient: ConversableAgent) -> None:
        func_name = "add_num"
        call_id = "call_12345xyz"
        arguments = {"num_to_be_added": 5}

        actual = ExecuteFunctionMessage(
            uuid=uuid, func_name=func_name, call_id=call_id, arguments=arguments, recipient=recipient
        )
        assert isinstance(actual, ExecuteFunctionEvent)


class TestExecutedFunctionMessage:
    def test_deprecated(self, uuid: UUID, recipient: ConversableAgent) -> None:
        func_name = "add_num"
        call_id = "call_12345xyz"
        arguments = {"num_to_be_added": 5}
        content = "15"

        actual = ExecutedFunctionMessage(
            uuid=uuid, func_name=func_name, call_id=call_id, arguments=arguments, content=content, recipient=recipient
        )
        assert isinstance(actual, ExecutedFunctionEvent)


class TestSelectSpeakerMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        agents = [
            ConversableAgent("bob", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
            ConversableAgent("charlie", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
        ]

        actual = SelectSpeakerMessage(uuid=uuid, agents=agents)  # type: ignore [arg-type]
        assert isinstance(actual, SelectSpeakerEvent)


class TestSelectSpeakerTryCountExceededMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        agents = [
            ConversableAgent("bob", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
            ConversableAgent("charlie", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
        ]
        try_count = 3

        actual = SelectSpeakerTryCountExceededMessage(uuid=uuid, try_count=try_count, agents=agents)  # type: ignore [arg-type]
        assert isinstance(actual, SelectSpeakerTryCountExceededEvent)


class TestSelectSpeakerInvalidInputMessage:
    def test_deprecated(self, uuid: UUID) -> None:
        agents = [
            ConversableAgent("bob", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
            ConversableAgent("charlie", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
        ]

        actual = SelectSpeakerInvalidInputMessage(uuid=uuid, agents=agents)  # type: ignore [arg-type]
        assert isinstance(actual, SelectSpeakerInvalidInputEvent)


class TestClearConversableAgentHistoryMessage:
    def test_deprecated(self, uuid: UUID, recipient: ConversableAgent) -> None:
        no_messages_preserved = 5

        actual = ClearConversableAgentHistoryMessage(
            uuid=uuid, agent=recipient, no_messages_preserved=no_messages_preserved
        )
        assert isinstance(actual, ClearConversableAgentHistoryEvent)


class TestClearConversableAgentHistoryWarningMessage:
    def test_deprecated(self, uuid: UUID, recipient: ConversableAgent) -> None:
        actual = ClearConversableAgentHistoryWarningMessage(uuid=uuid, recipient=recipient)
        assert isinstance(actual, ClearConversableAgentHistoryWarningEvent)


class TestGenerateCodeExecutionReplyMessage:
    def test_deprecated(
        self,
        uuid: UUID,
        sender: ConversableAgent,
        recipient: ConversableAgent,
    ) -> None:
        actual = GenerateCodeExecutionReplyMessage(
            uuid=uuid,
            code_blocks=[
                CodeBlock(code="print('hello world')", language="python"),
            ],
            sender=sender,
            recipient=recipient,
        )
        assert isinstance(actual, GenerateCodeExecutionReplyEvent)


class TestConversableAgentUsageSummaryNoCostIncurredMessage:
    def test_deprecation(
        self,
        uuid: UUID,
        recipient: ConversableAgent,
    ) -> None:
        actual = ConversableAgentUsageSummaryNoCostIncurredMessage(uuid=uuid, recipient=recipient)
        assert isinstance(actual, ConversableAgentUsageSummaryNoCostIncurredEvent)


class TestConversableAgentUsageSummaryMessage:
    def test_deprecation(
        self,
        uuid: UUID,
        recipient: ConversableAgent,
    ) -> None:
        actual = ConversableAgentUsageSummaryMessage(uuid=uuid, recipient=recipient)
        assert isinstance(actual, ConversableAgentUsageSummaryEvent)
