# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.reply_result import ReplyResult
from autogen.agentchat.group.transition_target import AfterWorkOptionTarget, AgentTarget, TransitionTarget


class TestReplyResult:
    def test_init_with_message_only(self) -> None:
        """Test initialisation with only a message."""
        message = "This is a test message"
        reply_result = ReplyResult(message=message)

        assert reply_result.message == message
        assert reply_result.target is None
        assert reply_result.context_variables is None

    def test_init_with_message_and_target(self) -> None:
        """Test initialisation with message and target."""
        message = "This is a test message"
        target = MagicMock(spec=TransitionTarget)

        reply_result = ReplyResult(message=message, target=target)

        assert reply_result.message == message
        assert reply_result.target == target
        assert reply_result.context_variables is None

    def test_init_with_message_and_context_variables(self) -> None:
        """Test initialisation with message and context variables."""
        message = "This is a test message"
        context_variables = ContextVariables(data={"key": "value"})

        reply_result = ReplyResult(message=message, context_variables=context_variables)

        assert reply_result.message == message
        assert reply_result.target is None
        assert reply_result.context_variables == context_variables
        assert reply_result.context_variables.get("key") == "value"

    def test_init_with_all_parameters(self) -> None:
        """Test initialisation with all parameters."""
        message = "This is a test message"
        target = MagicMock(spec=TransitionTarget)
        context_variables = ContextVariables(data={"key": "value"})

        reply_result = ReplyResult(message=message, target=target, context_variables=context_variables)

        assert reply_result.message == message
        assert reply_result.target == target
        assert reply_result.context_variables == context_variables
        assert reply_result.context_variables.get("key") == "value"

    def test_with_agent_target(self) -> None:
        """Test with AgentTarget."""
        message = "This is a test message"
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)

        reply_result = ReplyResult(message=message, target=target)

        assert reply_result.message == message
        assert reply_result.target == target
        assert reply_result.target.agent_name == "test_agent"

    def test_with_after_work_option_target(self) -> None:
        """Test with AfterWorkOptionTarget."""
        message = "This is a test message"
        target = AfterWorkOptionTarget(after_work_option="terminate")

        reply_result = ReplyResult(message=message, target=target)

        assert reply_result.message == message
        assert reply_result.target == target
        assert reply_result.target.after_work_option == "terminate"

    def test_with_empty_context_variables(self) -> None:
        """Test with empty ContextVariables."""
        message = "This is a test message"
        context_variables = ContextVariables()

        reply_result = ReplyResult(message=message, context_variables=context_variables)

        assert reply_result.message == message
        assert reply_result.context_variables == context_variables
        assert len(reply_result.context_variables.to_dict()) == 0

    def test_with_multiple_context_variables(self) -> None:
        """Test with multiple ContextVariables."""
        message = "This is a test message"
        context_variables = ContextVariables(
            data={"name": "John", "age": 30, "is_admin": True, "preferences": ["apples", "bananas"]}
        )

        reply_result = ReplyResult(message=message, context_variables=context_variables)

        assert reply_result.message == message
        assert reply_result.context_variables == context_variables
        assert reply_result.context_variables.get("name") == "John"
        assert reply_result.context_variables.get("age") == 30
        assert reply_result.context_variables.get("is_admin") is True
        assert reply_result.context_variables.get("preferences") == ["apples", "bananas"]

    def test_string_representation(self) -> None:
        """Test string representation."""
        message = "This is a test message"
        reply_result = ReplyResult(message=message)

        str_result = str(reply_result)
        assert isinstance(str_result, str)
        assert "ReplyResult" in str_result
        assert "message='" in str_result
        assert message in str_result
