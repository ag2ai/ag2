# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.after_work import (
    AfterWork,
    AfterWorkSelectionMessage,
    AfterWorkSelectionMessageContextStr,
    AfterWorkSelectionMessageString,
)
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.transition_target import (
    AfterWorkOptionTarget,
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
)


class TestAfterWorkSelectionMessage:
    def test_base_message_get_message(self) -> None:
        """Test that the base AfterWorkSelectionMessage class raises NotImplementedError when get_message is called."""
        message = AfterWorkSelectionMessage()
        with pytest.raises(NotImplementedError) as excinfo:
            message.get_message(None, [])
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestAfterWorkSelectionMessageString:
    def test_init(self) -> None:
        """Test initialisation with a string template."""
        template = "This is a test template"
        message = AfterWorkSelectionMessageString(template=template)
        assert message.template == template

    def test_get_message(self) -> None:
        """Test that get_message returns the template string."""
        template = "This is a test template"
        message = AfterWorkSelectionMessageString(template=template)
        result = message.get_message(None, [])
        assert result == template


class TestAfterWorkSelectionMessageContextStr:
    def test_init(self) -> None:
        """Test initialisation with a context string template."""
        template = "Hello, {name}!"
        message = AfterWorkSelectionMessageContextStr(context_str_template=template)
        assert message.context_str_template == template

    def test_get_message(self) -> None:
        """Test that get_message formats the template with context variables."""
        template = "Hello, {name}!"
        message = AfterWorkSelectionMessageContextStr(context_str_template=template)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"name": "World"})

        result = message.get_message(mock_agent, [])
        assert result == "Hello, World!"


class TestAfterWork:
    def test_init(self) -> None:
        """Test initialisation with different parameters."""
        # Test with target
        target = AfterWorkOptionTarget(after_work_option="terminate")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

        # Test with target and selection message
        message = AfterWorkSelectionMessageString(template="Test message")
        after_work = AfterWork(target=target, selection_message=message)
        assert after_work.target == target
        assert after_work.selection_message == message

    @patch("autogen.agentchat.group.after_work.ContextStr")
    def test_with_integrated_components(self, mock_context_str) -> None:
        """Test AfterWork with integrated components."""
        # Configure mock
        mock_context_str_instance = MagicMock()
        mock_context_str_instance.format.return_value = "Hello, Test Agent!"
        mock_context_str.return_value = mock_context_str_instance

        # Create components
        target = AfterWorkOptionTarget(after_work_option="terminate")
        message = AfterWorkSelectionMessageContextStr(context_str_template="Hello, {name}!")

        # Create AfterWork with components
        after_work = AfterWork(target=target, selection_message=message)

        # Create mock objects for resolve
        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"name": "Test Agent"})
        mock_groupchat = MagicMock()

        # Test resolve
        result = after_work.target.resolve(None, [], mock_groupchat)
        assert result == "terminate"

        # Test get_message
        message_result = after_work.selection_message.get_message(mock_agent, [])
        mock_context_str_instance.format.assert_called_once_with(mock_agent.context_variables)
        assert message_result == "Hello, Test Agent!"

    def test_init_with_option_target(self) -> None:
        """Test initialization with AfterWorkOptionTarget."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_agent_target(self) -> None:
        """Test initialization with AgentTarget."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_agent_name_target(self) -> None:
        """Test initialization with AgentNameTarget."""
        target = AgentNameTarget(agent_name="test_agent")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_nested_chat_target(self) -> None:
        """Test initialization with NestedChatTarget."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_selection_message(self) -> None:
        """Test initialization with target and selection message."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        message = AfterWorkSelectionMessageString(template="Test message")
        after_work = AfterWork(target=target, selection_message=message)
        assert after_work.target == target
        assert after_work.selection_message == message
