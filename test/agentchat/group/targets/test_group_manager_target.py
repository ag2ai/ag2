# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.speaker_selection_result import SpeakerSelectionResult
from autogen.agentchat.group.targets.group_manager_target import (
    GroupManagerSelectionMessage,
    GroupManagerSelectionMessageContextStr,
    GroupManagerSelectionMessageString,
    GroupManagerTarget,
    prepare_groupchat_auto_speaker,
)
from autogen.agentchat.group.targets.transition_utils import __AGENT_WRAPPER_PREFIX__
from autogen.agentchat.groupchat import SELECT_SPEAKER_PROMPT_TEMPLATE, GroupChat


class TestPrepareGroupchatAutoSpeaker:
    @pytest.fixture
    def mock_groupchat(self) -> MagicMock:
        """Create a mock GroupChat for testing."""
        groupchat = MagicMock(spec=GroupChat)
        groupchat.select_speaker_prompt.return_value = (
            "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
        )
        return groupchat

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock ConversableAgent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "MockAgent"
        return agent

    @pytest.fixture
    def mock_tool_executor(self) -> MagicMock:
        """Create a mock GroupToolExecutor for testing."""
        executor = MagicMock(spec=GroupToolExecutor)
        executor.name = "ToolExecutor"
        return executor

    @pytest.fixture
    def mock_wrapped_agent(self) -> MagicMock:
        """Create a mock wrapped agent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = f"{__AGENT_WRAPPER_PREFIX__}wrapped_agent"
        return agent

    def test_prepare_groupchat_auto_speaker_with_default_message(
        self, mock_groupchat: MagicMock, mock_agent: MagicMock
    ) -> None:
        """Test prepare_groupchat_auto_speaker with the default selection message."""
        # Setup agents in the groupchat
        mock_groupchat.agents = [mock_agent]

        # Call the function with no custom message
        prepare_groupchat_auto_speaker(mock_groupchat, mock_agent, None)

        # Verify that the default template was used
        assert mock_groupchat.select_speaker_prompt_template == SELECT_SPEAKER_PROMPT_TEMPLATE
        mock_groupchat.select_speaker_prompt.assert_called_once_with([mock_agent])

    def test_prepare_groupchat_auto_speaker_with_custom_message(
        self, mock_groupchat: MagicMock, mock_agent: MagicMock
    ) -> None:
        """Test prepare_groupchat_auto_speaker with a custom selection message."""
        # Setup agents in the groupchat
        mock_groupchat.agents = [mock_agent]

        # Create a mock selection message
        mock_selection_msg = MagicMock()
        mock_selection_msg.get_message.return_value = "Custom selection prompt"

        # Call the function with the custom message
        prepare_groupchat_auto_speaker(mock_groupchat, mock_agent, mock_selection_msg)

        # Verify that the custom message was used
        mock_selection_msg.get_message.assert_called_once_with(mock_agent)
        mock_groupchat.select_speaker_prompt.assert_called_once_with([mock_agent])

    def test_prepare_groupchat_auto_speaker_filters_agents(
        self,
        mock_groupchat: MagicMock,
        mock_agent: MagicMock,
        mock_tool_executor: MagicMock,
        mock_wrapped_agent: MagicMock,
    ) -> None:
        """Test prepare_groupchat_auto_speaker filters out tool executors and wrapped agents."""
        # Setup agents in the groupchat
        mock_groupchat.agents = [mock_agent, mock_tool_executor, mock_wrapped_agent]

        # Call the function
        prepare_groupchat_auto_speaker(mock_groupchat, mock_agent, None)

        # Verify that only the regular agent was passed to select_speaker_prompt
        mock_groupchat.select_speaker_prompt.assert_called_once()
        agents_list = mock_groupchat.select_speaker_prompt.call_args[0][0]
        assert len(agents_list) == 1
        assert agents_list[0] == mock_agent
        assert mock_tool_executor not in agents_list
        assert mock_wrapped_agent not in agents_list


class TestGroupManagerSelectionMessages:
    def test_base_class_get_message_raises_not_implemented(self) -> None:
        """Test that the base GroupManagerSelectionMessage's get_message raises NotImplementedError."""
        base_message = GroupManagerSelectionMessage()
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            base_message.get_message(mock_agent)
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_string_message_get_message(self) -> None:
        """Test that GroupManagerSelectionMessageString's get_message returns the message string."""
        message_text = "This is a test message"
        string_message = GroupManagerSelectionMessageString(message=message_text)
        mock_agent = MagicMock(spec=ConversableAgent)

        result = string_message.get_message(mock_agent)
        assert result == message_text

    def test_context_str_message_get_message(self) -> None:
        """Test that GroupManagerSelectionMessageContextStr's get_message formats the template."""
        template = "Hello, {name}!"
        context_str_message = GroupManagerSelectionMessageContextStr(context_str_template=template)

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables(data={"name": "World"})

        result = context_str_message.get_message(mock_agent)
        assert result == "Hello, World!"

    def test_context_str_message_with_agentlist_placeholder(self) -> None:
        """Test that GroupManagerSelectionMessageContextStr handles {agentlist} placeholder correctly."""
        template = "Please select an agent from {agentlist} based on {criteria}"
        context_str_message = GroupManagerSelectionMessageContextStr(context_str_template=template)

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables(data={"criteria": "expertise"})

        # Check that the agentlist placeholder was replaced and then restored
        result = context_str_message.get_message(mock_agent)
        assert result == "Please select an agent from {agentlist} based on expertise"
        assert "{agentlist}" in result  # The placeholder should be restored for later substitution

    def test_context_str_message_validator(self) -> None:
        """Test that the field validator replaces {agentlist} with a temporary placeholder."""
        # This directly tests the _replace_agentlist_placeholder validator
        template = "Select from {agentlist}"
        message = GroupManagerSelectionMessageContextStr(context_str_template=template)

        # The template should have been modified internally
        assert message.context_str_template == "Select from <<agent_list>>"

        # But the get_message should restore it
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables()
        result = message.get_message(mock_agent)

        assert result == "Select from {agentlist}"

    def test_context_str_message_with_missing_variables(self) -> None:
        """Test GroupManagerSelectionMessageContextStr with missing context variables."""
        template = "Hello, ${missing_var}!"
        context_str_message = GroupManagerSelectionMessageContextStr(context_str_template=template)

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.context_variables = ContextVariables()

        # With missing variables, it should just return the template as is
        result = context_str_message.get_message(mock_agent)
        assert result == template


class TestGroupManagerTarget:
    @pytest.fixture
    def mock_selection_message(self) -> MagicMock:
        """Create a mock selection message for testing."""
        message = MagicMock(spec=GroupManagerSelectionMessage)
        message.get_message.return_value = "Custom selection message"
        return message

    def test_init(self) -> None:
        """Test initialization of GroupManagerTarget with and without a selection message."""
        # Without selection message
        target = GroupManagerTarget()
        assert target.selection_message is None

        # With selection message
        mock_message = MagicMock(spec=GroupManagerSelectionMessage)
        target_with_message = GroupManagerTarget(selection_message=mock_message)
        assert target_with_message.selection_message is mock_message

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = GroupManagerTarget()
        assert target.can_resolve_for_speaker_selection() is True

    @patch("autogen.agentchat.group.targets.group_manager_target.prepare_groupchat_auto_speaker")
    def test_resolve_without_selection_message(self, mock_prepare: MagicMock) -> None:
        """Test resolve without a selection message."""
        target = GroupManagerTarget()

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        mock_user_agent = MagicMock(spec=ConversableAgent)

        result = target.resolve(mock_groupchat, mock_agent, mock_user_agent)

        # It should not call prepare_groupchat_auto_speaker
        mock_prepare.assert_not_called()

        # It should return a SpeakerSelectionResult with "auto"
        assert isinstance(result, SpeakerSelectionResult)
        assert result.speaker_selection_method == "auto"
        assert result.agent_name is None
        assert result.terminate is None

    @patch("autogen.agentchat.group.targets.group_manager_target.prepare_groupchat_auto_speaker")
    def test_resolve_with_selection_message(self, mock_prepare: MagicMock, mock_selection_message: MagicMock) -> None:
        """Test resolve with a selection message."""
        target = GroupManagerTarget(selection_message=mock_selection_message)

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        mock_user_agent = MagicMock(spec=ConversableAgent)

        result = target.resolve(mock_groupchat, mock_agent, mock_user_agent)

        # It should call prepare_groupchat_auto_speaker with the selection message
        mock_prepare.assert_called_once_with(mock_groupchat, mock_agent, mock_selection_message)

        # It should return a SpeakerSelectionResult with "auto"
        assert isinstance(result, SpeakerSelectionResult)
        assert result.speaker_selection_method == "auto"

    def test_display_name(self) -> None:
        """Test that display_name returns 'the group manager'."""
        target = GroupManagerTarget()
        assert target.display_name() == "the group manager"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the display_name."""
        target = GroupManagerTarget()
        assert target.normalized_name() == "the group manager"

    def test_str_representation(self) -> None:
        """Test the string representation of GroupManagerTarget."""
        target = GroupManagerTarget()
        assert str(target) == "Transfer to the group manager"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = GroupManagerTarget()
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = GroupManagerTarget()
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "GroupManagerTarget does not require wrapping" in str(excinfo.value)
