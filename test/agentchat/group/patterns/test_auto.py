# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.patterns.auto import AutoPattern
from autogen.agentchat.group.targets.group_manager_target import GroupManagerSelectionMessage, GroupManagerTarget


class TestAutoPattern:
    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock ConversableAgent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "mock_agent"
        agent._function_map = {}
        agent.handoffs = MagicMock()
        agent.handoffs.llm_conditions = []
        agent.handoffs.context_conditions = []
        agent.handoffs.after_work = None
        agent._group_is_established = False
        agent.description = "Mock agent description"
        agent.llm_config = {"config_list": [{"model": "test-model"}]}
        return agent

    @pytest.fixture
    def mock_initial_agent(self) -> MagicMock:
        """Create a mock initial agent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "initial_agent"
        agent._function_map = {}
        agent.handoffs = MagicMock()
        agent.handoffs.llm_conditions = []
        agent.handoffs.context_conditions = []
        agent.handoffs.after_work = None
        agent._group_is_established = False
        agent.description = "Initial agent description"
        agent.llm_config = {"config_list": [{"model": "test-model"}]}
        return agent

    @pytest.fixture
    def mock_user_agent(self) -> MagicMock:
        """Create a mock user agent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "user_agent"
        agent._function_map = {}
        agent.handoffs = MagicMock()
        agent.handoffs.llm_conditions = []
        agent.handoffs.context_conditions = []
        agent.handoffs.after_work = None
        return agent

    @pytest.fixture
    def context_variables(self) -> ContextVariables:
        """Create context variables for testing."""
        return ContextVariables(data={"test_key": "test_value"})

    @pytest.fixture
    def mock_selection_message(self) -> MagicMock:
        """Create a mock selection message for testing."""
        return MagicMock(spec=GroupManagerSelectionMessage)

    def test_init_with_minimal_params(self, mock_initial_agent: MagicMock, mock_agent: MagicMock) -> None:
        """Test initialization with minimal parameters."""
        agents = [mock_agent]

        # Create pattern
        pattern = AutoPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Check that group_after_work is a GroupManagerTarget
        assert isinstance(pattern.group_after_work, GroupManagerTarget)
        assert pattern.group_after_work.selection_message is None

        # Check that selection_message attribute is None
        assert pattern.selection_message is None

        # Check base class parameters
        assert pattern.initial_agent is mock_initial_agent
        assert pattern.agents == agents
        assert pattern.user_agent is None
        assert pattern.group_manager_args == {}
        assert isinstance(pattern.context_variables, ContextVariables)
        assert pattern.exclude_transit_message is True
        assert pattern.summary_method == "last_msg"

    def test_init_with_selection_message(
        self, mock_initial_agent: MagicMock, mock_agent: MagicMock, mock_selection_message: MagicMock
    ) -> None:
        """Test initialization with a selection message."""
        agents = [mock_agent]

        # Create pattern
        pattern = AutoPattern(
            initial_agent=mock_initial_agent,
            agents=cast(list[ConversableAgent], agents),
            selection_message=mock_selection_message,
        )

        # Check that group_after_work is a GroupManagerTarget with the selection message
        assert isinstance(pattern.group_after_work, GroupManagerTarget)
        assert pattern.group_after_work.selection_message is mock_selection_message

        # Check that selection_message attribute is set
        assert pattern.selection_message is mock_selection_message

    def test_init_with_all_params(
        self,
        mock_initial_agent: MagicMock,
        mock_agent: MagicMock,
        mock_user_agent: MagicMock,
        context_variables: ContextVariables,
        mock_selection_message: MagicMock,
    ) -> None:
        """Test initialization with all parameters."""
        agents = [mock_agent]
        group_manager_args = {"llm_config": {"model": "gpt-4"}}
        summary_method = "reflection"

        # Create pattern
        pattern = AutoPattern(
            initial_agent=mock_initial_agent,
            agents=cast(list[ConversableAgent], agents),
            user_agent=mock_user_agent,
            group_manager_args=group_manager_args,
            context_variables=context_variables,
            selection_message=mock_selection_message,
            exclude_transit_message=False,
            summary_method=summary_method,
        )

        # Check that group_after_work is a GroupManagerTarget with the selection message
        assert isinstance(pattern.group_after_work, GroupManagerTarget)
        assert pattern.group_after_work.selection_message is mock_selection_message

        # Check that selection_message attribute is set
        assert pattern.selection_message is mock_selection_message

        # Check base class parameters
        assert pattern.initial_agent is mock_initial_agent
        assert pattern.agents == agents
        assert pattern.user_agent is mock_user_agent
        assert pattern.group_manager_args == group_manager_args
        assert pattern.context_variables is context_variables
        assert pattern.exclude_transit_message is False
        assert pattern.summary_method == summary_method

    @patch("autogen.agentchat.group.patterns.auto.Pattern.prepare_group_chat")
    def test_prepare_group_chat(
        self,
        mock_super_prepare: MagicMock,
        mock_initial_agent: MagicMock,
        mock_agent: MagicMock,
        context_variables: ContextVariables,
    ) -> None:
        """Test the prepare_group_chat method."""
        # Setup
        agents = [mock_agent]

        # Mock return values from super().prepare_group_chat
        mock_super_prepare.return_value = (
            agents,  # agents
            [MagicMock(name="wrapped_agent")],  # wrapped_agents
            MagicMock(name="user_agent"),  # user_agent
            context_variables,  # context_variables
            mock_initial_agent,  # initial_agent
            MagicMock(),  # group_after_work (should be replaced)
            MagicMock(name="tool_executor"),  # tool_executor
            MagicMock(name="groupchat"),  # groupchat
            MagicMock(name="manager"),  # manager
            [{"role": "user", "content": "Hello"}],  # processed_messages
            mock_agent,  # last_agent
            ["mock_agent"],  # group_agent_names
            [],  # temp_user_list
        )

        # Create pattern
        pattern = AutoPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Call the method
        result = pattern.prepare_group_chat(max_rounds=10, messages="Hello")

        # Check super class method was called
        mock_super_prepare.assert_called_once_with(max_rounds=10, messages="Hello")

        # Check the returned tuple
        assert len(result) == 13

        # Verify that group_after_work (item 5) is the pattern's group_after_work
        assert result[5] is pattern.group_after_work
        assert isinstance(result[5], GroupManagerTarget)

    def test_prepare_group_chat_with_no_llm_config(self, mock_initial_agent: MagicMock, mock_agent: MagicMock) -> None:
        """Test prepare_group_chat with no LLM config raises ValueError."""
        # Setup agents with no LLM config
        mock_agent.llm_config = False
        mock_initial_agent.llm_config = False
        agents = [mock_agent]

        # Create pattern with empty group_manager_args
        pattern = AutoPattern(
            initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents), group_manager_args={}
        )

        # Call the method - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            pattern.prepare_group_chat(max_rounds=10, messages="Hello")

        assert "AutoPattern requires the group_manager_args to include an llm_config" in str(excinfo.value)

    def test_prepare_group_chat_with_llm_config_in_group_manager_args(
        self, mock_initial_agent: MagicMock, mock_agent: MagicMock
    ) -> None:
        """Test prepare_group_chat with LLM config in group_manager_args works."""
        # Setup agents with no LLM config
        mock_agent.llm_config = False
        mock_initial_agent.llm_config = False
        agents = [mock_agent]

        # Create pattern with LLM config in group_manager_args
        pattern = AutoPattern(
            initial_agent=mock_initial_agent,
            agents=cast(list[ConversableAgent], agents),
            group_manager_args={"llm_config": {"model": "gpt-4"}},
        )

        # Mock super().prepare_group_chat to avoid actually calling it
        with patch("autogen.agentchat.group.patterns.auto.Pattern.prepare_group_chat") as mock_super:
            mock_super.return_value = (agents, [], None, None, None, None, None, None, None, [], None, [], [])

            # Call the method - should not raise ValueError
            pattern.prepare_group_chat(max_rounds=10, messages="Hello")

            # Check super class method was called
            mock_super.assert_called_once()

    def test_check_agent_descriptions(self, mock_initial_agent: MagicMock) -> None:
        """Test that agent descriptions are set if missing."""
        # Create agents with missing descriptions
        agent1 = MagicMock(spec=ConversableAgent)
        agent1.name = "agent1"
        agent1.description = None
        agent1.llm_config = {"config_list": [{"model": "test-model"}]}  # Add llm_config

        agent2 = MagicMock(spec=ConversableAgent)
        agent2.name = "agent2"
        agent2.description = "Existing description"
        agent2.llm_config = False  # No llm_config for this agent

        agents = [agent1, agent2]

        # Create pattern
        pattern = AutoPattern(
            initial_agent=mock_initial_agent,
            agents=cast(list[ConversableAgent], agents),
            # Could also set group_manager_args={"llm_config": {"model": "test-model"}} instead
        )

        # Mock super().prepare_group_chat to avoid actually calling it
        with patch("autogen.agentchat.group.patterns.auto.Pattern.prepare_group_chat") as mock_super:
            mock_super.return_value = (agents, [], None, None, None, None, None, None, None, [], None, [], [])

            # Call the method
            pattern.prepare_group_chat(max_rounds=10, messages="Hello")

            # Check that agent1 description was set
            assert agent1.description == "Agent agent1"

            # Check that agent2 description was not changed
            assert agent2.description == "Existing description"
