# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.patterns.pattern import DefaultPattern
from autogen.agentchat.group.targets.transition_target import TerminateTarget


class TestDefaultPattern:
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

    def test_init(self, mock_initial_agent: MagicMock, mock_agent: MagicMock) -> None:
        """Test initialization."""
        agents = [mock_agent]

        # Create pattern with minimal parameters
        pattern = DefaultPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Check default's class parameters
        assert pattern.initial_agent is mock_initial_agent
        assert pattern.agents == agents
        assert pattern.user_agent is None
        assert pattern.group_manager_args == {}
        assert isinstance(pattern.context_variables, ContextVariables)
        assert isinstance(pattern.group_after_work, TerminateTarget)
        assert pattern.exclude_transit_message is True
        assert pattern.summary_method == "last_msg"

    @patch("autogen.agentchat.group.patterns.pattern.Pattern.prepare_group_chat")
    def test_prepare_group_chat(
        self,
        mock_super_prepare: MagicMock,
        mock_initial_agent: MagicMock,
        mock_agent: MagicMock,
        context_variables: ContextVariables,
    ) -> None:
        """Test the prepare_group_chat method of DefaultPattern."""
        # Setup
        agents = [mock_agent]

        # Mock return values from super().prepare_group_chat
        mock_super_prepare.return_value = (
            agents,  # agents
            [MagicMock(name="wrapped_agent")],  # wrapped_agents
            MagicMock(name="user_agent"),  # user_agent
            context_variables,  # context_variables
            mock_initial_agent,  # initial_agent
            MagicMock(spec=TerminateTarget),  # group_after_work (ignored by DefaultPattern)
            MagicMock(name="tool_executor"),  # tool_executor
            MagicMock(name="groupchat"),  # groupchat
            MagicMock(name="manager"),  # manager
            [{"role": "user", "content": "Hello"}],  # processed_messages
            mock_agent,  # last_agent
            ["mock_agent"],  # group_agent_names
            [],  # temp_user_list
        )

        # Create pattern
        pattern = DefaultPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Call the method
        result = pattern.prepare_group_chat(max_rounds=10, messages="Hello")

        # Check super class method was called
        mock_super_prepare.assert_called_once_with(max_rounds=10, messages="Hello")

        # Check the returned tuple
        assert len(result) == 13

        # Most important check: verify that group_after_work (item 5) is the pattern's group_after_work
        # and not what was returned by the super method
        assert result[5] is pattern.group_after_work
