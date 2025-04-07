# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.patterns.round_robin import RoundRobinPattern
from autogen.agentchat.group.targets.transition_target import AgentTarget, TerminateTarget, TransitionTarget


class TestRoundRobinPattern:
    @pytest.fixture
    def mock_agent1(self) -> MagicMock:
        """Create a mock ConversableAgent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "agent1"
        agent._function_map = {}
        agent.handoffs = MagicMock()
        agent.handoffs.llm_conditions = []
        agent.handoffs.context_conditions = []
        agent.handoffs.after_work = None
        agent._group_is_established = False
        return agent

    @pytest.fixture
    def mock_agent2(self) -> MagicMock:
        """Create another mock ConversableAgent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "agent2"
        agent._function_map = {}
        agent.handoffs = MagicMock()
        agent.handoffs.llm_conditions = []
        agent.handoffs.context_conditions = []
        agent.handoffs.after_work = None
        agent._group_is_established = False
        return agent

    @pytest.fixture
    def mock_agent3(self) -> MagicMock:
        """Create a third mock ConversableAgent for testing."""
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "agent3"
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

    def test_init(self, mock_initial_agent: MagicMock, mock_agent1: MagicMock) -> None:
        """Test initialization."""
        agents = [mock_agent1]

        # Create pattern
        pattern = RoundRobinPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Check base class parameters
        assert pattern.initial_agent is mock_initial_agent
        assert pattern.agents == agents
        assert pattern.user_agent is None
        assert pattern.group_manager_args == {}
        assert isinstance(pattern.context_variables, ContextVariables)
        assert isinstance(pattern.group_after_work, TerminateTarget)
        assert pattern.exclude_transit_message is True
        assert pattern.summary_method == "last_msg"

    def test_generate_handoffs_with_single_agent(self, mock_initial_agent: MagicMock) -> None:
        """Test the _generate_handoffs method with a single agent."""
        # Setup
        agents = [mock_initial_agent]
        user_agent = None

        # Create pattern
        pattern = RoundRobinPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Call the method
        pattern._generate_handoffs(mock_initial_agent, cast(list[ConversableAgent], agents), user_agent)

        # The initial agent should handoff to itself (only one agent)
        mock_initial_agent.handoffs.set_after_work.assert_called_once()
        args = mock_initial_agent.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_initial_agent.name

    def test_generate_handoffs_with_multiple_agents(
        self, mock_initial_agent: MagicMock, mock_agent1: MagicMock, mock_agent2: MagicMock
    ) -> None:
        """Test the _generate_handoffs method with multiple agents."""
        # Setup
        agents = [mock_initial_agent, mock_agent1, mock_agent2]
        user_agent = None

        # Create pattern
        pattern = RoundRobinPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Call the method
        pattern._generate_handoffs(mock_initial_agent, cast(list[ConversableAgent], agents), user_agent)

        # Check that each agent hands off to the next in sequence
        # initial_agent -> agent1
        mock_initial_agent.handoffs.set_after_work.assert_called_once()
        args = mock_initial_agent.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_agent1.name

        # agent1 -> agent2
        mock_agent1.handoffs.set_after_work.assert_called_once()
        args = mock_agent1.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_agent2.name

        # agent2 -> initial_agent (loop back to the beginning)
        mock_agent2.handoffs.set_after_work.assert_called_once()
        args = mock_agent2.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_initial_agent.name

    def test_generate_handoffs_with_user_agent(
        self, mock_initial_agent: MagicMock, mock_agent1: MagicMock, mock_user_agent: MagicMock
    ) -> None:
        """Test the _generate_handoffs method with a user agent."""
        # Setup
        agents = [mock_initial_agent, mock_agent1]

        # Create pattern
        pattern = RoundRobinPattern(
            initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents), user_agent=mock_user_agent
        )

        # Call the method
        pattern._generate_handoffs(mock_initial_agent, cast(list[ConversableAgent], agents), mock_user_agent)

        # Check that handoffs form a cycle: initial_agent -> agent1 -> user_agent -> initial_agent
        # initial_agent -> agent1
        mock_initial_agent.handoffs.set_after_work.assert_called_once()
        args = mock_initial_agent.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_agent1.name

        # agent1 -> user_agent
        mock_agent1.handoffs.set_after_work.assert_called_once()
        args = mock_agent1.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_user_agent.name

        # user_agent -> initial_agent
        mock_user_agent.handoffs.set_after_work.assert_called_once()
        args = mock_user_agent.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_initial_agent.name

    def test_generate_handoffs_with_initial_agent_not_first(
        self, mock_initial_agent: MagicMock, mock_agent1: MagicMock, mock_agent2: MagicMock
    ) -> None:
        """Test the _generate_handoffs method when the initial agent is not the first in the list."""
        # Setup - initial_agent in the middle
        agents = [mock_agent1, mock_initial_agent, mock_agent2]
        user_agent = None

        # Create pattern
        pattern = RoundRobinPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Call the method
        pattern._generate_handoffs(mock_initial_agent, cast(list[ConversableAgent], agents), user_agent)

        # According to RoundRobinPattern implementation, it should reorder the agents to:
        # [initial_agent, agent1, agent2] when creating the handoff chain
        # The handoff chain should be: initial_agent -> agent1 -> agent2 -> initial_agent

        # initial_agent -> agent1
        mock_initial_agent.handoffs.set_after_work.assert_called_once()
        args = mock_initial_agent.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_agent1.name

        # agent1 -> agent2
        mock_agent1.handoffs.set_after_work.assert_called_once()
        args = mock_agent1.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_agent2.name

        # agent2 -> initial_agent
        mock_agent2.handoffs.set_after_work.assert_called_once()
        args = mock_agent2.handoffs.set_after_work.call_args[1]
        assert "target" in args
        assert isinstance(args["target"], AgentTarget)
        assert args["target"].agent_name == mock_initial_agent.name

    @patch("autogen.agentchat.group.patterns.round_robin.Pattern.prepare_group_chat")
    def test_prepare_group_chat(
        self,
        mock_super_prepare: MagicMock,
        mock_initial_agent: MagicMock,
        mock_agent1: MagicMock,
        mock_agent2: MagicMock,
        context_variables: ContextVariables,
    ) -> None:
        """Test the prepare_group_chat method."""
        # Setup
        agents = [mock_initial_agent, mock_agent1, mock_agent2]
        user_agent = None

        # Mock return values from super().prepare_group_chat
        mock_super_prepare.return_value = (
            agents,  # agents
            [MagicMock(name="wrapped_agent")],  # wrapped_agents
            user_agent,  # user_agent
            context_variables,  # context_variables
            mock_initial_agent,  # initial_agent
            MagicMock(spec=TransitionTarget),  # group_after_work
            MagicMock(name="tool_executor"),  # tool_executor
            MagicMock(name="groupchat"),  # groupchat
            MagicMock(name="manager"),  # manager
            [{"role": "user", "content": "Hello"}],  # processed_messages
            mock_agent1,  # last_agent
            ["initial_agent", "agent1", "agent2"],  # group_agent_names
            [],  # temp_user_list
        )

        # Create pattern
        pattern = RoundRobinPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Mock _generate_handoffs
        with patch.object(pattern, "_generate_handoffs") as mock_generate_handoffs:
            # Call the method
            result = pattern.prepare_group_chat(max_rounds=10, messages="Hello")

            # Check super class method was called
            mock_super_prepare.assert_called_once_with(max_rounds=10, messages="Hello")

            # Check _generate_handoffs was called with correct args
            mock_generate_handoffs.assert_called_once_with(
                initial_agent=mock_initial_agent, agents=agents, user_agent=user_agent
            )

            # Check the returned tuple
            assert len(result) == 13

            # Verify the group_after_work value is passed through (not changed by RoundRobinPattern)
            assert result[5] is mock_super_prepare.return_value[5]

    @patch("autogen.agentchat.group.patterns.round_robin.Pattern.prepare_group_chat")
    def test_prepare_group_chat_with_user_agent(
        self,
        mock_super_prepare: MagicMock,
        mock_initial_agent: MagicMock,
        mock_agent1: MagicMock,
        mock_user_agent: MagicMock,
        context_variables: ContextVariables,
    ) -> None:
        """Test the prepare_group_chat method with a user agent."""
        # Setup
        agents = [mock_initial_agent, mock_agent1]

        # Mock return values from super().prepare_group_chat
        mock_super_prepare.return_value = (
            agents,  # agents
            [MagicMock(name="wrapped_agent")],  # wrapped_agents
            mock_user_agent,  # user_agent
            context_variables,  # context_variables
            mock_initial_agent,  # initial_agent
            MagicMock(spec=TransitionTarget),  # group_after_work
            MagicMock(name="tool_executor"),  # tool_executor
            MagicMock(name="groupchat"),  # groupchat
            MagicMock(name="manager"),  # manager
            [{"role": "user", "content": "Hello"}],  # processed_messages
            mock_agent1,  # last_agent
            ["initial_agent", "agent1"],  # group_agent_names
            [],  # temp_user_list
        )

        # Create pattern
        pattern = RoundRobinPattern(
            initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents), user_agent=mock_user_agent
        )

        # Mock _generate_handoffs
        with patch.object(pattern, "_generate_handoffs") as mock_generate_handoffs:
            # Call the method
            result = pattern.prepare_group_chat(max_rounds=10, messages="Hello")

            # Check super class method was called
            mock_super_prepare.assert_called_once_with(max_rounds=10, messages="Hello")

            # Check _generate_handoffs was called with correct args
            mock_generate_handoffs.assert_called_once_with(
                initial_agent=mock_initial_agent, agents=agents, user_agent=mock_user_agent
            )

            # Check the returned tuple
            assert len(result) == 13
