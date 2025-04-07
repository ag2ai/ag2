# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.patterns.pattern import DefaultPattern, Pattern
from autogen.agentchat.group.targets.transition_target import StayTarget, TerminateTarget, TransitionTarget

if TYPE_CHECKING:
    from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
    from autogen.agentchat.groupchat import GroupChat, GroupChatManager


# Create a concrete test subclass for testing Pattern's base functionality
class TestPatternImpl(Pattern):
    """Concrete implementation of Pattern for testing purposes."""

    def prepare_group_chat(
        self,
        max_rounds: int,
        messages: Union[list[dict[str, Any]], str],
    ) -> Tuple[
        list["ConversableAgent"],
        list["ConversableAgent"],
        Optional["ConversableAgent"],
        ContextVariables,
        "ConversableAgent",
        TransitionTarget,
        "GroupToolExecutor",
        "GroupChat",
        "GroupChatManager",
        list[dict[str, Any]],
        Any,
        list[str],
        list[Any],
    ]:
        """Concrete implementation that just calls the parent method."""
        return super().prepare_group_chat(max_rounds, messages)


class TestPattern:
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

    def test_init_with_minimal_params(self, mock_initial_agent: MagicMock, mock_agent: MagicMock) -> None:
        """Test initialization with minimal parameters."""
        agents = [mock_agent]
        pattern = TestPatternImpl(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))

        # Check required parameters
        assert pattern.initial_agent is mock_initial_agent
        assert pattern.agents == agents

        # Check defaults
        assert pattern.user_agent is None
        assert pattern.group_manager_args == {}
        assert isinstance(pattern.context_variables, ContextVariables)
        assert isinstance(pattern.group_after_work, TerminateTarget)
        assert pattern.exclude_transit_message is True
        assert pattern.summary_method == "last_msg"

    def test_init_with_all_params(
        self,
        mock_initial_agent: MagicMock,
        mock_agent: MagicMock,
        mock_user_agent: MagicMock,
        context_variables: ContextVariables,
    ) -> None:
        """Test initialization with all parameters."""
        agents = [mock_agent]
        group_manager_args = {"llm_config": {"model": "gpt-4"}}
        group_after_work = StayTarget()
        summary_method = "reflection"

        pattern = TestPatternImpl(
            initial_agent=mock_initial_agent,
            agents=cast(list[ConversableAgent], agents),
            user_agent=mock_user_agent,
            group_manager_args=group_manager_args,
            context_variables=context_variables,
            group_after_work=group_after_work,
            exclude_transit_message=False,
            summary_method=summary_method,
        )

        # Check all parameters are set correctly
        assert pattern.initial_agent is mock_initial_agent
        assert pattern.agents == agents
        assert pattern.user_agent is mock_user_agent
        assert pattern.group_manager_args == group_manager_args
        assert pattern.context_variables is context_variables
        assert pattern.group_after_work is group_after_work
        assert pattern.exclude_transit_message is False
        assert pattern.summary_method == summary_method

    @patch("autogen.agentchat.group.patterns.pattern.prepare_group_agents")
    @patch("autogen.agentchat.group.patterns.pattern.process_initial_messages")
    @patch("autogen.agentchat.group.patterns.pattern.create_group_transition")
    @patch("autogen.agentchat.group.patterns.pattern.create_group_manager")
    @patch("autogen.agentchat.group.patterns.pattern.setup_context_variables")
    @patch("autogen.agentchat.group.patterns.pattern.link_agents_to_group_manager")
    @patch("autogen.agentchat.groupchat.GroupChat")
    def test_prepare_group_chat(
        self,
        mock_group_chat: MagicMock,
        mock_link_agents: MagicMock,
        mock_setup_context: MagicMock,
        mock_create_manager: MagicMock,
        mock_create_transition: MagicMock,
        mock_process_messages: MagicMock,
        mock_prepare_agents: MagicMock,
        mock_initial_agent: MagicMock,
        mock_agent: MagicMock,
        mock_user_agent: MagicMock,
        context_variables: ContextVariables,
    ) -> None:
        """Test the prepare_group_chat method."""
        # Setup mocks
        agents = [mock_agent]
        group_after_work = StayTarget()

        # Mock return values
        mock_tool_executor = MagicMock(name="tool_executor")
        mock_wrapped_agents = [MagicMock(name="wrapped_agent")]
        mock_prepare_agents.return_value = (mock_tool_executor, mock_wrapped_agents)

        mock_processed_messages = [{"role": "user", "content": "Hello"}]
        mock_group_agent_names = ["mock_agent"]
        mock_temp_user_list: list[ConversableAgent] = []
        mock_last_agent = mock_agent
        mock_process_messages.return_value = (
            mock_processed_messages,
            mock_last_agent,
            mock_group_agent_names,
            mock_temp_user_list,
        )

        mock_transition_func = MagicMock(name="transition_func")
        mock_create_transition.return_value = mock_transition_func

        mock_groupchat_instance = MagicMock(name="groupchat")
        mock_group_chat.return_value = mock_groupchat_instance

        mock_manager = MagicMock(name="manager")
        mock_create_manager.return_value = mock_manager

        # Create pattern
        pattern = TestPatternImpl(
            initial_agent=mock_initial_agent,
            agents=cast(list[ConversableAgent], agents),
            user_agent=mock_user_agent,
            context_variables=context_variables,
            group_after_work=group_after_work,
        )

        # Call the method
        result = pattern.prepare_group_chat(max_rounds=10, messages="Hello")

        # Check method calls
        mock_prepare_agents.assert_called_once_with(agents, context_variables, True)
        mock_process_messages.assert_called_once()
        mock_create_transition.assert_called_once()
        mock_group_chat.assert_called_once()
        mock_create_manager.assert_called_once()
        mock_setup_context.assert_called_once()
        mock_link_agents.assert_called_once()

        # Check the returned tuple
        assert len(result) == 13
        assert result[0] == agents  # agents
        assert result[1] == mock_wrapped_agents  # wrapped_agents
        assert result[2] == mock_user_agent  # user_agent
        assert result[3] == context_variables  # context_variables
        assert result[4] == mock_initial_agent  # initial_agent
        assert result[5] == group_after_work  # group_after_work
        assert result[6] == mock_tool_executor  # tool_execution
        assert result[7] == mock_groupchat_instance  # groupchat
        assert result[8] == mock_manager  # manager
        assert result[9] == mock_processed_messages  # processed_messages
        assert result[10] == mock_last_agent  # last_agent
        assert result[11] == mock_group_agent_names  # group_agent_names
        assert result[12] == mock_temp_user_list  # temp_user_list


class TestDefaultPatternIntegration:
    """Test the integration between Pattern ABC and DefaultPattern."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "mock_agent"
        return agent

    @pytest.fixture
    def mock_initial_agent(self) -> MagicMock:
        agent = MagicMock(spec=ConversableAgent)
        agent.name = "initial_agent"
        return agent

    def test_default_pattern_is_pattern_subclass(self) -> None:
        """Test that DefaultPattern is a subclass of Pattern."""
        assert issubclass(DefaultPattern, Pattern)

    def test_can_instantiate_default_pattern(self, mock_initial_agent: MagicMock, mock_agent: MagicMock) -> None:
        """Test that we can instantiate DefaultPattern but not Pattern."""
        agents = [mock_agent]

        # Should be able to instantiate DefaultPattern
        pattern = DefaultPattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))
        assert isinstance(pattern, DefaultPattern)
        assert isinstance(pattern, Pattern)

        # Should NOT be able to instantiate Pattern directly
        with pytest.raises(TypeError):
            Pattern(initial_agent=mock_initial_agent, agents=cast(list[ConversableAgent], agents))  # type: ignore[abstract]
