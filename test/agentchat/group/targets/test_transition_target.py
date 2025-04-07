# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.handoffs import Handoffs
from autogen.agentchat.group.speaker_selection_result import SpeakerSelectionResult
from autogen.agentchat.group.targets.transition_target import (
    AgentNameTarget,
    AgentTarget,
    AskUserTarget,
    NestedChatTarget,
    RandomAgentTarget,
    RevertToUserTarget,
    StayTarget,
    TerminateTarget,
    TransitionTarget,
)
from autogen.agentchat.group.targets.transition_utils import __AGENT_WRAPPER_PREFIX__
from autogen.agentchat.groupchat import GroupChat


class TestTransitionTarget:
    def test_base_target_can_resolve_for_speaker_selection(self) -> None:
        """Test that the base TransitionTarget's can_resolve_for_speaker_selection returns False."""
        target = TransitionTarget()
        assert target.can_resolve_for_speaker_selection() is False

    def test_base_target_resolve(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError when resolve is called."""
        target = TransitionTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        with pytest.raises(NotImplementedError) as excinfo:
            target.resolve(mock_groupchat, mock_agent, None)
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_display_name(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for display_name."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.display_name()
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_normalized_name(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for normalized_name."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.normalized_name()
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_needs_agent_wrapper(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for needs_agent_wrapper."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.needs_agent_wrapper()
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_create_wrapper_agent(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for create_wrapper_agent."""
        target = TransitionTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestAgentTarget:
    def test_init(self) -> None:
        """Test initialisation with a mock agent."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.agent_name == "test_agent"

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"

        target = AgentTarget(agent=mock_agent)
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        result = target.resolve(mock_groupchat, mock_agent, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "test_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_display_name(self) -> None:
        """Test that display_name returns the agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.display_name() == "test_agent"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.normalized_name() == "test_agent"

    def test_str_representation(self) -> None:
        """Test the string representation of AgentTarget."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert str(target) == "Transfer to test_agent"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "AgentTarget does not require wrapping" in str(excinfo.value)


class TestAgentNameTarget:
    def test_init(self) -> None:
        """Test initialisation with an agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.agent_name == "test_agent"

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        result = target.resolve(mock_groupchat, mock_agent, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "test_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_display_name(self) -> None:
        """Test that display_name returns the agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.display_name() == "test_agent"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.normalized_name() == "test_agent"

    def test_str_representation(self) -> None:
        """Test the string representation of AgentNameTarget."""
        target = AgentNameTarget(agent_name="test_agent")
        assert str(target) == "Transfer to test_agent"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = AgentNameTarget(agent_name="test_agent")
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "AgentNameTarget does not require wrapping" in str(excinfo.value)


class TestNestedChatTarget:
    def test_init(self) -> None:
        """Test initialisation with a nested chat config."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.nested_chat_config == nested_chat_config

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns False."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.can_resolve_for_speaker_selection() is False

    def test_resolve_raises_error(self) -> None:
        """Test that resolve raises NotImplementedError."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)

        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        with pytest.raises(NotImplementedError) as excinfo:
            target.resolve(mock_groupchat, mock_agent, None)
        assert "NestedChatTarget does not support the resolve method" in str(excinfo.value)

    def test_display_name(self) -> None:
        """Test that display_name returns 'a nested chat'."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.display_name() == "a nested chat"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns 'nested_chat'."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.normalized_name() == "nested_chat"

    def test_str_representation(self) -> None:
        """Test the string representation of NestedChatTarget."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert str(target) == "Transfer to nested chat"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns True."""
        nested_chat_config = {"chat_queue": [{}], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.needs_agent_wrapper() is True

    def test_create_wrapper_agent(self) -> None:
        """Test creating a wrapper agent for a nested chat target."""
        # Set up the nested chat
        sample_agent = ConversableAgent(name="sample_agent")
        sample_agent_two = ConversableAgent(name="sample_agent_two")
        nested_chat_config = {
            "chat_queue": [
                {
                    "recipient": sample_agent,
                    "summary_method": "reflection_with_llm",
                    "summary_prompt": "Summarise the conversation into bullet points.",
                },
                {
                    "recipient": sample_agent_two,
                    "message": "Write a poem about the context.",
                    "max_turns": 1,
                    "summary_method": "last_msg",
                },
            ],
            "use_async": False,
        }
        target = NestedChatTarget(nested_chat_config=nested_chat_config)

        # Set up the parent agent
        parent_agent = ConversableAgent(name="parent_agent")
        parent_agent.handoffs = Handoffs()

        # Call create_wrapper_agent
        index = 2
        result = target.create_wrapper_agent(parent_agent, index)

        assert result.name == f"{__AGENT_WRAPPER_PREFIX__}nested_{parent_agent.name}_{index + 1}"


class TestTerminateTarget:
    def test_init(self) -> None:
        """Test initialization of TerminateTarget."""
        target = TerminateTarget()
        assert isinstance(target, TransitionTarget)

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = TerminateTarget()
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with terminate=True."""
        target = TerminateTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        result = target.resolve(mock_groupchat, mock_agent, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.terminate is True
        assert result.agent_name is None
        assert result.speaker_selection_method is None

    def test_display_name(self) -> None:
        """Test that display_name returns 'Terminate'."""
        target = TerminateTarget()
        assert target.display_name() == "Terminate"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns 'terminate'."""
        target = TerminateTarget()
        assert target.normalized_name() == "terminate"

    def test_str_representation(self) -> None:
        """Test the string representation of TerminateTarget."""
        target = TerminateTarget()
        assert str(target) == "Terminate"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = TerminateTarget()
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = TerminateTarget()
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "TerminateTarget does not require wrapping" in str(excinfo.value)


class TestStayTarget:
    def test_init(self) -> None:
        """Test initialization of StayTarget."""
        target = StayTarget()
        assert isinstance(target, TransitionTarget)

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = StayTarget()
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the current agent name."""
        target = StayTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "current_agent"
        mock_groupchat = MagicMock(spec=GroupChat)
        result = target.resolve(mock_groupchat, mock_agent, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "current_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_display_name(self) -> None:
        """Test that display_name returns 'Stay'."""
        target = StayTarget()
        assert target.display_name() == "Stay"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns 'stay'."""
        target = StayTarget()
        assert target.normalized_name() == "stay"

    def test_str_representation(self) -> None:
        """Test the string representation of StayTarget."""
        target = StayTarget()
        assert str(target) == "Stay with agent"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = StayTarget()
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = StayTarget()
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "StayTarget does not require wrapping" in str(excinfo.value)


class TestRevertToUserTarget:
    def test_init(self) -> None:
        """Test initialization of RevertToUserTarget."""
        target = RevertToUserTarget()
        assert isinstance(target, TransitionTarget)

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = RevertToUserTarget()
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the user agent name."""
        target = RevertToUserTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_user_agent = MagicMock(spec=ConversableAgent)
        mock_user_agent.name = "user_agent"
        mock_groupchat = MagicMock(spec=GroupChat)
        result = target.resolve(mock_groupchat, mock_agent, mock_user_agent)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "user_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_resolve_with_no_user_agent(self) -> None:
        """Test that resolve raises ValueError when user_agent is None."""
        target = RevertToUserTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)

        with pytest.raises(ValueError) as excinfo:
            target.resolve(mock_groupchat, mock_agent, None)
        assert "User agent must be provided" in str(excinfo.value)

    def test_display_name(self) -> None:
        """Test that display_name returns 'Revert to User'."""
        target = RevertToUserTarget()
        assert target.display_name() == "Revert to User"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns 'revert_to_user'."""
        target = RevertToUserTarget()
        assert target.normalized_name() == "revert_to_user"

    def test_str_representation(self) -> None:
        """Test the string representation of RevertToUserTarget."""
        target = RevertToUserTarget()
        assert str(target) == "Revert to User"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = RevertToUserTarget()
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = RevertToUserTarget()
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "RevertToUserTarget does not require wrapping" in str(excinfo.value)


class TestAskUserTarget:
    def test_init(self) -> None:
        """Test initialization of AskUserTarget."""
        target = AskUserTarget()
        assert isinstance(target, TransitionTarget)

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = AskUserTarget()
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with speaker_selection_method='manual'."""
        target = AskUserTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_groupchat = MagicMock(spec=GroupChat)
        result = target.resolve(mock_groupchat, mock_agent, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.speaker_selection_method == "manual"
        assert result.agent_name is None
        assert result.terminate is None

    def test_display_name(self) -> None:
        """Test that display_name returns 'Ask User'."""
        target = AskUserTarget()
        assert target.display_name() == "Ask User"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns 'ask_user'."""
        target = AskUserTarget()
        assert target.normalized_name() == "ask_user"

    def test_str_representation(self) -> None:
        """Test the string representation of AskUserTarget."""
        target = AskUserTarget()
        assert str(target) == "Ask User"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = AskUserTarget()
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = AskUserTarget()
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "AskUserTarget does not require wrapping" in str(excinfo.value)


class TestRandomAgentTarget:
    def test_init(self) -> None:
        """Test initialization with a list of agents."""
        mock_agent1 = MagicMock(spec=ConversableAgent)
        mock_agent1.name = "agent1"
        mock_agent2 = MagicMock(spec=ConversableAgent)
        mock_agent2.name = "agent2"

        target = RandomAgentTarget(agents=[mock_agent1, mock_agent2])
        assert target.agent_names == ["agent1", "agent2"]
        assert target.nominated_name == "<Not Randomly Selected Yet>"

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "agent1"
        target = RandomAgentTarget(agents=[mock_agent])
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with a randomly selected agent name."""
        # Setup
        mock_agent1 = MagicMock(spec=ConversableAgent)
        mock_agent1.name = "agent1"
        mock_agent2 = MagicMock(spec=ConversableAgent)
        mock_agent2.name = "agent2"

        target = RandomAgentTarget(agents=[mock_agent1, mock_agent2])
        mock_current_agent = MagicMock(spec=ConversableAgent)
        mock_current_agent.name = "current_agent"  # Different from the available agents
        mock_groupchat = MagicMock(spec=GroupChat)

        # Test with mocked random selection to ensure deterministic behavior
        with patch("random.choice", return_value="agent2"):
            result = target.resolve(mock_groupchat, mock_current_agent, None)

            # Assert the result is correct
            assert isinstance(result, SpeakerSelectionResult)
            assert result.agent_name == "agent2"
            assert result.terminate is None
            assert result.speaker_selection_method is None

            # Assert the nominated name was updated
            assert target.nominated_name == "agent2"

    def test_resolve_with_randomness(self) -> None:
        """Test that resolve randomly selects an agent name from the available options."""
        # Setup
        mock_agent1 = MagicMock(spec=ConversableAgent)
        mock_agent1.name = "agent1"
        mock_agent2 = MagicMock(spec=ConversableAgent)
        mock_agent2.name = "agent2"

        target = RandomAgentTarget(agents=[mock_agent1, mock_agent2])
        mock_current_agent = MagicMock(spec=ConversableAgent)
        mock_current_agent.name = "current_agent"  # Different from the available agents
        mock_groupchat = MagicMock(spec=GroupChat)

        # Call resolve with real randomness
        result = target.resolve(mock_groupchat, mock_current_agent, None)

        # Verify the result is one of the expected agent names
        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name in ["agent1", "agent2"]
        assert result.terminate is None
        assert result.speaker_selection_method is None
        assert target.nominated_name in ["agent1", "agent2"]

    def test_resolve_excludes_current_agent(self) -> None:
        """Test that resolve excludes the current agent from selection."""
        # Setup
        mock_agent1 = MagicMock(spec=ConversableAgent)
        mock_agent1.name = "agent1"
        mock_agent2 = MagicMock(spec=ConversableAgent)
        mock_agent2.name = "agent2"

        target = RandomAgentTarget(agents=[mock_agent1, mock_agent2])

        # Use agent1 as the current agent
        mock_current_agent = mock_agent1
        mock_groupchat = MagicMock(spec=GroupChat)

        with patch("random.choice") as mock_choice:
            mock_choice.return_value = "agent2"

            result = target.resolve(mock_groupchat, mock_current_agent, None)

            # Verify the argument passed to random.choice
            args = mock_choice.call_args[0][0]  # First positional argument
            assert len(args) == 1
            assert "agent1" not in args
            assert "agent2" in args

            # Verify the result
            assert result.agent_name == "agent2"

    def test_display_name(self) -> None:
        """Test that display_name returns the nominated agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "agent1"
        target = RandomAgentTarget(agents=[mock_agent])

        # Before resolution, should return the default value
        assert target.display_name() == "<Not Randomly Selected Yet>"

        # After resolution, should return the nominated name
        target.nominated_name = "selected_agent"
        assert target.display_name() == "selected_agent"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the display_name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "agent1"
        target = RandomAgentTarget(agents=[mock_agent])

        target.nominated_name = "selected_agent"
        assert target.normalized_name() == "selected_agent"

    def test_str_representation(self) -> None:
        """Test the string representation of RandomAgentTarget."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "agent1"
        target = RandomAgentTarget(agents=[mock_agent])

        target.nominated_name = "selected_agent"
        assert str(target) == "Transfer to selected_agent"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "agent1"
        target = RandomAgentTarget(agents=[mock_agent])
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "agent1"
        target = RandomAgentTarget(agents=[mock_agent])

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "RandomAgentTarget does not require wrapping" in str(excinfo.value)
