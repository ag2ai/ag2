# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.agent import Agent
from autogen.agentchat.group.events.handoff_events import (
    AfterWorksTransitionEvent,
    OnConditionLLMTransitionEvent,
    OnContextConditionTransitionEvent,
)
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.targets.transition_target import TransitionTarget


class TestHandoffEvents:
    @pytest.fixture
    def mock_agent(self) -> Agent:
        """Create a mock Agent for testing."""
        agent = MagicMock(spec=Agent)
        agent.name = "TestAgent"
        return agent

    @pytest.fixture
    def mock_transition_target(self) -> TransitionTarget:
        """Create a mock TransitionTarget for testing."""
        target = MagicMock(spec=TransitionTarget)
        target.display_name.return_value = "TestTarget"
        return target

    @pytest.fixture
    def executor(self) -> GroupToolExecutor:
        """Create a GroupToolExecutor for testing."""
        return GroupToolExecutor()


class TestAfterWorksTransitionEvent(TestHandoffEvents):
    @pytest.fixture
    def event(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> AfterWorksTransitionEvent:
        """Create an AfterWorksTransitionEvent for testing."""
        return AfterWorksTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

    def test_initialization(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> None:
        """Test that the event initializes correctly with valid parameters."""
        event = AfterWorksTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

        # The @wrap_event decorator wraps the actual event in a content field
        assert event.content.source_agent == mock_agent  # type: ignore[attr-defined]
        assert event.content.transition_target == mock_transition_target  # type: ignore[attr-defined]

    def test_event_type(self, event: AfterWorksTransitionEvent) -> None:
        """Test that the @wrap_event decorator sets the correct event type."""
        dumped = event.model_dump()
        assert dumped["type"] == "after_works_transition"

    def test_print_with_agent_name(self, event: AfterWorksTransitionEvent) -> None:
        """Test print method with agent that has a name."""
        mock_print_func = MagicMock()

        event.content.print(f=mock_print_func)  # type: ignore[attr-defined]

        assert mock_print_func.call_count == 1
        call_args = mock_print_func.call_args_list[0][0]
        assert "AfterWork handoff" in call_args[0]
        assert "TestAgent" in call_args[0]
        assert "TestTarget" in call_args[0]


class TestOnContextConditionTransitionEvent(TestHandoffEvents):
    @pytest.fixture
    def event(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> OnContextConditionTransitionEvent:
        """Create an OnContextConditionTransitionEvent for testing."""
        return OnContextConditionTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

    def test_initialization(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> None:
        """Test that the event initializes correctly with valid parameters."""
        event = OnContextConditionTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

        assert event.content.source_agent == mock_agent  # type: ignore[attr-defined]
        assert event.content.transition_target == mock_transition_target  # type: ignore[attr-defined]

    def test_event_type(self, event: OnContextConditionTransitionEvent) -> None:
        """Test that the @wrap_event decorator sets the correct event type."""
        dumped = event.model_dump()
        assert dumped["type"] == "on_context_condition_transition"

    def test_print_with_agent_name(self, event: OnContextConditionTransitionEvent) -> None:
        """Test print method with agent that has a name."""
        mock_print_func = MagicMock()

        event.content.print(f=mock_print_func)  # type: ignore[attr-defined]

        assert mock_print_func.call_count == 1
        call_args = mock_print_func.call_args_list[0][0]
        assert "OnContextCondition handoff" in call_args[0]
        assert "TestAgent" in call_args[0]
        assert "TestTarget" in call_args[0]


class TestOnConditionLLMTransitionEvent(TestHandoffEvents):
    @pytest.fixture
    def event(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> OnConditionLLMTransitionEvent:
        """Create an OnConditionLLMTransitionEvent for testing."""
        return OnConditionLLMTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

    def test_initialization(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> None:
        """Test that the event initializes correctly with valid parameters."""
        event = OnConditionLLMTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

        assert event.content.source_agent == mock_agent  # type: ignore[attr-defined]
        assert event.content.transition_target == mock_transition_target  # type: ignore[attr-defined]

    def test_event_type(self, event: OnConditionLLMTransitionEvent) -> None:
        """Test that the @wrap_event decorator sets the correct event type."""
        dumped = event.model_dump()
        assert dumped["type"] == "on_condition_l_l_m_transition"

    def test_print_with_agent_name(self, event: OnConditionLLMTransitionEvent) -> None:
        """Test print method with agent that has a name."""
        mock_print_func = MagicMock()

        event.content.print(f=mock_print_func)  # type: ignore[attr-defined]

        assert mock_print_func.call_count == 1
        call_args = mock_print_func.call_args_list[0][0]
        assert "LLM-based OnCondition handoff" in call_args[0]
        assert "TestAgent" in call_args[0]
        assert "TestTarget" in call_args[0]

    @patch("autogen.agentchat.group.group_tool_executor.IOStream")
    def test_send_llm_handoff_event(self, mock_iostream: MagicMock, executor: GroupToolExecutor) -> None:
        """Test _send_llm_handoff_event method."""
        # Setup mock iostream
        mock_iostream_instance = MagicMock()
        mock_iostream.get_default.return_value = mock_iostream_instance

        # Test case 1: Not a handoff function
        with patch.object(executor, "is_handoff_function", return_value=False):
            message = {"name": "SomeAgent", "content": "test"}
            transition_target = MagicMock(spec=TransitionTarget)
            executor._send_llm_handoff_event(message, transition_target)
            mock_iostream_instance.send.assert_not_called()

        # Test case 2: Handoff function but no sender agent
        with (
            patch.object(executor, "is_handoff_function", return_value=True),
            patch.object(executor, "get_sender_agent_for_message", return_value=None),
        ):
            message = {"name": "SomeAgent", "content": "test"}
            transition_target = MagicMock(spec=TransitionTarget)
            executor._send_llm_handoff_event(message, transition_target)
            mock_iostream_instance.send.assert_not_called()

        # Test case 3: Valid handoff function with sender agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        with (
            patch.object(executor, "is_handoff_function", return_value=True),
            patch.object(executor, "get_sender_agent_for_message", return_value=mock_agent),
        ):
            message = {"name": "TestAgent", "content": "test"}
            transition_target = MagicMock(spec=TransitionTarget)
            executor._send_llm_handoff_event(message, transition_target)

            # Verify IOStream.send was called
            mock_iostream_instance.send.assert_called_once()

            # Verify the event object passed to send
            sent_event = mock_iostream_instance.send.call_args[0][0]
            assert isinstance(sent_event, OnConditionLLMTransitionEvent)
            # The wrapped event has a 'content' field that contains the actual event data
            assert sent_event.content.source_agent == mock_agent  # type: ignore[attr-defined]
            assert sent_event.content.transition_target == transition_target  # type: ignore[attr-defined]
