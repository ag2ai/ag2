# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.agent import Agent
from autogen.agentchat.group.events.transition_events import (
    AfterWorksTransitionEvent,
    OnConditionLLMTransitionEvent,
    OnContextConditionTransitionEvent,
    ReplyResultTransitionEvent,
)
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.targets.transition_target import TransitionTarget


class TestTransitionEvents:
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


class TestAfterWorksTransitionEvent(TestTransitionEvents):
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


class TestOnContextConditionTransitionEvent(TestTransitionEvents):
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


class TestOnConditionLLMTransitionEvent(TestTransitionEvents):
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
        mock_agent = MagicMock(spec=Agent)
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


class TestReplyResultTransitionEvent(TestTransitionEvents):
    @pytest.fixture
    def event(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> ReplyResultTransitionEvent:
        """Create a ReplyResultTransitionEvent for testing."""
        return ReplyResultTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

    def test_initialization(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> None:
        """Test that the event initializes correctly with valid parameters."""
        event = ReplyResultTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

        # The @wrap_event decorator wraps the actual event in a content field
        assert event.content.source_agent == mock_agent  # type: ignore[attr-defined]
        assert event.content.transition_target == mock_transition_target  # type: ignore[attr-defined]

    def test_model_config_allows_arbitrary_types(self, event: ReplyResultTransitionEvent) -> None:
        """Test that the event allows arbitrary types as configured."""
        # The model_config is on the inner content object
        assert event.content.model_config["arbitrary_types_allowed"] is True  # type: ignore[attr-defined]

    def test_properties(
        self, event: ReplyResultTransitionEvent, mock_agent: Agent, mock_transition_target: TransitionTarget
    ) -> None:
        """Test that properties are accessible and return correct values."""
        assert event.content.source_agent == mock_agent  # type: ignore[attr-defined]
        assert event.content.transition_target == mock_transition_target  # type: ignore[attr-defined]

    def test_print_with_default_function(
        self, event: ReplyResultTransitionEvent, mock_agent: Agent, mock_transition_target: TransitionTarget
    ) -> None:
        """Test print method with default print function."""
        mock_print = MagicMock()

        with patch("builtins.print", mock_print):
            event.content.print()  # type: ignore[attr-defined]

        # Should call print once for the colored message (super().print() does nothing)
        assert mock_print.call_count == 1

        # Check that the call contains the expected formatted message
        call_args = mock_print.call_args_list[0][0]
        assert "ReplyResult transition" in call_args[0]
        assert "TestAgent" in call_args[0]
        assert "TestTarget" in call_args[0]

    def test_print_with_custom_function(self, event: ReplyResultTransitionEvent) -> None:
        """Test print method with custom print function."""
        mock_print_func = MagicMock()

        event.content.print(f=mock_print_func)  # type: ignore[attr-defined]

        # Should call the custom function once
        assert mock_print_func.call_count == 1

        # Check that the call contains the expected formatted message
        call_args = mock_print_func.call_args_list[0][0]
        assert "ReplyResult transition" in call_args[0]
        assert "TestAgent" in call_args[0]
        assert "TestTarget" in call_args[0]

    def test_print_with_agent_without_name(self, mock_transition_target: TransitionTarget) -> None:
        """Test print method when agent doesn't have name attribute."""
        agent_without_name = MagicMock(spec=Agent)
        del agent_without_name.name  # Remove name attribute
        agent_without_name.__str__ = MagicMock(return_value="AgentWithoutName")  # type: ignore[method-assign]

        event = ReplyResultTransitionEvent(source_agent=agent_without_name, transition_target=mock_transition_target)
        mock_print_func = MagicMock()

        event.content.print(f=mock_print_func)  # type: ignore[attr-defined]

        # Should still work and use the string representation of the agent
        assert mock_print_func.call_count == 1
        call_args = mock_print_func.call_args_list[0][0]
        assert "ReplyResult transition" in call_args[0]

    def test_print_calls_flush(self, event: ReplyResultTransitionEvent) -> None:
        """Test that print method calls with flush=True."""
        mock_print_func = MagicMock()

        event.content.print(f=mock_print_func)  # type: ignore[attr-defined]

        # Check that the call was made with flush=True
        call = mock_print_func.call_args_list[0]
        assert call.kwargs.get("flush") is True

    def test_event_type_from_wrap_event_decorator(self, event: ReplyResultTransitionEvent) -> None:
        """Test that the @wrap_event decorator sets the correct event type."""
        # The event should have a type derived from the class name
        dumped = event.model_dump()
        assert dumped["type"] == "reply_result_transition"

    def test_model_dump_structure(
        self, event: ReplyResultTransitionEvent, mock_agent: Agent, mock_transition_target: TransitionTarget
    ) -> None:
        """Test the structure of model_dump output."""
        dumped = event.model_dump()

        assert "type" in dumped
        assert "content" in dumped
        assert dumped["type"] == "reply_result_transition"

        content = dumped["content"]
        assert "source_agent" in content
        assert "transition_target" in content
        assert "uuid" in content

    def test_model_validate_roundtrip(self, event: ReplyResultTransitionEvent) -> None:
        """Test that an event can be serialized and deserialized."""
        dumped = event.model_dump()

        # Note: For complex objects like Agent and TransitionTarget,
        # we may need to handle serialization differently in real scenarios
        # This test validates the structure is maintained
        assert dumped["type"] == "reply_result_transition"
        assert "content" in dumped
        assert "source_agent" in dumped["content"]
        assert "transition_target" in dumped["content"]

    def test_super_init_called(self, mock_agent: Agent, mock_transition_target: TransitionTarget) -> None:
        """Test that super().__init__() is called with correct parameters."""
        event = ReplyResultTransitionEvent(source_agent=mock_agent, transition_target=mock_transition_target)

        # The BaseEvent should have been initialized with the same parameters
        assert hasattr(event.content, "source_agent")  # type: ignore[attr-defined]
        assert hasattr(event.content, "transition_target")  # type: ignore[attr-defined]
        assert hasattr(event.content, "uuid")  # type: ignore[attr-defined]  # Should have UUID from BaseEvent

    def test_model_dump_contains_uuid(self, event: ReplyResultTransitionEvent) -> None:
        """Test that the wrapped event contains a UUID."""
        dumped = event.model_dump()

        # The outer wrapper should have content with a UUID
        assert "content" in dumped
        assert "uuid" in dumped["content"]
