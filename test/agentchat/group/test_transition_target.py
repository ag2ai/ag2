# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.agentchat.group.transition_target import (
    AfterWorkOptionTarget,
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
    TransitionTarget,
)


class TestTransitionTarget:
    def test_base_target_resolve(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError when resolve is called."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.resolve(None, [], None)
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestAgentTarget:
    def test_init(self) -> None:
        """Test initialization with a mock agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.agent_name == "test_agent"

    def test_resolve(self) -> None:
        """Test that resolve returns the agent object from the groupchat."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_groupchat = MagicMock()
        mock_groupchat.agents = [mock_agent]

        target = AgentTarget(agent=mock_agent)
        result = target.resolve(None, [], mock_groupchat)
        assert result == mock_agent

    def test_resolve_agent_not_found(self) -> None:
        """Test that resolve raises ValueError when agent is not found in groupchat."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_groupchat = MagicMock()
        mock_groupchat.agents = []  # Empty list, so agent won't be found

        target = AgentTarget(agent=mock_agent)
        with pytest.raises(ValueError) as excinfo:
            target.resolve(None, [], mock_groupchat)
        assert "Agent with name 'test_agent' not found in groupchat" in str(excinfo.value)


class TestAgentNameTarget:
    def test_init(self) -> None:
        """Test initialization with an agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.agent_name == "test_agent"

    def test_resolve(self) -> None:
        """Test that resolve returns the agent name."""
        agent_name = "test_agent"
        target = AgentNameTarget(agent_name=agent_name)
        result = target.resolve(None, [], None)
        assert result == agent_name


class TestNestedChatTarget:
    def test_init(self) -> None:
        """Test initialization with a nested chat config."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.nested_chat_config == nested_chat_config

    def test_resolve(self) -> None:
        """Test that resolve returns the nested chat configuration."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        result = target.resolve(None, [], None)
        assert result == nested_chat_config


class TestAfterWorkOptionTarget:
    def test_init(self) -> None:
        """Test initialization with different options."""
        # Test with valid options
        for option in ["terminate", "revert_to_user", "stay", "group_manager"]:
            target = AfterWorkOptionTarget(after_work_option=option)
            assert target.after_work_option == option

    def test_resolve(self) -> None:
        """Test that resolve returns the option value."""
        option = "terminate"
        target = AfterWorkOptionTarget(after_work_option=option)
        result = target.resolve(None, [], None)
        assert result == option
