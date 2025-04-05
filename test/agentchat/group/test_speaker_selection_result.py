# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.agentchat.agent import Agent
from autogen.agentchat.group.speaker_selection_result import SpeakerSelectionResult


class TestSpeakerSelectionResult:
    def test_init_with_terminate(self) -> None:
        """Test initialisation with terminate=True."""
        result = SpeakerSelectionResult(terminate=True)
        assert result.terminate is True
        assert result.agent_name is None
        assert result.speaker_selection_method is None

    def test_init_with_agent_name(self) -> None:
        """Test initialisation with agent_name."""
        result = SpeakerSelectionResult(agent_name="test_agent")
        assert result.terminate is None
        assert result.agent_name == "test_agent"
        assert result.speaker_selection_method is None

    def test_init_with_speaker_selection_method(self) -> None:
        """Test initialisation with speaker_selection_method."""
        result = SpeakerSelectionResult(speaker_selection_method="auto")
        assert result.terminate is None
        assert result.agent_name is None
        assert result.speaker_selection_method == "auto"

    def test_init_with_multiple_params(self) -> None:
        """Test initialisation with multiple parameters (terminate takes precedence)."""
        result = SpeakerSelectionResult(terminate=True, agent_name="test_agent", speaker_selection_method="auto")
        assert result.terminate is True
        assert result.agent_name == "test_agent"
        assert result.speaker_selection_method == "auto"

    def test_init_with_no_params(self) -> None:
        """Test initialisation with no parameters."""
        result = SpeakerSelectionResult()
        assert result.terminate is None
        assert result.agent_name is None
        assert result.speaker_selection_method is None

    def test_get_speaker_selection_result_with_agent_name(self) -> None:
        """Test get_speaker_selection_result when agent_name is provided."""
        # Setup mock agents and groupchat
        mock_agent = MagicMock(spec=Agent)
        mock_agent.name = "test_agent"

        mock_groupchat = MagicMock()
        mock_groupchat.agents = [mock_agent]

        result = SpeakerSelectionResult(agent_name="test_agent")
        selected = result.get_speaker_selection_result(mock_groupchat)

        assert selected == mock_agent

    def test_get_speaker_selection_result_with_speaker_selection_method(self) -> None:
        """Test get_speaker_selection_result when speaker_selection_method is provided."""
        mock_groupchat = MagicMock()

        result = SpeakerSelectionResult(speaker_selection_method="auto")
        selected = result.get_speaker_selection_result(mock_groupchat)

        assert selected == "auto"

    def test_get_speaker_selection_result_with_terminate(self) -> None:
        """Test get_speaker_selection_result when terminate=True."""
        mock_groupchat = MagicMock()

        result = SpeakerSelectionResult(terminate=True)
        selected = result.get_speaker_selection_result(mock_groupchat)

        assert selected is None

    def test_get_speaker_selection_result_with_agent_not_found(self) -> None:
        """Test get_speaker_selection_result when the specified agent is not found in the groupchat."""
        mock_groupchat = MagicMock()
        mock_groupchat.agents = []  # No agents in the groupchat

        result = SpeakerSelectionResult(agent_name="nonexistent_agent")

        with pytest.raises(ValueError) as excinfo:
            result.get_speaker_selection_result(mock_groupchat)

        assert "Agent 'nonexistent_agent' not found in groupchat" in str(excinfo.value)

    def test_get_speaker_selection_result_with_no_selection_info(self) -> None:
        """Test get_speaker_selection_result when no selection information is provided."""
        mock_groupchat = MagicMock()

        result = SpeakerSelectionResult()

        with pytest.raises(ValueError) as excinfo:
            result.get_speaker_selection_result(mock_groupchat)

        assert "Unable to establish speaker selection result" in str(excinfo.value)

    def test_precedence_order(self) -> None:
        """Test the precedence order when multiple parameters are provided."""
        # Setup
        mock_agent = MagicMock(spec=Agent)
        mock_agent.name = "test_agent"

        mock_groupchat = MagicMock()
        mock_groupchat.agents = [mock_agent]

        # Test with agent_name and speaker_selection_method - agent_name should be used first
        result1 = SpeakerSelectionResult(agent_name="test_agent", speaker_selection_method="auto")
        selected1 = result1.get_speaker_selection_result(mock_groupchat)
        assert selected1 == mock_agent  # Should return the agent, not "auto"

        # Test with speaker selection method and terminate - terminate is last so should be speaker selection method
        result2 = SpeakerSelectionResult(
            speaker_selection_method="auto",
            terminate=True,
        )
        selected2 = result2.get_speaker_selection_result(mock_groupchat)
        assert selected2 == "auto"  # Should return the speaker selection method
