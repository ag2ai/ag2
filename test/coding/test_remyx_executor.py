# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RemyxCodeExecutor."""

import os
from unittest.mock import Mock, patch

import pytest

from autogen.coding import MarkdownCodeExtractor

try:
    # Check that remyxai dependencies are available
    from remyxai.api.search import Asset
    from remyxai.client.search import SearchClient

    from autogen.coding import RemyxCodeExecutor, RemyxCodeResult

    _has_remyx = Asset is not None and SearchClient is not None
except ImportError:
    _has_remyx = False

pytestmark = pytest.mark.skipif(not _has_remyx, reason="Remyx dependencies not installed")


@pytest.mark.skipif(not _has_remyx, reason="Remyx dependencies not installed")
class TestRemyxCodeExecutor:
    """Test suite for RemyxCodeExecutor."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clear environment variables
        if "REMYX_API_KEY" in os.environ:
            del os.environ["REMYX_API_KEY"]
        if "REMYXAI_API_KEY" in os.environ:
            del os.environ["REMYXAI_API_KEY"]

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_init_with_arxiv_id(self, mock_parent_init, mock_get_asset):
        """Test initialization with arXiv ID."""
        # Mock asset
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "docker_image": "remyxai/2010.11929v2:latest",
            "has_docker": True,
            "environment_vars": [],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        assert executor.arxiv_id == "2010.11929v2"
        assert executor._asset_metadata is not None
        mock_get_asset.assert_called_once_with("2010.11929v2")
        mock_parent_init.assert_called_once()

    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_init_with_direct_image(self, mock_parent_init):
        """Test initialization with direct Docker image."""
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(image="remyxai/test:latest")

        assert executor.arxiv_id is None
        assert executor._asset_metadata is None
        mock_parent_init.assert_called_once()

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_init_with_api_key_from_env(self, mock_parent_init, mock_get_asset):
        """Test initialization with API key from environment."""
        os.environ["REMYX_API_KEY"] = "test_key"
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {"arxiv_id": "2010.11929v2", "environment_vars": []}
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        assert executor.api_key == "test_key"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    def test_init_with_paper_not_found(self, mock_get_asset):
        """Test initialization with paper not found."""
        mock_get_asset.return_value = None

        with pytest.raises(ValueError, match="Paper .* not found in Remyx catalog"):
            RemyxCodeExecutor(arxiv_id="9999.99999v1")

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    def test_init_with_no_docker_image(self, mock_get_asset):
        """Test initialization with paper that has no Docker image."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = False
        mock_get_asset.return_value = mock_asset

        with pytest.raises(ValueError, match="does not have a Docker image"):
            RemyxCodeExecutor(arxiv_id="2010.11929v2")

    def test_init_with_no_arxiv_or_image(self):
        """Test initialization without arxiv_id or image raises error."""
        with pytest.raises(ValueError, match="Either arxiv_id or image must be provided"):
            RemyxCodeExecutor()

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_environment_variable_handling(self, mock_parent_init, mock_get_asset):
        """Test that environment variables are passed to container."""
        os.environ["HF_TOKEN"] = "test_hf_token"
        os.environ["WANDB_API_KEY"] = "test_wandb_key"

        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "environment_vars": ["HF_TOKEN", "WANDB_API_KEY"],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        # Check that parent init was called with environment
        call_kwargs = mock_parent_init.call_args[1]
        container_kwargs = call_kwargs.get("container_create_kwargs", {})
        assert "environment" in container_kwargs
        assert container_kwargs["environment"]["HF_TOKEN"] == "test_hf_token"
        assert container_kwargs["environment"]["WANDB_API_KEY"] == "test_wandb_key"
        assert executor.arxiv_id == "2010.11929v2"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_code_extractor_property(self, mock_parent_init, mock_get_asset):
        """Test code_extractor property returns MarkdownCodeExtractor."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {"arxiv_id": "2010.11929v2", "environment_vars": []}
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        assert isinstance(executor.code_extractor, MarkdownCodeExtractor)

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_paper_info_property(self, mock_parent_init, mock_get_asset):
        """Test paper_info property returns asset metadata."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "environment_vars": [],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        assert executor.paper_info is not None
        assert executor.paper_info["title"] == "Test Paper"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_get_paper_context(self, mock_parent_init, mock_get_asset):
        """Test get_paper_context returns formatted context."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "github_url": "https://github.com/test/repo",
            "working_directory": "/app",
            "reasoning": "Test reasoning",
            "quickstart_hint": "Run python test.py",
            "environment_vars": [],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        context = executor.get_paper_context()

        assert "Test Paper" in context
        assert "2010.11929v2" in context
        assert "https://github.com/test/repo" in context
        assert "Test reasoning" in context
        assert "Run python test.py" in context

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_get_paper_context_no_metadata(self, mock_parent_init, mock_get_asset):
        """Test get_paper_context with no metadata."""
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(image="remyxai/test:latest")
        context = executor.get_paper_context()

        assert context == "No paper metadata available."

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_build_system_message_default(self, mock_parent_init, mock_get_asset):
        """Test _build_system_message with default parameters."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "environment_vars": [],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        system_message = executor._build_system_message()

        assert "Test Paper" in system_message
        assert "Your Mission:" in system_message
        assert "Phase 1: Understanding" in system_message
        assert "Important Guidelines:" in system_message

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_build_system_message_with_custom_goal(self, mock_parent_init, mock_get_asset):
        """Test _build_system_message with custom goal."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {"arxiv_id": "2010.11929v2", "environment_vars": []}
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        system_message = executor._build_system_message(goal="Run all benchmarks")

        assert "Run all benchmarks" in system_message
        assert "Phase 1: Understanding" not in system_message

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_build_system_message_with_system_message(self, mock_parent_init, mock_get_asset):
        """Test _build_system_message with additional system message."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {"arxiv_id": "2010.11929v2", "environment_vars": []}
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        additional_message = "Keep responses concise. Use GPU when available."
        result = executor._build_system_message(system_message=additional_message)

        assert "Keep responses concise" in result
        assert "Use GPU when available" in result

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_repr_with_arxiv_id(self, mock_parent_init, mock_get_asset):
        """Test __repr__ with arxiv_id."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {"arxiv_id": "2010.11929v2", "environment_vars": []}
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        assert repr(executor) == "RemyxCodeExecutor(arxiv_id='2010.11929v2')"

    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_repr_with_image_only(self, mock_parent_init):
        """Test __repr__ with image only."""
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(image="remyxai/test:latest")
        executor._executor_image = "remyxai/test:latest"

        assert repr(executor) == "RemyxCodeExecutor(image='remyxai/test:latest')"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    @patch("autogen.ConversableAgent")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"})
    def test_create_agents(self, mock_agent, mock_parent_init, mock_get_asset):
        """Test create_agents method."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "environment_vars": [],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        mock_executor_agent = Mock()
        mock_writer_agent = Mock()
        mock_agent.side_effect = [mock_executor_agent, mock_writer_agent]

        executor_agent, writer_agent = executor.create_agents(
            goal="Test goal", llm_model="gpt-4o", human_input_mode="NEVER"
        )

        assert executor_agent == mock_executor_agent
        assert writer_agent == mock_writer_agent
        assert mock_agent.call_count == 2

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    @patch("autogen.ConversableAgent")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"})
    def test_create_agents_with_system_message(self, mock_agent, mock_parent_init, mock_get_asset):
        """Test create_agents method with additional system message."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "environment_vars": [],
        }
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        mock_executor_agent = Mock()
        mock_writer_agent = Mock()
        mock_agent.side_effect = [mock_executor_agent, mock_writer_agent]

        executor_agent, writer_agent = executor.create_agents(
            goal="Test goal",
            llm_model="gpt-4o",
            human_input_mode="NEVER",
            system_message="Output results as JSON",
        )

        # Verify system message was passed to writer agent
        writer_call_kwargs = mock_agent.call_args_list[1][1]
        assert "Output results as JSON" in writer_call_kwargs["system_message"]

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_format_chat_result(self, mock_parent_init, mock_get_asset):
        """Test format_chat_result static method."""
        mock_asset = Mock()
        mock_asset.arxiv_id = "2010.11929v2"
        mock_asset.has_docker = True
        mock_asset.docker_image = "remyxai/2010.11929v2:latest"
        mock_asset.to_dict.return_value = {"arxiv_id": "2010.11929v2", "environment_vars": []}
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        # Mock chat result
        mock_result = Mock()
        mock_result.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        mock_result.chat_id = "test_chat_123"
        mock_result.summary = "Test summary"
        mock_result.cost = {"usage_including_cached_inference": {"total_cost": 0.0123}}

        formatted = RemyxCodeExecutor.format_chat_result(mock_result)

        assert "📊 Exploration Session Summary" in formatted
        assert "Total messages: 2" in formatted
        assert "Chat ID: test_chat_123" in formatted
        assert "Cost: $0.0123" in formatted
        assert "Test summary" in formatted

    def test_format_chat_result_from_utils(self):
        """Test format_chat_result utility function directly."""
        from autogen.coding.utils import format_chat_result

        mock_result = Mock()
        mock_result.chat_history = [
            {"name": "agent1", "content": "Hello"},
        ]
        mock_result.chat_id = "utils_test_123"
        mock_result.summary = "Utils test"
        mock_result.cost = {"usage_including_cached_inference": {"total_cost": 0.05}}

        formatted = format_chat_result(mock_result)

        assert "Exploration Session Summary" in formatted
        assert "utils_test_123" in formatted


@pytest.mark.skipif(not _has_remyx, reason="Remyx dependencies not installed")
class TestRemyxCodeResult:
    """Test suite for RemyxCodeResult."""

    def test_code_result_creation(self):
        """Test RemyxCodeResult creation."""
        result = RemyxCodeResult(
            exit_code=0,
            output="Test output",
            arxiv_id="2010.11929v2",
            paper_title="Test Paper",
        )

        assert result.exit_code == 0
        assert result.output == "Test output"
        assert result.arxiv_id == "2010.11929v2"
        assert result.paper_title == "Test Paper"

    def test_code_result_without_paper_info(self):
        """Test RemyxCodeResult creation without paper info."""
        result = RemyxCodeResult(exit_code=1, output="Error output")

        assert result.exit_code == 1
        assert result.output == "Error output"
        assert result.arxiv_id is None
        assert result.paper_title is None
