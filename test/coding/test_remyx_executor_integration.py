# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for RemyxCodeExecutor with real API calls and Docker."""

import os
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autogen.coding import CodeBlock

try:
    import docker
    import dotenv
    from remyxai.client.search import SearchClient

    from autogen.coding import RemyxCodeExecutor

    _has_remyx = True
    _has_docker = True

    # Check if Docker is actually running
    try:
        docker.from_env().ping()
    except Exception:
        _has_docker = False
except ImportError:
    _has_remyx = False
    _has_docker = False

pytestmark = pytest.mark.skipif(not _has_remyx or not _has_docker, reason="Remyx dependencies or Docker not available")


@pytest.mark.skipif(not _has_remyx or not _has_docker, reason="Remyx/Docker not available")
@pytest.mark.integration
class TestRemyxCodeExecutorIntegration:
    """Integration test suite for RemyxCodeExecutor with real API calls."""

    def setup_method(self):
        """Setup method run before each test."""
        # Load environment variables from .env file
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            dotenv.load_dotenv(env_file)

        # Check for API key
        if not os.getenv("REMYX_API_KEY") and not os.getenv("REMYXAI_API_KEY"):
            pytest.skip("REMYX_API_KEY or REMYXAI_API_KEY environment variable not set")

    def test_search_papers(self):
        """Test searching for papers with Docker environments."""
        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=5)

        assert len(papers) > 0, "Should find at least one paper with Docker"

        # Check first paper has required fields
        paper = papers[0]
        assert paper.arxiv_id is not None
        assert paper.docker_image is not None
        assert paper.has_docker is True

    def test_init_with_arxiv_id(self):
        """Test initialization with a real arXiv ID."""
        # Search for a paper first
        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=1)

        if not papers:
            pytest.skip("No papers found with Docker")

        arxiv_id = papers[0].arxiv_id

        # Create executor
        executor = RemyxCodeExecutor(
            arxiv_id=arxiv_id,
            timeout=300,
            auto_remove=True,
            stop_container=True,
        )

        assert executor.arxiv_id == arxiv_id
        assert executor.paper_info is not None
        assert executor.paper_info["arxiv_id"] == arxiv_id

        # Clean up
        if hasattr(executor, "_container") and executor._container:
            try:
                executor._container.stop()
                executor._container.remove()
            except Exception:
                pass

    def test_execute_basic_python_code(self):
        """Test executing basic Python code in paper environment."""
        # Search for a paper
        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=1)

        if not papers:
            pytest.skip("No papers found with Docker")

        arxiv_id = papers[0].arxiv_id

        executor = RemyxCodeExecutor(
            arxiv_id=arxiv_id,
            timeout=300,
            auto_remove=True,
        )

        # Execute simple Python code
        code_blocks = [
            CodeBlock(
                language="python",
                code="""
import sys
print(f"Python version: {sys.version}")
print("Hello from research paper environment!")

# Test that we're in /app
import os
print(f"Working directory: {os.getcwd()}")
""",
            )
        ]

        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        assert "Python version:" in result.output
        assert "Hello from research paper environment!" in result.output
        assert "/app" in result.output or "Working directory:" in result.output

        # Clean up
        if hasattr(executor, "_container") and executor._container:
            try:
                executor._container.stop()
                executor._container.remove()
            except Exception:
                pass

    def test_execute_bash_code(self):
        """Test executing bash commands in paper environment."""
        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=1)

        if not papers:
            pytest.skip("No papers found with Docker")

        arxiv_id = papers[0].arxiv_id

        executor = RemyxCodeExecutor(
            arxiv_id=arxiv_id,
            timeout=300,
            auto_remove=True,
        )

        code_blocks = [
            CodeBlock(
                language="bash",
                code="""
echo "Testing bash execution"
ls -la /app | head -5
pwd
""",
            )
        ]

        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        assert "Testing bash execution" in result.output
        assert "/app" in result.output

        # Clean up
        if hasattr(executor, "_container") and executor._container:
            try:
                executor._container.stop()
                executor._container.remove()
            except Exception:
                pass

    def test_get_paper_context(self):
        """Test getting paper context."""
        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=1)

        if not papers:
            pytest.skip("No papers found with Docker")

        arxiv_id = papers[0].arxiv_id

        executor = RemyxCodeExecutor(arxiv_id=arxiv_id, timeout=300)

        context = executor.get_paper_context()

        assert context is not None
        assert arxiv_id in context
        assert "Title:" in context
        assert "arXiv ID:" in context

        # Clean up
        if hasattr(executor, "_container") and executor._container:
            try:
                executor._container.stop()
                executor._container.remove()
            except Exception:
                pass

    def test_error_handling(self):
        """Test error handling with invalid code."""
        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=1)

        if not papers:
            pytest.skip("No papers found with Docker")

        arxiv_id = papers[0].arxiv_id

        executor = RemyxCodeExecutor(
            arxiv_id=arxiv_id,
            timeout=300,
            auto_remove=True,
        )

        code_blocks = [
            CodeBlock(
                language="python",
                code="""
# This will raise a NameError
print(undefined_variable)
""",
            )
        ]

        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code != 0, "Expected non-zero exit code for error"
        assert "NameError" in result.output or "undefined_variable" in result.output

        # Clean up
        if hasattr(executor, "_container") and executor._container:
            try:
                executor._container.stop()
                executor._container.remove()
            except Exception:
                pass

    @pytest.mark.slow
    def test_create_agents(self):
        """Test creating agents for exploration."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        client = SearchClient()
        papers = client.search(query="CLIP", has_docker=True, max_results=1)

        if not papers:
            pytest.skip("No papers found with Docker")

        arxiv_id = papers[0].arxiv_id

        executor = RemyxCodeExecutor(
            arxiv_id=arxiv_id,
            timeout=300,
            auto_remove=True,
        )

        executor_agent, writer_agent = executor.create_agents(
            goal="List files in repository", llm_model="gpt-4o-mini", human_input_mode="NEVER"
        )

        assert executor_agent is not None
        assert writer_agent is not None
        assert executor_agent.name == "code_executor"
        assert writer_agent.name == "research_explorer"

        # Clean up
        if hasattr(executor, "_container") and executor._container:
            try:
                executor._container.stop()
                executor._container.remove()
            except Exception:
                pass


@pytest.fixture
def mock_asset():
    """Create a mock asset for testing."""
    asset = Mock()
    asset.arxiv_id = "2010.11929v2"
    asset.has_docker = True
    asset.docker_image = "remyxai/2010.11929v2:latest"
    asset.to_dict.return_value = {
        "arxiv_id": "2010.11929v2",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "github_url": "https://github.com/google-research/vision_transformer",
        "working_directory": "/app",
        "reasoning": "Vision Transformer (ViT) implementation",
        "quickstart_hint": "python train.py",
        "environment_vars": [],
    }
    return asset


@pytest.mark.skipif(not _has_remyx, reason="Remyx dependencies not installed")
@pytest.mark.integration
class TestRemyxGroupChatIntegration:
    """Integration tests for RemyxCodeExecutor with GroupChat."""

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor._setup_container")
    def test_groupchat_with_remyx_executor(self, mock_setup, mock_parent_init, mock_get_asset, mock_asset):
        """Test RemyxCodeExecutor in a GroupChat setting with multiple agents."""
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        # Create the executor
        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        # Verify executor is properly initialized
        assert executor.arxiv_id == "2010.11929v2"
        assert executor.paper_info is not None

        # Get paper context and verify content with regex
        paper_context = executor.get_paper_context()

        # Assert content using regular expressions as requested
        assert re.search(r"Title:\s+An Image is Worth", paper_context), "Expected paper title in context"
        assert re.search(r"arXiv ID:\s+2010\.11929v2", paper_context), "Expected arXiv ID in context"
        assert re.search(r"GitHub:\s+https://github\.com", paper_context), "Expected GitHub URL in context"
        assert re.search(r"Working Directory:\s+/app", paper_context), "Expected working directory in context"
        assert re.search(r"Vision Transformer", paper_context), "Expected reasoning content in context"
        assert re.search(r"train\.py", paper_context), "Expected quickstart hint in context"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    @patch("autogen.ConversableAgent")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_groupchat_agent_creation(self, mock_agent_class, mock_parent_init, mock_get_asset, mock_asset):
        """Test that create_agents produces valid agents for GroupChat."""
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        # Mock the ConversableAgent instances
        mock_executor_agent = Mock()
        mock_executor_agent.name = "code_executor"
        mock_writer_agent = Mock()
        mock_writer_agent.name = "research_explorer"
        mock_agent_class.side_effect = [mock_executor_agent, mock_writer_agent]

        # Create agents
        executor_agent, writer_agent = executor.create_agents(
            goal="Test the Vision Transformer implementation",
            llm_model="gpt-4o",
            human_input_mode="NEVER",
        )

        # Verify agents were created with correct names
        assert executor_agent.name == "code_executor"
        assert writer_agent.name == "research_explorer"

        # Verify ConversableAgent was called twice
        assert mock_agent_class.call_count == 2

        # Verify the executor agent has correct config
        executor_call_kwargs = mock_agent_class.call_args_list[0][1]
        assert executor_call_kwargs["llm_config"] is False
        assert "executor" in executor_call_kwargs["code_execution_config"]

        # Verify the writer agent has correct config
        writer_call_kwargs = mock_agent_class.call_args_list[1][1]
        assert "system_message" in writer_call_kwargs
        assert writer_call_kwargs["code_execution_config"] is False

        # Assert system message content with regex
        system_message = writer_call_kwargs["system_message"]
        assert re.search(r"Vision Transformer", system_message), "Expected paper context in system message"
        assert re.search(r"Test the Vision Transformer", system_message), "Expected goal in system message"
        assert re.search(r"Repository is at /app", system_message), "Expected guidelines in system message"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    @patch("autogen.ConversableAgent")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_groupchat_with_system_message(self, mock_agent_class, mock_parent_init, mock_get_asset, mock_asset):
        """Test create_agents with additional system message for domain-specific needs."""
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        mock_executor_agent = Mock()
        mock_writer_agent = Mock()
        mock_agent_class.side_effect = [mock_executor_agent, mock_writer_agent]

        additional_system_message = "Focus only on the attention mechanism. Output all findings as JSON."

        executor_agent, writer_agent = executor.create_agents(
            goal="Analyze the attention implementation",
            system_message=additional_system_message,
            human_input_mode="NEVER",
        )

        # Get the system message from the writer agent call
        writer_call_kwargs = mock_agent_class.call_args_list[1][1]
        full_system_message = writer_call_kwargs["system_message"]

        # Assert system message content is included with regex
        assert re.search(r"Focus only on the attention mechanism", full_system_message), (
            "Expected system message content"
        )
        assert re.search(r"Output all findings as JSON", full_system_message), "Expected JSON output instruction"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    @patch("autogen.ConversableAgent")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_groupchat_explore_with_mock_chat(self, mock_agent_class, mock_parent_init, mock_get_asset, mock_asset):
        """Test the explore method with mocked agent chat."""
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        # Mock chat result
        mock_chat_result = Mock()
        mock_chat_result.chat_history = [
            {"role": "user", "content": "Let's begin exploring this research paper."},
            {
                "role": "assistant",
                "content": "I'll start by examining the directory structure. ```bash\nls -la /app\n```",
            },
            {
                "role": "user",
                "content": "total 48\ndrwxr-xr-x 5 root root 4096 Jan 1 00:00 .\n-rw-r--r-- 1 root root 1234 Jan 1 00:00 train.py",
            },
            {"role": "assistant", "content": "I can see the Vision Transformer implementation. TERMINATE"},
        ]
        mock_chat_result.summary = "Explored the Vision Transformer repository structure."
        mock_chat_result.cost = {"usage_including_cached_inference": {"total_cost": 0.0042}}

        # Setup mock agents
        mock_executor_agent = Mock()
        mock_executor_agent.initiate_chat.return_value = mock_chat_result
        mock_writer_agent = Mock()
        mock_agent_class.side_effect = [mock_executor_agent, mock_writer_agent]

        # Run exploration
        result = executor.explore(
            goal="Explore the repository structure",
            interactive=False,
            verbose=False,
        )

        # Verify the result
        assert result == mock_chat_result
        assert len(result.chat_history) == 4

        # Assert chat content with regex
        chat_content = " ".join([msg["content"] for msg in result.chat_history])
        assert re.search(r"Let's begin exploring", chat_content), "Expected exploration start message"
        assert re.search(r"directory structure", chat_content), "Expected directory exploration"
        assert re.search(r"Vision Transformer", chat_content), "Expected paper recognition"
        assert re.search(r"TERMINATE", chat_content), "Expected termination"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_groupchat_default_goal(self, mock_parent_init, mock_get_asset, mock_asset):
        """Test that the default exploration goal is used correctly."""

        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        # Build system message with default goal
        system_message = executor._build_system_message()

        # Assert default goal content is present with regex
        assert re.search(r"Phase 1: Understanding", system_message), "Expected Phase 1 in default goal"
        assert re.search(r"Phase 2: Experimentation", system_message), "Expected Phase 2 in default goal"
        assert re.search(r"Phase 3: Analysis", system_message), "Expected Phase 3 in default goal"
        assert re.search(r"TERMINATE when exploration is complete", system_message), "Expected termination instruction"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_groupchat_system_message_structure(self, mock_parent_init, mock_get_asset, mock_asset):
        """Test the structure of the generated system message."""
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")

        custom_goal = "Run benchmarks and compare results"
        additional_system_message = "Use GPU acceleration when available"

        full_system_message = executor._build_system_message(
            goal=custom_goal,
            system_message=additional_system_message,
        )

        # Verify structure with regex - paper context should come first
        assert re.search(r"^Paper Information:", full_system_message), "Expected paper info at start"

        # Verify mission section
        assert re.search(r"\*\*Your Mission:\*\*\s+Run benchmarks", full_system_message), "Expected mission section"

        # Verify guidelines
        assert re.search(r"\*\*Important Guidelines:\*\*", full_system_message), "Expected guidelines section"
        assert re.search(r"Execute ONE command at a time", full_system_message), "Expected execution guideline"
        assert re.search(r"What You Can Do:", full_system_message), "Expected capabilities section"

        # Verify additional system message is appended
        assert re.search(r"Use GPU acceleration when available", full_system_message), (
            "Expected additional system message content"
        )


@pytest.mark.skipif(not _has_remyx, reason="Remyx dependencies not installed")
@pytest.mark.integration
class TestRemyxUtilsIntegration:
    """Integration tests for Remyx utility functions."""

    def test_format_chat_result_utility(self):
        """Test the format_chat_result utility function."""
        from autogen.coding.utils import format_chat_result

        # Create mock result
        mock_result = Mock()
        mock_result.chat_history = [
            {"name": "code_executor", "content": "Starting exploration..."},
            {"name": "research_explorer", "content": "Let me examine the code."},
            {"name": "code_executor", "content": "Execution complete."},
        ]
        mock_result.chat_id = "test_groupchat_123"
        mock_result.summary = "Successfully explored the Vision Transformer implementation."
        mock_result.cost = {"usage_including_cached_inference": {"total_cost": 0.0567}}

        formatted = format_chat_result(mock_result)

        # Assert formatting with regex
        assert re.search(r"Exploration Session Summary", formatted), "Expected summary header"
        assert re.search(r"Total messages:\s+3", formatted), "Expected message count"
        assert re.search(r"Chat ID:\s+test_groupchat_123", formatted), "Expected chat ID"
        assert re.search(r"Cost:\s+\$0\.0567", formatted), "Expected cost"
        assert re.search(r"Vision Transformer", formatted), "Expected summary content"
        assert re.search(r"Last 3 messages:", formatted), "Expected message preview section"
        assert re.search(r"\[code_executor\]:", formatted), "Expected executor agent messages"
        assert re.search(r"\[research_explorer\]:", formatted), "Expected explorer agent messages"

    @patch("autogen.coding.remyx_code_executor.remyxai_get_asset")
    @patch("autogen.coding.remyx_code_executor.DockerCommandLineCodeExecutor.__init__")
    def test_format_chat_result_via_executor(self, mock_parent_init, mock_get_asset, mock_asset):
        """Test format_chat_result accessed via executor static method."""
        mock_get_asset.return_value = mock_asset
        mock_parent_init.return_value = None

        # Create mock result
        mock_result = Mock()
        mock_result.chat_history = [{"role": "user", "content": "Hello"}]
        mock_result.chat_id = "test_123"
        mock_result.summary = "Test completed"
        mock_result.cost = {"usage_including_cached_inference": {"total_cost": 0.01}}

        formatted = RemyxCodeExecutor.format_chat_result(mock_result)

        # Verify it works via the static method
        assert re.search(r"Exploration Session Summary", formatted), "Expected summary via static method"
        assert re.search(r"test_123", formatted), "Expected chat ID via static method"
