# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for RemyxCodeExecutor with real API calls and Docker."""

import os
from pathlib import Path

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
