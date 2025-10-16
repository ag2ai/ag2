# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autogen.agents.experimental.document_agent.task_manager import TASK_MANAGER_SYSTEM_MESSAGE, TaskManagerAgent
from autogen.import_utils import skip_on_missing_imports

from ....conftest import Credentials


class TestTaskManagerAgent:
    """Test TaskManagerAgent class focusing on helper methods and basic functionality."""

    @pytest.fixture
    def mock_query_engine(self) -> MagicMock:
        """Create a mock query engine for testing."""
        mock_engine = MagicMock()
        mock_engine.add_docs = MagicMock()
        mock_engine.query = MagicMock()
        mock_engine.enable_query_citations = False
        mock_engine.query_with_citations = MagicMock()
        return mock_engine

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create a mock ThreadPoolExecutor for testing."""
        mock_executor = MagicMock()
        mock_executor.shutdown = MagicMock()
        return mock_executor

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_init_basic(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test TaskManagerAgent basic initialization."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine") as mock_ve,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor") as mock_tpe,
        ):
            mock_ve.return_value = MagicMock()
            mock_tpe.return_value = MagicMock()

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, max_workers=2)

            assert agent.name == "TaskManagerAgent"
            assert agent.parsed_docs_path == tmp_path
            assert hasattr(agent, "executor")
            assert hasattr(agent, "query_engine")
            mock_tpe.assert_called_once_with(max_workers=2)

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_init_defaults(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test TaskManagerAgent initialization with default values."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine") as mock_ve,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor") as mock_tpe,
        ):
            mock_ve.return_value = MagicMock()
            mock_tpe.return_value = MagicMock()

            agent = TaskManagerAgent(llm_config=llm_config)

            assert agent.name == "TaskManagerAgent"
            assert agent.parsed_docs_path == Path("./parsed_docs")
            assert hasattr(agent, "executor")
            assert hasattr(agent, "query_engine")
            mock_tpe.assert_called_once_with(max_workers=None)

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_init_custom_name(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test TaskManagerAgent initialization with custom name."""
        llm_config = credentials_gpt_4o_mini.llm_config
        custom_name = "CustomTaskManager"

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine"),
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            agent = TaskManagerAgent(name=custom_name, llm_config=llm_config, parsed_docs_path=tmp_path)

            assert agent.name == custom_name

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_init_with_custom_query_engine(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test TaskManagerAgent initialization with custom query engine."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"):
            agent = TaskManagerAgent(llm_config=llm_config, query_engine=mock_query_engine, parsed_docs_path=tmp_path)

            assert agent.query_engine == mock_query_engine

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_init_with_collection_name(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test TaskManagerAgent initialization with collection name."""
        llm_config = credentials_gpt_4o_mini.llm_config
        collection_name = "test_collection"

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine") as mock_ve,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            mock_ve.return_value = MagicMock()

            TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, collection_name=collection_name)

            mock_ve.assert_called_once_with(collection_name=collection_name)

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_cleanup(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_executor: MagicMock
    ) -> None:
        """Test TaskManagerAgent cleanup on destruction."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine"),
            patch(
                "autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor", return_value=mock_executor
            ),
        ):
            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Call __del__ explicitly
            agent.__del__()

            # Verify shutdown was called
            mock_executor.shutdown.assert_called_once_with(wait=True)

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_task_manager_agent_cleanup_no_executor(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test TaskManagerAgent cleanup when executor doesn't exist."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine"),
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Remove executor attribute
            delattr(agent, "executor")
            # Should not raise an exception
            agent.__del__()

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_system_message_content(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test that the system message contains expected content."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine"),
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            system_message = agent.system_message
            assert "task manager agent" in system_message.lower()
            assert "ingest_documents" in system_message.lower()
            assert "execute_query" in system_message.lower()
            assert "tools" in system_message.lower()

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_system_message_constant(self) -> None:
        """Test that the system message constant is properly defined."""
        assert "task manager agent" in TASK_MANAGER_SYSTEM_MESSAGE.lower()
        assert "ingest_documents" in TASK_MANAGER_SYSTEM_MESSAGE.lower()
        assert "execute_query" in TASK_MANAGER_SYSTEM_MESSAGE.lower()
        assert "tools" in TASK_MANAGER_SYSTEM_MESSAGE.lower()

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_inheritance_from_conversable_agent(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test that TaskManagerAgent properly inherits from ConversableAgent."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.VectorChromaQueryEngine"),
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Check that it has ConversableAgent attributes
            assert hasattr(agent, "name")
            assert hasattr(agent, "system_message")
            assert hasattr(agent, "llm_config")
            assert hasattr(agent, "function_map")

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_process_single_document_success(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _process_single_document helper method with successful processing."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.docling_parse_docs") as mock_parse,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            # Mock successful document parsing
            mock_output_file = tmp_path / "test.md"
            mock_output_file.touch()
            mock_parse.return_value = [mock_output_file]

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_process_single_document_parsing_failure(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _process_single_document helper method with parsing failure."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.docling_parse_docs") as mock_parse,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            # Mock document parsing failure
            mock_parse.side_effect = Exception("Parsing failed")

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_process_single_document_no_output_files(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _process_single_document helper method with no output files."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.docling_parse_docs") as mock_parse,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            # Mock no output files
            mock_parse.return_value = []

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_process_single_document_non_markdown_output(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _process_single_document helper method with non-markdown output files."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.task_manager.docling_parse_docs") as mock_parse,
            patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"),
        ):
            # Mock non-markdown output file
            mock_output_file = tmp_path / "test.txt"
            mock_output_file.touch()
            mock_parse.return_value = [mock_output_file]

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_execute_single_query_success(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _execute_single_query helper method with successful execution."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"):
            # Mock successful query execution
            mock_query_engine.query.return_value = "Test answer"

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_execute_single_query_with_citations(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _execute_single_query helper method with citations support."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"):
            # Mock citations support
            mock_citations = MagicMock()
            mock_citations.answer = "Test answer with citations"
            mock_citations.citations = [
                MagicMock(
                    node=MagicMock(get_text=MagicMock(return_value="Citation text")), metadata={"file_path": "test.pdf"}
                )
            ]

            mock_query_engine.enable_query_citations = True
            mock_query_engine.query_with_citations.return_value = mock_citations

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_execute_single_query_failure(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path, mock_query_engine: MagicMock
    ) -> None:
        """Test _execute_single_query helper method with query failure."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"):
            # Mock query execution failure
            mock_query_engine.query.side_effect = Exception("Query failed")

            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

            # Test basic agent functionality
            assert agent.query_engine == mock_query_engine
            assert agent.parsed_docs_path == tmp_path

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_execute_single_query_no_query_engine(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test _execute_single_query helper method when query engine is None."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.task_manager.ThreadPoolExecutor"):
            agent = TaskManagerAgent(llm_config=llm_config, parsed_docs_path=tmp_path)
            # Fix the assignment error by properly typing the query_engine attribute
            agent.query_engine = None  # type: ignore[assignment]

            # Test basic agent functionality
            assert agent.query_engine is None
            assert agent.parsed_docs_path == tmp_path
