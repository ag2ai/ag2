# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import Mock, mock_open, patch

import pytest
import requests

from autogen.agents.experimental.document_agent.task_manager_utils import (
    aggregate_rag_results,
    execute_single_query,
    extract_text_from_pdf,
    ingest_to_engines,
    process_single_document,
)


class TestExtractTextFromPDF:
    """Test extract_text_from_pdf function."""

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.fitz")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.requests.get")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.tempfile.TemporaryDirectory")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.LLMLingua")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.TextMessageCompressor")
    def test_extract_text_from_pdf_url_success(
        self, mock_compressor_class: Any, mock_llm_lingua_class: Any, mock_temp_dir: Any, mock_get: Any, mock_fitz: Any
    ) -> None:
        """Test successful PDF text extraction from URL."""
        # Setup mocks
        mock_response = Mock()
        mock_response.content = b"fake pdf content"
        mock_get.return_value = mock_response

        # Mock the temporary directory context manager properly
        mock_temp_dir_instance = Mock()
        mock_temp_dir_instance.__enter__ = Mock(return_value="/tmp/test")
        mock_temp_dir_instance.__exit__ = Mock(return_value=None)
        mock_temp_dir.return_value = mock_temp_dir_instance

        # Create a proper mock document that supports iteration
        mock_doc = Mock()
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Page 2 content"

        # Fix: Use side_effect instead of return_value for iteration
        mock_doc.__iter__ = Mock(return_value=iter([mock_page1, mock_page2]))
        mock_fitz.open.return_value.__enter__.return_value = mock_doc

        mock_compressor = Mock()
        mock_compressor.apply_transform.return_value = [{"content": "compressed text"}]
        mock_compressor_class.return_value = mock_compressor

        mock_llm_lingua = Mock()
        mock_llm_lingua_class.return_value = mock_llm_lingua

        # Mock urllib3 to return a URL with scheme
        with (
            patch(
                "autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url"
            ) as mock_parse_url,
            patch("builtins.open", mock_open()) as mock_file,
        ):
            mock_parsed_url = Mock()
            mock_parsed_url.scheme = "https"
            mock_parse_url.return_value = mock_parsed_url

            # Execute
            result = extract_text_from_pdf("https://example.com/test.pdf")

            # Assertions
            assert result == [{"content": "compressed text"}]
            mock_get.assert_called_once_with("https://example.com/test.pdf")
            mock_fitz.open.assert_called_once()
            mock_compressor.apply_transform.assert_called_once_with([{"content": "Page 1 contentPage 2 content"}])
            # Verify file was opened for writing - use Path object instead of string
            mock_file.assert_called_with(Path("/tmp/test/temp.pdf"), "wb")

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    def test_extract_text_from_pdf_non_url_raises_error(self, mock_parse_url: Any) -> None:
        """Test that non-URL input raises ValueError."""
        mock_parsed_url = Mock()
        mock_parsed_url.scheme = None
        mock_parse_url.return_value = mock_parsed_url

        with pytest.raises(ValueError, match="doc_path must be a string or a URL"):
            extract_text_from_pdf("not_a_url")

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.requests.get")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    def test_extract_text_from_pdf_request_error(self, mock_parse_url: Any, mock_get: Any) -> None:
        """Test handling of request errors."""
        mock_parsed_url = Mock()
        mock_parsed_url.scheme = "https"
        mock_parse_url.return_value = mock_parsed_url

        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException):
            extract_text_from_pdf("https://example.com/test.pdf")


class TestAggregateRagResults:
    """Test aggregate_rag_results function."""

    def test_aggregate_rag_results_empty(self) -> None:
        """Test aggregation with empty results."""
        result = aggregate_rag_results("test query", {})
        expected = "Query: test query\nAnswer: No results found from any RAG engine."
        assert result == expected

    def test_aggregate_rag_results_single_engine(self) -> None:
        """Test aggregation with single engine result."""
        results: dict[str, Any] = {"vector": {"answer": "Vector answer", "citations": []}}
        result = aggregate_rag_results("test query", results)
        expected = "Query: test query\n\nVECTOR Results:\nAnswer: Vector answer"
        assert result == expected

    def test_aggregate_rag_results_with_citations(self) -> None:
        """Test aggregation with citations."""
        results: dict[str, Any] = {
            "vector": {"answer": "Vector answer", "citations": [{"file_path": "doc1.pdf"}, {"file_path": "doc2.pdf"}]}
        }
        result = aggregate_rag_results("test query", results)
        assert "Citations:" in result
        assert "[1] doc1.pdf" in result
        assert "[2] doc2.pdf" in result

    def test_aggregate_rag_results_multiple_engines(self) -> None:
        """Test aggregation with multiple engines."""
        results: dict[str, Any] = {"vector": {"answer": "Vector answer"}, "graph": {"answer": "Graph answer"}}
        result = aggregate_rag_results("test query", results)
        assert "VECTOR Results:" in result
        assert "GRAPH Results:" in result
        assert "Vector answer" in result
        assert "Graph answer" in result

    def test_aggregate_rag_results_missing_answer(self) -> None:
        """Test aggregation when answer is missing."""
        results: dict[str, Any] = {"vector": {}}
        result = aggregate_rag_results("test query", results)
        assert "No answer available" in result


class TestIngestToEngines:
    """Test ingest_to_engines function."""

    @patch("autogen.agentchat.contrib.graph_rag.document.Document")
    @patch("autogen.agentchat.contrib.graph_rag.document.DocumentType")
    def test_ingest_to_engines_vector_only(self, mock_doc_type: Any, mock_document: Any) -> None:
        """Test ingestion to vector engine only."""
        # Setup mocks
        mock_doc_type.TEXT = "TEXT"
        mock_doc_type.PDF = "PDF"
        mock_doc = Mock()
        mock_document.return_value = mock_doc

        # Mock engines
        vector_engine = Mock()
        rag_engines = {"vector": vector_engine}
        rag_config: list[str] = ["vector"]

        # Execute
        ingest_to_engines("/path/to/file.md", "/path/to/input.pdf", rag_config, rag_engines)  # type: ignore

        # Assertions
        vector_engine.add_docs.assert_called_once_with(new_doc_paths_or_urls=["/path/to/file.md"])
        mock_document.assert_called_once_with(doctype="PDF", path_or_url="/path/to/file.md", data=None)

    @patch("autogen.agentchat.contrib.graph_rag.document.Document")
    @patch("autogen.agentchat.contrib.graph_rag.document.DocumentType")
    def test_ingest_to_engines_graph_uninitialized(self, mock_doc_type: Any, mock_document: Any) -> None:
        """Test ingestion to uninitialized graph engine."""
        # Setup mocks
        mock_doc_type.TEXT = "TEXT"
        mock_doc = Mock()
        mock_document.return_value = mock_doc

        # Mock engines
        graph_engine = Mock()
        # Fix: Remove _initialized attribute to simulate uninitialized state
        del graph_engine._initialized
        rag_engines = {"graph": graph_engine}
        rag_config: list[str] = ["graph"]

        # Execute
        ingest_to_engines("/path/to/file.md", "/path/to/input.txt", rag_config, rag_engines)  # type: ignore

        # Assertions
        graph_engine.init_db.assert_called_once_with([mock_doc])
        assert graph_engine._initialized is True

    @patch("autogen.agentchat.contrib.graph_rag.document.Document")
    @patch("autogen.agentchat.contrib.graph_rag.document.DocumentType")
    def test_ingest_to_engines_graph_initialized(self, mock_doc_type: Any, mock_document: Any) -> None:
        """Test ingestion to initialized graph engine."""
        # Setup mocks
        mock_doc_type.TEXT = "TEXT"
        mock_doc = Mock()
        mock_document.return_value = mock_doc

        # Mock engines
        graph_engine = Mock()
        graph_engine._initialized = True  # Already initialized
        rag_engines = {"graph": graph_engine}
        rag_config: list[str] = ["graph"]

        # Execute
        ingest_to_engines("/path/to/file.md", "/path/to/input.txt", rag_config, rag_engines)  # type: ignore

        # Assertions
        graph_engine.init_db.assert_not_called()
        graph_engine.add_records.assert_called_once_with([mock_doc])

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.logger")
    @patch("autogen.agentchat.contrib.graph_rag.document.Document")
    @patch("autogen.agentchat.contrib.graph_rag.document.DocumentType")
    def test_ingest_to_engines_error_handling(self, mock_doc_type: Any, mock_document: Any, mock_logger: Any) -> None:
        """Test error handling during ingestion."""
        # Setup mocks
        mock_doc_type.TEXT = "TEXT"
        mock_doc = Mock()
        mock_document.return_value = mock_doc

        # Mock engines with error
        vector_engine = Mock()
        vector_engine.add_docs.side_effect = Exception("Ingestion failed")
        rag_engines = {"vector": vector_engine}
        rag_config: list[str] = ["vector"]

        # Execute
        ingest_to_engines("/path/to/file.md", "/path/to/input.txt", rag_config, rag_engines)  # type: ignore

        # Assertions
        mock_logger.warning.assert_called_once_with("Failed to ingest to vector engine: Ingestion failed")

    def test_ingest_to_engines_missing_engine(self) -> None:
        """Test handling of missing engine."""
        rag_engines: dict[str, Any] = {}
        rag_config: list[str] = ["vector"]

        # Should not raise any error
        ingest_to_engines("/path/to/file.md", "/path/to/input.txt", rag_config, rag_engines)  # type: ignore


class TestProcessSingleDocument:
    """Test process_single_document function."""

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.requests.get")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.tempfile.TemporaryDirectory")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.fitz")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.compress_and_save_text")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.ingest_to_engines")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_pdf_url_success(
        self,
        mock_file_open: Any,
        mock_ingest: Any,
        mock_compress: Any,
        mock_fitz: Any,
        mock_temp_dir: Any,
        mock_get: Any,
        mock_parse_url: Any,
    ) -> None:
        """Test successful processing of PDF URL."""
        # Setup mocks
        mock_parsed_url = Mock()
        mock_parsed_url.scheme = "https"
        mock_parse_url.return_value = mock_parsed_url

        mock_response = Mock()
        mock_response.content = b"fake pdf content"
        mock_get.return_value = mock_response

        # Fix: Mock the actual temp directory path properly
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"

        # Create a proper mock document that supports iteration
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "PDF content"

        # Fix: Use side_effect instead of return_value for iteration
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz.open.return_value.__enter__.return_value = mock_doc

        mock_compress.return_value = "/path/to/output.md"

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": Mock()}

        # Execute
        result = process_single_document("https://example.com/test.pdf", Path("/output"), rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result == ("https://example.com/test.pdf", True, "")
        mock_compress.assert_called_once()
        mock_ingest.assert_called_once()

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.fitz")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.compress_and_save_text")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.ingest_to_engines")
    def test_process_local_pdf_success(
        self, mock_ingest: Any, mock_compress: Any, mock_fitz: Any, mock_parse_url: Any
    ) -> None:
        """Test successful processing of local PDF file."""
        # Setup mocks
        mock_parsed_url = Mock()
        mock_parsed_url.scheme = None
        mock_parse_url.return_value = mock_parsed_url

        # Create a proper mock document that supports iteration
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "PDF content"

        # Fix: Use side_effect instead of return_value for iteration
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz.open.return_value.__enter__.return_value = mock_doc

        mock_compress.return_value = "/path/to/output.md"

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": Mock()}

        # Execute
        result = process_single_document("/path/to/test.pdf", Path("/output"), rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result == ("/path/to/test.pdf", True, "")
        mock_compress.assert_called_once()
        mock_ingest.assert_called_once()

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.docling_parse_docs")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.ingest_to_engines")
    def test_process_non_pdf_success(self, mock_ingest: Any, mock_docling: Any, mock_parse_url: Any) -> None:
        """Test successful processing of non-PDF document."""
        # Setup mocks
        mock_parsed_url = Mock()
        mock_parsed_url.scheme = None
        mock_parse_url.return_value = mock_parsed_url

        mock_output_file = Path("/output/test.md")
        mock_docling.return_value = [mock_output_file]

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": Mock()}

        # Execute
        result = process_single_document("/path/to/test.txt", Path("/output"), rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result == ("/path/to/test.txt", True, "")
        mock_docling.assert_called_once_with(
            input_file_path="/path/to/test.txt", output_dir_path=Path("/output"), output_formats=["markdown"]
        )
        mock_ingest.assert_called_once()

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    @patch("autogen.agents.experimental.document_agent.task_manager_utils.docling_parse_docs")
    def test_process_non_pdf_no_output(self, mock_docling: Any, mock_parse_url: Any) -> None:
        """Test processing non-PDF document with no output."""
        # Setup mocks
        mock_parsed_url = Mock()
        mock_parsed_url.scheme = None
        mock_parse_url.return_value = mock_parsed_url

        mock_docling.return_value = []

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": Mock()}

        # Execute
        result = process_single_document("/path/to/test.txt", Path("/output"), rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result == ("/path/to/test.txt", False, "No valid markdown output generated")

    @patch("autogen.agents.experimental.document_agent.task_manager_utils.urllib3.util.url.parse_url")
    def test_process_document_exception(self, mock_parse_url: Any) -> None:
        """Test handling of exceptions during processing."""
        # Setup mocks
        mock_parse_url.side_effect = Exception("Parse error")

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": Mock()}

        # Execute
        result = process_single_document("/path/to/test.txt", Path("/output"), rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result == ("/path/to/test.txt", False, "Parse error")


class TestExecuteSingleQuery:
    """Test execute_single_query function."""

    def test_execute_query_vector_with_citations(self) -> None:
        """Test vector query with citations."""
        # Mock engine with citations
        mock_engine = Mock()
        mock_engine.enable_query_citations = True
        mock_answer_with_citations = Mock()
        mock_answer_with_citations.answer = "Test answer"

        mock_source = Mock()
        mock_source.node.get_text.return_value = "Source text"
        mock_source.metadata.get.return_value = "source.pdf"
        mock_answer_with_citations.citations = [mock_source]

        mock_engine.query_with_citations.return_value = mock_answer_with_citations

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": mock_engine}

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "Test answer" in result[1]
        assert "VECTOR Results:" in result[1]
        mock_engine.query_with_citations.assert_called_once_with("test query")

    def test_execute_query_vector_without_citations(self) -> None:
        """Test vector query without citations."""
        # Mock engine without citations
        mock_engine = Mock()
        # Fix: Ensure the engine doesn't have citation capabilities
        mock_engine.enable_query_citations = False
        mock_engine.query.return_value = "Test answer"

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": mock_engine}

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "Test answer" in result[1]
        mock_engine.query.assert_called_once_with("test query")

    def test_execute_query_graph_success(self) -> None:
        """Test graph query success."""
        # Mock graph engine
        mock_engine = Mock()
        mock_engine.index = "some_index"  # Has index, so already connected

        mock_graph_result = Mock()
        mock_graph_result.answer = "Graph answer"
        mock_graph_result.results = {"nodes": []}
        mock_engine.query.return_value = mock_graph_result

        rag_config: list[str] = ["graph"]
        rag_engines = {"graph": mock_engine}

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "Graph answer" in result[1]
        assert "GRAPH Results:" in result[1]
        mock_engine.query.assert_called_once_with("test query")

    def test_execute_query_graph_connect_error(self) -> None:
        """Test graph query with connection error."""
        # Mock graph engine without index
        mock_engine = Mock()
        # Fix: Remove index attribute to simulate unconnected state
        del mock_engine.index
        mock_engine.connect_db.side_effect = Exception("Connection failed")

        rag_config: list[str] = ["graph"]
        rag_engines = {"graph": mock_engine}

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "Error connecting to graph: Connection failed" in result[1]

    def test_execute_query_engine_error(self) -> None:
        """Test query with engine error."""
        # Mock engine that raises error
        mock_engine = Mock()
        # Fix: Ensure the engine doesn't have citation capabilities
        mock_engine.enable_query_citations = False
        mock_engine.query.side_effect = Exception("Query failed")

        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": mock_engine}

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "Error querying vector: Query failed" in result[1]

    def test_execute_query_missing_engine(self) -> None:
        """Test query with missing engine."""
        rag_config: list[str] = ["vector"]
        rag_engines: dict[str, Any] = {}  # No engines

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "No results found from any RAG engine" in result[1]

    def test_execute_query_general_exception(self) -> None:
        """Test query with general exception."""
        rag_config: list[str] = ["vector"]
        rag_engines = {"vector": Mock()}

        with patch(
            "autogen.agents.experimental.document_agent.task_manager_utils.aggregate_rag_results"
        ) as mock_aggregate:
            mock_aggregate.side_effect = Exception("Aggregation failed")

            # Execute
            result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

            # Assertions
            assert result[0] == "test query"
            assert "Error executing query: Aggregation failed" in result[1]

    def test_execute_query_multiple_engines(self) -> None:
        """Test query with multiple engines."""
        # Mock vector engine
        mock_vector_engine = Mock()
        # Fix: Ensure the engine doesn't have citation capabilities
        mock_vector_engine.enable_query_citations = False
        mock_vector_engine.query.return_value = "Vector answer"

        # Mock graph engine
        mock_graph_engine = Mock()
        mock_graph_engine.index = "some_index"
        mock_graph_result = Mock()
        mock_graph_result.answer = "Graph answer"
        mock_graph_result.results = {"nodes": []}
        mock_graph_engine.query.return_value = mock_graph_result

        rag_config: list[str] = ["vector", "graph"]
        rag_engines = {"vector": mock_vector_engine, "graph": mock_graph_engine}

        # Execute
        result = execute_single_query("test query", rag_config, rag_engines)  # type: ignore

        # Assertions
        assert result[0] == "test query"
        assert "Vector answer" in result[1]
        assert "Graph answer" in result[1]
        assert "VECTOR Results:" in result[1]
        assert "GRAPH Results:" in result[1]
