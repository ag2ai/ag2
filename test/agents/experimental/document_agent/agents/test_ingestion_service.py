# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autogen.agents.experimental.document_agent.agents.ingestion_service import DocumentIngestionService
from autogen.agents.experimental.document_agent.core.base_interfaces import RAGQueryEngine
from autogen.agents.experimental.document_agent.core.config import DocAgentConfig, ProcessingConfig
from autogen.agents.experimental.document_agent.ingestion.document_processor import DoclingDocumentProcessor


class TestDocumentIngestionService:
    """Test cases for DocumentIngestionService."""

    @pytest.fixture
    def mock_query_engine(self) -> Mock:
        """Create a mock RAG query engine."""
        mock_engine = Mock(spec=RAGQueryEngine)
        # Explicitly mock the add_docs method to ensure it's a Mock object
        mock_engine.add_docs = Mock()
        return mock_engine

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create a mock DocAgentConfig."""
        config = Mock(spec=DocAgentConfig)
        # Create a mock ProcessingConfig
        processing_config = Mock(spec=ProcessingConfig)
        processing_config.output_dir = Path("/tmp/output")
        processing_config.chunk_size = 1000
        processing_config.supported_formats = ["txt", "pdf", "docx"]

        # Set the processing attribute
        config.processing = processing_config
        return config

    @pytest.fixture
    def mock_document_processor(self) -> Mock:
        """Create a mock DoclingDocumentProcessor."""
        return Mock(spec=DoclingDocumentProcessor)

    @pytest.fixture
    def service(self, mock_query_engine: Mock, mock_config: Mock) -> DocumentIngestionService:
        """Create a DocumentIngestionService instance for testing."""
        with patch(
            "autogen.agents.experimental.document_agent.agents.ingestion_service.DoclingDocumentProcessor"
        ) as mock_processor_class:
            mock_processor_class.return_value = Mock(spec=DoclingDocumentProcessor)
            return DocumentIngestionService(mock_query_engine, mock_config)

    def test_init_with_query_engine_and_config(self, mock_query_engine: Mock, mock_config: Mock) -> None:
        """Test initialization with both query engine and config."""
        with patch(
            "autogen.agents.experimental.document_agent.agents.ingestion_service.DoclingDocumentProcessor"
        ) as mock_processor_class:
            mock_processor_class.return_value = Mock(spec=DoclingDocumentProcessor)

            service = DocumentIngestionService(mock_query_engine, mock_config)

            assert service.query_engine == mock_query_engine
            assert service.config == mock_config
            mock_processor_class.assert_called_once_with(
                output_dir=mock_config.processing.output_dir, chunk_size=mock_config.processing.chunk_size
            )

    def test_init_with_default_config(self, mock_query_engine: Mock) -> None:
        """Test initialization with default config."""
        with (
            patch(
                "autogen.agents.experimental.document_agent.agents.ingestion_service.DocAgentConfig"
            ) as mock_config_class,
            patch(
                "autogen.agents.experimental.document_agent.agents.ingestion_service.DoclingDocumentProcessor"
            ) as mock_processor_class,
        ):
            # Create a proper mock config with nested structure
            mock_config_instance = Mock(spec=DocAgentConfig)
            mock_processing_config = Mock(spec=ProcessingConfig)
            mock_processing_config.output_dir = Path("./parsed_docs")
            mock_processing_config.chunk_size = 512
            mock_config_instance.processing = mock_processing_config

            mock_config_class.return_value = mock_config_instance
            mock_processor_class.return_value = Mock(spec=DoclingDocumentProcessor)

            service = DocumentIngestionService(mock_query_engine)

            assert service.query_engine == mock_query_engine
            assert service.config == mock_config_instance
            mock_config_class.assert_called_once()

    def test_ingest_document_success(self, service: DocumentIngestionService, mock_document_processor: Mock) -> None:
        """Test successful document ingestion."""
        # Setup
        document_path = "/path/to/document.pdf"
        processed_files: list[Path] = [Path("/tmp/output/doc1.txt"), Path("/tmp/output/doc2.txt")]

        service.document_processor = mock_document_processor
        mock_document_processor.process_document.return_value = processed_files

        # Execute
        result = service.ingest_document(document_path)

        # Assert
        mock_document_processor.process_document.assert_called_once_with(
            document_path, service.config.processing.output_dir
        )

        assert "Successfully ingested 2 document(s)" in result
        assert "doc1.txt" in result
        assert "doc2.txt" in result

    def test_ingest_document_no_processed_files(
        self, service: DocumentIngestionService, mock_document_processor: Mock
    ) -> None:
        """Test document ingestion when no files are processed."""
        # Setup
        document_path = "/path/to/document.pdf"
        mock_document_processor.process_document.return_value = []
        service.document_processor = mock_document_processor

        # Execute
        result = service.ingest_document(document_path)

        # Assert
        assert result == "No documents were processed."

    def test_ingest_document_exception(self, service: DocumentIngestionService, mock_document_processor: Mock) -> None:
        """Test document ingestion when an exception occurs."""
        # Setup
        document_path = "/path/to/document.pdf"
        mock_document_processor.process_document.side_effect = Exception("Processing failed")
        service.document_processor = mock_document_processor

        # Execute
        result = service.ingest_document(document_path)

        # Assert
        assert result == "Error ingesting document: Processing failed"

    def test_ingest_documents_multiple_success(self, service: DocumentIngestionService) -> None:
        """Test successful ingestion of multiple documents."""
        # Setup
        document_paths: list[str] = ["/path/to/doc1.pdf", "/path/to/doc2.txt"]

        with patch.object(service, "ingest_document") as mock_ingest:
            mock_ingest.side_effect = ["Success 1", "Success 2"]

            # Execute
            results = service.ingest_documents(document_paths)

            # Assert
            assert results == ["Success 1", "Success 2"]
            assert mock_ingest.call_count == 2
            mock_ingest.assert_any_call("/path/to/doc1.pdf")
            mock_ingest.assert_any_call("/path/to/doc2.txt")

    def test_ingest_documents_empty_sequence(self, service: DocumentIngestionService) -> None:
        """Test ingestion of empty document sequence."""
        # Setup
        document_paths: list[str] = []

        # Execute
        results = service.ingest_documents(document_paths)

        # Assert
        assert results == []

    def test_ingest_directory_success(self, service: DocumentIngestionService) -> None:
        """Test successful directory ingestion."""
        # Setup
        directory_path = "/path/to/documents"
        mock_directory = Mock(spec=Path)
        mock_directory.exists.return_value = True
        mock_directory.is_dir.return_value = True

        # Mock finding files - need to properly mock the glob method for each extension
        # The method calls glob for each supported format (txt, pdf, docx)
        mock_directory.glob.side_effect = [
            [Path("doc1.txt")],  # First call for *.txt
            [Path("doc2.pdf")],  # Second call for *.pdf
            [Path("doc3.docx")],  # Third call for *.docx
            [],  # Fourth call for *.TXT (uppercase)
            [],  # Fifth call for *.PDF (uppercase)
            [],  # Sixth call for *.DOCX (uppercase)
        ]

        with (
            patch(
                "autogen.agents.experimental.document_agent.agents.ingestion_service.Path", return_value=mock_directory
            ),
            patch.object(service, "ingest_documents") as mock_ingest_docs,
        ):
            mock_ingest_docs.return_value = [
                "Successfully ingested 1 document(s): ['doc1.txt']",
                "Successfully ingested 1 document(s): ['doc2.pdf']",
                "Successfully ingested 1 document(s): ['doc3.docx']",
            ]

            # Execute
            result = service.ingest_directory(directory_path)

            # Assert
            assert "Ingestion complete: 3 successful, 0 failed" in result
            mock_ingest_docs.assert_called_once()

    def test_ingest_directory_not_found(self, service: DocumentIngestionService) -> None:
        """Test directory ingestion when directory doesn't exist."""
        # Setup
        directory_path = "/nonexistent/path"
        mock_directory = Mock(spec=Path)
        mock_directory.exists.return_value = False

        with patch(
            "autogen.agents.experimental.document_agent.agents.ingestion_service.Path", return_value=mock_directory
        ):
            # Execute
            result = service.ingest_directory(directory_path)

            # Assert
            assert result == f"Directory not found: {directory_path}"

    def test_ingest_directory_no_supported_files(self, service: DocumentIngestionService) -> None:
        """Test directory ingestion when no supported files are found."""
        # Setup
        directory_path = "/path/to/documents"
        mock_directory = Mock(spec=Path)
        mock_directory.exists.return_value = True
        mock_directory.is_dir.return_value = True

        # Mock finding no files
        mock_directory.glob.return_value = []

        with patch(
            "autogen.agents.experimental.document_agent.agents.ingestion_service.Path", return_value=mock_directory
        ):
            # Execute
            result = service.ingest_directory(directory_path)

            # Assert
            assert result == f"No supported documents found in directory: {directory_path}"

    def test_ingest_directory_exception(self, service: DocumentIngestionService) -> None:
        """Test directory ingestion when an exception occurs."""
        # Setup
        directory_path = "/path/to/documents"

        with patch(
            "autogen.agents.experimental.document_agent.agents.ingestion_service.Path",
            side_effect=Exception("Path error"),
        ):
            # Execute
            result = service.ingest_directory(directory_path)

            # Assert
            assert result == "Error ingesting directory: Path error"

    def test_get_ingestion_status(self, service: DocumentIngestionService) -> None:
        """Test getting ingestion service status."""
        # Execute
        status = service.get_ingestion_status()

        # Assert
        assert status["query_engine_configured"] is True
        assert status["output_directory"] == str(service.config.processing.output_dir)
        assert status["chunk_size"] == service.config.processing.chunk_size
        assert status["supported_formats"] == service.config.processing.supported_formats

    def test_get_ingestion_status_no_query_engine(self, mock_config: Mock) -> None:
        """Test getting status when no query engine is configured."""
        # Setup - need to patch the DoclingDocumentProcessor import to avoid dependency issues
        with patch(
            "autogen.agents.experimental.document_agent.agents.ingestion_service.DoclingDocumentProcessor"
        ) as mock_processor_class:
            mock_processor_class.return_value = Mock(spec=DoclingDocumentProcessor)

            service = DocumentIngestionService(None, mock_config)  # type: ignore[arg-type]

            # Execute
            status = service.get_ingestion_status()

            # Assert
            assert status["query_engine_configured"] is False

    def test_set_query_engine(self, service: DocumentIngestionService, mock_query_engine: Mock) -> None:
        """Test setting a new query engine."""
        # Setup
        new_query_engine = Mock(spec=RAGQueryEngine)

        # Execute
        service.set_query_engine(new_query_engine)

        # Assert
        assert service.query_engine == new_query_engine

    def test_set_query_engine_logs_info(self, service: DocumentIngestionService, mock_query_engine: Mock) -> None:
        """Test that setting query engine logs an info message."""
        # Setup
        new_query_engine = Mock(spec=RAGQueryEngine)

        with patch("autogen.agents.experimental.document_agent.agents.ingestion_service.logger") as mock_logger:
            # Execute
            service.set_query_engine(new_query_engine)

            # Assert
            mock_logger.info.assert_called_once_with("Query engine updated for ingestion service")
