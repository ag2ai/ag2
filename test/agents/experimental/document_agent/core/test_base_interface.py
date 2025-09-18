# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from autogen.agents.experimental.document_agent.core.base_interfaces import (
    DocumentMetadata,
    DocumentProcessor,
    QueryResult,
    RAGQueryEngine,
    StorageBackend,
)


class TestRAGQueryEngine:
    """Test cases for RAGQueryEngine abstract base class."""

    def test_rag_query_engine_is_abstract(self) -> None:
        """Test that RAGQueryEngine cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RAGQueryEngine()  # type: ignore[abstract]

    def test_rag_query_engine_has_required_methods(self) -> None:
        """Test that RAGQueryEngine has all required abstract methods."""
        required_methods = {"query", "add_docs", "connect_db"}
        assert all(hasattr(RAGQueryEngine, method) for method in required_methods)

    def test_rag_query_engine_methods_are_abstract(self) -> None:
        """Test that RAGQueryEngine methods are properly abstract."""
        # Check that methods exist and are abstract
        assert hasattr(RAGQueryEngine, "query")
        assert hasattr(RAGQueryEngine, "add_docs")
        assert hasattr(RAGQueryEngine, "connect_db")
        # Check that the class itself is abstract
        assert inspect.isabstract(RAGQueryEngine)


class TestDocumentProcessor:
    """Test cases for DocumentProcessor abstract base class."""

    def test_document_processor_is_abstract(self) -> None:
        """Test that DocumentProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DocumentProcessor()  # type: ignore[abstract]

    def test_document_processor_has_required_methods(self) -> None:
        """Test that DocumentProcessor has all required abstract methods."""
        required_methods = {"process_document", "chunk_document"}
        assert all(hasattr(DocumentProcessor, method) for method in required_methods)

    def test_document_processor_methods_are_abstract(self) -> None:
        """Test that DocumentProcessor methods are properly abstract."""
        # Check that methods exist and are abstract
        assert hasattr(DocumentProcessor, "process_document")
        assert hasattr(DocumentProcessor, "chunk_document")
        # Check that the class itself is abstract
        assert inspect.isabstract(DocumentProcessor)


class TestStorageBackend:
    """Test cases for StorageBackend abstract base class."""

    def test_storage_backend_is_abstract(self) -> None:
        """Test that StorageBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            StorageBackend()  # type: ignore[abstract]

    def test_storage_backend_has_required_methods(self) -> None:
        """Test that StorageBackend has all required abstract methods."""
        required_methods = {"store_document", "retrieve_document", "list_documents"}
        assert all(hasattr(StorageBackend, method) for method in required_methods)

    def test_storage_backend_methods_are_abstract(self) -> None:
        """Test that StorageBackend methods are properly abstract."""
        # Check that methods exist and are abstract
        assert hasattr(StorageBackend, "store_document")
        assert hasattr(StorageBackend, "retrieve_document")
        assert hasattr(StorageBackend, "list_documents")
        # Check that the class itself is abstract
        assert inspect.isabstract(StorageBackend)


class TestQueryResult:
    """Test cases for QueryResult model."""

    def test_query_result_creation_with_defaults(self) -> None:
        """Test QueryResult creation with only required fields."""
        result = QueryResult(answer="Test answer")
        assert result.answer == "Test answer"
        assert result.confidence == 0.0
        assert result.sources == []
        assert result.metadata == {}

    def test_query_result_creation_with_all_fields(self) -> None:
        """Test QueryResult creation with all fields specified."""
        metadata = {"source": "test", "timestamp": "2024-01-01"}
        result = QueryResult(
            answer="Test answer",
            confidence=0.95,
            sources=["doc1.pdf", "doc2.pdf"],
            metadata=metadata,
        )
        assert result.answer == "Test answer"
        assert result.confidence == 0.95
        assert result.sources == ["doc1.pdf", "doc2.pdf"]
        assert result.metadata == metadata

    def test_query_result_confidence_validation(self) -> None:
        """Test QueryResult confidence field validation."""
        # Test valid confidence values
        result1 = QueryResult(answer="Test", confidence=0.0)
        result2 = QueryResult(answer="Test", confidence=1.0)
        result3 = QueryResult(answer="Test", confidence=0.5)

        assert result1.confidence == 0.0
        assert result2.confidence == 1.0
        assert result3.confidence == 0.5

    def test_query_result_sources_validation(self) -> None:
        """Test QueryResult sources field validation."""
        sources = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        result = QueryResult(answer="Test", sources=sources)
        assert result.sources == sources
        assert len(result.sources) == 3

    def test_query_result_metadata_validation(self) -> None:
        """Test QueryResult metadata field validation."""
        metadata = {
            "source": "test_document",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0",
            "tags": ["important", "reference"],
        }
        result = QueryResult(answer="Test", metadata=metadata)
        assert result.metadata == metadata
        assert "source" in result.metadata
        assert "timestamp" in result.metadata

    def test_query_result_immutability(self) -> None:
        """Test that QueryResult fields can be modified after creation (Pydantic v2 default behavior)."""
        result = QueryResult(answer="Test answer")

        # Test that fields are mutable (Pydantic v2 default behavior)
        # In Pydantic v2, models are mutable by default unless frozen=True is set
        result.answer = "Modified answer"
        result.confidence = 0.8
        result.sources = ["doc1.pdf"]
        result.metadata = {"key": "value"}

        assert result.answer == "Modified answer"
        assert result.confidence == 0.8
        assert result.sources == ["doc1.pdf"]
        assert result.metadata == {"key": "value"}


class TestDocumentMetadata:
    """Test cases for DocumentMetadata model."""

    def test_document_metadata_creation_with_required_fields(self) -> None:
        """Test DocumentMetadata creation with required fields only."""
        metadata = DocumentMetadata(
            document_id="doc_123",
            file_path="/path/to/document.pdf",
            file_type="pdf",
            file_size=1024,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
        )

        assert metadata.document_id == "doc_123"
        assert metadata.file_path == "/path/to/document.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.file_size == 1024
        assert metadata.created_at == "2024-01-01T00:00:00Z"
        assert metadata.processed_at == "2024-01-01T01:00:00Z"
        assert metadata.chunk_count == 0  # default value
        assert metadata.metadata == {}  # default value

    def test_document_metadata_creation_with_all_fields(self) -> None:
        """Test DocumentMetadata creation with all fields specified."""
        custom_metadata = {
            "author": "John Doe",
            "keywords": ["AI", "documentation"],
            "version": "1.0",
        }

        metadata = DocumentMetadata(
            document_id="doc_456",
            file_path="/path/to/document.docx",
            file_type="docx",
            file_size=2048,
            created_at="2024-01-02T00:00:00Z",
            processed_at="2024-01-02T01:00:00Z",
            chunk_count=5,
            metadata=custom_metadata,
        )

        assert metadata.document_id == "doc_456"
        assert metadata.file_path == "/path/to/document.docx"
        assert metadata.file_type == "docx"
        assert metadata.file_size == 2048
        assert metadata.created_at == "2024-01-02T00:00:00Z"
        assert metadata.processed_at == "2024-01-02T01:00:00Z"
        assert metadata.chunk_count == 5
        assert metadata.metadata == custom_metadata

    def test_document_metadata_file_size_validation(self) -> None:
        """Test DocumentMetadata file_size field validation."""
        # Test valid file sizes
        metadata1 = DocumentMetadata(
            document_id="doc1",
            file_path="/path/to/doc1.pdf",
            file_type="pdf",
            file_size=0,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
        )

        metadata2 = DocumentMetadata(
            document_id="doc2",
            file_path="/path/to/doc2.pdf",
            file_type="pdf",
            file_size=1000000,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
        )

        assert metadata1.file_size == 0
        assert metadata2.file_size == 1000000

    def test_document_metadata_chunk_count_validation(self) -> None:
        """Test DocumentMetadata chunk_count field validation."""
        metadata = DocumentMetadata(
            document_id="doc",
            file_path="/path/to/doc.pdf",
            file_type="pdf",
            file_size=1024,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
            chunk_count=10,
        )

        assert metadata.chunk_count == 10

    def test_document_metadata_custom_metadata_validation(self) -> None:
        """Test DocumentMetadata custom metadata field validation."""
        custom_metadata = {
            "language": "en",
            "category": "technical",
            "priority": "high",
            "reviewed": True,
            "reviewer": "Jane Smith",
        }

        metadata = DocumentMetadata(
            document_id="doc",
            file_path="/path/to/doc.pdf",
            file_type="pdf",
            file_size=1024,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
            metadata=custom_metadata,
        )

        assert metadata.metadata == custom_metadata
        assert metadata.metadata["language"] == "en"
        assert metadata.metadata["category"] == "technical"
        assert metadata.metadata["priority"] == "high"
        assert metadata.metadata["reviewed"] is True
        assert metadata.metadata["reviewer"] == "Jane Smith"

    def test_document_metadata_immutability(self) -> None:
        """Test that DocumentMetadata fields can be modified after creation (Pydantic v2 default behavior)."""
        metadata = DocumentMetadata(
            document_id="doc",
            file_path="/path/to/doc.pdf",
            file_type="pdf",
            file_size=1024,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
        )

        # Test that fields are mutable (Pydantic v2 default behavior)
        # In Pydantic v2, models are mutable by default unless frozen=True is set
        metadata.document_id = "modified_doc"
        metadata.file_path = "/modified/path/doc.pdf"
        metadata.file_type = "docx"
        metadata.file_size = 2048
        metadata.chunk_count = 5
        metadata.metadata = {"modified": True}

        assert metadata.document_id == "modified_doc"
        assert metadata.file_path == "/modified/path/doc.pdf"
        assert metadata.file_type == "docx"
        assert metadata.file_size == 2048
        assert metadata.chunk_count == 5
        assert metadata.metadata == {"modified": True}


class TestConcreteImplementations:
    """Test concrete implementations of abstract base classes."""

    class MockRAGQueryEngine(RAGQueryEngine):
        """Concrete implementation of RAGQueryEngine for testing."""

        def __init__(self) -> None:
            self.documents: list[str] = []
            self.connected = False

        def query(self, question: str) -> str:
            """Mock query implementation."""
            if not self.connected:
                raise RuntimeError("Not connected to database")
            return f"Answer to: {question}"

        def add_docs(
            self,
            new_doc_dir: Path | str | None = None,
            new_doc_paths_or_urls: Sequence[Path | str] | None = None,
        ) -> None:
            """Mock add_docs implementation."""
            if new_doc_paths_or_urls:
                self.documents.extend(str(doc) for doc in new_doc_paths_or_urls)

        def connect_db(self, *args: Any, **kwargs: Any) -> bool:
            """Mock connect_db implementation."""
            self.connected = True
            return True

    class MockDocumentProcessor(DocumentProcessor):
        """Concrete implementation of DocumentProcessor for testing."""

        def process_document(self, input_path: Path | str, output_dir: Path | str) -> list[Path]:
            """Mock process_document implementation."""
            output_path = Path(output_dir) / f"processed_{Path(input_path).name}"
            return [output_path]

        def chunk_document(self, document_path: Path | str, chunk_size: int = 512) -> list[str]:
            """Mock chunk_document implementation."""
            return [f"chunk_{i}" for i in range(3)]

    class MockStorageBackend(StorageBackend):
        """Concrete implementation of StorageBackend for testing."""

        def __init__(self) -> None:
            self.storage: dict[str, tuple[str, dict[str, Any]]] = {}

        def store_document(self, document_id: str, content: str, metadata: dict[str, Any]) -> bool:
            """Mock store_document implementation."""
            self.storage[document_id] = (content, metadata)
            return True

        def retrieve_document(self, document_id: str) -> str | None:
            """Mock retrieve_document implementation."""
            if document_id in self.storage:
                return self.storage[document_id][0]
            return None

        def list_documents(self) -> list[str]:
            """Mock list_documents implementation."""
            return list(self.storage.keys())

    def test_mock_rag_query_engine_implementation(self) -> None:
        """Test that MockRAGQueryEngine properly implements RAGQueryEngine."""
        engine = self.MockRAGQueryEngine()

        # Test initial state
        assert not engine.connected
        assert len(engine.documents) == 0

        # Test connection
        assert engine.connect_db() is True
        assert engine.connected is True

        # Test querying
        answer = engine.query("What is AI?")
        assert answer == "Answer to: What is AI?"

        # Test adding documents
        engine.add_docs(new_doc_paths_or_urls=["doc1.pdf", "doc2.pdf"])
        assert len(engine.documents) == 2
        assert "doc1.pdf" in engine.documents
        assert "doc2.pdf" in engine.documents

    def test_mock_document_processor_implementation(self) -> None:
        """Test that MockDocumentProcessor properly implements DocumentProcessor."""
        processor = self.MockDocumentProcessor()

        # Test document processing
        output_paths = processor.process_document("input.pdf", "/output")
        assert len(output_paths) == 1
        assert output_paths[0].name == "processed_input.pdf"

        # Test document chunking
        chunks = processor.chunk_document("document.pdf", chunk_size=256)
        assert len(chunks) == 3
        assert chunks[0] == "chunk_0"
        assert chunks[1] == "chunk_1"
        assert chunks[2] == "chunk_2"

    def test_mock_storage_backend_implementation(self) -> None:
        """Test that MockStorageBackend properly implements StorageBackend."""
        backend = self.MockStorageBackend()

        # Test initial state
        assert len(backend.list_documents()) == 0

        # Test storing documents
        metadata = {"author": "John Doe", "created": "2024-01-01"}
        assert backend.store_document("doc1", "Content 1", metadata) is True
        assert backend.store_document("doc2", "Content 2", {}) is True

        # Test listing documents
        documents = backend.list_documents()
        assert len(documents) == 2
        assert "doc1" in documents
        assert "doc2" in documents

        # Test retrieving documents
        content1 = backend.retrieve_document("doc1")
        content2 = backend.retrieve_document("doc2")
        assert content1 == "Content 1"
        assert content2 == "Content 2"

        # Test retrieving non-existent document
        assert backend.retrieve_document("nonexistent") is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_query_result_with_empty_strings(self) -> None:
        """Test QueryResult with empty strings."""
        result = QueryResult(answer="")
        assert result.answer == ""
        assert result.sources == []
        assert result.metadata == {}

    def test_document_metadata_with_special_characters(self) -> None:
        """Test DocumentMetadata with special characters in strings."""
        metadata = DocumentMetadata(
            document_id="doc_123-456_789",
            file_path="/path/with spaces/and-special-chars/file (1).pdf",
            file_type="pdf",
            file_size=1024,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
            metadata={"special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?"},
        )

        assert metadata.document_id == "doc_123-456_789"
        assert metadata.file_path == "/path/with spaces/and-special-chars/file (1).pdf"
        assert "special_chars" in metadata.metadata

    def test_document_metadata_with_unicode(self) -> None:
        """Test DocumentMetadata with unicode characters."""
        metadata = DocumentMetadata(
            document_id="doc_unicode_测试",
            file_path="/path/with/unicode/测试文档.pdf",
            file_type="pdf",
            file_size=1024,
            created_at="2024-01-01T00:00:00Z",
            processed_at="2024-01-01T01:00:00Z",
            metadata={"unicode": "测试", "emoji": "🚀📚"},
        )

        assert metadata.document_id == "doc_unicode_测试"
        assert metadata.file_path == "/path/with/unicode/测试文档.pdf"
        assert metadata.metadata["unicode"] == "测试"
        assert metadata.metadata["emoji"] == "🚀📚"
