# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RAGQueryEngine(ABC):
    """Abstract base class for RAG query engines."""

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the RAG engine with a question."""
        pass

    @abstractmethod
    def add_docs(
        self,
        new_doc_dir: Path | str | None = None,
        new_doc_paths_or_urls: Sequence[Path | str] | None = None,
    ) -> None:
        """Add documents to the RAG engine."""
        pass

    @abstractmethod
    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """Connect to the underlying database."""
        pass


class DocumentProcessor(ABC):
    """Abstract base class for document processing."""

    @abstractmethod
    def process_document(self, input_path: Path | str, output_dir: Path | str) -> list[Path]:
        """Process a document and return output file paths."""
        pass

    @abstractmethod
    def chunk_document(self, document_path: Path | str, chunk_size: int = 512) -> list[str]:
        """Chunk a document into smaller pieces."""
        pass


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store_document(self, document_id: str, content: str, metadata: dict[str, Any]) -> bool:
        """Store a document in the backend."""
        pass

    @abstractmethod
    def retrieve_document(self, document_id: str) -> str | None:
        """Retrieve a document from the backend."""
        pass

    @abstractmethod
    def list_documents(self) -> list[str]:
        """List all document IDs in the backend."""
        pass


class QueryResult(BaseModel):
    """Base class for query results."""

    answer: str
    confidence: float = 0.0
    sources: list[str] = []
    metadata: dict[str, Any] = {}


class DocumentMetadata(BaseModel):
    """Base class for document metadata."""

    document_id: str
    file_path: str
    file_type: str
    file_size: int
    created_at: str
    processed_at: str
    chunk_count: int = 0
    metadata: dict[str, Any] = {}
