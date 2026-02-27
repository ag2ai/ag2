# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .....doc_utils import export_module
from ..core.base_interfaces import RAGQueryEngine
from ..core.config import DocAgentConfig
from ..ingestion.document_processor import DoclingDocumentProcessor

__all__ = ["DocumentIngestionService"]

logger = logging.getLogger(__name__)


@export_module("autogen.agents.experimental.document_agent")
class DocumentIngestionService:
    """Service for document ingestion and processing.

    This is a support service, not an agent. It handles document processing,
    chunking, and adding documents to the RAG backend.
    """

    def __init__(
        self,
        query_engine: RAGQueryEngine,
        config: DocAgentConfig | None = None,
    ) -> None:
        """Initialize the DocumentIngestionService.

        Args:
            query_engine: The RAG query engine to add documents to
            config: Configuration for the service
        """
        self.config = config or DocAgentConfig()
        self.query_engine = query_engine
        self.document_processor = DoclingDocumentProcessor(
            output_dir=self.config.processing.output_dir, chunk_size=self.config.processing.chunk_size
        )

    def ingest_document(self, document_path: str | Path) -> str:
        """Ingest a single document.

        Args:
            document_path: Path to the document to ingest

        Returns:
            Status message about the ingestion
        """
        try:
            logger.info(f"Starting ingestion of document: {document_path}")

            # Process the document
            processed_files = self.document_processor.process_document(document_path, self.config.processing.output_dir)

            if processed_files and self.query_engine:
                # Add processed documents to the query engine
                self.query_engine.add_docs(new_doc_paths_or_urls=processed_files)
                logger.info(f"Successfully ingested {len(processed_files)} document(s)")
                return f"Successfully ingested {len(processed_files)} document(s): {[f.name for f in processed_files]}"
            else:
                logger.warning("No documents were processed")
                return "No documents were processed."

        except Exception as e:
            logger.error(f"Ingestion failed for {document_path}: {e}")
            return f"Error ingesting document: {str(e)}"

    def ingest_documents(self, document_paths: Sequence[str | Path]) -> list[str]:
        """Ingest multiple documents.

        Args:
            document_paths: Sequence of paths to documents to ingest

        Returns:
            List of status messages for each document
        """
        results = []
        for doc_path in document_paths:
            result = self.ingest_document(doc_path)
            results.append(result)
        return results

    def ingest_directory(self, directory_path: str | Path) -> str:
        """Ingest all documents in a directory.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Status message about the ingestion
        """
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return f"Directory not found: {directory_path}"

            # Find all supported files
            supported_extensions = self.config.processing.supported_formats
            document_files: list[Path] = []

            for ext in supported_extensions:
                document_files.extend(directory.glob(f"*.{ext}"))
                document_files.extend(directory.glob(f"*.{ext.upper()}"))

            if not document_files:
                return f"No supported documents found in directory: {directory_path}"

            logger.info(f"Found {len(document_files)} documents to ingest in {directory_path}")

            # Ingest all documents - convert List[Path] to Sequence[Path]
            results = self.ingest_documents(document_files)

            successful = sum(1 for r in results if "Successfully" in r)
            failed = len(results) - successful

            return f"Ingestion complete: {successful} successful, {failed} failed"

        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}")
            return f"Error ingesting directory: {str(e)}"

    def get_ingestion_status(self) -> dict[str, Any]:
        """Get the current status of the ingestion service.

        Returns:
            Dictionary with status information
        """
        return {
            "query_engine_configured": self.query_engine is not None,
            "output_directory": str(self.config.processing.output_dir),
            "chunk_size": self.config.processing.chunk_size,
            "supported_formats": self.config.processing.supported_formats,
        }

    def set_query_engine(self, query_engine: RAGQueryEngine) -> None:
        """Set the query engine for this service."""
        self.query_engine = query_engine
        logger.info("Query engine updated for ingestion service")
