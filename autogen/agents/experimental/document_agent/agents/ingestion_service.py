# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .....doc_utils import export_module
from ..core.base_interfaces import DocumentSource, RAGQueryEngine
from ..core.config import DocAgentConfig
from ..ingestion.document_processor import DoclingDocumentProcessor
from ..sources.source_factory import DocumentSourceFactory

__all__ = ["DocumentIngestionService"]

logger = logging.getLogger(__name__)


@export_module("autogen.agents.experimental.document_agent")
class DocumentIngestionService:
    """DocumentIngestionService supporting multiple input sources with backward compatibility.

    This service handles document processing from various sources (local, S3, etc.)
    and maintains full backward compatibility with the original DocumentIngestionService.
    """

    def __init__(
        self,
        query_engine: RAGQueryEngine,
        config: DocAgentConfig | None = None,
        document_source: DocumentSource | None = None,
    ) -> None:
        """Initialize the DocumentIngestionService.

        Args:
            query_engine: The RAG query engine to add documents to
            config: Configuration for the service
            document_source: Optional document source (will create from config if None)
        """
        self.config = config or DocAgentConfig()
        self.query_engine = query_engine

        # Initialize document source
        self.document_source = document_source or DocumentSourceFactory.create_source(self.config.source)

        self.document_processor = DoclingDocumentProcessor(
            output_dir=self.config.processing.output_dir, chunk_size=self.config.processing.chunk_size
        )

    # ============================================================================
    # NEW SOURCE-AWARE METHODS
    # ============================================================================

    def ingest_document_from_source(self, document_key: str) -> str:
        """Ingest a document from the configured document source.

        Args:
            document_key: Key/path of document in the source (e.g., "reports/Q3-2024.pdf")

        Returns:
            Status message about the ingestion
        """
        try:
            # Check if document exists in source
            if not self.document_source.document_exists(document_key):
                return f"Document not found in source: {document_key}"

            logger.info(f"Starting ingestion of document from source: {document_key}")

            # For local source, we can process directly if same location
            if self.config.source.source_type == "local":
                source_path = self.config.source.base_path / document_key
                return self._process_local_document(source_path, document_key)
            else:
                # For cloud sources, download to temporary location first
                return self._process_remote_document(document_key)

        except Exception as e:
            logger.error(f"Ingestion failed for {document_key}: {e}")
            return f"Error ingesting document: {str(e)}"

    def list_available_documents(self, prefix: str | None = None) -> list[str]:
        """List all available documents from the source."""
        return self.document_source.list_documents(prefix)

    def ingest_documents_by_pattern(self, pattern: str) -> list[str]:
        """Ingest documents matching a pattern."""
        available_docs = self.list_available_documents()

        # Simple pattern matching (you could enhance with regex)
        matching_docs = [doc for doc in available_docs if pattern in doc]

        results = []
        for doc_key in matching_docs:
            result = self.ingest_document_from_source(doc_key)
            results.append(f"{doc_key}: {result}")

        return results

    def get_document_info(self, document_key: str) -> dict[str, Any]:
        """Get detailed information about a specific document."""
        try:
            if not self.document_source.document_exists(document_key):
                return {"error": f"Document not found: {document_key}"}

            metadata = self.document_source.get_document_metadata(document_key)

            return {
                "document_key": document_key,
                "exists": True,
                "source_type": self.config.source.source_type,
                "metadata": metadata,
            }
        except Exception as e:
            return {"error": f"Failed to get document info: {str(e)}"}

    def bulk_ingest_from_source(self, document_keys: list[str]) -> dict[str, Any]:
        """Ingest multiple documents from source by their keys."""
        results: dict[str, Any] = {"successful": [], "failed": [], "summary": {}}

        for doc_key in document_keys:
            try:
                result = self.ingest_document_from_source(doc_key)
                if "Successfully" in result:
                    results["successful"].append({"key": doc_key, "result": result})
                else:
                    results["failed"].append({"key": doc_key, "error": result})
            except Exception as e:
                results["failed"].append({"key": doc_key, "error": str(e)})

        results["summary"] = {
            "total": len(document_keys),
            "successful": len(results["successful"]),
            "failed": len(results["failed"]),
        }

        return results

    def ingest_document(self, document_path: str | Path) -> str:
        """Ingest a single document - BACKWARD COMPATIBLE with original API.

        This method maintains the exact same signature and behavior as the original,
        but now uses the source layer internally for enhanced capabilities.

        Args:
            document_path: Path to the document to ingest

        Returns:
            Status message about the ingestion
        """
        if self.config.source.source_type == "local":
            # Convert absolute/relative path to document key for source layer
            path = Path(document_path)
            try:
                # Try to make it relative to the base path
                document_key = str(path.relative_to(self.config.source.base_path)) if path.is_absolute() else str(path)
            except ValueError:
                # Path is outside base_path, fall back to original direct processing
                return self._process_document_directly(document_path)
        else:
            # For non-local sources, assume document_path is the key
            document_key = str(document_path)

        return self.ingest_document_from_source(document_key)

    def ingest_documents(self, document_paths: Sequence[str | Path]) -> list[str]:
        """Ingest multiple documents - BACKWARD COMPATIBLE.

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
        """Ingest all documents in a directory - ENHANCED but BACKWARD COMPATIBLE.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Status message about the ingestion
        """
        try:
            if self.config.source.source_type == "local":
                # Enhanced: Use source layer for local directories
                try:
                    directory_key = str(Path(directory_path).relative_to(self.config.source.base_path))
                    available_docs = self.document_source.list_documents(directory_key)

                    if not available_docs:
                        return f"No supported documents found in directory: {directory_path}"

                    logger.info(f"Found {len(available_docs)} documents to ingest in {directory_path}")

                    # Ingest all documents using source keys
                    results = []
                    for doc_key in available_docs:
                        result = self.ingest_document_from_source(doc_key)
                        results.append(result)

                    successful = sum(1 for r in results if "Successfully" in r)
                    failed = len(results) - successful

                    return f"Ingestion complete: {successful} successful, {failed} failed"

                except ValueError:
                    # Fall back to original directory processing if path is outside base
                    return self._process_directory_directly(directory_path)
            else:
                # For cloud sources, directory_path is a prefix
                available_docs = self.document_source.list_documents(str(directory_path))

                if not available_docs:
                    return f"No supported documents found in directory: {directory_path}"

                # Ingest all documents using source keys
                results = []
                for doc_key in available_docs:
                    result = self.ingest_document_from_source(doc_key)
                    results.append(result)

                successful = sum(1 for r in results if "Successfully" in r)
                failed = len(results) - successful

                return f"Ingestion complete: {successful} successful, {failed} failed"

        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}")
            return f"Error ingesting directory: {str(e)}"

    def get_ingestion_status(self) -> dict[str, Any]:
        """Get the current status of the ingestion service - ENHANCED.

        Returns:
            Dictionary with status information (enhanced with source info)
        """
        try:
            available_docs = str(len(self.document_source.list_documents()))
        except Exception:
            available_docs = "unknown"

        # Enhanced status with source information
        status = {
            "query_engine_configured": self.query_engine is not None,
            "output_directory": str(self.config.processing.output_dir),
            "chunk_size": self.config.processing.chunk_size,
            "supported_formats": self.config.processing.supported_formats,
            # NEW: Source layer information
            "document_source": type(self.document_source).__name__,
            "source_type": self.config.source.source_type,
            "available_documents": available_docs,
        }

        # Add source-specific info
        if self.config.source.source_type == "local":
            status["source_base_path"] = str(self.config.source.base_path)
        else:
            status["source_bucket"] = self.config.source.bucket_name

        return status

    def set_query_engine(self, query_engine: RAGQueryEngine) -> None:
        """Set the query engine for this service - BACKWARD COMPATIBLE."""
        self.query_engine = query_engine
        logger.info("Query engine updated for ingestion service")

    # ============================================================================
    # ENHANCED METHODS (INTERNAL USE ONLY)
    # ============================================================================

    def set_document_source(self, document_source: DocumentSource) -> None:
        """Set the document source for this service."""
        self.document_source = document_source
        logger.info(f"Document source updated: {type(document_source).__name__}")

    # ============================================================================
    # INTERNAL HELPER METHODS
    # ============================================================================

    def _process_local_document(self, source_path: Path, document_key: str) -> str:
        """Process a local document directly."""
        try:
            # Process the document directly from source
            processed_files = self.document_processor.process_document(source_path, self.config.processing.output_dir)

            if processed_files and self.query_engine:
                # Add processed documents to the query engine
                self.query_engine.add_docs(new_doc_paths_or_urls=processed_files)
                logger.info(f"Successfully ingested {len(processed_files)} document(s) from {document_key}")
                return f"Successfully ingested {len(processed_files)} document(s): {[f.name for f in processed_files]}"
            else:
                logger.warning("No documents were processed")
                return "No documents were processed"

        except Exception as e:
            logger.error(f"Processing failed for {document_key}: {e}")
            return f"Error processing document: {str(e)}"

    def _process_remote_document(self, document_key: str) -> str:
        """Process a document from remote source (S3, etc.)."""
        try:
            # Download to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(document_key).suffix) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                # Download document
                if not self.document_source.download_document(document_key, tmp_path):
                    return f"Failed to download document: {document_key}"

                # Process the downloaded document
                processed_files = self.document_processor.process_document(tmp_path, self.config.processing.output_dir)

                if processed_files and self.query_engine:
                    # Add to RAG engine
                    self.query_engine.add_docs(new_doc_paths_or_urls=processed_files)
                    logger.info(f"Successfully ingested {len(processed_files)} document(s) from {document_key}")
                    return f"Successfully ingested {len(processed_files)} document(s) from {document_key}"
                else:
                    return "No documents were processed"

            finally:
                # Clean up temp file
                if tmp_path.exists():
                    tmp_path.unlink()

        except Exception as e:
            logger.error(f"Remote processing failed for {document_key}: {e}")
            return f"Error processing remote document: {str(e)}"

    def _process_document_directly(self, document_path: str | Path) -> str:
        """Fallback: Process document directly without source layer (original behavior)."""
        try:
            logger.info(f"Starting direct ingestion of document: {document_path}")

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

    def _process_directory_directly(self, directory_path: str | Path) -> str:
        """Fallback: Process directory directly without source layer (original behavior)."""
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
