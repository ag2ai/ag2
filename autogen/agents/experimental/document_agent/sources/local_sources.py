# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from pathlib import Path
from typing import Any

from ..core.base_interfaces import DocumentSource
from ..core.config import SourceConfig

logger = logging.getLogger(__name__)


class LocalDocumentSource(DocumentSource):
    """Local filesystem document source."""

    def __init__(self, config: SourceConfig):
        self.config = config
        self.base_path = Path(config.base_path)

        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")

        logger.info(f"LocalDocumentSource initialized at {self.base_path}")

    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List documents in local directory."""
        try:
            search_path = self.base_path
            if prefix:
                search_path = search_path / prefix

            if not search_path.exists():
                return []

            documents = []
            # Get supported formats from processing config if available
            supported_extensions = [
                "pdf",
                "docx",
                "pptx",
                "xlsx",
                "html",
                "md",
                "txt",
                "json",
                "csv",
                "xml",
                "adoc",
                "png",
                "jpg",
                "jpeg",
                "tiff",
            ]

            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    # Check if file has supported extension
                    ext = file_path.suffix.lower().lstrip(".")
                    if ext in supported_extensions:
                        # Return relative path from base_path
                        rel_path = file_path.relative_to(self.base_path)
                        documents.append(str(rel_path))

            return documents

        except Exception as e:
            logger.error(f"Failed to list local documents: {e}")
            return []

    def download_document(self, document_key: str, local_path: Path) -> bool:
        """For local source, copy file to destination if different locations."""
        try:
            source_path = self.base_path / document_key

            if not source_path.exists():
                logger.error(f"Source document does not exist: {source_path}")
                return False

            # Create destination directory
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if source_path.resolve() != local_path.resolve():
                # Copy file if different locations
                shutil.copy2(source_path, local_path)
                logger.debug(f"Copied {source_path} to {local_path}")
            else:
                logger.debug(f"Source and destination are the same: {source_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to download document {document_key}: {e}")
            return False

    def get_document_metadata(self, document_key: str) -> dict[str, Any]:
        """Get file metadata."""
        try:
            file_path = self.base_path / document_key
            if file_path.exists():
                stat = file_path.stat()
                return {
                    "size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "created_time": stat.st_ctime,
                    "file_type": file_path.suffix.lower(),
                    "absolute_path": str(file_path.absolute()),
                }
            return {}

        except Exception as e:
            logger.error(f"Failed to get metadata for {document_key}: {e}")
            return {}

    def document_exists(self, document_key: str) -> bool:
        """Check if local file exists."""
        try:
            return (self.base_path / document_key).exists()
        except Exception:
            return False
