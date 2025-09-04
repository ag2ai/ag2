# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ..core.base_interfaces import DocumentSource
from ..core.config import SourceConfig
from .local_sources import LocalDocumentSource


class DocumentSourceFactory:
    """Factory for creating document source backends."""

    @staticmethod
    def create_source(config: SourceConfig) -> DocumentSource:
        """Create a document source based on configuration."""
        source_type = config.source_type.lower()

        if source_type == "local":
            return LocalDocumentSource(config)

        elif source_type in ["s3", "minio"]:
            from .s3_source import S3DocumentSource

            return S3DocumentSource(config)

        elif source_type == "gcs":
            # Future: return GCSDocumentSource(config)
            raise NotImplementedError("GCS source not yet implemented")

        elif source_type == "azure":
            # Future: return AzureDocumentSource(config)
            raise NotImplementedError("Azure source not yet implemented")

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    @staticmethod
    def get_supported_types() -> list[str]:
        """Get list of supported source types."""
        return ["local", "s3", "minio"]
