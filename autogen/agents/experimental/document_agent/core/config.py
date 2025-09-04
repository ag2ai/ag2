# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RAGConfig:
    """Configuration for RAG backends."""

    rag_type: str = "vector"  # "vector", "structured", "graph"
    backend: str = "chromadb"  # "chromadb", "weaviate", "neo4j", "inmemory"
    collection_name: str | None = None
    db_path: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class SourceConfig:
    """Configuration for document input sources (where documents come from)."""

    source_type: str = "local"  # "local", "s3", "minio", "gcs", "azure"

    # Local settings
    base_path: Path = field(default_factory=lambda: Path("./documents"))

    # Cloud settings
    bucket_name: str | None = None
    prefix: str | None = None  # S3 prefix/folder path
    region: str | None = None
    endpoint_url: str | None = None  # For MinIO

    # Authentication
    credentials: dict[str, Any] | None = None
    access_key: str | None = None
    secret_key: str | None = None


@dataclass
class StorageConfig:
    """Configuration for storage backends."""

    storage_type: str = "local"  # "local", "s3", "azure", "gcs", "minio"
    base_path: Path = field(default_factory=lambda: Path("./storage"))
    bucket_name: str | None = None
    credentials: dict[str, Any] | None = None


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""

    output_dir: Path = field(default_factory=lambda: Path("./parsed_docs"))
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    supported_formats: list[str] = field(
        default_factory=lambda: [
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
    )


@dataclass
class DocAgentConfig:
    """Main configuration for DocAgent."""

    rag: RAGConfig = field(default_factory=RAGConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
