# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from autogen.agents.experimental.document_agent.core.config import (
    DocAgentConfig,
    ProcessingConfig,
    RAGConfig,
    StorageConfig,
)


class TestRAGConfig:
    """Test cases for RAGConfig class."""

    def test_default_values(self) -> None:
        """Test that RAGConfig has correct default values."""
        config = RAGConfig()

        assert config.rag_type == "vector"
        assert config.backend == "chromadb"
        assert config.collection_name is None
        assert config.db_path is None
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_custom_values(self) -> None:
        """Test that RAGConfig can be initialized with custom values."""
        config = RAGConfig(
            rag_type="structured",
            backend="neo4j",
            collection_name="test_collection",
            db_path="/path/to/db",
            embedding_model="custom-model",
            chunk_size=1024,
            chunk_overlap=100,
        )

        assert config.rag_type == "structured"
        assert config.backend == "neo4j"
        assert config.collection_name == "test_collection"
        assert config.db_path == "/path/to/db"
        assert config.embedding_model == "custom-model"
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100

    def test_partial_customization(self) -> None:
        """Test that RAGConfig can be partially customized."""
        config = RAGConfig(rag_type="graph", chunk_size=256)

        # Custom values
        assert config.rag_type == "graph"
        assert config.chunk_size == 256

        # Default values should remain
        assert config.backend == "chromadb"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_overlap == 50


class TestStorageConfig:
    """Test cases for StorageConfig class."""

    def test_default_values(self) -> None:
        """Test that StorageConfig has correct default values."""
        config = StorageConfig()

        assert config.storage_type == "local"
        assert config.base_path == Path("./storage")
        assert config.bucket_name is None
        assert config.credentials is None

    def test_custom_values(self) -> None:
        """Test that StorageConfig can be initialized with custom values."""
        custom_path = Path("/custom/storage")
        custom_creds = {"access_key": "test", "secret_key": "test"}

        config = StorageConfig(
            storage_type="s3", base_path=custom_path, bucket_name="test-bucket", credentials=custom_creds
        )

        assert config.storage_type == "s3"
        assert config.base_path == custom_path
        assert config.bucket_name == "test-bucket"
        assert config.credentials == custom_creds

    def test_partial_customization(self) -> None:
        """Test that StorageConfig can be partially customized."""
        config = StorageConfig(storage_type="azure", bucket_name="azure-container")

        # Custom values
        assert config.storage_type == "azure"
        assert config.bucket_name == "azure-container"

        # Default values should remain
        assert config.base_path == Path("./storage")
        assert config.credentials is None

    def test_base_path_is_path_object(self) -> None:
        """Test that base_path is always a Path object."""
        config = StorageConfig()
        assert isinstance(config.base_path, Path)

        config = StorageConfig(base_path=Path("./relative/path"))
        assert isinstance(config.base_path, Path)
        assert config.base_path == Path("./relative/path")


class TestProcessingConfig:
    """Test cases for ProcessingConfig class."""

    def test_default_values(self) -> None:
        """Test that ProcessingConfig has correct default values."""
        config = ProcessingConfig()

        assert config.output_dir == Path("./parsed_docs")
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.max_file_size == 100 * 1024 * 1024  # 100MB
        assert "pdf" in config.supported_formats
        assert "docx" in config.supported_formats
        assert "txt" in config.supported_formats
        assert len(config.supported_formats) == 15

    def test_custom_values(self) -> None:
        """Test that ProcessingConfig can be initialized with custom values."""
        custom_output = Path("/custom/output")
        custom_formats = ["pdf", "docx", "txt"]

        config = ProcessingConfig(
            output_dir=custom_output,
            chunk_size=1024,
            chunk_overlap=100,
            max_file_size=50 * 1024 * 1024,  # 50MB
            supported_formats=custom_formats,
        )

        assert config.output_dir == custom_output
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.max_file_size == 50 * 1024 * 1024
        assert config.supported_formats == custom_formats

    def test_partial_customization(self) -> None:
        """Test that ProcessingConfig can be partially customized."""
        config = ProcessingConfig(
            chunk_size=256,
            max_file_size=25 * 1024 * 1024,  # 25MB
        )

        # Custom values
        assert config.chunk_size == 256
        assert config.max_file_size == 25 * 1024 * 1024

        # Default values should remain
        assert config.output_dir == Path("./parsed_docs")
        assert config.chunk_overlap == 50
        assert len(config.supported_formats) == 15

    def test_output_dir_is_path_object(self) -> None:
        """Test that output_dir is always a Path object."""
        config = ProcessingConfig()
        assert isinstance(config.output_dir, Path)

        config = ProcessingConfig(output_dir=Path("./relative/output"))
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("./relative/output")

    def test_supported_formats_list(self) -> None:
        """Test that supported_formats is always a list."""
        config = ProcessingConfig()
        assert isinstance(config.supported_formats, list)

        custom_formats = ["pdf", "docx"]
        config = ProcessingConfig(supported_formats=custom_formats)
        assert isinstance(config.supported_formats, list)
        assert config.supported_formats == custom_formats


class TestDocAgentConfig:
    """Test cases for DocAgentConfig class."""

    def test_default_values(self) -> None:
        """Test that DocAgentConfig has correct default values."""
        config = DocAgentConfig()

        # Check that nested configs are created with defaults
        assert isinstance(config.rag, RAGConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.processing, ProcessingConfig)

        # Check default values of nested configs
        assert config.rag.rag_type == "vector"
        assert config.storage.storage_type == "local"
        assert config.processing.chunk_size == 512

    def test_custom_nested_configs(self) -> None:
        """Test that DocAgentConfig can be initialized with custom nested configs."""
        custom_rag = RAGConfig(rag_type="graph", backend="neo4j")
        custom_storage = StorageConfig(storage_type="s3", bucket_name="test-bucket")
        custom_processing = ProcessingConfig(chunk_size=1024, max_file_size=50 * 1024 * 1024)

        config = DocAgentConfig(rag=custom_rag, storage=custom_storage, processing=custom_processing)

        assert config.rag == custom_rag
        assert config.storage == custom_storage
        assert config.processing == custom_processing

        # Verify the custom values
        assert config.rag.rag_type == "graph"
        assert config.rag.backend == "neo4j"
        assert config.storage.storage_type == "s3"
        assert config.storage.bucket_name == "test-bucket"
        assert config.processing.chunk_size == 1024
        assert config.processing.max_file_size == 50 * 1024 * 1024

    def test_partial_nested_customization(self) -> None:
        """Test that DocAgentConfig can be partially customized."""
        custom_rag = RAGConfig(rag_type="structured")

        config = DocAgentConfig(rag=custom_rag)

        # Custom nested config
        assert config.rag.rag_type == "structured"

        # Default nested configs should remain
        assert config.storage.storage_type == "local"
        assert config.processing.chunk_size == 512

    def test_nested_configs_are_instances(self) -> None:
        """Test that nested configs are proper instances of their classes."""
        config = DocAgentConfig()

        assert isinstance(config.rag, RAGConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.processing, ProcessingConfig)

    def test_nested_configs_independence(self) -> None:
        """Test that nested configs are independent instances."""
        config1 = DocAgentConfig()
        config2 = DocAgentConfig()

        # Modifying one config shouldn't affect the other
        config1.rag.chunk_size = 1024
        assert config2.rag.chunk_size == 512  # Default value unchanged

        config1.storage.base_path = Path("/custom/path")
        assert config2.storage.base_path == Path("./storage")  # Default value unchanged


class TestConfigIntegration:
    """Integration tests for configuration classes."""

    def test_config_immutability(self) -> None:
        """Test that config objects can be modified after creation."""
        config = DocAgentConfig()

        # Modify nested config values
        config.rag.chunk_size = 2048
        config.storage.bucket_name = "modified-bucket"
        config.processing.supported_formats.append("new-format")

        # Verify modifications
        assert config.rag.chunk_size == 2048
        assert config.storage.bucket_name == "modified-bucket"
        assert "new-format" in config.processing.supported_formats

    def test_config_copy_independence(self) -> None:
        """Test that config objects can be copied independently."""
        from copy import deepcopy

        original = DocAgentConfig()
        copied = deepcopy(original)

        # Modify original
        original.rag.chunk_size = 9999
        original.storage.bucket_name = "original-bucket"

        # Copied should remain unchanged
        assert copied.rag.chunk_size == 512  # Default value
        assert copied.storage.bucket_name is None  # Default value

    def test_config_serialization(self) -> None:
        """Test that config objects can be converted to dictionaries."""
        config = DocAgentConfig()

        # This test ensures the dataclass can be converted to dict
        # (useful for serialization/deserialization)
        config_dict: dict[str, Any] = {
            "rag": {
                "rag_type": config.rag.rag_type,
                "backend": config.rag.backend,
                "collection_name": config.rag.collection_name,
                "db_path": config.rag.db_path,
                "embedding_model": config.rag.embedding_model,
                "chunk_size": config.rag.chunk_size,
                "chunk_overlap": config.rag.chunk_overlap,
            },
            "storage": {
                "storage_type": config.storage.storage_type,
                "base_path": str(config.storage.base_path),
                "bucket_name": config.storage.bucket_name,
                "credentials": config.storage.credentials,
            },
            "processing": {
                "output_dir": str(config.processing.output_dir),
                "chunk_size": config.processing.chunk_size,
                "chunk_overlap": config.processing.chunk_overlap,
                "max_file_size": config.processing.max_file_size,
                "supported_formats": config.processing.supported_formats,
            },
        }

        # Verify the structure
        assert config_dict["rag"]["rag_type"] == "vector"
        assert config_dict["storage"]["storage_type"] == "local"
        assert config_dict["processing"]["chunk_size"] == 512
