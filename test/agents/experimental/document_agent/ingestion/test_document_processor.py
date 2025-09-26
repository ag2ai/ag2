# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autogen.agents.experimental.document_agent.ingestion.document_processor import DoclingDocumentProcessor
from autogen.import_utils import skip_on_missing_imports


class TestDoclingDocumentProcessor:
    """Test cases for DoclingDocumentProcessor class."""

    @pytest.fixture
    def mock_docling_imports(self) -> Any:
        """Mock docling imports for testing."""
        with patch(
            "autogen.agents.experimental.document_agent.ingestion.document_processor.optional_import_block"
        ) as mock_block:
            mock_block.return_value.__enter__ = MagicMock()
            mock_block.return_value.__exit__ = MagicMock()
            yield mock_block

    @pytest.fixture
    def mock_docling_modules(self) -> Any:
        """Mock docling modules and classes."""
        with (
            patch(
                "autogen.agents.experimental.document_agent.ingestion.document_processor.InputFormat"
            ) as mock_input_format,
            patch(
                "autogen.agents.experimental.document_agent.ingestion.document_processor.AcceleratorDevice"
            ) as mock_acc_device,
            patch(
                "autogen.agents.experimental.document_agent.ingestion.document_processor.AcceleratorOptions"
            ) as mock_acc_options,
            patch(
                "autogen.agents.experimental.document_agent.ingestion.document_processor.PdfPipelineOptions"
            ) as mock_pdf_options,
            patch(
                "autogen.agents.experimental.document_agent.ingestion.document_processor.DocumentConverter"
            ) as mock_converter,
            patch(
                "autogen.agents.experimental.document_agent.ingestion.document_processor.PdfFormatOption"
            ) as mock_pdf_format,
        ):
            # Setup mock enums
            mock_input_format.PDF = "pdf"
            mock_acc_device.AUTO = "auto"

            # Setup mock classes
            mock_acc_options.return_value.num_threads = 4
            mock_acc_options.return_value.device = "auto"

            mock_pdf_options.return_value.do_ocr = True
            mock_pdf_options.return_value.do_table_structure = True
            mock_pdf_options.return_value.table_structure_options.do_cell_matching = True
            mock_pdf_options.return_value.ocr_options.lang = ["en"]
            mock_pdf_options.return_value.accelerator_options = mock_acc_options.return_value

            # Mock OCR options
            mock_pdf_options.return_value.ocr_options.use_gpu = False

            yield {
                "input_format": mock_input_format,
                "acc_device": mock_acc_device,
                "acc_options": mock_acc_options,
                "pdf_options": mock_pdf_options,
                "converter": mock_converter,
                "pdf_format": mock_pdf_format,
            }

    @pytest.fixture
    def mock_handle_input(self) -> Any:
        """Mock handle_input function."""
        with patch("autogen.agents.experimental.document_agent.ingestion.document_processor.handle_input") as mock:
            yield mock

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def temp_input_file(self, tmp_path: Path) -> Path:
        """Create a temporary input file."""
        input_file = tmp_path / "test_document.pdf"
        input_file.write_text("test content")
        return input_file

    @skip_on_missing_imports(["docling"], "rag")
    def test_init_with_output_dir(self, mock_docling_imports: Any, tmp_path: Path) -> None:
        """Test initialization with custom output directory."""
        output_dir = tmp_path / "custom_output"
        processor = DoclingDocumentProcessor(output_dir=str(output_dir), chunk_size=1024)

        assert processor.output_dir == output_dir
        assert processor.chunk_size == 1024
        assert output_dir.exists()

    @skip_on_missing_imports(["docling"], "rag")
    def test_init_without_output_dir(self, mock_docling_imports: Any) -> None:
        """Test initialization without output directory (uses default)."""
        processor = DoclingDocumentProcessor()

        expected_dir = Path.cwd() / "output"
        assert processor.output_dir == expected_dir
        assert processor.chunk_size == 512
        assert expected_dir.exists()

    @skip_on_missing_imports(["docling"], "rag")
    def test_init_with_path_object(self, mock_docling_imports: Any, tmp_path: Path) -> None:
        """Test initialization with Path object."""
        output_dir = tmp_path / "path_output"
        processor = DoclingDocumentProcessor(output_dir=output_dir)

        assert processor.output_dir == output_dir
        assert output_dir.exists()

    @skip_on_missing_imports(["docling"], "rag")
    def test_process_document_with_custom_output_dir(
        self,
        mock_docling_imports: Any,
        mock_docling_modules: Any,
        mock_handle_input: Any,
        temp_input_file: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test process_document with custom output directory."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nContent here"
        mock_result.document.export_to_dict.return_value = {"title": "Test Document", "content": "Content here"}

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        result = processor.process_document(temp_input_file, temp_output_dir)

        assert len(result) == 1
        assert result[0].name == "test_document.md"
        assert temp_output_dir.exists()

    @skip_on_missing_imports(["docling"], "rag")
    def test_process_document_with_default_output_dir(
        self, mock_docling_imports: Any, mock_docling_modules: Any, mock_handle_input: Any, temp_input_file: Path
    ) -> None:
        """Test process_document with default output directory."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nContent here"
        mock_result.document.export_to_dict.return_value = {"title": "Test Document", "content": "Content here"}

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        result = processor.process_document(temp_input_file)

        assert len(result) == 1
        assert result[0].name == "test_document.md"
        assert processor.output_dir.exists()

    @skip_on_missing_imports(["docling"], "rag")
    def test_chunk_document_with_default_chunk_size(self, mock_docling_imports: Any, temp_input_file: Path) -> None:
        """Test chunk_document with default chunk size."""
        processor = DoclingDocumentProcessor()

        # Create a file with content longer than default chunk size
        content = "a" * 1000  # 1000 characters
        temp_input_file.write_text(content)

        chunks = processor.chunk_document(temp_input_file)

        assert len(chunks) == 2  # 1000 chars / 512 chars = 2 chunks
        assert len(chunks[0]) == 512
        assert len(chunks[1]) == 488

    @skip_on_missing_imports(["docling"], "rag")
    def test_chunk_document_with_custom_chunk_size(self, mock_docling_imports: Any, temp_input_file: Path) -> None:
        """Test chunk_document with custom chunk size."""
        processor = DoclingDocumentProcessor(chunk_size=100)

        # Create a file with content
        content = "a" * 250  # 250 characters
        temp_input_file.write_text(content)

        chunks = processor.chunk_document(temp_input_file, chunk_size=50)

        assert len(chunks) == 5  # 250 chars / 50 chars = 5 chunks
        assert all(len(chunk) == 50 for chunk in chunks[:-1])
        assert len(chunks[-1]) == 50

    @skip_on_missing_imports(["docling"], "rag")
    def test_chunk_document_with_content_shorter_than_chunk_size(
        self, mock_docling_imports: Any, temp_input_file: Path
    ) -> None:
        """Test chunk_document with content shorter than chunk size."""
        processor = DoclingDocumentProcessor(chunk_size=1000)

        # Create a file with short content
        content = "Short content"
        temp_input_file.write_text(content)

        chunks = processor.chunk_document(temp_input_file)

        assert len(chunks) == 1
        assert chunks[0] == content

    @skip_on_missing_imports(["docling"], "rag")
    def test_chunk_document_with_empty_file(self, mock_docling_imports: Any, temp_input_file: Path) -> None:
        """Test chunk_document with empty file."""
        processor = DoclingDocumentProcessor()

        # Create an empty file
        temp_input_file.write_text("")

        chunks = processor.chunk_document(temp_input_file)

        assert chunks[0] == ""

    @skip_on_missing_imports(["docling"], "rag")
    def test_docling_parse_docs_with_markdown_output(
        self,
        mock_docling_imports: Any,
        mock_docling_modules: Any,
        mock_handle_input: Any,
        temp_input_file: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test _docling_parse_docs with markdown output format."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nContent here"

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        result = processor._docling_parse_docs(temp_input_file, temp_output_dir, ["markdown"])

        assert len(result) == 1
        assert result[0].name == "test_document.md"
        assert result[0].exists()

        # Verify markdown content
        with open(result[0]) as f:
            content = f.read()
        assert content == "# Test Document\n\nContent here"

    @skip_on_missing_imports(["docling"], "rag")
    def test_docling_parse_docs_with_json_output(
        self,
        mock_docling_imports: Any,
        mock_docling_modules: Any,
        mock_handle_input: Any,
        temp_input_file: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test _docling_parse_docs with JSON output format."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_dict.return_value = {"title": "Test Document", "content": "Content here"}

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        result = processor._docling_parse_docs(temp_input_file, temp_output_dir, ["json"])

        assert len(result) == 1
        assert result[0].name == "test_document.json"
        assert result[0].exists()

        # Verify JSON content
        with open(result[0]) as f:
            content = json.load(f)
        assert content == {"title": "Test Document", "content": "Content here"}

    @skip_on_missing_imports(["docling"], "rag")
    def test_docling_parse_docs_with_both_output_formats(
        self,
        mock_docling_imports: Any,
        mock_docling_modules: Any,
        mock_handle_input: Any,
        temp_input_file: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test _docling_parse_docs with both markdown and JSON output formats."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nContent here"
        mock_result.document.export_to_dict.return_value = {"title": "Test Document", "content": "Content here"}

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        result = processor._docling_parse_docs(temp_input_file, temp_output_dir, ["markdown", "json"])

        assert len(result) == 2
        assert any(f.name == "test_document.md" for f in result)
        assert any(f.name == "test_document.json" for f in result)

    @skip_on_missing_imports(["docling"], "rag")
    def test_docling_parse_docs_with_no_documents_found(
        self, mock_docling_imports: Any, mock_handle_input: Any
    ) -> None:
        """Test _docling_parse_docs when no documents are found."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return empty list
        mock_handle_input.return_value = []

        with pytest.raises(ValueError, match="No documents found."):
            processor._docling_parse_docs("nonexistent", "output")

    @skip_on_missing_imports(["docling"], "rag")
    def test_docling_parse_docs_with_custom_table_output_format(
        self,
        mock_docling_imports: Any,
        mock_docling_modules: Any,
        mock_handle_input: Any,
        temp_input_file: Path,
        temp_output_dir: Path,
    ) -> None:
        """Test _docling_parse_docs with custom table output format."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nContent here"

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        result = processor._docling_parse_docs(temp_input_file, temp_output_dir, table_output_format="csv")

        assert len(result) == 1
        assert result[0].name == "test_document.md"

    @skip_on_missing_imports(["docling"], "rag")
    def test_docling_parse_docs_creates_output_directory(
        self,
        mock_docling_imports: Any,
        mock_docling_modules: Any,
        mock_handle_input: Any,
        temp_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test _docling_parse_docs creates output directory if it doesn't exist."""
        processor = DoclingDocumentProcessor()

        # Mock handle_input to return a list of paths
        mock_handle_input.return_value = [temp_input_file]

        # Mock document conversion result
        mock_result = MagicMock()
        mock_result.input.file.stem = "test_document"
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nContent here"

        mock_docling_modules["converter"].return_value.convert_all.return_value = [mock_result]

        # Use a non-existent output directory
        non_existent_dir = tmp_path / "new_output_dir"
        assert not non_existent_dir.exists()

        result = processor._docling_parse_docs(temp_input_file, non_existent_dir)

        assert non_existent_dir.exists()
        assert len(result) == 1
