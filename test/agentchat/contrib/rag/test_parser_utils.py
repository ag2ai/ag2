# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.document import ConversionResult, InputDocument

from autogen.agentchat.contrib.rag.parser_utils import docling_parse_docs


@pytest.fixture
def mock_document_input():
    mock_input = MagicMock(spec=InputDocument)
    mock_input.file = Path("input_file_path")
    return mock_input


@pytest.fixture
def mock_conversion_result(mock_document_input):
    mock_result = MagicMock(spec=ConversionResult)
    mock_result.input = mock_document_input
    mock_result.document = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Mock Markdown"
    mock_result.document.export_to_dict.return_value = {"mock": "data"}
    mock_result.document.tables = [MagicMock()]
    mock_result.document.tables[0].export_to_html.return_value = "<table></table>"
    return mock_result


def test_no_documents_found() -> None:
    """Test that ValueError is raised when no documents are found."""
    with patch("autogen.agentchat.contrib.rag.parser_utils.handle_input", return_value=[]):
        with pytest.raises(ValueError):
            list(docling_parse_docs("input_file_path", "output_dir_path"))


def test_returns_iterator_of_conversion_results(mock_conversion_result) -> None:
    """Test that function returns iterator of ConversionResult."""
    input_file_path = Path("input_file_path")
    output_dir_path = Path("output_dir_path")
    with patch("autogen.agentchat.contrib.rag.parser_utils.handle_input", return_value=[input_file_path]):
        with patch(
            "autogen.agentchat.contrib.rag.parser_utils.DocumentConverter.convert_all",
            return_value=iter([mock_conversion_result]),
        ):
            results = docling_parse_docs(str(input_file_path), str(output_dir_path))
            assert isinstance(results, list)
            assert isinstance(results[0], ConversionResult)


def test_exports_converted_documents(tmp_path: Path, mock_conversion_result) -> None:
    """
    Test that the function exports converted documents to the specified output directory.

    This test ensures that the function saves the converted documents in markdown and
    json formats to the specified output directory.
    """
    input_file_path = tmp_path / "input_file_path"
    output_dir_path = tmp_path
    with patch("autogen.agentchat.contrib.rag.parser_utils.handle_input", return_value=[input_file_path]):
        with patch(
            "autogen.agentchat.contrib.rag.parser_utils.DocumentConverter.convert_all",
            return_value=iter([mock_conversion_result]),
        ):
            docling_parse_docs(str(input_file_path), str(output_dir_path))

            md_path = output_dir_path / "input_file_path.md"
            json_path = output_dir_path / "input_file_path.json"
            html_path = output_dir_path / "input_file_path-table-1.html"

            assert md_path.exists()
            assert json_path.exists()
            assert html_path.exists()

            with md_path.open("r") as md_file:
                assert md_file.read() == "# Mock Markdown"

            with json_path.open("r") as json_file:
                assert json_file.read() == '{"mock": "data"}'

            with html_path.open("r") as html_file:
                assert html_file.read() == "<table></table>"


def test_logs_conversion_time_and_document_conversion_info(caplog, mock_conversion_result):
    """
    Test that the function logs conversion time and document conversion info.

    This test ensures that the function logs the conversion time and the document
    conversion information at the INFO level.
    """
    with patch("autogen.agentchat.contrib.rag.parser_utils.handle_input", return_value=[Path("input_file_path")]):
        with patch(
            "autogen.agentchat.contrib.rag.parser_utils.DocumentConverter.convert_all",
            return_value=[mock_conversion_result],
        ):
            with caplog.at_level(logging.INFO):
                docling_parse_docs("input_file_path", "output_dir_path")
                assert "Document converted in" in caplog.text
                assert "Document input_file_path converted.\nSaved markdown output to: output_dir_path" in caplog.text


def test_handles_invalid_input_file_paths_and_output_directory_paths():
    """
    Test that the function handles invalid input file paths and output directory paths.

    This test ensures that the function raises a ValueError when the input file path is invalid
    and a FileNotFoundError when the output directory path is invalid.
    """
    with pytest.raises(ValueError):
        docling_parse_docs("invalid_input_file_path", "output_dir_path")
