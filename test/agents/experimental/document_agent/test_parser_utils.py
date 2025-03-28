# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from pytest import LogCaptureFixture, fixture, raises

from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen.agents.experimental.document_agent.parser_utils import docling_parse_docs
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.tools.tool import Tool

from ....conftest import Credentials

with optional_import_block():
    from docling.datamodel.document import ConversionResult, InputDocument


@run_for_optional_imports(["docling", "requests", "selenium", "webdriver_manager"], "rag")
class TestDoclingParseDocs:
    @fixture
    def mock_document_input(self) -> MagicMock:
        mock_input = MagicMock(spec=InputDocument)
        mock_input.file = Path("input_file_path")
        return mock_input

    @fixture
    def mock_conversion_result(self, mock_document_input: MagicMock) -> MagicMock:
        mock_result = MagicMock(spec=ConversionResult)
        mock_result.input = mock_document_input
        mock_result.document = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Mock Markdown"
        mock_result.document.export_to_dict.return_value = {"mock": "data"}
        mock_result.document.tables = [MagicMock()]
        mock_result.document.tables[0].export_to_html.return_value = "<table></table>"
        return mock_result

    def test_no_documents_found(self) -> None:
        """Test that ValueError is raised when no documents are found."""
        with patch("autogen.agents.experimental.document_agent.parser_utils.handle_input", return_value=[]):  # noqa: SIM117
            with raises(ValueError, match="No documents found."):
                list(docling_parse_docs("input_file_path", "output_dir_path"))

    def test_returns_iterator_of_conversion_results(self, tmp_path: Path, mock_conversion_result: MagicMock) -> None:
        """Test that function returns iterator of ConversionResult."""
        input_file_path = tmp_path / "input_file_path"
        output_dir_path = tmp_path / "output"
        with (
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.handle_input", return_value=[input_file_path]
            ),
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.DocumentConverter.convert_all",
                return_value=iter([mock_conversion_result]),
            ),
        ):
            results = docling_parse_docs(input_file_path, output_dir_path)
            assert isinstance(results, list)
            assert isinstance(results[0], Path)

    def test_exports_converted_documents(self, tmp_path: Path, mock_conversion_result: MagicMock) -> None:
        """Test that the function exports converted documents to the specified output directory.

        This test ensures that the function saves the converted documents in markdown and
        json formats to the specified output directory.
        """
        input_file_path = tmp_path / "input_file_path"
        output_dir_path = tmp_path / "output"
        with (
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.handle_input", return_value=[input_file_path]
            ),
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.DocumentConverter.convert_all",
                return_value=iter([mock_conversion_result]),
            ),
        ):
            docling_parse_docs(input_file_path, output_dir_path, output_formats=["markdown", "json"])

            md_path = output_dir_path / "input_file_path.md"
            json_path = output_dir_path / "input_file_path.json"
            # html_path = output_dir_path / "input_file_path-table-1.html"

            assert md_path.exists()
            assert json_path.exists()
            # assert html_path.exists()

            with md_path.open("r") as md_file:
                assert md_file.read() == "# Mock Markdown", "Markdown file content does not match expected content."

            with json_path.open("r") as json_file:
                assert json_file.read() == '{"mock": "data"}', "JSON file content does not match expected content."

            """ HTML tables not being output.
            with html_path.open("r") as html_file:
                assert html_file.read() == "<table></table>", "HTML file content does not match expected content."
            """

    def test_logs_conversion_time_and_document_conversion_info(
        self, tmp_path: Path, caplog: LogCaptureFixture, mock_conversion_result: MagicMock
    ) -> None:
        """Test that the function logs conversion time and document conversion info.

        This test ensures that the function logs the conversion time and the document
        conversion information at the INFO level.
        """
        input_file_path = tmp_path / "input_file_path"
        output_dir_path = tmp_path / "output"
        caplog.set_level(logging.DEBUG)

        with (
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.handle_input",
                return_value=[Path("input_file_path")],
            ),
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.DocumentConverter.convert_all",
                return_value=iter([mock_conversion_result]),
            ),
        ):
            docling_parse_docs(input_file_path, output_dir_path)
            assert "Document converted in" in caplog.text
            assert f"Document input_file_path converted.\nSaved markdown output to: {output_dir_path}" in caplog.text

    def test_handles_invalid_input_file_paths_and_output_directory_paths(self, tmp_path: Path) -> None:
        """Test that the function handles invalid input file paths and output directory paths.

        This test ensures that the function raises a ValueError when the input file path is invalid
        and a FileNotFoundError when the output directory path is invalid.
        """
        invalid_input_file_path = tmp_path / "invalid_input_file_path"
        output_dir_path = tmp_path / "output"

        with raises(ValueError, match="The input provided does not exist."):
            docling_parse_docs(invalid_input_file_path, output_dir_path)

    def test_register_docling_parse_docs_as_a_tool(self, tmp_path: Path, mock_credentials: Credentials) -> None:
        input_file_path = (tmp_path / "input_file_path.md").resolve()
        output_dir_path = (tmp_path / "output").resolve()

        input_file_path.write_text("# Mock Markdown")

        parser_tool = Tool(
            name="docling_parse_docs",
            description="Use this tool to parse and understand text.",
            func_or_tool=docling_parse_docs,
        )

        user_agent = UserProxyAgent(
            name="UserAgent",
            human_input_mode="ALWAYS",
            code_execution_config=False,
        )

        parser_tool.register_for_execution(user_agent)

        results = user_agent.function_map["docling_parse_docs"](
            input_file_path=input_file_path, output_dir_path=output_dir_path
        )

        assert isinstance(results, str)
        assert str(output_dir_path).replace("\\", "/") in results

        assistant = AssistantAgent(
            name="AssistantAgent",
            llm_config=mock_credentials.llm_config,
        )
        parser_tool.register_for_llm(assistant)

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Use this tool to parse and understand text.",
                    "name": "docling_parse_docs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input_file_path": {
                                "anyOf": [{"format": "path", "type": "string"}, {"type": "string"}],
                                "description": "Path to the input file or directory",
                            },
                            "output_dir_path": {
                                "anyOf": [{"format": "path", "type": "string"}, {"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "Path to the output directory",
                            },
                            "output_formats": {
                                "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
                                "default": None,
                                "description": "List of output formats (markdown, json)",
                            },
                            "table_output_format": {
                                "default": "html",
                                "description": "table_output_format",
                                "type": "string",
                            },
                        },
                        "required": ["input_file_path"],
                    },
                },
            }
        ]
        assert assistant.llm_config and "tools" in assistant.llm_config
        assert assistant.llm_config["tools"] == expected_tools

    def test_default_output_dir_path(self, tmp_path: Path, mock_conversion_result: MagicMock) -> None:
        """Test that the function uses './output' as the default output directory path when None is provided."""
        input_file_path = tmp_path / "input_file_path"

        with (
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.handle_input", return_value=[input_file_path]
            ),
            patch(
                "autogen.agents.experimental.document_agent.parser_utils.DocumentConverter.convert_all",
                return_value=iter([mock_conversion_result]),
            ),
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.open", mock_open()) as mock_file,
        ):
            # Call the function with output_dir_path=None
            docling_parse_docs(input_file_path.resolve(), output_dir_path=None)

            # Check that Path('./output') was created
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

            # Verify that the correct output directory was used
            file_open_calls = [call[0][0] for call in mock_file.call_args_list]
            for file_path in file_open_calls:
                assert str(file_path).startswith(str(Path("./output")))
