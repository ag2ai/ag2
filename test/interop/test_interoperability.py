# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
from tempfile import TemporaryDirectory

import pytest

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.interop import Interoperability

from ..conftest import MOCK_OPEN_AI_API_KEY

with optional_import_block():
    from crewai_tools import FileReadTool


@pytest.mark.interop
class TestInteroperability:
    @skip_on_missing_imports(["crewai_tools", "langchain", "pydantic_ai"], "interop")
    def test_supported_types(self) -> None:
        actual = Interoperability.get_supported_types()

        if sys.version_info < (3, 9):
            assert actual == []

        if sys.version_info >= (3, 9) and sys.version_info < (3, 10):
            assert actual == ["langchain", "pydanticai"]

        if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
            assert actual == ["crewai", "langchain", "pydanticai"]

        if sys.version_info >= (3, 13):
            assert actual == ["langchain", "pydanticai"]

    @skip_on_missing_imports("crewai_tools", "interop-crewai")
    def test_crewai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)

        crewai_tool = FileReadTool()

        tool = Interoperability.convert_tool(type="crewai", tool=crewai_tool)

        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            assert tool.name == "Read_a_file_s_content"
            assert (
                tool.description
                == "A tool that reads the content of a file. To use this tool, provide a 'file_path' parameter with the path to the file you want to read. (IMPORTANT: When using arguments, put them all in an `args` dictionary)"
            )

            model_type = crewai_tool.args_schema

            args = model_type(file_path=file_path)

            assert tool.func(args=args) == "Hello, World!"

    @pytest.mark.skip("This test is not yet implemented")
    @skip_on_missing_imports("langchain", "interop-langchain")
    def test_langchain(self) -> None:
        raise NotImplementedError("This test is not yet implemented")
