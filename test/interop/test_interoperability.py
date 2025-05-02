# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from tempfile import TemporaryDirectory

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop import Interoperability

from ..conftest import MOCK_OPEN_AI_API_KEY

with optional_import_block():
    from crewai_tools import FileReadTool


@pytest.mark.interop
class TestInteroperability:
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

    @pytest.mark.skipif(
        sys.version_info < (3, 10) or sys.version_info >= (3, 13),
        reason="This test is only supported in Python 3.10-3.12",
    )
    def test_crewai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)

        crewai_tool = FileReadTool()

        tool = Interoperability.convert_tool(type="crewai", tool=crewai_tool)

        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            assert tool.name == "Read_a_file_s_content"
            assert "A tool that reads the content of a file" in tool.description

            model_type = crewai_tool.args_schema

            args = model_type(file_path=file_path)

            assert tool.func(args=args) == "Hello, World!"

    @pytest.mark.skip("This test is not yet implemented")
    @run_for_optional_imports("langchain", "interop-langchain")
    def test_langchain(self) -> None:
        raise NotImplementedError("This test is not yet implemented")
