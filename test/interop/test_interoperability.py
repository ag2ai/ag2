# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from tempfile import TemporaryDirectory

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop import Interoperability
from test.const import MOCK_OPEN_AI_API_KEY

with optional_import_block():
    from crewai_tools import FileReadTool


@pytest.mark.interop
class TestInteroperability:
    def test_supported_types(self) -> None:
        actual = Interoperability.get_supported_types()

        # Build expected list based on what should actually be available
        expected = []

        # Check if crewai should be available (Python 3.10-3.12 and crewai is installed)
        if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
            from autogen.interop.crewai.crewai import CrewAIInteroperability

            if CrewAIInteroperability.get_unsupported_reason() is None:
                expected.append("crewai")

        # Check if langchain should be available
        from autogen.interop.langchain.langchain_tool import LangChainInteroperability

        if LangChainInteroperability.get_unsupported_reason() is None:
            expected.append("langchain")

        # Check if pydantic_ai should be available
        from autogen.interop.pydantic_ai.pydantic_ai import PydanticAIInteroperability

        if PydanticAIInteroperability.get_unsupported_reason() is None:
            expected.append("pydanticai")

        expected.sort()  # get_supported_types() returns sorted list
        assert actual == expected

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
