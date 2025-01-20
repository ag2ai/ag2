# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
from tempfile import TemporaryDirectory

import pytest

from autogen.interop import Interoperability

from ..conftest import MOCK_OPEN_AI_API_KEY


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
        sys.version_info < (3, 10) or sys.version_info >= (3, 13), reason="Only Python 3.10, 3.11, 3.12 are supported"
    )
    @pytest.mark.skipif(sys.platform == "win32", reason="This test is not supported on Windows")
    def test_crewai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)
        from crewai_tools import FileReadTool

        crewai_tool = FileReadTool()

        tool = Interoperability.convert_tool(type="crewai", tool=crewai_tool)

        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            assert tool.name == "Read_a_file_s_content"
            assert (
                tool.description
                == "A tool that can be used to read None's content. (IMPORTANT: When using arguments, put them all in an `args` dictionary)"
            )

            model_type = crewai_tool.args_schema

            args = model_type(file_path=file_path)

            assert tool.func(args=args) == "Hello, World!"

    @pytest.mark.skipif(
        sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
    )
    @pytest.mark.skip(reason="This test is not yet implemented")
    def test_langchain(self) -> None:
        raise NotImplementedError("This test is not yet implemented")
