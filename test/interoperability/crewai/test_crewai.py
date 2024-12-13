# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory
from typing import Protocol

import pytest
from conftest import reason, skip_openai
from crewai_tools import FileReadTool

from autogen.interoperability import Interoperable
from autogen.interoperability.crewai import CrewAIInteroperability


class TestCrewAIInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.crewai_interop = CrewAIInteroperability()

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = self.crewai_interop
        # runtime check
        assert isinstance(interop, Interoperable)

    def test_init(self) -> None:
        assert isinstance(self.crewai_interop, Interoperable)

    def test_convert_tool(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            crewai_tool = FileReadTool()
            model_type = crewai_tool.args_schema

            tool = self.crewai_interop.convert_tool(crewai_tool)

            assert tool.name == "Read_a_file_s_content"
            assert tool.description == "A tool that can be used to read a file's content."

            args = model_type(file_path=file_path)

            assert tool.func(args=args) == "Hello, World!"

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self) -> None:
        assert False, "Test not implemented"
