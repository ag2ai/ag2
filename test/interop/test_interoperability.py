# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from autogen.interop import Interoperability, Interoperable
from autogen.tools.tool import Tool


class TestInteroperability:
    def test_supported_types(self) -> None:
        actual = Interoperability.supported_types()

        if sys.version_info < (3, 9):
            assert actual == []

        if sys.version_info >= (3, 9) and sys.version_info < (3, 10):
            assert actual == ["langchain", "pydanticai"]

        if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
            assert actual == ["crewai", "langchain", "pydanticai"]

        if sys.version_info >= (3, 13):
            assert actual == ["langchain", "pydanticai"]

    def test_register_interoperability_class(self) -> None:
        org_interoperability_classes = Interoperability._interoperability_classes
        try:

            class MyInteroperability:
                def convert_tool(self, tool: Any, **kwargs: Any) -> Tool:
                    return Tool(name="test", description="test description", func=tool)

            Interoperability.register_interoperability_class("my_interop", MyInteroperability)
            assert Interoperability.get_interoperability_class("my_interop") == MyInteroperability

            interop = Interoperability()
            tool = interop.convert_tool(type="my_interop", tool=lambda x: x)
            assert tool.name == "test"
            assert tool.description == "test description"
            assert tool.func("hello") == "hello"

        finally:
            Interoperability._interoperability_classes = org_interoperability_classes

    @pytest.mark.skipif(
        sys.version_info < (3, 10) or sys.version_info >= (3, 13), reason="Only Python 3.10, 3.11, 3.12 are supported"
    )
    def test_crewai(self) -> None:
        from crewai_tools import FileReadTool

        crewai_tool = FileReadTool()

        interoperability = Interoperability()
        tool = interoperability.convert_tool(type="crewai", tool=crewai_tool)

        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            assert tool.name == "Read_a_file_s_content"
            assert (
                tool.description
                == "A tool that can be used to read a file's content. (IMPORTANT: When using arguments, put them all in an `args` dictionary)"
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
