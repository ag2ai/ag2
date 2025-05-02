# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Type

import pytest
from pydantic import BaseModel

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop import CrewAIInteroperability, Interoperable
from autogen.tools import Tool

from ...conftest import Credentials

with optional_import_block():
    from crewai_tools import FileReadTool


# skip if python version is not in [3.10, 3.11, 3.12]
@pytest.mark.interop
@run_for_optional_imports("crewai", "interop-crewai")
class TestCrewAIInteroperability:
    # @pytest.fixture(autouse=True)
    # def setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
    #     monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)

    #     crewai_tool = FileReadTool()
    #     self.model_type = crewai_tool.args_schema
    #     self.tool = CrewAIInteroperability.convert_tool(crewai_tool)

    @pytest.fixture(scope="session")
    def crewai_tool(self) -> "FileReadTool":  # type: ignore[no-any-unimported]
        return FileReadTool()

    @pytest.fixture(scope="session")
    def model_type(self, crewai_tool: "FileReadTool") -> Type[BaseModel]:  # type: ignore[no-any-unimported]
        return crewai_tool.args_schema  # type: ignore[no-any-return]

    @pytest.fixture(scope="session")
    def tool(self, crewai_tool: "FileReadTool") -> Tool:  # type: ignore[no-any-unimported]
        return CrewAIInteroperability.convert_tool(crewai_tool)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = CrewAIInteroperability()

        # runtime check
        assert isinstance(interop, Interoperable)

    def test_convert_tool(self, tool: Tool, model_type: Type[BaseModel]) -> None:
        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            assert tool.name == "Read_a_file_s_content"
            # Check CrewAI tool description here: https://github.com/crewAIInc/crewAI-tools/blob/974b224eb7c4c2148571787cb987460a585d7df9/crewai_tools/tools/file_read_tool/file_read_tool.py#L40C25-L40C282
            assert (
                tool.description
                == "A tool that reads the content of a file. To use this tool, provide a 'file_path' parameter with the path to the file you want to read. Optionally, provide 'start_line' to start reading from a specific line and 'line_count' to limit the number of lines read. (IMPORTANT: When using arguments, put them all in an `args` dictionary)"
            )

            args = model_type(file_path=file_path)

            assert tool.func(args=args) == "Hello, World!"

    @run_for_optional_imports("openai", "openai")
    def test_with_llm(
        self, tool: Tool, credentials_gpt_4o_mini: Credentials, user_proxy: UserProxyAgent, tmp_path: Path
    ) -> None:
        chatbot = AssistantAgent(
            name="chatbot",
            llm_config=credentials_gpt_4o_mini.llm_config,
        )

        tool.register_for_execution(user_proxy)
        tool.register_for_llm(chatbot)

        file_path = tmp_path / "test.txt"
        with file_path.open("w") as file:
            file.write("Hello, World!")

        user_proxy.initiate_chat(recipient=chatbot, message=f"Read the content of the file at {file_path}", max_turns=2)

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] == "Hello, World!"
                return

        assert False, "Tool response not found in chat messages"

    @pytest.mark.skipif(
        not (sys.version_info >= (3, 10) or sys.version_info < (3, 13)),
        reason="Crew AI Interoperability is not supported",
    )
    def test_get_unsupported_reason(self) -> None:
        assert CrewAIInteroperability.get_unsupported_reason() is None


@pytest.mark.interop
@pytest.mark.skipif(
    sys.version_info >= (3, 10) or sys.version_info < (3, 13), reason="Crew AI Interoperability is supported"
)
@run_for_optional_imports("crewai", "interop-crewai")
class TestCrewAIInteroperabilityIfNotSupported:
    def test_get_unsupported_reason(self) -> None:
        assert (
            CrewAIInteroperability.get_unsupported_reason()
            == "This submodule is only supported for Python versions 3.10, 3.11, and 3.12"
        )
