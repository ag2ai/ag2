# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from tempfile import TemporaryDirectory

import pytest
from conftest import reason, skip_openai

if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
    from crewai_tools import FileReadTool
else:
    FileReadTool = unittest.mock.MagicMock()

from autogen import AssistantAgent, UserProxyAgent
from autogen.interoperability import Interoperable

if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
    from autogen.interoperability.crewai import CrewAIInteroperability
else:
    CrewAIInteroperability = unittest.mock.MagicMock()


# skip if python version is not in [3.10, 3.11, 3.12]
@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 13), reason="Only Python 3.10, 3.11, 3.12 are supported"
)
class TestCrewAIInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.crewai_interop = CrewAIInteroperability()

        crewai_tool = FileReadTool()
        self.model_type = crewai_tool.args_schema
        self.tool = self.crewai_interop.convert_tool(crewai_tool)

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

            assert self.tool.name == "Read_a_file_s_content"
            assert (
                self.tool.description
                == "A tool that can be used to read a file's content. (IMPORTANT: When using arguments, put them all in an `args` dictionary)"
            )

            args = self.model_type(file_path=file_path)

            assert self.tool.func(args=args) == "Hello, World!"

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self) -> None:
        config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
        )

        chatbot = AssistantAgent(
            name="chatbot",
            llm_config={"config_list": config_list},
        )

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            user_proxy.initiate_chat(
                recipient=chatbot, message=f"Read the content of the file at {file_path}", max_turns=2
            )

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] == "Hello, World!"
                return

        assert False, "Tool response not found in chat messages"
