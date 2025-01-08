# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.interop import Interoperable

from ...agentchat.test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST
from ...conftest import MOCK_OPEN_AI_API_KEY, reason, skip_openai

if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
    from autogen.interop.crewai import CrewAIInteroperability
else:
    CrewAIInteroperability = MagicMock()


# skip if python version is not in [3.10, 3.11, 3.12]
@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 13), reason="Only Python 3.10, 3.11, 3.12 are supported"
)
class TestCrewAIInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)
        from crewai_tools import FileReadTool

        crewai_tool = FileReadTool()
        self.model_type = crewai_tool.args_schema
        self.tool = CrewAIInteroperability.convert_tool(crewai_tool)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = CrewAIInteroperability()

        # runtime check
        assert isinstance(interop, Interoperable)

    def test_convert_tool(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            file_path = f"{tmp_dir}/test.txt"
            with open(file_path, "w") as file:
                file.write("Hello, World!")

            assert self.tool.name == "Read_a_file_s_content"
            assert (
                self.tool.description
                == "A tool that can be used to read None's content. (IMPORTANT: When using arguments, put them all in an `args` dictionary)"
            )

            args = self.model_type(file_path=file_path)

            assert self.tool.func(args=args) == "Hello, World!"

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self) -> None:
        config_list = autogen.config_list_from_json(
            OAI_CONFIG_LIST,
            filter_dict={
                "tags": ["gpt-4o-mini"],
            },
            file_location=KEY_LOC,
        )
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

    def test_get_unsupported_reason(self) -> None:
        assert CrewAIInteroperability.get_unsupported_reason() is None


@pytest.mark.skipif(
    sys.version_info >= (3, 10) or sys.version_info < (3, 13), reason="Crew AI Interoperability is supported"
)
class TestCrewAIInteroperabilityIfNotSupported:
    def test_get_unsupported_reason(self) -> None:
        assert (
            CrewAIInteroperability.get_unsupported_reason()
            == "This submodule is only supported for Python versions 3.10, 3.11, and 3.12"
        )
