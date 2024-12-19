# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest

import pytest
from conftest import reason, skip_openai
from pydantic import BaseModel, Field

from autogen import AssistantAgent, UserProxyAgent
from autogen.interop import Interoperable

if sys.version_info >= (3, 9):
    from langchain.tools import tool as langchain_tool
else:
    langchain_tool = unittest.mock.MagicMock()

from autogen.interop.langchain import LangChainInteroperability


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
class TestLangChainInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class SearchInput(BaseModel):
            query: str = Field(description="should be a search query")

        @langchain_tool("search-tool", args_schema=SearchInput, return_direct=True)  # type: ignore[misc]
        def search(query: SearchInput) -> str:
            """Look up things online."""
            return "LangChain Integration"

        self.model_type = search.args_schema
        self.tool = LangChainInteroperability.convert_tool(search)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = LangChainInteroperability()

        # runtime check
        assert isinstance(interop, Interoperable)

    def test_convert_tool(self) -> None:
        assert self.tool.name == "search-tool"
        assert self.tool.description == "Look up things online."

        tool_input = self.model_type(query="LangChain")
        assert self.tool.func(tool_input=tool_input) == "LangChain Integration"

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

        user_proxy.initiate_chat(recipient=chatbot, message="search for LangChain", max_turns=2)

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] == "LangChain Integration"
                return

        assert False, "No tool response found in chat messages"

    def test_get_unsupported_reason(self) -> None:
        assert LangChainInteroperability.get_unsupported_reason() is None


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
class TestLangChainInteroperabilityWithoutPydanticInput:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        @langchain_tool
        def search(query: str, max_length: int) -> str:
            """Look up things online."""
            return f"LangChain Integration, max_length: {max_length}"

        self.tool = LangChainInteroperability.convert_tool(search)
        self.model_type = search.args_schema

    def test_convert_tool(self) -> None:
        assert self.tool.name == "search"
        assert self.tool.description == "Look up things online."

        tool_input = self.model_type(query="LangChain", max_length=100)
        assert self.tool.func(tool_input=tool_input) == "LangChain Integration, max_length: 100"

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

        user_proxy.initiate_chat(recipient=chatbot, message="search for LangChain, Use max 100 characters", max_turns=2)

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] == "LangChain Integration, max_length: 100"
                return

        assert False, "No tool response found in chat messages"


@pytest.mark.skipif(sys.version_info >= (3, 9), reason="LangChain Interoperability is supported")
class TestLangChainInteroperabilityIfNotSupported:
    def test_get_unsupported_reason(self) -> None:
        assert (
            LangChainInteroperability.get_unsupported_reason()
            == "This submodule is only supported for Python versions 3.9 and above"
        )
