# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from unittest.mock import MagicMock

import pytest
from langchain.tools import tool as langchain_tool
from pydantic import BaseModel, Field

import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.interop import Interoperable
from autogen.interop.langchain import LangChainInteroperability

from ...conftest import Credentials, reason, skip_openai


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
class TestLangChainInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class SearchInput(BaseModel):
            query: str = Field(description="should be a search query")

        self.mock = MagicMock()

        @langchain_tool("search-tool", args_schema=SearchInput, return_direct=True)  # type: ignore[misc]
        def search(query: SearchInput) -> str:
            """Look up things online."""
            self.mock(query)
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

        tool_input = self.model_type(query="LangChain")  # type: ignore[misc]
        assert self.tool.func(tool_input=tool_input) == "LangChain Integration"

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self, credentials_gpt_4o: Credentials) -> None:
        llm_config = credentials_gpt_4o.llm_config

        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
        )

        chatbot = AssistantAgent(
            name="chatbot",
            llm_config=llm_config,
        )

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        user_proxy.initiate_chat(recipient=chatbot, message="search for LangChain", max_turns=5)

        self.mock.assert_called()

    def test_get_unsupported_reason(self) -> None:
        assert LangChainInteroperability.get_unsupported_reason() is None


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
class TestLangChainInteroperabilityWithoutPydanticInput:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mock = MagicMock()

        @langchain_tool
        def search(query: str, max_length: int) -> str:
            """Look up things online."""
            self.mock(query, max_length)
            return f"LangChain Integration, max_length: {max_length}"

        self.tool = LangChainInteroperability.convert_tool(search)
        self.model_type = search.args_schema

    def test_convert_tool(self) -> None:
        assert self.tool.name == "search"
        assert self.tool.description == "Look up things online."

        tool_input = self.model_type(query="LangChain", max_length=100)  # type: ignore[misc]
        assert self.tool.func(tool_input=tool_input) == "LangChain Integration, max_length: 100"

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self, credentials_gpt_4o: Credentials) -> None:
        llm_config = credentials_gpt_4o.llm_config
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
        )

        chatbot = AssistantAgent(
            name="chatbot",
            llm_config=llm_config,
            system_message="""
When using the search tool, input should be:
{
    "tool_input": {
        "query": ...,
        "max_length": ...
    }
}
""",
        )

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        user_proxy.initiate_chat(recipient=chatbot, message="search for LangChain, Use max 100 characters", max_turns=5)

        self.mock.assert_called()


@pytest.mark.skipif(sys.version_info >= (3, 9), reason="LangChain Interoperability is supported")
class TestLangChainInteroperabilityIfNotSupported:
    def test_get_unsupported_reason(self) -> None:
        assert (
            LangChainInteroperability.get_unsupported_reason()
            == "This submodule is only supported for Python versions 3.9 and above"
        )
