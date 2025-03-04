# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop import Interoperable
from autogen.interop.langchain import LangChainInteroperability

from ...conftest import Credentials

with optional_import_block():
    from langchain.tools import tool as langchain_tool


# skip if python version is not >= 3.9
@pytest.mark.interop
@run_for_optional_imports("langchain", "interop-langchain")
class TestLangChainInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mock = MagicMock()

        class SearchInput(BaseModel):
            query: str = Field(description="should be a search query")

        @langchain_tool("search-tool", args_schema=SearchInput, return_direct=True)  # type: ignore[misc]
        def search_tool(query: SearchInput) -> str:
            """Look up things online."""
            self.mock(query)
            return "LangChain Integration"

        self.search_tool = search_tool

        self.tool = LangChainInteroperability.convert_tool(search_tool)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = LangChainInteroperability()

        # runtime check
        assert isinstance(interop, Interoperable)

    def test_convert_tool(self) -> None:
        assert self.tool.name == "search-tool"
        assert self.tool.description == "Look up things online."

        model_type = self.search_tool.get_input_schema()
        issubclass(model_type, BaseModel)

        tool_input = model_type(query="LangChain")  # type: ignore[misc]
        assert self.tool.func(tool_input=tool_input) == "LangChain Integration"

    @run_for_optional_imports("openai", "openai")
    def test_with_llm(self, credentials_gpt_4o: Credentials, user_proxy: UserProxyAgent) -> None:
        llm_config = credentials_gpt_4o.llm_config

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


@run_for_optional_imports("langchain", "interop-langchain")
class TestLangChainInteroperabilityWithoutPydanticInput:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mock = MagicMock()

        @langchain_tool  # type: ignore[misc]
        def search_tool(query: str, max_length: int) -> str:
            """Look up things online."""
            self.mock(query, max_length)
            return f"LangChain Integration, max_length: {max_length}"

        self.search_tool = search_tool

        self.tool = LangChainInteroperability.convert_tool(search_tool)

    def test_convert_tool(self) -> None:
        assert self.tool.name == "search_tool"
        assert self.tool.description == "Look up things online."

        model_type = self.search_tool.get_input_schema()
        assert issubclass(model_type, BaseModel)
        tool_input = model_type(query="LangChain", max_length=100)
        assert self.tool.func(tool_input=tool_input) == "LangChain Integration, max_length: 100"

    @run_for_optional_imports("openai", "openai")
    def test_with_llm(self, credentials_gpt_4o: Credentials, user_proxy: UserProxyAgent) -> None:
        llm_config = credentials_gpt_4o.llm_config
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
