# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop import Interoperable
from autogen.interop.langchain import LangChainInteroperability
from test.credentials import Credentials

with optional_import_block():
    from langchain.tools import tool as langchain_tool
    from langchain_core.tools import BaseTool as LangchainBaseTool


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
class TestLangChainInteroperabilityAsync:
    """Async langchain tools must be converted to async AG2 tools.

    Regression for https://github.com/ag2ai/ag2/issues/1402 — async langchain
    tools were previously wrapped with the synchronous ``run`` dispatcher,
    blocking the event loop when invoked from ``a_initiate_chat``.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mock = MagicMock()

        @langchain_tool  # type: ignore[misc]
        async def search_tool(query: str) -> str:
            """Look up things online asynchronously."""
            self.mock(query)
            return "Async LangChain Integration"

        self.search_tool = search_tool
        self.tool = LangChainInteroperability.convert_tool(search_tool)

    def test_async_tool_wrapper_is_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(self.tool.func)

    def test_async_tool_executes_via_arun(self) -> None:
        model_type = self.search_tool.get_input_schema()
        tool_input = model_type(query="LangChain")  # type: ignore[misc]
        result = asyncio.run(self.tool.func(tool_input=tool_input))
        assert result == "Async LangChain Integration"
        self.mock.assert_called_once_with("LangChain")


@run_for_optional_imports("langchain", "interop-langchain")
class TestLangChainInteroperabilityClassBasedAsync:
    """``BaseTool`` subclasses that override ``_arun`` must also use the async path."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        mock = MagicMock()

        class AsyncSearchInput(BaseModel):
            query: str = Field(description="search query")

        class AsyncSearchTool(LangchainBaseTool):  # type: ignore[no-any-unimported,misc]
            name: str = "async-search-tool"
            description: str = "Look up things online asynchronously."
            args_schema: type[BaseModel] = AsyncSearchInput

            def _run(self, query: str) -> str:
                raise NotImplementedError("sync path must not be used")

            async def _arun(self, query: str) -> str:
                mock(query)
                return f"async:{query}"

        self.mock = mock
        self.search_tool = AsyncSearchTool()
        self.tool = LangChainInteroperability.convert_tool(self.search_tool)

    def test_async_tool_wrapper_is_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(self.tool.func)

    def test_async_tool_executes_via_arun(self) -> None:
        model_type = self.search_tool.get_input_schema()
        tool_input = model_type(query="hello")  # type: ignore[misc]
        result = asyncio.run(self.tool.func(tool_input=tool_input))
        assert result == "async:hello"
        self.mock.assert_called_once_with("hello")


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
