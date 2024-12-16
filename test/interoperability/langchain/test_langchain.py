# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from conftest import reason, skip_openai
from langchain.tools import tool
from pydantic import BaseModel, Field

from autogen import AssistantAgent, UserProxyAgent
from autogen.interoperability import Interoperable
from autogen.interoperability.langchain import LangchainInteroperability


class TestLangchainInteroperability:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class SearchInput(BaseModel):
            query: str = Field(description="should be a search query")

        @tool("search-tool", args_schema=SearchInput, return_direct=True)  # type: ignore[misc]
        def search(query: SearchInput) -> str:
            """Look up things online."""
            return "LangChain Integration"

        self.langchain_interop = LangchainInteroperability()
        self.model_type = search.args_schema
        self.tool = self.langchain_interop.convert_tool(search)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = self.langchain_interop
        # runtime check
        assert isinstance(interop, Interoperable)

    def test_init(self) -> None:
        assert isinstance(self.langchain_interop, Interoperable)

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
