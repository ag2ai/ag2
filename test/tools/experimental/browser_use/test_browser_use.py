# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Callable
from unittest.mock import MagicMock

import pytest

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.tools.experimental.browser_use import BrowserUseResult, BrowserUseTool

from ....conftest import Credentials

with optional_import_block():
    from browser_use import Agent
    from browser_use.browser.browser import Browser, BrowserConfig
    from langchain_openai import ChatOpenAI


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires Python 3.11 or higher")
@skip_on_missing_imports(["langchain_openai", "browser_use"], "browser-use")
@pytest.mark.browser_use  # todo: remove me after we merge the PR that ads it automatically
class TestBrowserUseToolOpenai:
    def _use_imports(self) -> None:
        self._ChatOpenAI = ChatOpenAI
        self._Agent = Agent

    def test_broser_use_tool_init(self) -> None:
        browser_use_tool = BrowserUseTool(api_key="api_key")
        assert browser_use_tool.name == "browser_use"
        assert browser_use_tool.description == "Use the browser to perform a task."
        assert isinstance(browser_use_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Use the browser to perform a task.",
            "name": "browser_use",
            "parameters": {
                "properties": {"task": {"description": "The task to perform.", "type": "string"}},
                "required": ["task"],
                "type": "object",
            },
        }
        assert browser_use_tool.function_schema == expected_schema

    @pytest.fixture()
    def browser_use_tool(self, credentials_gpt_4o: Credentials) -> BrowserUseTool:
        api_key = credentials_gpt_4o.api_key
        browser_config = BrowserConfig(
            headless=True,
        )
        browser = Browser(config=browser_config)
        return BrowserUseTool(api_key=api_key, browser=browser)

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_browser_use_tool(self, browser_use_tool: BrowserUseTool) -> None:
        result = await browser_use_tool(
            task="Go to Reddit, search for 'ag2' in the search bar, click on the first post and return the first comment."
        )
        assert isinstance(result, BrowserUseResult)
        assert len(result.extracted_content) > 0

    @pytest.mark.openai
    def test_end2end(self, browser_use_tool: BrowserUseTool, credentials_gpt_4o: Credentials) -> None:
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")
        assistant = AssistantAgent(name="assistant", llm_config=credentials_gpt_4o.llm_config)

        user_proxy.register_for_execution()(browser_use_tool)
        assistant.register_for_llm()(browser_use_tool)

        # Wrap the function so we can check if it was called
        mock = MagicMock(wraps=browser_use_tool.func)
        browser_use_tool._func = mock

        user_proxy.initiate_chat(
            recipient=assistant,
            message="Go to Reddit, search for 'ag2' in the search bar, click on the first post and return the first comment.",
            max_turns=2,
        )

        mock.assert_called_once()
