# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from autogen import UserProxyAgent
from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.tools.experimental import BrowserUseTool

from ....conftest import Credentials

with optional_import_block():
    from browser_use import Agent
    from langchain_openai import ChatOpenAI


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires Python 3.11 or higher")
@skip_on_missing_imports(["langchain_openai", "browser_use"], "browser-use")
@pytest.mark.browser_use  # todo: remove me after we merge the PR that ads it automatically
class TestBrowserUseTool:
    def _use_imports(self) -> None:
        self._ChatOpenAI = ChatOpenAI
        self._Agent = Agent

    @pytest.fixture()
    def browser_use_tool(self, credentials_gpt_4o: Credentials) -> BrowserUseTool:
        api_key = credentials_gpt_4o.api_key
        return BrowserUseTool(api_key=api_key)

    @pytest.fixture()
    def user_proxy(self) -> UserProxyAgent:
        return UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

    @pytest.mark.asyncio
    async def test_browser_use_tool(self, browser_use_tool: BrowserUseTool) -> None:
        result = await browser_use_tool(
            task="Go to Reddit, search for 'ag2' in the search bar, click on the first post and return the first comment."
        )
        print("*" * 80)
        print(result)
        assert len(result) > 0
        assert "error" not in result.lower()
