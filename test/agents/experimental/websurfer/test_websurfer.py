# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import pytest

from autogen.agentchat import UserProxyAgent
from autogen.agentchat.chat import ChatResult
from autogen.agents.experimental import WebSurferAgent
from autogen.import_utils import skip_on_missing_imports

from ....conftest import Credentials


class WebSurferTestHelper:
    @staticmethod
    def _check_tool_called(result: ChatResult, tool_name: str) -> bool:
        for message in result.chat_history:
            if "tool_calls" in message and message["tool_calls"][0]["function"]["name"] == tool_name:
                return True

        return False

    def test_init(
        self, credentials: Credentials, web_tool: Literal["browser_use", "crawl4ai"], expected: list[dict[str, Any]]
    ) -> None:
        websurfer = WebSurferAgent(name="WebSurfer", llm_config=credentials.llm_config, web_tool=web_tool)
        assert websurfer.llm_config is not False, "llm_config should not be False"
        assert isinstance(websurfer.llm_config, dict), "llm_config should be a dictionary"
        assert websurfer.llm_config["tools"] == expected

    def test_end2end(self, credentials: Credentials, web_tool: Literal["browser_use", "crawl4ai"]) -> None:
        websurfer = WebSurferAgent(name="WebSurfer", llm_config=credentials.llm_config, web_tool=web_tool)
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

        websurfer_tools = websurfer.tools
        for tool in websurfer_tools:
            tool.register_for_execution(user_proxy)

        result = user_proxy.initiate_chat(
            recipient=websurfer,
            message="Get info from https://docs.ag2.ai/docs/Home",
            max_turns=2,
        )

        assert self._check_tool_called(result, web_tool)


@skip_on_missing_imports(["crawl4ai"], "crawl4ai")
class TestCrawl4AIWebSurfer(WebSurferTestHelper):
    def test_init(
        self,
        mock_credentials: Credentials,
        web_tool: Literal["browser_use", "crawl4ai"],
        expected: list[dict[str, Any]],
    ) -> None:
        expected = [
            {
                "function": {
                    "description": "Crawl a website and extract information.",
                    "name": "crawl4ai",
                    "parameters": {
                        "properties": {
                            "instruction": {
                                "description": "The instruction to provide on how and what to extract.",
                                "type": "string",
                            },
                            "url": {"description": "The url to crawl and extract information from.", "type": "string"},
                        },
                        "required": ["url", "instruction"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ]
        super().test_init(mock_credentials, "crawl4ai", expected)

    @pytest.mark.openai
    def test_end2end(self, credentials_gpt_4o_mini: Credentials, web_tool: Literal["browser_use", "crawl4ai"]) -> None:
        super().test_end2end(credentials_gpt_4o_mini, "crawl4ai")


@skip_on_missing_imports(["langchain_openai", "browser_use"], "browser-use")
class TestBrowserUseWebSurfer(WebSurferTestHelper):
    def test_init(
        self,
        mock_credentials: Credentials,
        web_tool: Literal["browser_use", "crawl4ai"],
        expected: list[dict[str, Any]],
    ) -> None:
        expected = [
            {
                "function": {
                    "description": "Use the browser to perform a task.",
                    "name": "browser_use",
                    "parameters": {
                        "properties": {"task": {"description": "The task to perform.", "type": "string"}},
                        "required": ["task"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ]
        super().test_init(mock_credentials, "browser_use", expected)

    @pytest.mark.openai
    def test_end2end(self, credentials_gpt_4o_mini: Credentials, web_tool: Literal["browser_use", "crawl4ai"]) -> None:
        super().test_end2end(credentials_gpt_4o_mini, "browser_use")
