# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import pytest

from autogen.agentchat import UserProxyAgent
from autogen.agentchat.chat import ChatResult
from autogen.agents.experimental import WebSurferAgent
from autogen.import_utils import run_for_optional_imports
from autogen.llm_config.config import LLMConfig

from ....conftest import Credentials


class WebSurferTestHelper:
    @staticmethod
    def _check_tool_called(result: ChatResult, tool_name: str) -> bool:
        for message in result.chat_history:
            if "tool_calls" in message and message["tool_calls"][0]["function"]["name"] == tool_name:
                return True

        return False

    def test_init(
        self,
        credentials: Credentials,
        web_tool: Literal["browser_use", "crawl4ai", "firecrawl"],
        expected: list[dict[str, Any]],
    ) -> None:
        websurfer = WebSurferAgent(name="WebSurfer", llm_config=credentials.llm_config, web_tool=web_tool)
        assert websurfer.llm_config is not False, "llm_config should not be False"
        assert isinstance(websurfer.llm_config, (dict, LLMConfig)), "llm_config should be a dictionary or LLMConfig"
        assert websurfer.llm_config["tools"] == expected

    def test_end2end(self, credentials: Credentials, web_tool: Literal["browser_use", "crawl4ai", "firecrawl"]) -> None:
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


@run_for_optional_imports(["crawl4ai"], "crawl4ai")
class TestCrawl4AIWebSurfer(WebSurferTestHelper):
    @pytest.mark.parametrize("web_tool", ["crawl4ai"])
    @pytest.mark.skip(reason="This test is failing, TODO: fix it")
    def test_init(
        self,
        mock_credentials: Credentials,
        expected: list[dict[str, Any]],
        web_tool: Literal["browser_use", "crawl4ai", "firecrawl"],
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

    @run_for_optional_imports("openai", "openai")
    @pytest.mark.parametrize("web_tool", ["crawl4ai"])
    def test_end2end(
        self, credentials_gpt_4o_mini: Credentials, web_tool: Literal["browser_use", "crawl4ai", "firecrawl"]
    ) -> None:
        super().test_end2end(credentials_gpt_4o_mini, "crawl4ai")


@run_for_optional_imports(["langchain_openai", "browser_use"], "browser-use")
class TestBrowserUseWebSurfer(WebSurferTestHelper):
    @pytest.mark.skip(reason="This test is failing, TODO: fix it")
    @pytest.mark.parametrize("web_tool", ["browser_use"])
    def test_init(
        self,
        mock_credentials: Credentials,
        expected: list[dict[str, Any]],
        web_tool: Literal["browser_use", "crawl4ai", "firecrawl"],
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

    @run_for_optional_imports("openai", "openai")
    @pytest.mark.parametrize("web_tool", ["browser_use"])
    def test_end2end(
        self, credentials_gpt_4o_mini: Credentials, web_tool: Literal["browser_use", "crawl4ai", "firecrawl"]
    ) -> None:
        super().test_end2end(credentials_gpt_4o_mini, "browser_use")


@run_for_optional_imports(["firecrawl-py"], "firecrawl")
class TestFirecrawlWebSurfer(WebSurferTestHelper):
    @pytest.mark.parametrize("web_tool", ["firecrawl"])
    @pytest.mark.skip(reason="This test requires API credentials")
    def test_init(
        self,
        mock_credentials: Credentials,
        expected: list[dict[str, Any]],
        web_tool: Literal["browser_use", "crawl4ai", "firecrawl"],
    ) -> None:
        expected = [
            {
                "function": {
                    "description": "Use the Firecrawl API to scrape content from a single URL.",
                    "name": "firecrawl_scrape",
                    "parameters": {
                        "properties": {
                            "url": {"description": "The URL to scrape.", "type": "string"},
                            "formats": {
                                "description": "Output formats (e.g., ['markdown', 'html'])",
                                "items": {"type": "string"},
                                "type": "array",
                            },
                            "include_tags": {
                                "description": "HTML tags to include",
                                "items": {"type": "string"},
                                "type": "array",
                            },
                            "exclude_tags": {
                                "description": "HTML tags to exclude",
                                "items": {"type": "string"},
                                "type": "array",
                            },
                            "headers": {
                                "description": "HTTP headers to use",
                                "type": "object",
                            },
                            "wait_for": {
                                "description": "Time to wait for page load in milliseconds",
                                "type": "integer",
                            },
                            "timeout": {
                                "description": "Request timeout in milliseconds",
                                "type": "integer",
                            },
                        },
                        "required": ["url"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ]
        # Test with Firecrawl API key
        websurfer = WebSurferAgent(
            name="WebSurfer",
            llm_config=mock_credentials.llm_config,
            web_tool="firecrawl",
            web_tool_kwargs={"firecrawl_api_key": "test_key"},
        )
        assert websurfer.llm_config is not False, "llm_config should not be False"
        assert isinstance(websurfer.llm_config, (dict, LLMConfig)), "llm_config should be a dictionary or LLMConfig"
        assert websurfer.llm_config["tools"] == expected

    @run_for_optional_imports("openai", "openai")
    @pytest.mark.parametrize("web_tool", ["firecrawl"])
    @pytest.mark.skip(reason="This test requires API credentials")
    def test_end2end(
        self, credentials_gpt_4o_mini: Credentials, web_tool: Literal["browser_use", "crawl4ai", "firecrawl"]
    ) -> None:
        # Note: This test would require a valid Firecrawl API key
        websurfer = WebSurferAgent(
            name="WebSurfer",
            llm_config=credentials_gpt_4o_mini.llm_config,
            web_tool="firecrawl",
            web_tool_kwargs={"firecrawl_api_key": "test_key"},
        )
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

        websurfer_tools = websurfer.tools
        for tool in websurfer_tools:
            tool.register_for_execution(user_proxy)

        result = user_proxy.initiate_chat(
            recipient=websurfer,
            message="Get info from https://docs.ag2.ai/docs/Home",
            max_turns=2,
        )

        assert self._check_tool_called(result, "firecrawl_scrape")
