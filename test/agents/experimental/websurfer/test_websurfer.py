# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.agentchat import UserProxyAgent
from autogen.agentchat.chat import ChatResult
from autogen.agents.experimental import WebSurfer
from autogen.import_utils import skip_on_missing_imports

from ....conftest import Credentials


def _check_tool_called(result: ChatResult, tool_name: str) -> bool:
    for message in result.chat_history:
        if "tool_calls" in message and message["tool_calls"][0]["function"]["name"] == tool_name:
            return True

    return False


@skip_on_missing_imports(["crawl4ai"], "crawl4ai")
class TestCrawl4AIWebSurfer:
    def test_init(self, mock_credentials: Credentials) -> None:
        websurfer = WebSurfer(name="WebSurfer", llm_config=mock_credentials.llm_config, web_tool="crawl4ai")
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
        assert websurfer.llm_config["tools"] == expected

    @pytest.mark.openai
    def test_end2end(self, credentials_gpt_4o_mini: Credentials) -> None:
        websurfer = WebSurfer(name="WebSurfer", llm_config=credentials_gpt_4o_mini.llm_config, web_tool="crawl4ai")
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

        websurfer_tools = websurfer.tools
        for tool in websurfer_tools:
            tool.register_for_execution(user_proxy)

        result = user_proxy.initiate_chat(
            recipient=websurfer,
            message="Get info from https://docs.ag2.ai/docs/Home",
            max_turns=2,
        )

        assert _check_tool_called(result, "crawl4ai")


@skip_on_missing_imports(["langchain_openai", "browser_use"], "browser-use")
class TestBrowserUseWebSurfer:
    def test_init(self, mock_credentials: Credentials) -> None:
        websurfer = WebSurfer(name="WebSurfer", llm_config=mock_credentials.llm_config, web_tool="browser-use")
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
        assert websurfer.llm_config["tools"] == expected, websurfer.llm_config["tools"]

    @pytest.mark.openai
    def test_end2end(self, credentials_gpt_4o_mini: Credentials) -> None:
        websurfer = WebSurfer(name="WebSurfer", llm_config=credentials_gpt_4o_mini.llm_config, web_tool="browser-use")
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

        websurfer_tools = websurfer.tools
        for tool in websurfer_tools:
            tool.register_for_execution(user_proxy)

        result = user_proxy.initiate_chat(
            recipient=websurfer,
            message="Get info from https://docs.ag2.ai/docs/Home",
            max_turns=2,
        )

        assert _check_tool_called(result, "browser_use")
