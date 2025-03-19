# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Callable

import pytest

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.tools.experimental.browser_use import BrowserUseResult, BrowserUseTool, ExtractedContent

from ....conftest import Credentials, credentials_browser_use

with optional_import_block():
    from browser_use import Controller


class TestExtractedContent:
    @pytest.mark.parametrize(
        ("url", "expected_url"),
        [
            ("https://docs.ag2.ai/docs/Home", "https://docs.ag2.ai/docs/Home"),
            ("about:blank", None),
        ],
    )
    def test_url_is_proprely_set(self, url: str, expected_url: str) -> None:
        extracted_content = ExtractedContent(content="content", url=url)
        assert extracted_content.url == expected_url


@run_for_optional_imports(
    [
        "langchain_anthropic",
        "langchain_google_genai",
        "langchain_ollama",
        "langchain_openai",
        "langchain_core",
        "browser_use",
    ],
    "browser-use",
)
class TestBrowserUseToolOpenai:
    def test_browser_use_tool_init(self, mock_credentials: Credentials) -> None:
        browser_use_tool = BrowserUseTool(llm_config=mock_credentials.llm_config)
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

    @pytest.mark.parametrize(
        "credentials_from_test_param",
        credentials_browser_use,
        indirect=True,
    )
    @pytest.mark.asyncio
    async def test_browser_use_tool(self, credentials_from_test_param: Credentials) -> None:
        api_type = credentials_from_test_param.api_type
        if api_type == "deepseek":
            pytest.skip("Deepseek currently does not work too well with the browser-use")
        if api_type == "openai":
            pytest.skip("This test case will be covered by the test_end2end test case.")

        # If we decide to test with deepseek, we need to set use_vision to False
        agent_kwargs = {"use_vision": False, "max_steps": 100} if api_type == "deepseek" else {"max_steps": 100}
        browser_use_tool = BrowserUseTool(llm_config=credentials_from_test_param.llm_config, agent_kwargs=agent_kwargs)
        task = "Get info from https://docs.ag2.ai/docs/Home"

        result = await browser_use_tool(
            task=task,
        )
        assert isinstance(result, BrowserUseResult)
        assert len(result.extracted_content) > 0

    @pytest.fixture()
    def browser_use_tool(self, credentials_gpt_4o_mini: Credentials) -> BrowserUseTool:
        return BrowserUseTool(llm_config=credentials_gpt_4o_mini.llm_config)

    def test_get_controller(self, mock_credentials: Credentials) -> None:
        controller = BrowserUseTool._get_controller(llm_config=mock_credentials.llm_config)
        assert isinstance(controller, Controller)

    @run_for_optional_imports("openai", "openai")
    def test_end2end(self, browser_use_tool: BrowserUseTool, credentials_gpt_4o: Credentials) -> None:
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")
        assistant = AssistantAgent(name="assistant", llm_config=credentials_gpt_4o.llm_config)

        browser_use_tool.register_for_execution(user_proxy)
        browser_use_tool.register_for_llm(assistant)

        result = user_proxy.initiate_chat(
            recipient=assistant,
            message="Get info from https://docs.ag2.ai/docs/Home",
            max_turns=2,
        )

        result_validated = False
        for message in result.chat_history:
            if "role" in message and message["role"] == "tool":
                # Convert JSON string to Python dictionary
                data = json.loads(message["content"])
                assert isinstance(BrowserUseResult(**data), BrowserUseResult)
                result_validated = True
                break

        assert result_validated, "No valid result found in the chat history."
