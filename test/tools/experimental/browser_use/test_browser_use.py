# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Callable

import pytest

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import skip_on_missing_imports
from autogen.tools.experimental.browser_use import BrowserUseResult, BrowserUseTool

from ....conftest import Credentials, credentials_browser_use


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "browser_use"],
    "browser-use",
)
class TestBrowserUseToolOpenai:
    def test_broser_use_tool_init(self, mock_credentials: Credentials) -> None:
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
        ("config_list", "llm_class_name"),
        [
            (
                [
                    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test"},
                ],
                "ChatOpenAI",
            ),
            (
                [
                    {"api_type": "deepseek", "model": "deepseek-model", "api_key": "test", "base_url": "test"},
                ],
                "ChatOpenAI",
            ),
            (
                [
                    {
                        "api_type": "azure",
                        "model": "gpt-4o-mini",
                        "api_key": "test",
                        "base_url": "test",
                        "api_version": "test",
                    },
                ],
                "AzureChatOpenAI",
            ),
            (
                [
                    {"api_type": "google", "model": "gemini", "api_key": "test"},
                ],
                "ChatGoogleGenerativeAI",
            ),
            (
                [
                    {"api_type": "anthropic", "model": "sonnet", "api_key": "test"},
                ],
                "ChatAnthropic",
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b-instruct-v0.3-q6_K"}],
                "ChatOllama",
            ),
        ],
    )
    def test_get_llm(  # type: ignore[no-any-unimported]
        self,
        config_list: list[dict[str, str]],
        llm_class_name: str,
    ) -> None:
        llm = BrowserUseTool._get_llm(llm_config={"config_list": config_list})
        assert llm.__class__.__name__ == llm_class_name

    @pytest.mark.parametrize(
        ("config_list", "error_msg"),
        [
            (
                [
                    {"api_type": "deepseek", "model": "gpt-4o-mini", "api_key": "test"},
                ],
                "base_url is required for deepseek api type.",
            ),
            (
                [
                    {"api_type": "azure", "model": "gpt-4o-mini", "api_key": "test", "base_url": "test"},
                ],
                "api_version is required for azure api type.",
            ),
        ],
    )
    def test_get_llm_raises_if_mandatory_key_missing(self, config_list: list[dict[str, str]], error_msg: str) -> None:
        with pytest.raises(ValueError, match=error_msg):
            BrowserUseTool._get_llm(llm_config={"config_list": config_list})

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

        # If we decide to test with deepseek, we need to set use_vision to False
        agent_kwargs = {"use_vision": False} if api_type == "deepseek" else {}
        browser_use_tool = BrowserUseTool(llm_config=credentials_from_test_param.llm_config, agent_kwargs=agent_kwargs)
        task = "Go to Reddit, search for 'ag2' in the search bar, click on the first post and return the first comment."

        result = await browser_use_tool(
            task=task,
        )
        assert isinstance(result, BrowserUseResult)
        assert len(result.extracted_content) > 0

    @pytest.fixture()
    def browser_use_tool(self, credentials_gpt_4o_mini: Credentials) -> BrowserUseTool:
        return BrowserUseTool(llm_config=credentials_gpt_4o_mini.llm_config)

    @pytest.mark.openai
    def test_end2end(self, browser_use_tool: BrowserUseTool, credentials_gpt_4o: Credentials) -> None:
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")
        assistant = AssistantAgent(name="assistant", llm_config=credentials_gpt_4o.llm_config)

        browser_use_tool.register_for_execution(user_proxy)
        browser_use_tool.register_for_llm(assistant)

        result = user_proxy.initiate_chat(
            recipient=assistant,
            message="Go to Reddit, search for 'ag2' in the search bar, click on the first post and return the first comment.",
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
