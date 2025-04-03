# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import patch

import pytest

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental import GoogleSearchTool
from autogen.tools.experimental.google_search.google_search import _google_search

from ....conftest import Credentials


class TestGoogleSearchTool:
    @pytest.mark.parametrize("use_internal_llm_tool_if_available", [True, False])
    def test_init(self, use_internal_llm_tool_if_available: bool) -> None:
        if use_internal_llm_tool_if_available:
            google_search_tool = GoogleSearchTool()
        else:
            google_search_tool = GoogleSearchTool(search_api_key="api_key", search_engine_id="engine_id")

        assert google_search_tool.name == "prebuilt_google_search"
        assert google_search_tool.description == "Use the Google Search API to perform a search."

    @pytest.fixture
    def expected_search_result(self) -> dict[str, Any]:
        return {
            "items": [
                {
                    "title": "DeepSeek sparks AI stock selloff; Nvidia posts record market-cap ...",
                    "link": "https://www.reuters.com/technology/chinas-deepseek-sets-off-ai-market-rout-2025-01-27/",
                    "snippet": "Jan 27, 2025 ... DEEPSEEK 'SPUTNIK MOMENT'. After the release of the first Chinese ChatGPT equivalent, made by search engine giant Baidu (9888.HK)\xa0...",
                    "some_other_key": "some_other_value",
                },
                {
                    "title": "Nvidia shares sink as Chinese AI app DeepSeek spooks US markets",
                    "link": "https://www.bbc.com/news/articles/c0qw7z2v1pgo",
                    "snippet": "Jan 27, 2025 ... US tech giant Nvidia lost over a sixth of its value after the ... After DeepSeek-R1 was launched earlier this month, the company\xa0...",
                    "some_other_key": "some_other_value",
                },
            ],
        }

    def test_google_search_f(self, expected_search_result: dict[str, Any]) -> None:
        with patch(
            "autogen.tools.experimental.google_search.google_search._execute_query",
            return_value=expected_search_result,
        ):
            search_results = _google_search(
                query="DeepSeek",
                search_api_key="api_key",
                search_engine_id="engine_id",
                num_results=2,
            )
        assert len(search_results) == 2

    def _test_end_to_end(
        self,
        google_search_tool: GoogleSearchTool,
        credentials: Credentials,
        expected_search_result: dict[str, Any],
        execute_query_called: bool,
    ) -> None:
        assistant = AssistantAgent(
            name="assistant",
            llm_config=credentials.llm_config,
        )

        google_search_tool.register_for_llm(assistant)

        with patch(
            "autogen.tools.experimental.google_search.google_search._execute_query",
            return_value=expected_search_result,
        ) as mock_execute_query:
            run_response = assistant.run(
                message="Get me the latest news on DeepSeek",
                tools=assistant.tools,
                max_turns=3,
                user_input=False,
            )
            run_response.process()
            assert mock_execute_query.called == execute_query_called

    @run_for_optional_imports("openai", "openai")
    def test_end_to_end_openai(
        self, credentials_gpt_4o_mini: Credentials, expected_search_result: dict[str, Any]
    ) -> None:
        google_search_tool = GoogleSearchTool(search_api_key="api_key", search_engine_id="engine_id")
        self._test_end_to_end(
            google_search_tool=google_search_tool,
            credentials=credentials_gpt_4o_mini,
            expected_search_result=expected_search_result,
            execute_query_called=True,
        )

    @run_for_optional_imports(["google", "vertexai", "PIL", "jsonschema", "jsonschema"], "gemini")
    def test_end_to_end_gemini(
        self, credentials_gemini_flash_exp: Credentials, expected_search_result: dict[str, Any]
    ) -> None:
        google_search_tool = GoogleSearchTool(use_internal_llm_tool_if_available=True)
        self._test_end_to_end(
            google_search_tool=google_search_tool,
            credentials=credentials_gemini_flash_exp,
            expected_search_result=expected_search_result,
            execute_query_called=False,
        )
