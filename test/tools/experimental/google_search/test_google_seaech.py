# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental import GoogleSearchTool


@run_for_optional_imports(
    [
        "googleapiclient",
    ],
    "google-search",
)
class TestGoogleSearchTool:
    @pytest.mark.parametrize("use_genai_search_tool", [True, False])
    def test_init(self, use_genai_search_tool: bool) -> None:
        if use_genai_search_tool:
            google_search_tool = GoogleSearchTool(use_genai_search_tool=True)
            tool_name = "gemini_google_search"
            assert google_search_tool.name == tool_name
        else:
            google_search_tool = GoogleSearchTool(search_api_key="api_key", search_engine_id="engine_id")
            tool_name = "google_search"
            assert google_search_tool.name == tool_name

        assert google_search_tool.description == "Use the Google Search API to perform a search."
        expected_schema = {
            "description": "Use the Google Search API to perform a search.",
            "name": tool_name,
            "parameters": {
                "properties": {
                    "num_results": {
                        "default": 10,
                        "description": "The number of results to return.",
                        "type": "integer",
                    },
                    "query": {"description": "The search query.", "type": "string"},
                },
                "required": ["query"],
                "type": "object",
            },
        }
        assert google_search_tool.function_schema == expected_schema
