# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Annotated, Any, Optional

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from googleapiclient.discovery import build


@require_optional_import(
    [
        "googleapiclient",
    ],
    "google-search",
)
def _google_search(
    query: str,
    search_api_key: str,
    search_engine_id: str,
    num_results: int,
) -> list[dict[str, Any]]:
    service = build("customsearch", "v1", developerKey=search_api_key)
    res = service.cse().list(q=query, cx=search_engine_id, num=num_results).execute()

    results = []
    if "items" in res:
        for item in res["items"]:
            results.append({"title": item["title"], "link": item["link"], "snippet": item.get("snippet", "")})
    return results


@export_module("autogen.tools.experimental")
class GoogleSearchTool(Tool):
    """GoogleSearchTool is a tool that uses the Google Search API to perform a search."""

    def __init__(
        self,
        *,
        search_api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        use_genai_search_tool: bool = False,
    ):
        """GoogleSearchTool is a tool that uses the Google Search API to perform a search.

        Args:
            search_api_key: The API key for the Google Search API.
            search_engine_id: The search engine ID for the Google Search API.
            use_genai_search_tool: Whether to use the predefined Gemini search tool. This can only be used for agents with the Gemini (GenAI) configuration.
        """
        self.search_api_key = search_api_key
        self.search_engine_id = search_engine_id
        self.use_predefined_gemini_tool = use_genai_search_tool

        if not use_genai_search_tool and (search_api_key is None or search_engine_id is None):
            raise ValueError(
                "search_api_key and search_engine_id must be provided if use_predefined_gemini_tool is False"
            )

        if use_genai_search_tool and (search_api_key is not None or search_engine_id is not None):
            logging.warning("search_api_key and search_engine_id will be ignored as use_predefined_gemini_tool is True")

        def google_search(
            query: Annotated[str, "The search query."],
            search_api_key: Annotated[Optional[str], Depends(on(search_api_key))],
            search_engine_id: Annotated[Optional[str], Depends(on(search_engine_id))],
            use_genai_search_tool: Annotated[bool, Depends(on(use_genai_search_tool))],
            num_results: Annotated[int, "The number of results to return."] = 10,
        ) -> list[dict[str, Any]]:
            if use_genai_search_tool or search_api_key is None or search_engine_id is None:
                raise ValueError(
                    "Your agent is not configured to use the Gemini (GenAI) LLM.\n"
                    "If you want to use different LLM providers, GoogleSearchTool must be configured with: use_genai_search_tool=False and provided search_api_key and search_engine_id.\n"
                )
            return _google_search(query, search_api_key, search_engine_id, num_results)

        super().__init__(
            # GeminiClient will look for a tool with the name "gemini_google_search"
            name="gemini_google_search" if use_genai_search_tool else "google_search",
            description="Google Search",
            func_or_tool=google_search,
        )
