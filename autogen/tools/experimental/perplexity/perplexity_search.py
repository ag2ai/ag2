# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Dict, List, Optional

import requests

from autogen.tools import Tool

from .schemas import PerplexityChatCompletionResponse, SearchResponse


class PerplexitySearchTool(Tool):
    def __init__(
        self,
        model: str = "sonar",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        search_domain_filter: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self._validate_tool_config(model, self.api_key, max_tokens, search_domain_filter)
        self.url = "https://api.perplexity.ai/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.search_domain_filters = search_domain_filter
        super().__init__(
            name="perplexity-search",
            description="Perplexity AI search tool for web search, news search, and conversational search "
            "for finding answers to everyday questions, conducting in-depth research and analysis.",
            func_or_tool=self.search,
        )

    @staticmethod
    def _validate_tool_config(
        model: str, api_key: Optional[str], max_tokens: int, search_domain_filter: Optional[List[str]]
    ) -> None:
        if not api_key:
            raise ValueError("Perplexity API key is missing")
        if model is None or model == "":
            raise ValueError("model cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if search_domain_filter is not None and not isinstance(search_domain_filter, list):
            raise ValueError("search_domain_filter must be a list")

    def _execute_query(self, payload: Dict[str, Any]) -> PerplexityChatCompletionResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, json=payload, headers=headers)
        response_json = response.json()
        perp_resp = PerplexityChatCompletionResponse(**response_json)
        return perp_resp

    def search(self, query: Annotated[str, "The search query."]) -> SearchResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "Be precise and concise."}, {"role": "user", "content": query}],
            "max_tokens": self.max_tokens,
            "search_domain_filter": self.search_domain_filters,
            "web_search_options": {"search_context_size": "high"},
        }
        perplexity_response = self._execute_query(payload)
        content = perplexity_response.choices[0].message.content
        citations = perplexity_response.citations
        response = SearchResponse(content=content, citations=citations)
        return response
