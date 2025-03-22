# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Dict, List, Optional

import requests

from autogen.import_utils import optional_import_block
from autogen.tools import Tool
from autogen.tools.experimental.perplexity.schemas import PerplexityChatCompletionResponse

with optional_import_block():
    pass


class PerplexitySearchTool(Tool):
    def __init__(
        self,
        model: str = "sonar",
        ppx_api_key: Optional[str] = None,
        max_tokens: int = 1000,
        search_domain_filter: Optional[List[str]] = None,
    ):
        self.url = "https://api.perplexity.ai/chat/completions"
        self.model = model
        self.perplexity_api_key = ppx_api_key
        self.max_tokens = max_tokens
        self.search_domain_filter = search_domain_filter

        if not ppx_api_key:
            raise ValueError(
                "Perplexity API key is missing. Please provide a valid API key to use the Perplexity service."
            )

        super().__init__(
            name="perplexity-search",
            description="Perplexity AI search tool to retrieve real-time, conversational answers with cited sources from the web.",
            func_or_tool=self.search,
        )

    def _execute_query(self, payload: Dict[str, Any]) -> PerplexityChatCompletionResponse:
        headers = {"Authorization": f"Bearer {self.perplexity_api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, json=payload, headers=headers)
        response_json = response.json()
        perp_resp = PerplexityChatCompletionResponse(**response_json)
        return perp_resp

    def search(self, query: Annotated[str, "The search query."]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "Be precise and concise."}, {"role": "user", "content": query}],
            "max_tokens": self.max_tokens,
            "search_domain_filter": self.search_domain_filter,
            "web_search_options": {"search_context_size": "high"},
        }

        perplexity_response = self._execute_query(payload)
        content = perplexity_response.choices[0].message.content
        citations = perplexity_response.citations

        return {"content": content, "citations": citations}
