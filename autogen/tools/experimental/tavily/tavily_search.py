import os
from typing import Annotated, Any, Optional, Union

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from tavily import TavilyClient


@require_optional_import(
    [
        "tavily",
    ],
    "tavily",
)
def _execute_tavily_query(
    query: str,
    tavily_api_key: str,
    search_depth: str = "basic",
    topic: str = "general",
    include_answer: str = "basic",
    include_raw_content: bool = False,
    include_domains: list[str] = [],
    num_results: int = 5,
) -> Any:
    tavily_client = TavilyClient(api_key=tavily_api_key)
    return tavily_client.search(
        query=query,
        search_depth=search_depth,
        topic=topic,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        max_results=num_results,
    )


def _tavily_search(
    query: str,
    tavily_api_key: str,
    search_depth: str = "basic",
    topic: str = "general",
    include_answer: str = "basic",
    include_raw_content: bool = False,
    include_domains: list[str] = [],
    num_results: int = 5,
) -> list[dict[str, Any]]:
    res = _execute_tavily_query(
        query=query,
        tavily_api_key=tavily_api_key,
        search_depth=search_depth,
        topic=topic,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        num_results=num_results,
    )

    return [
        {"title": item.get("title", ""), "link": item.get("url", ""), "snippet": item.get("content", "")}
        for item in res.get("results", [])
    ]


@export_module("autogen.tools.experimental")
class TavilySearchTool(Tool):
    """TavilySearchTool is a tool that uses the Tavily Search API to perform a search."""

    def __init__(
        self, *, llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None, tavily_api_key: Optional[str] = None
    ):
        """TavilySearchTool is a tool that uses the Tavily Search API to perform a search.

        Args:
            tavily_api_key: The API key for the Tavily Search API.
        """
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

        if tavily_api_key is None:
            raise ValueError("tavily_api_key must be provided")

        def tavily_search(
            query: Annotated[str, "The search query."],
            tavily_api_key: Annotated[Optional[str], Depends(on(tavily_api_key))],
            search_depth: Annotated[Optional[str], "Either 'advanced' or 'basic'"] = "basic",
            include_answer: Annotated[Optional[str], "Either 'advanced' or 'basic'"] = "basic",
            include_raw_content: Annotated[Optional[bool], "Include the raw contents"] = False,
            include_domains: Annotated[Optional[list[str]], "Specific web domains to search"] = [],
            num_results: Annotated[int, "The number of results to return."] = 10,
        ) -> list[dict[str, Any]]:
            if tavily_api_key is None:
                raise ValueError("Please provide tavily_api_key.\n")
            return _tavily_search(query, tavily_api_key, str(num_results))

        super().__init__(
            name="tavily_search",
            description="Use the Tavily Search API to perform a search.",
            func_or_tool=tavily_search,
        )
