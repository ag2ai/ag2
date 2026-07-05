# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("tavily")

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.search.tavily import SearchResponse, SearchResult, TavilySearchTool

TAVILY_BASE_URL = "https://api.tavily.com"

SAMPLE_RAW: dict[str, Any] = {
    "query": "AG2 framework",
    "answer": "AG2 is an open-source multi-agent framework.",
    "results": [
        {
            "title": "AG2 Framework",
            "url": "https://ag2.ai",
            "content": "AG2 is an agent framework.",
            "score": 0.95,
            "raw_content": "# AG2\nFull text",
            "favicon": "https://ag2.ai/favicon.ico",
        },
        {
            "title": "GitHub - AG2",
            "url": "https://github.com/ag2ai/ag2",
            "content": "Open source repo.",
            "score": 0.82,
            "raw_content": None,
            "favicon": None,
        },
    ],
    "images": ["https://ag2.ai/img.png"],
}


def _make_config(query: str, *, final_reply: str = "done", tool_name: str = "tavily_search") -> TestConfig:
    call = ToolCallEvent(arguments=json.dumps({"query": query}), name=tool_name)
    return TestConfig(ModelResponse(tool_calls=ToolCallsEvent([call])), final_reply)


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schema(self, context: Context) -> None:
        tavily = TavilySearchTool(api_key="test")

        [schema] = await tavily.schemas(context)

        assert schema.function.name == "tavily_search"
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_schema(self, context: Context) -> None:
        tavily = TavilySearchTool(api_key="test", name="my_search", description="Custom search tool.")

        [schema] = await tavily.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_search_returns_structured_results(self) -> None:
        respx.post(f"{TAVILY_BASE_URL}/search").mock(return_value=httpx.Response(200, json=SAMPLE_RAW))
        tavily = TavilySearchTool(api_key="test")

        config = TrackingConfig(_make_config("AG2 framework"))
        agent = Agent("a", config=config, tools=[tavily])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchResponse(
                query="AG2 framework",
                results=[
                    SearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        content="AG2 is an agent framework.",
                        score=0.95,
                    ),
                    SearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        content="Open source repo.",
                        score=0.82,
                    ),
                ],
                answer="AG2 is an open-source multi-agent framework.",
                images=["https://ag2.ai/img.png"],
            )
        )

    @respx.mock
    async def test_search_empty_results(self) -> None:
        respx.post(f"{TAVILY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"query": "nothing", "results": []})
        )
        tavily = TavilySearchTool(api_key="test")

        config = TrackingConfig(_make_config("nothing"))
        agent = Agent("a", config=config, tools=[tavily])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(SearchResponse(query="nothing", results=[]))

    @respx.mock
    async def test_all_params_forwarded_to_client(self) -> None:
        route = respx.post(f"{TAVILY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"query": "q", "results": []})
        )
        tavily = TavilySearchTool(
            api_key="test",
            max_results=3,
            search_depth="advanced",
            topic="news",
            include_answer=True,
            include_raw_content="markdown",
            include_images=True,
            time_range="week",
            start_date="2024-01-01",
            end_date="2024-12-31",
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            country="US",
            auto_parameters=True,
            include_favicon=True,
        )
        agent = Agent("a", config=_make_config("q"), tools=[tavily])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "query": "q",
            "max_results": 3,
            "search_depth": "advanced",
            "topic": "news",
            "include_answer": True,
            "include_raw_content": "markdown",
            "include_images": True,
            "time_range": "week",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "include_domains": ["arxiv.org"],
            "exclude_domains": ["medium.com"],
            "country": "US",
            "auto_parameters": True,
            "include_favicon": True,
        })

    @respx.mock
    async def test_none_params_omitted(self) -> None:
        route = respx.post(f"{TAVILY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"query": "q", "results": []})
        )
        tavily = TavilySearchTool(api_key="test")
        agent = Agent("a", config=_make_config("q"), tools=[tavily])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"query": "q"})

    @respx.mock
    async def test_client_kwargs_forwarded_to_sdk(self) -> None:
        custom_url = "https://custom.tavily.example"
        route = respx.post(f"{custom_url}/search").mock(
            return_value=httpx.Response(200, json={"query": "q", "results": []})
        )
        tavily = TavilySearchTool(api_key="test", api_base_url=custom_url)
        agent = Agent("a", config=_make_config("q"), tools=[tavily])
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_custom_tool_name_in_agent(self) -> None:
        route = respx.post(f"{TAVILY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"query": "q", "results": []})
        )
        tavily = TavilySearchTool(api_key="test", name="web_search")

        config = TrackingConfig(_make_config("AG2 framework", tool_name="web_search"))
        agent = Agent("a", config=config, tools=[tavily])
        await agent.ask("search")

        assert route.called


@pytest.mark.asyncio
class TestTavilySearchToolVariable:
    @respx.mock
    async def test_resolved(self) -> None:
        route = respx.post(f"{TAVILY_BASE_URL}/search").mock(return_value=httpx.Response(200, json=SAMPLE_RAW))
        tavily = TavilySearchTool(
            api_key="test",
            search_depth=Variable("user_depth"),
            topic=Variable(),
        )

        agent = Agent(
            "a",
            config=_make_config("test query"),
            tools=[tavily],
            variables={"user_depth": "basic", "topic": "general"},
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "query": "test query",
            "search_depth": "basic",
            "topic": "general",
        })

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(f"{TAVILY_BASE_URL}/search").mock(return_value=httpx.Response(200, json=SAMPLE_RAW))
        tavily = TavilySearchTool(api_key="test", topic=Variable())

        agent = Agent("a", config=_make_config("test query"), tools=[tavily])

        with pytest.raises(KeyError, match="topic"):
            await agent.ask("search")
