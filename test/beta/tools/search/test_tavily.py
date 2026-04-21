# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsJson, IsPartialDict

pytest.importorskip("tavily")

from autogen.beta import Agent, Variable
from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from autogen.beta.events.types import ModelResponse
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.search.tavily import SearchResponse, SearchResult, TavilySearchTool

SAMPLE_RAW = {
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


def _make_config(query: str, final_reply: str = "done", tool_name: str = "tavily_search") -> TestConfig:
    call = ToolCallEvent(arguments=json.dumps({"query": query}), name=tool_name)
    return TestConfig(ModelResponse(tool_calls=ToolCallsEvent([call])), final_reply)


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schema(self, context: ConversationContext) -> None:
        tavily = TavilySearchTool(client=MagicMock())

        [schema] = await tavily.schemas(context)

        assert schema.function.name == "tavily_search"
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_schema(self, context: ConversationContext) -> None:
        tavily = TavilySearchTool(client=MagicMock(), name="my_search", description="Custom search tool.")

        [schema] = await tavily.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_search_returns_structured_results(self, mock: MagicMock) -> None:
        # arrange tool
        mock.search.return_value = SAMPLE_RAW
        tavily = TavilySearchTool(client=mock)

        # arrange agent
        config = TrackingConfig(_make_config("AG2 framework"))
        agent = Agent("a", config=config, tools=[tavily])

        # act
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].content == IsJson(
            asdict(
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
        )

    async def test_search_empty_results(self, mock: MagicMock) -> None:
        # arrange tool
        mock.search.return_value = {"query": "nothing", "results": []}
        tavily = TavilySearchTool(client=mock)

        # arrange agent
        config = TrackingConfig(_make_config("nothing"))
        agent = Agent("a", config=config, tools=[tavily])

        # act
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].content == IsJson(
            asdict(
                SearchResponse(
                    query="nothing",
                    results=[],
                )
            )
        )

    async def test_all_params_forwarded_to_client(self, mock: MagicMock) -> None:
        mock.search.return_value = {"query": "q", "results": []}

        tavily = TavilySearchTool(
            client=mock,
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

        mock.search.assert_called_once_with(
            "q",
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

    async def test_none_params_omitted(self, mock: MagicMock) -> None:
        # All optional params default to None and must not be forwarded to the SDK
        # so Tavily applies its own server-side defaults.
        mock.search.return_value = {"query": "q", "results": []}

        tavily = TavilySearchTool(client=mock)
        agent = Agent("a", config=_make_config("q"), tools=[tavily])

        await agent.ask("search")

        mock.search.assert_called_once_with("q")

    async def test_custom_tool_name_in_agent(self, mock: MagicMock) -> None:
        # arrange tool
        mock.search.return_value = {"query": "q", "results": []}
        ddg = TavilySearchTool(client=mock, name="web_search")

        # arrange agent
        config = TrackingConfig(_make_config("AG2 framework", tool_name="web_search"))
        agent = Agent("a", config=config, tools=[ddg])

        # act
        await agent.ask("search")

        # assert tool called
        mock.search.assert_called_once()


@pytest.mark.asyncio
class TestTavilykSearchToolVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        # arrange tool
        mock.search.return_value = SAMPLE_RAW
        tavily = TavilySearchTool(
            client=mock,
            search_depth=Variable("user_depth"),
            topic=Variable(),
        )

        # arrange agent
        agent = Agent(
            "a",
            config=_make_config("test query"),
            tools=[tavily],
            variables={
                "user_depth": "basic",
                "topic": "general",
            },
        )

        # act
        await agent.ask("search")

        # assert variables resolved
        mock.search.assert_called_once_with(
            "test query",
            search_depth="basic",
            topic="general",
        )

    async def test_missing_raises(self, mock: MagicMock) -> None:
        mock.search.return_value = SAMPLE_RAW
        tavily = TavilySearchTool(
            client=mock,
            topic=Variable(),
        )

        agent = Agent("a", config=_make_config("test query"), tools=[tavily])

        with pytest.raises(KeyError, match="topic"):
            await agent.ask("search")
