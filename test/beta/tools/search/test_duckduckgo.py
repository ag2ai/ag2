# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsJson, IsPartialDict

pytest.importorskip("ddgs")
final_reply: str = "done"
from autogen.beta import Agent, Variable
from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from autogen.beta.events.types import ModelResponse
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.search.duckduckgo import DuckDuckSearchTool, SearchResponse, SearchResult

SAMPLE_RESULTS = [
    {"title": "AG2 Framework", "href": "https://ag2.ai", "body": "AG2 is an agent framework."},
    {"title": "GitHub - AG2", "href": "https://github.com/ag2ai/ag2", "body": "Open source repo."},
]


def _make_config(
    query: str,
    final_reply: str = "done",
    tool_name: str = "duckduckgo_search",
) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(
                    arguments=json.dumps({"query": query}),
                    name=tool_name,
                )
            ]),
        ),
        final_reply,
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schema(self, context: ConversationContext) -> None:
        ddg = DuckDuckSearchTool(client=MagicMock())

        [schema] = await ddg.schemas(context)

        assert schema.function.name == "duckduckgo_search"
        assert schema.function.parameters == IsPartialDict({
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
            "required": ["query"],
        })

    async def test_custom_schema(self, context: ConversationContext) -> None:
        ddg = DuckDuckSearchTool(client=MagicMock(), name="my_search", description="Custom search tool.")

        [schema] = await ddg.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."
        assert schema.function.parameters == IsPartialDict({
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
            "required": ["query"],
        })


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_search_returns_structured_results(self, mock: MagicMock) -> None:
        # arrange tool
        mock.text.return_value = SAMPLE_RESULTS
        ddg = DuckDuckSearchTool(client=mock)

        # arrange agent
        config = TrackingConfig(_make_config("AG2 framework"))
        agent = Agent("a", config=config, tools=[ddg])

        # act
        await agent.ask("search")

        # assert tool result
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].content == IsJson(
            asdict(
                SearchResponse(
                    query="AG2 framework",
                    results=[
                        SearchResult(title="AG2 Framework", href="https://ag2.ai", body="AG2 is an agent framework."),
                        SearchResult(
                            title="GitHub - AG2", href="https://github.com/ag2ai/ag2", body="Open source repo."
                        ),
                    ],
                )
            )
        )

    async def test_search_empty_results(self, mock: MagicMock) -> None:
        # arrange tool
        mock.text.return_value = []
        ddg = DuckDuckSearchTool(client=mock)

        # arrange agent
        config = TrackingConfig(_make_config("nonexistent query"))
        agent = Agent("a", config=config, tools=[ddg])

        # act
        await agent.ask("search")

        # assert tool result
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        config = TrackingConfig(_make_config("AG2 framework"))
        agent = Agent("a", config=config, tools=[ddg])

        # act
        await agent.ask("search")

        #
        assert tool_results_event.results[0].content == IsJson(
            asdict(
                SearchResponse(
                    query="nonexistent query",
                    results=[],
                )
            )
        )

    async def test_custom_client_used(self, mock: MagicMock) -> None:
        mock.text.return_value = SAMPLE_RESULTS
        ddg = DuckDuckSearchTool(client=mock, max_results=3, region="us-en", safesearch="off")

        agent = Agent("a", config=_make_config("test query"), tools=[ddg])

        await agent.ask("search")

        mock.text.assert_called_once_with(
            "test query",
            region="us-en",
            safesearch="off",
            max_results=3,
        )

    async def test_custom_tool_name_in_agent(self, mock: MagicMock) -> None:
        # arrange tool
        mock.text.return_value = SAMPLE_RESULTS
        ddg = DuckDuckSearchTool(client=mock, name="web_search")

        # arrange agent
        config = TrackingConfig(_make_config("AG2 framework", tool_name="web_search"))
        agent = Agent("a", config=config, tools=[ddg])

        # act
        await agent.ask("search")

        # assert tool called
        mock.text.assert_called_once()


@pytest.mark.asyncio
class TestDuckDuckSearchToolVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        # arrange tool
        mock.text.return_value = SAMPLE_RESULTS
        ddg = DuckDuckSearchTool(
            client=mock,
            region=Variable("user_region"),
            safesearch=Variable(),
        )

        # arrange agent
        agent = Agent(
            "a",
            config=_make_config("test query"),
            tools=[ddg],
            variables={
                "user_region": "us-en",
                "safesearch": "off",
            },
        )

        # act
        await agent.ask("search")

        # assert variables resolved
        mock.text.assert_called_once_with(
            "test query",
            region="us-en",
            safesearch="off",
            max_results=5,
        )

    async def test_missing_raises(self, mock: MagicMock) -> None:
        mock.text.return_value = SAMPLE_RESULTS
        ddg = DuckDuckSearchTool(
            client=mock,
            safesearch=Variable(),
        )

        agent = Agent("a", config=_make_config("test query"), tools=[ddg])

        with pytest.raises(KeyError, match="safesearch"):
            await agent.ask("search")
