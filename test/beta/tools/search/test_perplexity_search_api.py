# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("perplexity")

from autogen.beta import Agent, DataInput, Variable
from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.search.perplexity_search_api import (
    PerplexitySearchAPITool,
    SearchResponse,
    SearchResult,
)

SAMPLE_RAW = SimpleNamespace(
    id="search-xxx",
    results=[
        SimpleNamespace(
            title="AG2 Framework",
            url="https://ag2.ai",
            snippet="AG2 is an agent framework.",
            date="2026-01-01",
        ),
        SimpleNamespace(
            title="GitHub - AG2",
            url="https://github.com/ag2ai/ag2",
            snippet="Open source repo.",
            date=None,
        ),
    ],
)


def _make_config(query: str, final_reply: str = "done", tool_name: str = "perplexity_search_api") -> TestConfig:
    call = ToolCallEvent(arguments=json.dumps({"query": query}), name=tool_name)
    return TestConfig(ModelResponse(tool_calls=ToolCallsEvent([call])), final_reply)


def _empty_response() -> SimpleNamespace:
    return SimpleNamespace(id="search-empty", results=[])


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schema(self, context: ConversationContext) -> None:
        perp = PerplexitySearchAPITool(client=MagicMock())

        [schema] = await perp.schemas(context)

        assert schema.function.name == "perplexity_search_api"
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_schema(self, context: ConversationContext) -> None:
        perp = PerplexitySearchAPITool(
            client=MagicMock(), name="my_search", description="Custom search tool."
        )

        [schema] = await perp.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_search_returns_structured_results(self, mock: MagicMock) -> None:
        mock.search.create.return_value = SAMPLE_RAW
        perp = PerplexitySearchAPITool(client=mock)

        config = TrackingConfig(_make_config("AG2 framework"))
        agent = Agent("a", config=config, tools=[perp])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(
            SearchResponse(
                query="AG2 framework",
                results=[
                    SearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        date="2026-01-01",
                    ),
                    SearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet="Open source repo.",
                        date=None,
                    ),
                ],
            )
        )

    async def test_search_empty_results(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_response()
        perp = PerplexitySearchAPITool(client=mock)

        config = TrackingConfig(_make_config("nothing"))
        agent = Agent("a", config=config, tools=[perp])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [tool_result] = tool_results_event.results
        [part] = tool_result.result.parts
        assert part == DataInput(SearchResponse(query="nothing", results=[]))

    async def test_all_params_forwarded_to_client(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_response()

        perp = PerplexitySearchAPITool(
            client=mock,
            max_results=5,
            max_tokens_per_page=512,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_recency_filter="week",
            search_after_date_filter="1/1/2025",
            search_before_date_filter="12/31/2025",
        )
        agent = Agent("a", config=_make_config("q"), tools=[perp])

        await agent.ask("search")

        mock.search.create.assert_called_once_with(
            query="q",
            max_results=5,
            max_tokens_per_page=512,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_recency_filter="week",
            search_after_date_filter="1/1/2025",
            search_before_date_filter="12/31/2025",
        )

    async def test_none_params_omitted(self, mock: MagicMock) -> None:
        # Optional params default to None and must not be forwarded so the API
        # uses its own server-side defaults.
        mock.search.create.return_value = _empty_response()

        perp = PerplexitySearchAPITool(client=mock)
        agent = Agent("a", config=_make_config("q"), tools=[perp])

        await agent.ask("search")

        mock.search.create.assert_called_once_with(query="q")

    async def test_custom_tool_name_in_agent(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_response()
        perp = PerplexitySearchAPITool(client=mock, name="web_search")

        config = TrackingConfig(_make_config("AG2 framework", tool_name="web_search"))
        agent = Agent("a", config=config, tools=[perp])

        await agent.ask("search")

        mock.search.create.assert_called_once()


@pytest.mark.asyncio
class TestPerplexitySearchAPIToolVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_response()
        perp = PerplexitySearchAPITool(
            client=mock,
            max_results=Variable("user_max"),
            search_recency_filter=Variable(),
        )

        agent = Agent(
            "a",
            config=_make_config("test query"),
            tools=[perp],
            variables={
                "user_max": 7,
                "search_recency_filter": "day",
            },
        )

        await agent.ask("search")

        mock.search.create.assert_called_once_with(
            query="test query",
            max_results=7,
            search_recency_filter="day",
        )

    async def test_missing_raises(self, mock: MagicMock) -> None:
        mock.search.create.return_value = _empty_response()
        perp = PerplexitySearchAPITool(client=mock, max_results=Variable())

        agent = Agent("a", config=_make_config("test query"), tools=[perp])

        with pytest.raises(KeyError, match="max_results"):
            await agent.ask("search")
