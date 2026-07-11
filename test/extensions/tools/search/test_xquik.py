# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search.xquik import XquikSearchToolkit, XquikTweetSearchResponse
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

XQUIK_BASE_URL = "https://xquik.com"


def _tool_call_config(
    arguments: dict[str, object],
    *,
    tool_name: str = "xquik_tweet_search",
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(arguments=json.dumps(arguments), name=tool_name),
            ]),
        ),
        final_reply,
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schema(self, context: Context) -> None:
        toolkit = XquikSearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        [schema] = schemas

        assert isinstance(schema, FunctionToolSchema)
        assert schema.function.name == "xquik_tweet_search"
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_name_and_description(self, context: Context) -> None:
        toolkit = XquikSearchToolkit(api_key="test")
        custom = toolkit.search(name="search_x", description="Search X posts.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "search_x"
        assert schema.function.description == "Search X posts."


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.get(f"{XQUIK_BASE_URL}/api/v1/x/tweets/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "tweets": [{"id": "123", "text": "AG2 release"}],
                    "has_next_page": True,
                    "next_cursor": "cursor-2",
                },
            )
        )
        toolkit = XquikSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            XquikTweetSearchResponse(
                query="AG2",
                tweets=[{"id": "123", "text": "AG2 release"}],
                has_next_page=True,
                next_cursor="cursor-2",
            )
        )

    @respx.mock
    async def test_forwards_search_defaults_and_auth_header(self) -> None:
        route = respx.get(f"{XQUIK_BASE_URL}/api/v1/x/tweets/search").mock(
            return_value=httpx.Response(200, json={"tweets": []})
        )
        toolkit = XquikSearchToolkit(
            api_key="test-key",
            query_type="Top",
            cursor="cursor-1",
            since_time="2026-07-01T00:00:00Z",
            until_time="2026-07-02T00:00:00Z",
            limit=25,
        )
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {
            "q": "AG2",
            "queryType": "Top",
            "cursor": "cursor-1",
            "sinceTime": "2026-07-01T00:00:00Z",
            "untilTime": "2026-07-02T00:00:00Z",
            "limit": "25",
        }
        assert route.calls.last.request.headers["x-api-key"] == "test-key"

    @respx.mock
    async def test_omits_unset_params(self) -> None:
        route = respx.get(f"{XQUIK_BASE_URL}/api/v1/x/tweets/search").mock(
            return_value=httpx.Response(200, json={"tweets": []})
        )
        toolkit = XquikSearchToolkit(api_key="test")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {"q": "AG2"}

    async def test_empty_api_key_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            XquikSearchToolkit("")


@pytest.mark.asyncio
class TestVariables:
    @respx.mock
    async def test_resolves_runtime_values(self) -> None:
        route = respx.get(f"{XQUIK_BASE_URL}/api/v1/x/tweets/search").mock(
            return_value=httpx.Response(200, json={"tweets": []})
        )
        toolkit = XquikSearchToolkit(api_key="test")
        search_tool = toolkit.search(query_type=Variable(), limit=Variable("result_limit"))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[search_tool],
            variables={"query_type": "Latest", "result_limit": 10},
        )

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {
            "q": "AG2",
            "queryType": "Latest",
            "limit": "10",
        }
