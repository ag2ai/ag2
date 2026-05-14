# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("tinyfish")

from autogen.beta import Agent, Context, DataInput, Variable
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.search.tinyfish import (
    TinyFishFetchError,
    TinyFishFetchResponse,
    TinyFishFetchResult,
    TinyFishSearchResponse,
    TinyFishSearchResult,
    TinyFishSearchToolkit,
)

TINYFISH_BASE_URL = "https://agent.tinyfish.ai"

SAMPLE_SEARCH_RAW: dict[str, Any] = {
    "query": "AG2 framework",
    "total_results": 2,
    "results": [
        {
            "position": 1,
            "site_name": "ag2.ai",
            "title": "AG2 Framework",
            "snippet": "AG2 is an agent framework.",
            "url": "https://ag2.ai",
        },
        {
            "position": 2,
            "site_name": "github.com",
            "title": "GitHub - AG2",
            "snippet": "Open source repo.",
            "url": "https://github.com/ag2ai/ag2",
        },
    ],
}

SAMPLE_FETCH_RAW: dict[str, Any] = {
    "results": [
        {
            "url": "https://ag2.ai",
            "final_url": "https://ag2.ai/",
            "title": "AG2",
            "description": "Agent framework",
            "language": "en",
            "author": "ag2ai",
            "published_date": "2026-01-01",
            "text": "# AG2\nFull text",
            "format": "markdown",
            "links": ["https://github.com/ag2ai/ag2"],
            "image_links": ["https://ag2.ai/logo.png"],
            "latency_ms": 123.4,
        }
    ],
    "errors": [{"url": "https://bad.example", "error": "target_unreachable"}],
}


def _tool_call_config(
    arguments: dict[str, Any],
    *,
    tool_name: str,
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)])),
        final_reply,
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = TinyFishSearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["tinyfish_search", "tinyfish_fetch"]

    async def test_search_schema_has_query_param(self, context: Context) -> None:
        toolkit = TinyFishSearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "tinyfish_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_fetch_schema_has_urls_param(self, context: Context) -> None:
        toolkit = TinyFishSearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        fetch_schema = next(s for s in schemas if s.function.name == "tinyfish_fetch")

        assert fetch_schema.function.parameters == IsPartialDict({
            "required": ["urls"],
            "properties": IsPartialDict({"urls": IsPartialDict({"type": "array"})}),
        })

    async def test_custom_tool_name_and_description(self, context: Context) -> None:
        toolkit = TinyFishSearchToolkit(api_key="test")
        custom = toolkit.search(name="web_search", description="Custom TinyFish search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "web_search"
        assert schema.function.description == "Custom TinyFish search."


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.get(f"{TINYFISH_BASE_URL}/v1/search").mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = TinyFishSearchToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="tinyfish_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            TinyFishSearchResponse(
                query="AG2 framework",
                total_results=2,
                results=[
                    TinyFishSearchResult(
                        position=1,
                        site_name="ag2.ai",
                        title="AG2 Framework",
                        snippet="AG2 is an agent framework.",
                        url="https://ag2.ai",
                    ),
                    TinyFishSearchResult(
                        position=2,
                        site_name="github.com",
                        title="GitHub - AG2",
                        snippet="Open source repo.",
                        url="https://github.com/ag2ai/ag2",
                    ),
                ],
            )
        )

    @respx.mock
    async def test_search_params_forwarded(self) -> None:
        route = respx.get(f"{TINYFISH_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="tinyfish_search"),
            tools=[toolkit.search(location="US", language="en")],
        )
        await agent.ask("search")

        assert route.calls.last.request.url.params["query"] == "q"
        assert route.calls.last.request.url.params["location"] == "US"
        assert route.calls.last.request.url.params["language"] == "en"

    @respx.mock
    async def test_none_search_params_omitted(self) -> None:
        route = respx.get(f"{TINYFISH_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="tinyfish_search"), tools=[toolkit])
        await agent.ask("search")

        assert set(route.calls.last.request.url.params.keys()) == {"query"}


@pytest.mark.asyncio
class TestFetchExecution:
    @respx.mock
    async def test_returns_structured_results_and_errors(self) -> None:
        route = respx.post(f"{TINYFISH_BASE_URL}/v1/fetch").mock(
            return_value=httpx.Response(200, json=SAMPLE_FETCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")

        config = TrackingConfig(
            _tool_call_config(
                {"urls": ["https://ag2.ai", "https://bad.example"]},
                tool_name="tinyfish_fetch",
            )
        )
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("fetch")

        body = json.loads(route.calls.last.request.content)
        assert body == {"urls": ["https://ag2.ai", "https://bad.example"]}

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            TinyFishFetchResponse(
                results=[
                    TinyFishFetchResult(
                        url="https://ag2.ai",
                        final_url="https://ag2.ai/",
                        title="AG2",
                        description="Agent framework",
                        language="en",
                        author="ag2ai",
                        published_date="2026-01-01",
                        text="# AG2\nFull text",
                        format="markdown",
                        links=["https://github.com/ag2ai/ag2"],
                        image_links=["https://ag2.ai/logo.png"],
                        latency_ms=123.4,
                    )
                ],
                errors=[TinyFishFetchError(url="https://bad.example", error="target_unreachable")],
            )
        )

    @respx.mock
    async def test_fetch_params_forwarded(self) -> None:
        route = respx.post(f"{TINYFISH_BASE_URL}/v1/fetch").mock(
            return_value=httpx.Response(200, json=SAMPLE_FETCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="tinyfish_fetch"),
            tools=[toolkit.fetch(format="html", links=True, image_links=True)],
        )
        await agent.ask("fetch")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "urls": ["https://ag2.ai"],
            "format": "html",
            "links": True,
            "image_links": True,
        }

    @respx.mock
    async def test_toolkit_level_fetch_params_applied_to_default_fetch(self) -> None:
        route = respx.post(f"{TINYFISH_BASE_URL}/v1/fetch").mock(
            return_value=httpx.Response(200, json=SAMPLE_FETCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test", format="markdown", links=True)

        agent = Agent(
            "a", config=_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="tinyfish_fetch"), tools=[toolkit]
        )
        await agent.ask("fetch")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "urls": ["https://ag2.ai"],
            "format": "markdown",
            "links": True,
        }


@pytest.mark.asyncio
class TestTinyFishToolkitVariable:
    @respx.mock
    async def test_search_resolved(self) -> None:
        route = respx.get(f"{TINYFISH_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")
        search_tool = toolkit.search(location=Variable("user_location"), language=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="tinyfish_search"),
            tools=[search_tool],
            variables={"user_location": "US", "language": "en"},
        )
        await agent.ask("search")

        assert route.calls.last.request.url.params["location"] == "US"
        assert route.calls.last.request.url.params["language"] == "en"

    @respx.mock
    async def test_fetch_resolved(self) -> None:
        route = respx.post(f"{TINYFISH_BASE_URL}/v1/fetch").mock(
            return_value=httpx.Response(200, json=SAMPLE_FETCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")
        fetch_tool = toolkit.fetch(format=Variable("fetch_format"), links=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="tinyfish_fetch"),
            tools=[fetch_tool],
            variables={"fetch_format": "markdown", "links": True},
        )
        await agent.ask("fetch")

        body = json.loads(route.calls.last.request.content)
        assert body == {"urls": ["https://ag2.ai"], "format": "markdown", "links": True}

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.get(f"{TINYFISH_BASE_URL}/v1/search").mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = TinyFishSearchToolkit(api_key="test")
        search_tool = toolkit.search(location=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="tinyfish_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="location"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_search_tool_passed_alone(self) -> None:
        route = respx.get(f"{TINYFISH_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="tinyfish_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_fetch_tool_passed_alone(self) -> None:
        route = respx.post(f"{TINYFISH_BASE_URL}/v1/fetch").mock(
            return_value=httpx.Response(200, json=SAMPLE_FETCH_RAW)
        )
        toolkit = TinyFishSearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="tinyfish_fetch"),
            tools=[toolkit.fetch()],
        )
        await agent.ask("fetch")

        assert route.called
